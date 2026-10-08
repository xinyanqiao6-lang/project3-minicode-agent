"""执行轨迹（Trace）：记录 Agent 每一步的 action / 耗时 / token / 观察结果。

为什么需要 trace：
1. **bad case 可定位**——"它为什么修错了"如果只看 final 文本永远说不清，
   必须能看到它当时看到了什么、动了什么；
2. **成本可量化**——单任务花了多少 token / 多少钱，是评估 Agent 可用性的前提；
3. **回归可比对**——优化 Prompt 前后要有同口径的轨迹才能对比。

设计：Trace 只是一个数据容器 + 几个统计方法，不依赖 LLM，可离线单测。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field


@dataclass
class StepTrace:
    """一步执行的完整记录。"""
    index: int                      # 第几轮（从 1 开始）
    type: str                       # "action" | "final"
    action: str = ""                # 工具名
    args: dict = field(default_factory=dict)
    ok: bool = False                # 工具是否成功
    observation: str = ""           # 工具返回给模型的观察（截断）
    llm_ms: float = 0.0             # 本轮 LLM 调用耗时
    tool_ms: float = 0.0            # 本轮工具执行耗时
    prompt_tokens: int = 0
    completion_tokens: int = 0
    verified: bool | None = None    # 结果校验是否通过（仅 final 轮有值）

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Trace:
    """一次 Agent 任务的完整轨迹。"""
    task: str
    model: str = ""
    steps: list[StepTrace] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    final_answer: str = ""
    stop_reason: str = ""            # final / max_iterations / upstream_error / verify_failed
    verify_enabled: bool = False     # 是否启用了结果校验步骤
    verify_attempts: int = 0        # 结果校验被触发的次数

    # ---------- 统计 ----------
    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def action_steps(self) -> list[StepTrace]:
        return [s for s in self.steps if s.type == "action"]

    def tool_usage(self) -> dict[str, int]:
        """各工具被调用次数——用来发现"模型其实只会 write_file+run_python"。"""
        out: dict[str, int] = {}
        for s in self.action_steps:
            out[s.action] = out.get(s.action, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def add(self, step: StepTrace) -> None:
        self.steps.append(step)

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "model": self.model,
            "steps": [asdict(s) for s in self.steps],
            "elapsed_s": round(self.elapsed_s, 2),
            "final_answer": self.final_answer,
            "stop_reason": self.stop_reason,
            "verify_enabled": self.verify_enabled,
            "verify_attempts": self.verify_attempts,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_usage": self.tool_usage(),
        }

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def render(self, max_obs: int = 160) -> str:
        """人类可读的轨迹文本，面试时可以直接展示。"""
        lines = [f"任务：{self.task}",
                 f"模型：{self.model} | 轮数：{len(self.steps)} | "
                 f"token：{self.total_tokens} | 耗时：{self.elapsed_s:.1f}s",
                 f"工具调用：{self.tool_usage()}",
                 "-" * 60]
        for s in self.steps:
            if s.type == "final":
                mark = "✓" if s.verified is not False else "✗校验未过"
                lines.append(f"[{s.index}] {mark} final: {s.final_answer if hasattr(s,'final_answer') else ''}")
            else:
                obs = (s.observation or "").replace("\n", " ")[:max_obs]
                lines.append(
                    f"[{s.index}] {'ok ' if s.ok else 'ERR'} {s.action}"
                    f"  (llm {s.llm_ms:.0f}ms / tool {s.tool_ms:.0f}ms"
                    f" / {s.total_tokens} tok)  → {obs}"
                )
        return "\n".join(lines)