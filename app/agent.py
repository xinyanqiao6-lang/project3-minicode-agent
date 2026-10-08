"""Agent 循环 v2：Planner-Action-Observation + 自愈闭环 + **结果校验步骤**。

## 为什么要加结果校验（v1 的最大缺陷）

v1 实测发现：5 个 bug 里 Agent 有 **5 次都自报「完成」，但实际只有 3 次真修对**
——幻觉式汇报率 40%。v1 只能靠**人工**在事后跑脚本判定，Agent 自己不知道。

v2 的做法：把「用事实判定」这条原则**内建进 Agent 循环**——
当模型想输出 final 时，先不采信，而是调用一个**验证器（verifier）**去实际执行
并检查结果；验证不通过就把失败原因喂回模型，让它继续修。

    模型想 final
        │
        ├─ 无 verifier（v1 行为）→ 直接采信 ← 这就是幻觉的来源
        │
        └─ 有 verifier（v2）
             ├─ verifier() 通过 → 采信 final
             └─ verifier() 失败 → 把「实际输出 vs 期望输出」喂回模型 → 继续修

这正是 Coding Agent 工程里最容易被忽视、也最致命的一点：
**如果信了模型的"已完成"，上线就是事故。**

## 另一个改进：token 成本记账

每轮记录 prompt/completion token，Trace 可汇总出单任务成本。
没有成本数据的 Agent 评测是无意义的（换个模型就换算不出收益）。
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from . import config, llm_client, tools
from .trace import StepTrace, Trace

logger = logging.getLogger("minicode.agent")

# 观察结果送回模型时的最大长度——太长会挤爆上下文，太短会丢关键信息
MAX_OBS_CHARS = 1200


# ---------- 系统提示词 ----------

def build_system_prompt() -> str:
    tool_desc = "\n".join(
        f"- {name}({', '.join(params)}): {desc}"
        for name, (_fn, params, desc) in tools.TOOLS.items()
    )
    return f"""你是一个轻量级 Coding Agent。你的任务是在一个沙箱 workspace 内通过调用工具完成任务。

可用工具：
{tool_desc}

输出规则（严格遵守，每次只输出一个 JSON 对象，不要输出其他文字）：
1. 如果需要调用工具，输出：
   {{"type": "action", "thought": "你的思考", "action": "工具名", "args": {{...}}}}
2. 如果任务已经完成，输出：
   {{"type": "final", "content": "最终回答（总结你做了什么、结果如何）"}}

约束：
- 每次只调用一个工具；
- 工具执行结果会作为「观察」反馈给你，若报错请根据错误信息修复后重试；
- 运行 Python 前先 write_file 写好代码；
- 不要臆测文件内容，先 read_file / list_files 看清楚再动手；
- 不要「脑补」结果：只有当 run_python 的观察里明确打印出你声称的数值时，才能在 final 里引用该数值；否则只能陈述你实际做了什么、看到了什么。
- 你的 final 会被**自动验证器**核对。如果验证不通过，你会收到实际输出与期望输出的对比，
  此时必须继续修复，而不是重复宣告完成。
"""


# ---------- 结果模型 ----------

@dataclass
class AgentResult:
    success: bool
    final_answer: str
    steps: list[dict] = field(default_factory=list)
    iterations: int = 0
    error: str = ""
    # ---- v2 新增 ----
    verify_enabled: bool = False    # 是否启用了结果校验步骤
    verify_attempts: int = 0        # 结果校验被触发了几次
    verify_rejected: int = 0        # 其中有多少次验证失败（把 Agent 打回去继续修）
    claimed_done: bool = False      # 模型是否曾宣告完成（事实是否成立由评测器判定）
    total_tokens: int = 0
    elapsed_s: float = 0.0
    stop_reason: str = ""
    trace: "Trace | None" = None    # 完整执行轨迹

    # 注：这里**故意不提供** `hallucinated` 属性。
    # 幻觉 = "模型自报完成" ∧ "事实不成立"，而后半个条件 Agent 自己无从知晓——
    # 它没有独立于自己的判定器。把它放在 Agent 层会诱导人用 success 当真相，
    # 而 success 在无 verifier 时恰恰等于"模型说了算"。
    # 正确做法：由评测器在 Agent 结束后独立执行脚本判定（见 eval/runner.py）。


# ---------- JSON 解析（容错） ----------

def parse_model_output(raw: str) -> "tuple[str, dict, str]":
    """把模型输出解析成 (type, payload, 原始文本)。容错点同 v1。

    额外返回 raw 原文——v2 的结果校验步骤需要用**模型自己的输出**作为
    assistant 消息回填历史，保持上下文连贯（而不是伪造一段助手消息）。
    """
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "final", {"content": text or "（模型未返回有效 JSON）"}, raw or ""
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return "final", {"content": text[start:end + 1][:200] + "（JSON 解析失败）"}, raw or ""

    t = obj.get("type", "action")
    if t == "final":
        return "final", {"content": str(obj.get("content", ""))}, raw or ""
    return "action", {
        "thought": str(obj.get("thought", "")),
        "action": str(obj.get("action", "")),
        "args": obj.get("args") if isinstance(obj.get("args"), dict) else {},
    }, raw or ""


# ---------- Agent 主循环（v2） ----------

async def run(
    task: str,
    max_iterations: int | None = None,
    verifier: Callable[[], tuple[bool, str]] | None = None,
) -> AgentResult:
    """执行一次 Agent 任务。

    参数
    ----
    verifier : 可选的验证器，无参可调用，返回 (是否通过, 详情文本)。
               传入后启用「结果校验步骤」：模型宣告完成时先做事实校验，
               不通过则把实际 vs 期望的差异喂回模型继续修。
               这是 v2 对 v1「幻觉式汇报」问题的核心修复。
    """
    max_iter = max_iterations or config.MAX_ITERATIONS
    system = build_system_prompt()
    history: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"任务：{task}"},
    ]
    result = AgentResult(success=False, final_answer="", verify_enabled=verifier is not None)
    trace = Trace(task=task, model=config.MODEL, verify_enabled=verifier is not None)
    t_start = time.perf_counter()

    for i in range(1, max_iter + 1):
        # 1) Planner
        t0 = time.perf_counter()
        try:
            raw, usage = await llm_client.chat_completion_with_usage(history)
        except llm_client.UpstreamUnavailableError as e:
            result.error = f"上游不可用：{e}"
            result.stop_reason = "upstream_error"
            result.iterations = i
            break
        llm_ms = (time.perf_counter() - t0) * 1000

        # 2) Parser
        stype, payload, raw_text = parse_model_output(raw)
        result.total_tokens += int(usage.get("total_tokens", 0) or 0)

        if stype == "final":
            result.claimed_done = True

            # ★ 结果校验步骤：先验事实，再采信 final ★
            if verifier is not None:
                result.verify_attempts += 1
                try:
                    v_ok, v_detail = verifier()
                except Exception as e:      # 验证器自身出错也不能放过
                    v_ok, v_detail = False, f"验证器异常：{e}"
                if not v_ok:
                    result.verify_rejected += 1
                    logger.info("[轮 %d] 结果校验未通过，打回继续修", i)
                    step = StepTrace(
                        index=i, type="final", ok=False, llm_ms=llm_ms,
                        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                        observation=v_detail[:MAX_OBS_CHARS], verified=False,
                    )
                    trace.add(step)
                    history.append({"role": "assistant", "content": raw_text})
                    history.append({
                        "role": "user",
                        "content": (
                            f"❌ 自动验证未通过。实际执行结果：\n{v_detail}\n\n"
                            f"请根据上面的实际输出继续修复，不要重复宣告完成。"
                        ),
                    })
                    continue
                # 验证通过，采信
                step = StepTrace(
                    index=i, type="final", ok=True, llm_ms=llm_ms,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    observation="（结果校验通过）", verified=True,
                )
                trace.add(step)
                result.steps.append({"type": "final", "content": payload["content"]})
                result.success = True
                result.final_answer = payload["content"]
                result.iterations = i
                result.stop_reason = "verified_final" if verifier else "final"
                break

            # 无 verifier：v1 行为，直接采信（保留以便做对照实验）
            trace.add(StepTrace(
                index=i, type="final", ok=True, llm_ms=llm_ms,
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                observation="（未启用结果校验）", verified=None,
            ))
            result.steps.append({"type": "final", "content": payload["content"]})
            result.success = True
            result.final_answer = payload["content"]
            result.iterations = i
            result.stop_reason = "final"
            break

        # 3) Action
        action, args, thought = payload["action"], payload["args"], payload["thought"]
        logger.info("[轮 %d] action=%s", i, action)
        t1 = time.perf_counter()
        ok, obs = tools.call_tool(action, args)
        tool_ms = (time.perf_counter() - t1) * 1000

        step = StepTrace(
            index=i, type="action", action=action, args=args, ok=ok,
            observation=obs, llm_ms=llm_ms, tool_ms=tool_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )
        trace.add(step)
        result.steps.append({
            "type": "action", "thought": thought, "action": action,
            "args": args, "ok": ok, "observation": obs,
        })

        # 4) Observation → 自愈闭环
        history.append({"role": "assistant", "content": raw_text})
        history.append({
            "role": "user",
            "content": (
                f"工具执行结果（{'成功' if ok else '失败'}）：\n{obs[:MAX_OBS_CHARS]}\n\n"
                f"请继续：调用下一个工具，或输出 final 结束。"
            ),
        })
    else:
        result.error = f"达到最大轮数 {max_iter}，任务未完成"
        result.stop_reason = "max_iterations"
        result.iterations = max_iter

    result.elapsed_s = time.perf_counter() - t_start
    result.total_tokens = trace.total_tokens
    trace.elapsed_s = result.elapsed_s
    trace.final_answer = result.final_answer
    trace.stop_reason = result.stop_reason
    trace.verify_attempts = result.verify_attempts
    result.trace = trace          # 挂上便于调用方保存/展示
    return result


def run_sync(task: str, max_iterations: int | None = None,
             verifier: Callable[[], tuple[bool, str]] | None = None) -> AgentResult:
    """同步入口，便于脚本 / 测试直接调用。"""
    return asyncio.run(run(task, max_iterations, verifier))