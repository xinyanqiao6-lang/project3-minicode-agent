"""Agent 循环：Planner-Action-Observation + 错误反馈自愈闭环。

核心流程（面试讲法）：
    while 未结束 and 未超最大轮数:
        1. Planner：把「任务 + 历史观察」拼成 prompt，让模型输出结构化 JSON
           （thought / action / args 或 final）
        2. Parser：解析模型输出，容错（截取 JSON、去代码围栏、补全）
        3. Action：调用 tools.call_tool 执行工具
        4. Observation：把工具结果（含报错）追加到历史，作为下一轮的反馈
        5. 判断：模型返回 final 或达到护栏上限则停止

「自愈」体现在：run_python 失败时返回的 stderr/returncode 会原样喂回模型，
模型据此生成修复动作（edit_file 改代码 → 再 run_python），形成闭环。
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from . import config, llm_client, tools

logger = logging.getLogger("minicode.agent")


# ---------- 系统提示词：约束模型输出 JSON + 定义工具 ----------

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
"""


# ---------- 结果模型 ----------

@dataclass
class AgentStep:
    type: str          # "action" | "final"
    thought: str = ""
    action: str = ""   # action 时的工具名
    args: dict = field(default_factory=dict)
    content: str = ""  # final 时的最终回答


@dataclass
class AgentResult:
    success: bool
    final_answer: str
    steps: list[dict] = field(default_factory=list)
    iterations: int = 0
    error: str = ""


# ---------- JSON 解析（容错） ----------

def parse_model_output(raw: str) -> AgentStep:
    """把模型原始输出解析成 AgentStep。

    容错点（真实模型输出常带杂讯）：
    1. 去掉 ```json ... ``` 代码围栏；
    2. 截取第一个 { 到最后一个 } 之间的 JSON；
    3. 解析失败时返回一个空 action，交由上层降级处理。
    """
    text = raw.strip()
    # 去围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 截取 JSON 主体
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return AgentStep(type="final", content=text or "（模型未返回有效 JSON）")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return AgentStep(type="final", content=text[start:end + 1][:200] + "（JSON 解析失败）")

    t = obj.get("type", "action")
    if t == "final":
        return AgentStep(type="final", content=str(obj.get("content", "")))
    return AgentStep(
        type="action",
        thought=str(obj.get("thought", "")),
        action=str(obj.get("action", "")),
        args=obj.get("args") if isinstance(obj.get("args"), dict) else {},
    )


# ---------- Agent 主循环 ----------

async def run(task: str, max_iterations: int | None = None) -> AgentResult:
    """执行一次 Agent 任务，返回最终结果与完整轨迹。"""
    max_iter = max_iterations or config.MAX_ITERATIONS
    system = build_system_prompt()
    history: list[dict] = [{"role": "system", "content": system}]
    steps: list[dict] = []

    # 首轮：只给任务，不带观察
    history.append({"role": "user", "content": f"任务：{task}"})

    for i in range(1, max_iter + 1):
        # 1) Planner：调 LLM 拿结构化输出
        try:
            raw = await llm_client.chat_completion(history)
        except llm_client.UpstreamUnavailableError as e:
            return AgentResult(False, "", steps, i, f"上游不可用：{e}")

        # 2) Parser
        step = parse_model_output(raw)
        logger.info("[轮 %d] type=%s action=%s", i, step.type, step.action)

        if step.type == "final":
            steps.append({"type": "final", "content": step.content})
            return AgentResult(True, step.content, steps, i)

        # 3) Action
        ok, obs = tools.call_tool(step.action, step.args)
        steps.append({
            "type": "action",
            "thought": step.thought,
            "action": step.action,
            "args": step.args,
            "ok": ok,
            "observation": obs,
        })

        # 4) Observation：把工具结果（含报错）作为 assistant + user 反馈，形成自愈闭环
        history.append({"role": "assistant", "content": raw})
        history.append({"role": "user", "content": f"工具执行结果（{'成功' if ok else '失败'}）：\n{obs}\n\n请继续：调用下一个工具，或输出 final 结束。"})

    # 达到最大轮数仍未 final
    return AgentResult(False, "", steps, max_iter, f"达到最大轮数 {max_iter}，任务未完成")


# ---------- 便捷入口 ----------

def run_sync(task: str, max_iterations: int | None = None) -> AgentResult:
    """同步入口，便于脚本 / 测试直接调用。"""
    return asyncio.run(run(task, max_iterations))
