"""MiniCode 核心单元测试（v2）。

用「脚本化的假模型」（monkeypatch chat_completion_with_usage）驱动 Agent 循环，
验证 Planner-Action-Observation、自愈闭环、**结果校验步骤**，
全部不依赖真实 API Key。

注意：v2 起模型调用改为返回 (text, usage) 以便记账 token，
测试里的假模型也必须返回同样结构——这是有意的接口约束，
避免"测试能过但生产拿不到 usage"。
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import agent, tools


# ---------- 测试工具：伪造模型输出 ----------

def fake_model_script(responses: list[str]):
    """返回一个按顺序吐出固定 JSON 的 async 假模型。

    v2：返回 (text, usage) 二元组，与真实调用同构。
    """
    idx = 0

    async def _fake(messages, temperature=None, max_tokens=None):
        nonlocal idx
        if idx >= len(responses):
            return (json.dumps({"type": "final", "content": "脚本耗尽，结束"}),
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        r = responses[idx]
        idx += 1
        return r, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    return _fake


def monkeypatch_model(monkeypatch, responses):
    fake = fake_model_script(responses)
    monkeypatch.setattr(agent.llm_client, "chat_completion_with_usage", fake)
    return fake


def monkeypatch_workspace(d):
    import app.config as config
    config.WORKSPACE = d


# ---------- 1. JSON 解析容错（v2 改为三元组返回） ----------

def test_parse_plain_action():
    t, payload, _ = agent.parse_model_output(
        '{"type":"action","thought":"查看文件","action":"list_files","args":{"path":""}}')
    assert t == "action"
    assert payload["action"] == "list_files"


def test_parse_fenced_json():
    """模型常输出 ```json ... ``` 围栏，需能剥掉。"""
    raw = '```json\n{"type":"final","content":"完成"}\n```'
    t, payload, _ = agent.parse_model_output(raw)
    assert t == "final"
    assert payload["content"] == "完成"


def test_parse_with_noise():
    """模型在 JSON 前后带解释文字，需截取主体。"""
    raw = '好的，我来操作：{"type":"action","action":"read_file","args":{"name":"a.py"}} 就这个'
    t, payload, _ = agent.parse_model_output(raw)
    assert payload["action"] == "read_file"


def test_parse_invalid_fallback():
    """解析失败降级为 final，不抛异常。"""
    t, payload, _ = agent.parse_model_output("这不是 JSON")
    assert t == "final"


def test_parse_returns_raw_text():
    """v2 需要把模型原始输出回填历史，保持上下文连贯。"""
    raw = '{"type":"final","content":"ok"}'
    _, _, original = agent.parse_model_output(raw)
    assert original == raw


# ---------- 2. 工具层 ----------

def test_write_and_read_file():
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        ok, _ = tools.write_file("a.py", "print('hi')")
        assert ok
        ok, out = tools.read_file("a.py")
        assert ok and "print" in out


def test_run_python_success():
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        tools.write_file("ok.py", "print(1+1)")
        ok, out = tools.run_python("ok.py")
        assert ok and "2" in out


def test_run_python_error_feedback():
    """执行报错时，stderr 要能反馈给模型（自愈闭环的关键）。"""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        tools.write_file("bad.py", "print(1/0)")
        ok, out = tools.run_python("bad.py")
        assert not ok
        assert "ZeroDivisionError" in out


def test_path_escape_blocked():
    """沙箱护栏：不能写到 workspace 之外。"""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        ok, _ = tools.write_file("../escape.py", "x=1")
        assert not ok


def test_command_whitelist():
    """命令白名单：危险命令必须被拦截。"""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        ok, _ = tools.run_command("rm -rf /")
        assert not ok


# ---------- 3. Agent 自愈闭环（核心） ----------

def test_agent_self_healing_loop(monkeypatch):
    """模拟：写 bug 代码 -> 执行报错 -> 修复 -> 再执行成功 -> final。"""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        script = [
            json.dumps({"type": "action", "thought": "先写代码", "action": "write_file",
                        "args": {"name": "bad.py", "content": "print(1/0)"}}),
            json.dumps({"type": "action", "thought": "运行看看", "action": "run_python",
                        "args": {"name": "bad.py"}}),
            json.dumps({"type": "action", "thought": "报错了，修复", "action": "edit_file",
                        "args": {"name": "bad.py", "old": "print(1/0)", "new": "print(1+1)"}}),
            json.dumps({"type": "action", "thought": "再运行", "action": "run_python",
                        "args": {"name": "bad.py"}}),
            json.dumps({"type": "final", "content": "已完成：修复 ZeroDivisionError 后输出 2"}),
        ]
        monkeypatch_model(monkeypatch, script)
        result = agent.run_sync("写个能正确运行的脚本")

        assert result.success
        assert result.final_answer == "已完成：修复 ZeroDivisionError 后输出 2"
        action_steps = [s for s in result.steps if s["type"] == "action"]
        assert [s["action"] for s in action_steps] == [
            "write_file", "run_python", "edit_file", "run_python"]
        assert [s["ok"] for s in action_steps if s["action"] == "run_python"] == [False, True]


def test_agent_max_iteration_guard(monkeypatch):
    """模型一直输出 action 不 final，应被最大轮数护栏终止。"""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        script = [json.dumps({"type": "action", "action": "list_files", "args": {}})] * 20
        monkeypatch_model(monkeypatch, script)
        result = agent.run_sync("随便", max_iterations=5)
        assert not result.success
        assert result.iterations == 5
        assert "最大轮数" in result.error
        assert result.stop_reason == "max_iterations"


def test_agent_counts_tokens(monkeypatch):
    """v2 新增：token 成本必须可记账，否则评测无意义。"""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        monkeypatch_model(monkeypatch, [
            json.dumps({"type": "action", "action": "list_files", "args": {}}),
            json.dumps({"type": "final", "content": "done"}),
        ])
        result = agent.run_sync("看看文件", max_iterations=5)
        assert result.total_tokens > 0
        assert result.trace is not None
        assert result.trace.total_tokens == result.total_tokens


# ---------- 4. 结果校验步骤（v2 核心新增） ----------

def test_verifier_blocks_false_claim(monkeypatch):
    """**本项目最重要的测试**：模型谎报完成时，验证器必须拦住它。

    场景：模型写了个错代码，然后立刻 final 宣称"已修复"。
    没有 verifier 时这就是一次幻觉式汇报（v1 实测率 40%）；
    有了 verifier，Agent 应该被打回去继续修。
    """
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        # 第 1 轮：写错误代码；第 2 轮：直接 final 谎报完成；第 3 轮：真正修好；第 4 轮：final
        script = [
            json.dumps({"type": "action", "action": "write_file",
                        "args": {"name": "t.py", "content": "print(1/0)"}}),
            json.dumps({"type": "final", "content": "已修复，输出 2"}),   # ← 谎报
            json.dumps({"type": "action", "action": "edit_file",
                        "args": {"name": "t.py", "old": "print(1/0)", "new": "print(1+1)"}}),
            json.dumps({"type": "final", "content": "现在真的修好了，输出 2"}),
        ]
        monkeypatch_model(monkeypatch, script)

        calls = {"n": 0}

        def _verify():
            calls["n"] += 1
            ok, _ = tools.run_python("t.py")
            return ok, "实际输出: 2" if ok else "实际输出: ZeroDivisionError"

        result = agent.run_sync("修好 t.py", verifier=_verify)

        assert result.verify_enabled
        assert result.verify_attempts >= 2          # 谎报被拦下，重验一次
        assert result.verify_rejected >= 1         # 第一次 final 未通过验证
        assert result.success                       # 最终真的修好了
        assert result.stop_reason == "verified_final"


def test_verifier_passes_and_agent_stops(monkeypatch):
    """验证通过时 Agent 应正常结束，不多消耗轮数。"""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        monkeypatch_model(monkeypatch, [
            json.dumps({"type": "action", "action": "write_file",
                        "args": {"name": "ok.py", "content": "print(42)"}}),
            json.dumps({"type": "final", "content": "写好了"}),
        ])
        result = agent.run_sync("写 ok.py", verifier=lambda: (True, "输出 42"))
        assert result.success
        assert result.verify_rejected == 0
        assert result.iterations == 2


def test_agent_claims_done_without_verifier(monkeypatch):
    """不开 verifier 时，谎报不会被拦——这正是 v1 的 40% 幻觉来源。

    注意断言的是**可观测事实**：Agent 采信了模型的 final（claimed_done=True、
    verify_enabled=False）。而"这次宣称是否是真是假"，Agent 自己不知道，
    必须由评测器事后独立执行脚本判定——这也是为什么 `AgentResult` 上
    刻意不提供 `hallucinated` 属性。
    """
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        monkeypatch_model(monkeypatch, [
            json.dumps({"type": "action", "action": "write_file",
                        "args": {"name": "x.py", "content": "print(1/0)"}}),
            json.dumps({"type": "final", "content": "搞定了"}),
        ])
        result = agent.run_sync("修 x.py")
        assert result.claimed_done is True        # 模型确实宣称完成了
        assert result.verify_enabled is False     # 但没有任何东西校验它
        assert result.verify_attempts == 0
        assert not hasattr(result, "hallucinated")  # 真相由评测器判定，不在 Agent 层
        # 外部事实判定会看出它是假的：脚本仍然会抛 ZeroDivisionError
        ok, detail = tools.run_python("x.py")
        assert ok is False and "ZeroDivisionError" in detail
        assert result.verify_enabled is False


def test_verifier_exception_does_not_crash(monkeypatch):
    """验证器自身抛异常时不能放行（fail-closed），否则等于没有校验。"""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        monkeypatch_model(monkeypatch, [
            json.dumps({"type": "final", "content": "done"}),
        ] * 3)

        def _boom():
            raise RuntimeError("验证器炸了")

        result = agent.run_sync("任务", max_iterations=3, verifier=_boom)
        # 3 次 final 都被验证器拦下，最终因超轮数结束
        assert result.verify_rejected == 3
        assert not result.success


# ---------- 5. Trace ----------

def test_trace_records_tool_usage():
    from app.trace import StepTrace, Trace
    tr = Trace(task="t", model="m")
    tr.add(StepTrace(index=1, type="action", action="write_file", ok=True,
                     prompt_tokens=10, completion_tokens=5))
    tr.add(StepTrace(index=2, type="action", action="run_python", ok=False,
                     prompt_tokens=20, completion_tokens=8))
    tr.add(StepTrace(index=3, type="action", action="edit_file", ok=True,
                     prompt_tokens=30, completion_tokens=12))
    assert tr.total_tokens == 85
    assert tr.tool_usage() == {"edit_file": 1, "run_python": 1, "write_file": 1}


def test_trace_to_dict_serializable():
    from app.trace import StepTrace, Trace
    import json as _json
    tr = Trace(task="t", model="m")
    tr.add(StepTrace(index=1, type="final", ok=True, verified=True))
    d = tr.to_dict()
    assert _json.loads(_json.dumps(d))["steps"][0]["verified"] is True


# ---------- 6. 基准集完整性 ----------

def test_benchmark_cases_are_wellformed():
    """基准集自身的质量也要有测试：expect 为空、分类拼错都会让实验失真。"""
    from eval.cases import CASES, CATEGORY_ORDER
    assert len(CASES) >= 30, f"基准集只有 {len(CASES)} 例，样本量不足"
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)), "用例 id 有重复"
    for c in CASES:
        assert c["category"] in CATEGORY_ORDER, f"{c['id']} 分类非法"
        assert c["expect"], f"{c['id']} 缺少期望输出"
        assert c["code"].strip(), f"{c['id']} 代码为空"
        assert c["task"].strip(), f"{c['id']} 任务描述为空"
        assert c["id"] in c["task"], f"{c['id']} 的任务描述没提到文件名，Agent 找不到目标"


def test_all_categories_present():
    from eval.cases import CASES, CATEGORY_ORDER
    seen = {c["category"] for c in CASES}
    assert seen == set(CATEGORY_ORDER), f"缺少类别：{set(CATEGORY_ORDER) - seen}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))