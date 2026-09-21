"""MiniCode 核心单元测试。

用「脚本化的假模型」（monkeypatch chat_completion）驱动 Agent 循环，
验证 Planner-Action-Observation 与自愈闭环逻辑，不依赖真实 API Key。
"""
import asyncio
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import agent, tools


# ---------- 测试工具：伪造模型输出 ----------

def fake_model_script(responses: list[str]):
    """返回一个按顺序吐出固定 JSON 的 async 假 chat_completion。

    responses: 每轮一个模型原始输出（JSON 字符串）。
    用完后再调用会抛 StopIteration（测试里可用 max_iter 兜底）。
    """
    idx = 0

    async def _fake(messages, temperature=None, max_tokens=None):
        nonlocal idx
        if idx >= len(responses):
            # 模型没词了，返回一个 final 兜底，避免死循环
            return json.dumps({"type": "final", "content": "脚本耗尽，结束"})
        r = responses[idx]
        idx += 1
        return r

    return _fake


def monkeypatch_model(monkeypatch, responses):
    fake = fake_model_script(responses)
    monkeypatch.setattr(agent.llm_client, "chat_completion", fake)
    return fake


# ---------- 1. JSON 解析容错 ----------

def test_parse_plain_action():
    s = agent.parse_model_output('{"type":"action","thought":"查看文件","action":"list_files","args":{"path":""}}')
    assert s.type == "action"
    assert s.action == "list_files"


def test_parse_fenced_json():
    """模型常输出 ```json ... ``` 围栏，需能剥掉。"""
    raw = '```json\n{"type":"final","content":"完成"}\n```'
    s = agent.parse_model_output(raw)
    assert s.type == "final"
    assert s.content == "完成"


def test_parse_with_noise():
    """模型在 JSON 前后带解释文字，需截取主体。"""
    raw = '好的，我来操作：{"type":"action","action":"read_file","args":{"name":"a.py"}} 就这个'
    s = agent.parse_model_output(raw)
    assert s.action == "read_file"


def test_parse_invalid_fallback():
    s = agent.parse_model_output("这不是 JSON")
    assert s.type == "final"  # 解析失败降级为 final，不抛异常


# ---------- 2. 工具层 ----------

def test_write_and_read_file():
    import tempfile
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
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        ok, out = tools.write_file("../escape.py", "x=1")
        assert not ok  # 越界被拦截


def monkeypatch_workspace(d):
    import app.config as config
    config.WORKSPACE = d


# ---------- 3. Agent 自愈闭环（核心） ----------

def test_agent_self_healing_loop(monkeypatch):
    """模拟：写 bug 代码 -> 执行报错 -> 修复 -> 再执行成功 -> final。

    这是整个项目「多轮自愈调试」的端到端证明。
    """
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)

        # 4 轮脚本：
        # 1) 写有 bug 的代码（1/0）
        # 2) 执行 -> 会报 ZeroDivisionError
        # 3) 修复（改成 1+1）
        # 4) 再执行 -> 成功
        # 5) final
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
        # 验证轨迹里有「失败 -> 修复 -> 成功」的闭环
        action_steps = [s for s in result.steps if s["type"] == "action"]
        actions = [s["action"] for s in action_steps]
        assert actions == ["write_file", "run_python", "edit_file", "run_python"]
        # 第一次 run_python 失败，第二次成功
        run_results = [s["ok"] for s in action_steps if s["action"] == "run_python"]
        assert run_results == [False, True]


def test_agent_max_iteration_guard(monkeypatch):
    """模型一直输出 action 不 final，应被最大轮数护栏终止。"""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch_workspace(d)
        # 无限输出 list_files 的脚本（用 20 个够多了）
        script = [
            json.dumps({"type": "action", "action": "list_files", "args": {}})
        ] * 20
        monkeypatch_model(monkeypatch, script)

        result = agent.run_sync("随便", max_iterations=5)
        assert not result.success
        assert result.iterations == 5  # 被护栏截断
        assert "最大轮数" in result.error
