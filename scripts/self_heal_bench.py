"""自愈成功率实验：预置 N 个不同类型的 bug，让 Agent 自主修复，统计成功率 + 平均轮数。

实验设计（面试讲法）：
- 每个 case 是一个「含 bug 的脚本 + 期望正确输出 + 任务描述」；
- Agent 不知道 bug 具体在哪，只知道"运行它、报错就修、让它输出正确结果"；
- 判定标准：Agent 结束后，脚本能跑出「期望输出」= 修复成功（用事实判定，不信模型的 final 自报）；
- 记录：成功与否、实际轮数、最终脚本是否通过。

用法：
    python scripts/self_heal_bench.py
可选参数：
    --cases 3     只跑前 N 个 case
    --json out.json  把结果存成 JSON
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import agent, config


# ---------- 测试用例：不同类型的 bug ----------

CASES = [
    {
        "id": "syntax_error",
        "name": "语法错误（缺冒号）",
        "code": "def greet(name)\n    return f'Hello, {name}'\n\nprint(greet('World'))\n",
        "expect": "Hello, World",
        "task": "运行 workspace 里的 case_syntax_error.py，如果报错请修复，让它输出 'Hello, World'",
    },
    {
        "id": "runtime_zero_division",
        "name": "运行时错误（除零）",
        "code": "def divide(a, b):\n    return a / b\n\nprint(divide(10, 0))\n",
        "expect": "除数不能为零",
        "task": "运行 workspace 里的 case_runtime_zero_division.py，如果报错请诊断并修复，让它不再报错",
    },
    {
        "id": "logic_wrong_op",
        "name": "逻辑错误（加法写成减法）",
        "code": "def add(a, b):\n    return a - b\n\nprint(add(3, 5))\n",
        "expect": "8",
        "task": "workspace 里的 case_logic_wrong_op.py 计算结果不对，请找出并修复，让 add(3,5) 输出 8",
    },
    {
        "id": "name_error",
        "name": "NameError（变量未定义）",
        "code": "def calc():\n    result = total + 10\n    return result\n\nprint(calc())\n",
        "expect": "30",
        "task": "运行 workspace 里的 case_name_error.py，报错请修复，让程序输出 30",
    },
    {
        "id": "indent_error",
        "name": "缩进错误",
        "code": "def loop():\nfor i in range(3):\n    print(i)\nloop()\n",
        "expect": "0",
        "task": "运行 workspace 里的 case_indent_error.py，报错请修复，让它能正常打印 0 1 2",
    },
]


def _workspace_file(name: str, code: str):
    """把 case 代码写入 workspace，返回文件名。"""
    ws = config.WORKSPACE
    os.makedirs(ws, exist_ok=True)
    path = os.path.join(ws, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def _run_and_check(path: str) -> tuple[bool, str]:
    """用事实判定：直接跑脚本，看 stdout 是否包含期望输出。"""
    try:
        proc = subprocess.run(
            ["python", path], capture_output=True, text=True, timeout=15
        )
        return proc.returncode == 0, (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "（判定超时）"


async def run_one(case: dict) -> dict:
    """跑一个 case，返回结果 dict。"""
    filename = f"case_{case['id']}.py"
    path = _workspace_file(filename, case["code"])

    result = await agent.run(case["task"])

    # 用事实判定（不信 final 自报）
    ok, out = _run_and_check(path)
    success = ok and (case["expect"] in out)

    return {
        "id": case["id"],
        "name": case["name"],
        "success": success,
        "iterations": result.iterations,
        "agent_said_done": result.success,
        "final_output": out[:200],
        "expected": case["expect"],
    }


async def main(n_cases: int | None = None):
    cases = CASES[:n_cases] if n_cases else CASES
    results = []
    print(f"=== 自愈成功率实验（{len(cases)} 个 case）===\n")
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['name']} ...", end=" ", flush=True)
        t0 = time.time()
        r = await run_one(case)
        r["elapsed_s"] = round(time.time() - t0, 1)
        results.append(r)
        mark = "✅ 成功" if r["success"] else "❌ 失败"
        print(f"{mark}（{r['iterations']} 轮，{r['elapsed_s']}s）")

    # 汇总
    success_n = sum(1 for r in results if r["success"])
    avg_iter = sum(r["iterations"] for r in results) / len(results) if results else 0
    print("\n=== 汇总 ===")
    print(f"成功率：{success_n}/{len(results)} = {success_n/len(results)*100:.0f}%")
    print(f"平均轮数：{avg_iter:.1f}")
    print("\n=== 明细 ===")
    for r in results:
        print(f"  {r['id']:28s} {r['name']:20s} {'成功' if r['success'] else '失败'} "
              f"({r['iterations']}轮) agent自报={'是' if r['agent_said_done'] else '否'}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=None)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    results = asyncio.run(main(args.cases))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已存：{args.json}")
