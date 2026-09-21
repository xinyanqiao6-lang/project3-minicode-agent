"""MiniCode 命令行入口。

用法：
    python -m app.cli "在 workspace 里写一个斐波那契函数并运行验证"

或直接跑一个演示任务（无 Key 时用 MOCK_MODE 走通链路）：
    python -m app.cli --demo
"""
import argparse
import logging

from . import agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="MiniCode 轻量级 Coding Agent")
    parser.add_argument("task", nargs="?", help="任务描述")
    parser.add_argument("--demo", action="store_true", help="运行内置演示任务")
    parser.add_argument("--max-iter", type=int, default=None, help="最大执行轮数")
    args = parser.parse_args()

    task = args.task
    if args.demo or not task:
        task = "在 workspace 中创建一个 fib.py，实现斐波那契数列函数 fib(n)，然后用 python 运行验证 fib(10) == 55，最后告诉我结果。"

    print(f"\n=== 任务 ===\n{task}\n")
    result = agent.run_sync(task)

    print("\n=== 执行轨迹 ===")
    for i, s in enumerate(result.steps, 1):
        if s["type"] == "final":
            print(f"[{i}] FINAL: {s['content']}")
        else:
            print(f"[{i}] ACTION {s['action']}({s['args']}) -> {'OK' if s['ok'] else 'FAIL'}")
            print(f"     观察: {s['observation'][:120]}")

    print(f"\n=== 结果 ===")
    print(f"成功: {result.success}  轮数: {result.iterations}")
    if result.error:
        print(f"错误: {result.error}")
    if result.final_answer:
        print(f"最终回答:\n{result.final_answer}")


if __name__ == "__main__":
    main()
