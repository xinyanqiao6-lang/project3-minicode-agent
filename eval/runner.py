"""评测执行器：跑基准集 → 判定 → 聚合指标 → 出报告。

这是 v2 的核心产出：一条命令跑完整套实验，输出**可写进简历**的数字。

    # A/B 对照：关闭 / 开启结果校验步骤，量化幻觉率下降
    python -m eval.runner --ab

    # 单模型跑全量 30 例
    python -m eval.runner --model Qwen/Qwen2.5-7B-Instruct

    # 跨模型对比（证明"模型能力是 Agent 上限"）
    python -m eval.runner --models Qwen/Qwen2.5-7B-Instruct,deepseek-ai/DeepSeek-V3

## 关键设计：一切用事实判定
判定器在 Agent 循环**内部**与**外部**各跑一次脚本：
- 内部（verifier）：喂回 Agent，作为自愈反馈；
- 外部（judge）：Agent 结束后独立判定，用于统计成功率与幻觉率。
两者独立，避免"自己判自己及格"。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import agent, config          # noqa: E402
from eval.cases import CASES, CATEGORY_DESC, CATEGORY_ORDER  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------- 事实判定器 ----------
def judge_case(path: str, expect: str) -> tuple[bool, str]:
    """直接跑脚本，比对 stdout 是否包含期望值。

    这是整个评测的信任基础：**不信 Agent 的 final 自报，只信脚本的实际输出。**
    """
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=15
        )
    except subprocess.TimeoutExpired:
        return False, "（判定超时）"
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        return False, f"[exit {proc.returncode}] {out[-400:]}"
    if expect in out:
        return True, out[-400:]
    return False, f"实际输出缺少期望值 {expect!r}；实际：{out[-300:]}"


def make_verifier(path: str, expect: str):
    """构造给 Agent 循环用的闭包验证器（无参可调用）。"""
    def _verify() -> tuple[bool, str]:
        return judge_case(path, expect)
    return _verify


def write_case_file(case: dict) -> str:
    """把用例代码写入 workspace，返回文件绝对路径。"""
    ws = Path(config.WORKSPACE).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    p = ws / f"case_{case['id']}.py"
    p.write_text(case["code"], encoding="utf-8")
    return str(p)


# ---------- 单例执行 ----------
def _ckpt_path(label: str, model: str) -> Path:
    """断点文件名。

    注意必须净化 **两处**：model 和 label。
    label 形如「模型 deepseek-ai/DeepSeek-V3」里也含模型名，
    只净化 model 会在 Windows 上把 "/" 当成目录分隔符，
    报 FileNotFoundError（实测踩过）。
    """
    def _safe(s: str) -> str:
        return (s.replace("/", "_").replace("\\", "_")
                 .replace(":", "").replace(" ", "_")
                 .replace("：", "_").replace("·", "_"))

    return RESULTS_DIR / f"records_{_safe(model)}_{_safe(label)}.jsonl"


def load_ckpt(path: Path) -> dict[str, dict]:
    """读取断点文件，返回 {case_id: record}。文件不存在则返回空。"""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            out[r["id"]] = r
        except json.JSONDecodeError:
            continue          # 崩溃时最后一行可能是半截的，忽略即可
    return out


async def run_case(case: dict, model: str, use_verifier: bool) -> tuple[dict, object]:
    """跑一个用例，返回 (结果记录, AgentResult)。"""
    path = write_case_file(case)
    task = case["task"]

    t0 = time.perf_counter()
    if use_verifier:
        r = await agent.run(task, verifier=make_verifier(path, case["expect"]))
    else:
        r = await agent.run(task)
    elapsed = time.perf_counter() - t0

    # 外部独立判定（不信 Agent 自报，也不信内部 verifier）
    ok, detail = judge_case(path, case["expect"])
    success = ok
    hallucinated = r.claimed_done and not success

    rec = {
        "id": case["id"],
        "category": case["category"],
        "name": case["name"],
        "model": model,
        "success": success,
        "agent_claimed_done": r.claimed_done,
        "hallucinated": hallucinated,       # ← v2 核心指标
        "verify_enabled": use_verifier,
        "verify_rejected": r.verify_rejected,
        "iterations": r.iterations,
        "stop_reason": r.stop_reason,
        "tokens": r.total_tokens,
        "elapsed_s": round(elapsed, 1),
        "judge_detail": detail[:300],
        "final_answer": (r.final_answer or "")[:200],
        "tool_usage": r.trace.tool_usage() if r.trace else {},
    }
    return rec, r


# ---------- 指标聚合 ----------
def summarize(records: list[dict]) -> dict:
    """把逐例记录聚合成可引用指标。

    注意耗时用**中位数**而不是均值：实测中机器休眠过一次，单条记录被拉到
    10 小时以上，均值会被这一个异常值彻底污染（曾出现"平均 1246.9s"这种
    完全没意义的数字）。中位数才是真实体验，均值只作参考并列出来。
    """
    n = len(records)
    if n == 0:
        return {}
    succ = sum(1 for r in records if r["success"])
    hallu = sum(1 for r in records if r["hallucinated"])
    claimed = sum(1 for r in records if r["agent_claimed_done"])
    verified = sum(1 for r in records if r.get("verify_enabled"))
    rejected = sum(r.get("verify_rejected", 0) for r in records)

    elapseds = sorted(r["elapsed_s"] for r in records)
    median_elapsed = elapseds[len(elapseds) // 2] if elapseds else 0.0
    # 超过中位数 5 倍的算异常值（通常是机器休眠/网络挂起），单独计数不污染均值
    outliers = [r for r in records if r["elapsed_s"] > max(median_elapsed * 5, 60)]

    return {
        "n_cases": n,
        "success_rate": round(succ / n, 4),
        "success_n": succ,
        # 幻觉率：自报完成里有多少是假的。这是 v1 发现的核心问题。
        "hallucination_rate": round(hallu / n, 4),
        "hallucination_n": hallu,
        # 宣称率：模型有多爱说"做完了"
        "claim_rate": round(claimed / n, 4),
        # 精确率：模型说"做完了"且真做对了的比例（越接近 1 越可信）
        "claim_precision": round((claimed - hallu) / claimed, 4) if claimed else None,
        "avg_iterations": round(sum(r["iterations"] for r in records) / n, 2),
        "avg_tokens": round(sum(r["tokens"] for r in records) / n, 1),
        "total_tokens": sum(r["tokens"] for r in records),
        "median_elapsed_s": round(median_elapsed, 1),
        "mean_elapsed_s": round(sum(elapseds) / n, 1),
        "elapsed_outliers": [
            {"id": r["id"], "elapsed_s": r["elapsed_s"]} for r in outliers
        ],
        "verify_enabled": bool(verified),
        "verify_rejected_total": rejected,
    }


def by_category(records: list[dict]) -> dict[str, dict]:
    """分维度结论：哪类能修、哪类修不了。"""
    out = {}
    for cat in CATEGORY_ORDER:
        items = [r for r in records if r["category"] == cat]
        if items:
            s = summarize(items)
            s["desc"] = CATEGORY_DESC[cat]
            out[cat] = s
    return out


def fmt_pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


# ---------- 报告 ----------
def render_report(payload: dict) -> str:
    """人可读报告（可直接贴进简历/面试讲稿）。"""
    L = []
    L.append("# MiniCode Agent 评测报告")
    L.append("")
    L.append(f"生成时间：{payload['generated_at']}")
    L.append(f"用例集：{payload['n_cases']} 个（{len(CATEGORY_ORDER)} 类）")
    L.append(f"判定方式：**事实判定**——Agent 结束后实跑脚本比对 stdout，不信模型自报")
    L.append("")

    for exp in payload["experiments"]:
        s = exp["summary"]
        tag = "结果校验 ON" if s["verify_enabled"] else "结果校验 OFF（v1 行为）"
        L.append(f"## {exp['label']}（{tag}）")
        L.append("")
        L.append("| 指标 | 值 |")
        L.append("|---|---|")
        L.append(f"| 模型 | `{exp['model']}` |")
        L.append(f"| 样本 | {s['n_cases']} |")
        L.append(f"| **自愈成功率** | **{fmt_pct(s['success_rate'])}**（{s['success_n']}/{s['n_cases']}）|")
        L.append(f"| **幻觉式汇报率** | **{fmt_pct(s['hallucination_rate'])}**（{s['hallucination_n']}/{s['n_cases']}）|")
        L.append(f"| 宣称完成率 | {fmt_pct(s['claim_rate'])} |")
        L.append(f"| 宣称精确率（说完了且真对） | {fmt_pct(s['claim_precision'])} |")
        L.append(f"| 平均轮数 | {s['avg_iterations']} |")
        L.append(f"| 平均 token/任务 | {s['avg_tokens']} |")
        L.append(f"| 耗时中位数 | {s['median_elapsed_s']}s |")
        if s["verify_enabled"]:
            L.append(f"| 校验打回次数 | {s['verify_rejected_total']} |")
        L.append("")

        if s.get("elapsed_outliers"):
            bad = "、".join(f"`{o['id']}`({o['elapsed_s']}s)" for o in s["elapsed_outliers"])
            L.append(f"> ⚠️ 耗时异常值已剔除统计口径：{bad} —— 通常是机器休眠或网络挂起，"
                     f"不代表真实性能。表中用**中位数**而非均值。")
            L.append("")

        cats = exp["by_category"]
        if cats:
            L.append("### 分维度成功率（哪类能修、哪类修不了）")
            L.append("")
            L.append("| 类别 | 说明 | 成功率 | 幻觉率 | 平均轮数 |")
            L.append("|---|---|---|---|---|")
            for cat in CATEGORY_ORDER:
                c = cats.get(cat)
                if not c:
                    continue
                L.append(f"| `{cat}` | {c['desc']} | {fmt_pct(c['success_rate'])} | "
                         f"{fmt_pct(c['hallucination_rate'])} | {c['avg_iterations']} |")
            L.append("")

        if exp.get("failures"):
            L.append("### 失败用例")
            L.append("")
            for r in exp["failures"]:
                L.append(f"- `{r['id']}`（{r['name']}）：{r['judge_detail'][:120]}")
            L.append("")

    if payload.get("conclusion"):
        L.append("## 结论")
        L.append("")
        for c in payload["conclusion"]:
            L.append(f"- {c}")
        L.append("")
    return "\n".join(L)


# ---------- 主流程 ----------
async def run_experiment(label: str, model: str, cases: list[dict],
                         use_verifier: bool, resume: bool = True) -> dict:
    """跑一个实验。**每跑完一例立即落盘**，配合断点文件可续跑。

    为什么必须这么做：这组实验要跑 60 次真实 Agent 任务、十几分钟，
    中途被会话中断是常态（实测已丢过一次数据）。没有断点机制，
    一次重启就要重跑全部，评测成本翻倍。
    """
    print(f"\n>>> 实验：{label} | model={model} | verifier={'ON' if use_verifier else 'OFF'} | "
          f"{len(cases)} 例", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    ckpt = _ckpt_path(label, model)
    done = load_ckpt(ckpt) if resume else {}
    if done:
        print(f"[*] 断点续跑：已完成 {len(done)}/{len(cases)} 例", flush=True)

    records: list[dict] = []
    t0 = time.perf_counter()
    with ckpt.open("a", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            if case["id"] in done:
                records.append(done[case["id"]])
                continue

            rec, r = await run_case(case, model, use_verifier)
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()          # 立刻落盘：崩了也不丢已完成的结果

            mark = "✅" if rec["success"] else "❌"
            flag = " [幻觉]" if rec["hallucinated"] else ""
            print(f"  [{i}/{len(cases)}] {mark} {case['name']:22s} "
                  f"({rec['iterations']}轮 {rec['tokens']}tok {rec['elapsed_s']}s){flag}", flush=True)

            if r.trace is not None:
                td = RESULTS_DIR / "traces"
                td.mkdir(exist_ok=True)
                r.trace.save(td / f"{model.replace('/', '_')}_{case['id']}.json")

    return {
        "label": label,
        "model": model,
        "verify_enabled": use_verifier,
        "summary": summarize(records),
        "by_category": by_category(records),
        "failures": [r for r in records if not r["success"]],
        "records": records,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


async def main():
    ap = argparse.ArgumentParser(description="MiniCode Agent 评测执行器")
    ap.add_argument("--model", default=config.MODEL, help="单个模型")
    ap.add_argument("--models", default="", help="多模型逗号分隔（跨模型对比）")
    ap.add_argument("--ab", action="store_true", help="A/B：同一模型开关结果校验步骤")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个用例（冒烟）")
    ap.add_argument("--per-category", type=int, default=0, help="每类取 N 个（快速冒烟）")
    ap.add_argument("--no-verify", action="store_true", help="全部关闭结果校验（复现 v1 行为）")
    ap.add_argument("--out", default="", help="报告输出路径（md）")
    ap.add_argument("--json", default="", help="原始数据输出路径（json）")
    ap.add_argument("--report-only", action="store_true",
                    help="不跑任务，直接用断点文件(results/records_*.jsonl)重建报告")
    args = ap.parse_args()

    if args.per_category:
        from eval.cases import subset
        cases = subset(args.per_category)
    else:
        cases = CASES[: args.limit] if args.limit else CASES

    models = [m.strip() for m in args.models.split(",") if m.strip()] or [args.model]

    # 断点重建模式：复用已跑完的记录，只重算指标与报告。
    # 价值：统计口径一改（比如剔除异常值），不需要重跑 60 次真实任务。
    if args.report_only:
        experiments = []
        for label, model, use_v in ([("A: 无结果校验", models[0], False),
                                     ("B: 启用结果校验", models[0], True)] if args.ab
                                    else [(f"模型 {m}", m, not args.no_verify)
                                          for m in models]):
            recs = list(load_ckpt(_ckpt_path(label, model)).values())
            if not recs:
                print(f"[!] 无断点数据：{label} / {model}")
                continue
            print(f"[*] 复用断点 {len(recs)} 例：{label}")
            experiments.append({
                "label": label, "model": model, "verify_enabled": use_v,
                "summary": summarize(recs), "by_category": by_category(recs),
                "failures": [r for r in recs if not r["success"]],
                "records": recs, "wall_s": 0,
            })
        return _finish(experiments, cases, args)

    experiments = []
    for m in models:
        config.MODEL = m                      # 切模型
        if args.ab and len(models) == 1:
            experiments.append(await run_experiment("A: 无结果校验", m, cases, False))
            experiments.append(await run_experiment("B: 启用结果校验", m, cases, True))
        else:
            experiments.append(await run_experiment(
                f"模型 {m}", m, cases, not args.no_verify))

    return _finish(experiments, cases, args)


def _finish(experiments: list[dict], cases: list[dict], args) -> None:
    """汇总 payload → 写文件 → 打印报告。两条路径共用，保证口径一致。"""
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(cases),
        "experiments": experiments,
    }

    # 自动写结论
    conclusion = []
    if len(experiments) >= 2 and experiments[0]["verify_enabled"] != experiments[1]["verify_enabled"]:
        a, b = experiments[0]["summary"], experiments[1]["summary"]
        conclusion.append(
            f"**结果校验步骤把幻觉式汇报率从 {fmt_pct(a['hallucination_rate'])} 降到 "
            f"{fmt_pct(b['hallucination_rate'])}**（宣称精确率 "
            f"{fmt_pct(a['claim_precision'])} → {fmt_pct(b['claim_precision'])}），"
            f"自愈成功率同时从 {fmt_pct(a['success_rate'])} 提升到 {fmt_pct(b['success_rate'])}。"
        )
        conclusion.append(
            f"平均代价：每任务多 {round(b['avg_iterations'] - a['avg_iterations'], 2)} 轮、"
            f"{round(b['avg_tokens'] - a['avg_tokens'])} token。"
        )
    if len(experiments) >= 2 and experiments[0]["model"] != experiments[1]["model"]:
        a, b = experiments[0]["summary"], experiments[1]["summary"]
        conclusion.append(
            f"跨模型对比：{experiments[1]['model']} 成功率 {fmt_pct(b['success_rate'])} vs "
            f"{experiments[0]['model']} {fmt_pct(a['success_rate'])}，"
            f"证明模型能力是 Coding Agent 的能力上限。"
        )
    # 最难的两类要看**最终部署配置**（启用结果校验那组），不是对照组。
    # 只挑"未满分"的类别——把 100% 的类别排进"最难"是自相矛盾。
    final = experiments[-1]
    cats = final["by_category"]
    imperfect = {k: v for k, v in cats.items() if v["success_rate"] < 1.0}
    if imperfect:
        worst = sorted(imperfect.items(), key=lambda kv: kv[1]["success_rate"])[:2]
        conclusion.append(
            "启用结果校验后，仍未 100% 的类别是 "
            + "、".join(f"`{k}`（{fmt_pct(v['success_rate'])}）" for k, v in worst)
            + "——共同点是「不报语法错、但需要理解代码意图」，"
            "靠让模型自查补不上，必须靠外部事实验证器兜底。"
        )
    # 耗时离群检测：平均耗时异常长通常意味着上游限流排队，会污染对比结论
    slow = [e for e in experiments if e["summary"]["avg_elapsed_s"] > 120]
    if slow:
        for e in slow:
            conclusion.append(
                f"⚠️ {e['model']} 平均耗时 {e['summary']['avg_elapsed_s']}s/任务，"
                f"明显异常（其他组 {experiments[0]['summary']['avg_elapsed_s']}s 量级），"
                f"疑似上游限流排队；本组数字可参考，但**不宜与其余组直接横向比较**。"
            )
    payload["conclusion"] = conclusion

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(args.json) if args.json else RESULTS_DIR / f"eval_{stamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_report(payload)
    md_path = Path(args.out) if args.out else RESULTS_DIR / f"eval_{stamp}.md"
    md_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 70)
    print(report)
    print(f"\n[*] 报告：{md_path}")
    print(f"[*] 原始数据：{json_path}")


if __name__ == "__main__":
    asyncio.run(main())