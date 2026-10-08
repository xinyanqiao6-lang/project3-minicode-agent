# MiniCode · 轻量级 Coding Agent（v2）

> 在 project1 网关「韧性调用」模式之上，实现 Coding Agent 的 Planner-Action-Observation 循环，
> 并在 v2 中解决了 v1 发现的**幻觉式汇报**问题：把「用事实判定」这条原则内建进 Agent 循环。

**核心结论（30 个 bug 用例实测，见 `results/eval_*.md`）**：

| 配置 | 自愈成功率 | 幻觉式汇报率 | 宣称精确率 | 平均轮数 | 平均 token |
|---|---|---|---|---|---|
| v1 行为（无结果校验） | 77% | **17%** | 82% | 4.37 | 3184 |
| **v2（启用结果校验）** | **87%** | **3%** | **96%** | 4.40 | 3329 |

**代价：多 0.03 轮、多 145 token。** 这是本项目最有价值的一组数字——
用几乎零成本换掉了 14 个百分点的幻觉率。

---

## v2 做了什么

### 1. 结果校验步骤（核心改进）

v1 实测发现：**5 个 bug 里 Agent 有 5 次都自报「完成」，但只有 3 次真修对**。
根因是 v1 只能事后人工判定，Agent 自己不知道它报错了。

v2 把验证器内建进循环：

```
模型想输出final
    │
    ├─ 无 verifier（v1 行为）→ 直接采信← 幻觉的来源
    │
    └─ 有 verifier（v2）
         ├─ verifier() 通过 → 采信 final
         └─ verifier() 失败 → 把「实际输出 vs 期望输出」喂回模型 → 继续修
```

验证器是 `fail-closed` 的：验证器自身抛异常也视为不通过
（见 `test_verifier_exception_does_not_crash`）。

### 2. 基准集从 5 例扩到 30 例（5 类）

| 类别 | 说明 | v2 成功率 |
|---|---|---|
| `syntax` | 语法错误（stderr 直接指明） | 83% |
| `runtime` | 运行时异常 | **100%** |
| `logic` | 逻辑错误（不报错但结果错） | **100%** |
| `dependency` | 依赖/接口错误 | **100%** |
| `boundary` | 边界条件（空、除零、负数） | 50% |

结论清晰：**能被 stderr 指明的错误基本能修，需要理解意图或预判边界条件的修不了。**

### 3. Trace 与成本记账

`app/trace.py` 记录每一步的 action / 耗时 / token / 观察结果 / 校验结果，
输出 `results/traces/*.json`。这是「bad case 可定位」和「成本可量化」的前提。

### 4. 断点续跑

评测要跑 60 次真实任务、十几分钟，中断是常态。`eval/runner.py` 每跑完一例立即落盘
（`results/records_*.jsonl`），重跑自动跳过已完成的用例；
`--report-only` 还能只重算指标与报告而不重跑任务（改统计口径时省掉整轮重跑）。

---

## 架构

```
用户任务 → Agent 循环
   ├─ Planner：调 LLM 拿结构化 JSON（复用 project1 的超时/重试/熔断）
   ├─ Parser：容错解析（代码围栏 / 前后杂讯 / JSON 错误降级）
   ├─ Action：7 个工具（文件 + 执行），统一返回 (ok, observation)
   ├─ Observation：工具结果（含 stderr）原样喂回 → 自愈闭环
   └─ ★ 结果校验：模型想 final 时先跑 verifier，不通过就打回继续修
```

---

## 快速开始

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env        # 填 SILICONFLOW_API_KEY
```

```bash
# 跑一个任务
python -m app.cli "在 workspace 里写一个 fib.py，实现 fib(n)，运行验证 fib(10)==55"

# 跑完整 A/B 实验（30 例 × 开关结果校验）
python -m eval.runner --ab

# 只重建报告（复用断点数据，不调API）
python -m eval.runner --ab --report-only

# 跨模型对比
python -m eval.runner --models Qwen/Qwen2.5-7B-Instruct,deepseek-ai/DeepSeek-V3

# 冒烟（每类 1 例，5 例，约 1 分钟）
python -m eval.runner --per-category 1

# 单测（21 个，不需要 Key）
python -m pytest tests/ -v
```

## 项目结构

```
├── app/
│   ├── agent.py        # Agent 循环 v2：结果校验步骤 + token 记账
│   ├── trace.py        # 执行轨迹：action / 耗时 / token / 校验结果
│   ├── tools.py        # 7 个工具 + 三层安全护栏（沙箱/白名单/最大轮数）
│   ├── llm_client.py   # LLM 客户端：返回 (text, usage)，跨事件循环重建
│   └── config.py
├── eval/
│   ├── cases.py        # 30 个 bug 用例，5 类
│   └── runner.py       # 评测执行器：A/B 对照 + 断点续跑 + 报告生成
├── tests/test_agent.py # 21 个单测（含结果校验关键路径）
└── results/
    ├── eval_*.md       # 评测报告
    ├── records_*.jsonl # 断点数据（可复现/可重算）
    ├── traces/         # 逐步执行轨迹
    └── 自愈成功率实验报告.md   # v1 时期的 5 例实验（历史留档）
```

## 面试要点

**Q：为什么不直接信模型说「已完成」？**
A：Coding Agent 里这是最危险的坑。v1 实测幻觉率 17%——模型会写个错代码然后自信地说"已修复"。
上线信了它就是事故。所以 v2 把「跑一遍看实际输出」做成循环里的一个步骤，验证不过就打回。

**Q：结果校验会不会让 Agent 死循环？**
A：不会。有最大轮数护栏（默认 10 轮），实测平均只多 0.03 轮，多数情况第一次就通过。

**Q：幻觉率怎么定义的？**
A：`模型宣告完成 ∧ 外部独立执行脚本判定失败`。这个判定**必须在 Agent 之外**做——
Agent 无法知道自己是否真修好，所以 `AgentResult` 上刻意不提供 `hallucinated` 属性
（见 `test_agent_claims_done_without_verifier`）。

**Q：哪些错误修不了？**
A：`boundary` 类（空列表、除零、负索引）成功率最低。这类 bug 不报错，
模型缺少"应该输出什么"的信号，容易"消除症状"而非"修好问题"——
v1 阶段就发现过：除零 case 里模型把 `a/b` 改成 `return None`，报错没了但语义错了。

**Q：为什么用 char bigram 而不是 jieba？**
A：技术文档主体是未登录术语（FastAPI、pydantic），词切分反而切不准。
字符二元组对这些词召回更好，且不引入额外依赖。

## 已知边界（诚实说明）

- 只测了 `Qwen2.5-7B-Instruct`；`deepseek-ai/DeepSeek-V3` 的对比数据待跑。
- 评测语料是脚本化 bug，与真实仓库里的 bug 分布有差异，成功率不能外推。
- 耗时统计已剔除异常值（实测有一次 10 小时记录，是机器休眠导致），报告中位数而非均值。
- `logic`/`dependency` 类 100% 的成功率受样本量限制（各 6 例），置信区间较宽。