# MiniCode · 轻量级 Coding Agent

> 目标：在现有 LLM 网关（project1）之上，实现一个轻量级 Coding Agent，证明「Agent 开发 + Function Calling + 多轮自愈调试」三项能力。
> 对标：样本简历 1 的 MiniCode 项目思路，但复用 project1 的韧性调用模式，形成「网关 → Agent」的进阶叙事。

## 架构

```
flowchart LR
    U[用户任务] --> AG{Agent 循环}
    AG -->|Planner| LLM[chat_completion<br/>复用 project1 韧性调用]
    LLM -->|结构化 JSON| P[Parser<br/>容错解析]
    P -->|action| EX[工具执行器 tools.call_tool]
    EX --> O{Observation}
    O -->|成功| AG
    O -->|报错 stderr/returncode| AG
    AG -->|final| R[最终回答]

    EX --> F1[文件工具<br/>list/read/write/edit/search]
    EX --> F2[执行工具<br/>run_python/run_command<br/>subprocess]
```

## 核心设计（面试讲法）

| 模块 | 做法 | 解决什么 |
|---|---|---|
| **Planner** | 把「系统提示 + 任务 + 历史观察」拼成 messages，调 LLM | 让模型基于上下文做下一步决策 |
| **Action 执行器** | 7 个工具（文件 + 执行），统一 `(ok, output)` 返回 | 让模型能真正「动手」改代码、跑代码 |
| **Observation 解析** | 工具结果原样作为下一轮 user 反馈 | 形成「看到结果 → 再决策」的闭环 |
| **JSON Function Calling** | 约束模型输出 `{"type":"action","action":...,"args":{...}}` 或 `{"type":"final",...}` | 结构化、可解析、可约束 |
| **自愈闭环** | `run_python` 失败时把 stderr/returncode 喂回模型，模型 edit_file 修复后再跑 | 多轮自动调试，无需人工介入 |
| **安全护栏** | 路径沙箱（`../` 越界拦截）+ 命令白名单 + 最大轮数 | 防止 Agent 失控 / 死循环 |

## 与 project1 网关的集成方式

MiniCode **不直接 import project1 的包**（避免跨项目耦合），而是：

1. **复用同一套环境变量**：`SILICONFLOW_API_KEY` / `BASE_URL` / `MODEL` 两个项目共用一把 Key；
2. **提炼韧性调用模式**：把 project1 `llm_client.chat_completion` 的「超时 + 有限重试」精简复制到 `app/llm_client.py`，Agent 的每次 Planner 调用都走这条韧性路径；
3. **叙事上的衔接**：project1 解决「怎么稳定调 LLM」，project3 解决「怎么让 LLM 自主做事」——面试可讲「从网关到 Agent 的进阶」。

> 若要深度集成，也可把 MiniCode 作为网关的一个 `/agent` 端点挂载（扩展点），但本项目保持独立可跑、独立可测。

## 快速开始

### 1) 环境
```bash
pip install -r requirements.txt
```

### 2) 配置
```bash
cp .env.example .env
# 编辑 .env 填入 SILICONFLOW_API_KEY
```

### 3) 跑测试（10 个用例，无需 Key）
```bash
python -m pytest tests/ -v
```

### 4) 跑一个真实任务
```bash
# 演示任务：写斐波那契函数并运行验证
python -m app.cli "在 workspace 里写一个 fib.py，实现 fib(n)，运行验证 fib(10)==55，告诉我结果"
```

### 5) 无 Key 先跑通链路（Mock）
```bash
MOCK_MODE=true python -m app.cli --demo
```

## 端到端验证用例

| 用例 | 验证什么 | 结果 |
|---|---|---|
| `test_parse_fenced_json` | JSON 围栏容错 | ✅ |
| `test_parse_with_noise` | 前后杂讯截取 | ✅ |
| `test_write_and_read_file` | 文件工具 | ✅ |
| `test_run_python_error_feedback` | 报错反馈（stderr 捕获） | ✅ |
| `test_path_escape_blocked` | 沙箱越界拦截 | ✅ |
| `test_agent_self_healing_loop` | **自愈闭环**：写 bug→报错→修复→成功 | ✅ |
| `test_agent_max_iteration_guard` | 最大轮数护栏 | ✅ |

## 项目结构

```
project3-minicode-agent/
├── app/
│   ├── config.py      # 环境变量配置（护栏参数）
│   ├── llm_client.py  # 韧性调用（复用 project1 模式）
│   ├── tools.py       # 7 个工具（文件 + 执行）+ 注册表
│   ├── agent.py       # Agent 循环 + JSON 解析 + 自愈闭环
│   └── cli.py         # 命令行入口
├── tests/
│   └── test_agent.py  # 10 个单测（含自愈闭环端到端）
├── workspace/         # 沙箱工作目录（Agent 读写在此）
├── .env.example
└── requirements.txt
```

## 面试必考题参考

1. **Agent Loop 是什么？** → 见 `agent.py` 头部注释：Planner（决策）→ Action（执行）→ Observation（反馈）循环，直到 final。
2. **为什么用 JSON 约束而不是让模型自由输出？** → 可解析、可校验、可路由；配合容错解析（围栏/截取）兼顾模型输出不稳。
3. **自愈是怎么实现的？** → `run_python` 失败时把 stderr/returncode 原样作为 user 消息喂回，模型据此 edit_file 修复再跑。
4. **怎么防 Agent 失控？** → 三层：路径沙箱（越界拦截）+ 命令白名单 + 最大轮数护栏。
5. **和 project1 网关什么关系？** → 复用其韧性调用模式（超时+重试），把「稳定调 LLM」的能力延伸到「让 LLM 自主执行任务」。
