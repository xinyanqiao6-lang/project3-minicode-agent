"""MiniCode 配置：全部从环境变量读取，与 project1 网关保持一致风格。

设计要点：
- 复用 project1 的 SILICONFLOW_API_KEY / BASE_URL / MODEL 环境变量，两项目可共用同一把 Key；
- MOCK_MODE 用于无 Key 时先跑通 Agent 循环（数字不进简历）；
- 各护栏参数（最大轮数、超时、退避）都做成可配，便于单测与调参。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---- 硅基流动（SiliconFlow），与 project1 网关共用 ----
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")

# ---- Agent 循环护栏 ----
MAX_ITERATIONS = int(os.getenv("MINICODE_MAX_ITERATIONS", "10"))  # 最大执行轮数（防死循环）
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))                # 单次 LLM 请求超时
LLM_TEMPERATURE = float(os.getenv("MINICODE_TEMPERATURE", "0.2"))  # Agent 要稳定，温度调低
LLM_MAX_TOKENS = int(os.getenv("MINICODE_MAX_TOKENS", "2048"))

# ---- 执行器 ----
EXEC_TIMEOUT = float(os.getenv("MINICODE_EXEC_TIMEOUT", "10"))     # subprocess 单次执行超时
WORKSPACE = os.getenv("MINICODE_WORKSPACE", "./workspace")         # 沙箱工作目录（相对路径）

# ---- 运行模式 ----
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() in ("1", "true", "yes")

# 调上游是否走系统代理。默认 false = 直连。
# 与 project1 同理：走代理会给每次调用叠加不确定的延迟，Agent 评测数据不可信。
UPSTREAM_TRUST_ENV = os.getenv("UPSTREAM_TRUST_ENV", "false").lower() in ("1", "true", "yes")

# ---- 评测 ----
# 基准集默认使用托管 venv 的解释器跑被测脚本，避免 "python" 指向不同的环境
BENCH_PYTHON = os.getenv("MINICODE_BENCH_PYTHON", "")
