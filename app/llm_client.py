"""LLM 客户端：复用 project1 网关的「超时 + 重试 + 熔断」三层韧性，独立成文件。

为什么独立而不 import project1：project1 的 llm_client 依赖其 config 包，这里
希望 MiniCode 是一个**可独立跑、可独立测试**的项目，所以把 chat_completion
精简复制一份（保留韧性核心），避免跨项目耦合。面试讲法：从项目1 提炼出的
「韧性调用」模式复用到项目3，体现工程能力的迁移。
"""
import asyncio
import logging
from typing import AsyncGenerator

import httpx

from . import config

logger = logging.getLogger("minicode.llm")

_client: httpx.AsyncClient | None = None
_client_loop: object | None = None   # 记录 client 创建时所在的事件循环


def get_client() -> httpx.AsyncClient:
    """获取共享的 AsyncClient，**但跨事件循环时必须重建**。

    这是实测踩到的坑：httpx.AsyncClient 绑定在创建时的事件循环上，
    如果在一次 `asyncio.run()` 里创建、另一次 `asyncio.run()` 里复用，
    会直接抛 `RuntimeError: Event loop is closed`。
    单测（每个用例一次 run_sync）与评测脚本（多次调用）都会触发，
    所以这里记录创建时的 loop，发现变了就重建。
    """
    global _client, _client_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if _client is None or _client.is_closed or (loop is not None and loop is not _client_loop):
        if _client is not None and not _client.is_closed:
            try:
                loop.run_until_complete(_client.aclose())
            except Exception:
                pass
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.LLM_TIMEOUT, connect=5.0),
            # 不走系统代理：走代理会给每次调用叠加不确定的延迟，
            # 评测数据不可信；且部分本机代理对 absolute-URI 请求直接回 404。
            trust_env=config.UPSTREAM_TRUST_ENV,
        )
        _client_loop = loop
    return _client


async def close_client():
    global _client, _client_loop
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
    _client_loop = None


class UpstreamUnavailableError(Exception):
    """上游不可用 / 重试耗尽时抛出，上层据此终止 Agent 或降级。"""


# ---------- Mock 模式 ----------
async def mock_chat(messages, temperature, max_tokens) -> tuple[str, dict]:
    """无 Key 时返回一个「立即结束」的 final，让 Agent 循环能跑通（仅验证链路）。

    返回 (文本, usage)：usage 与真实调用同构，便于下游 trace 代码无需分支。
    """
    await asyncio.sleep(0.02)
    return json_final("（MOCK）未配置 API Key，Agent 直接结束。配置 SILICONFLOW_API_KEY 后重试。"), \
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ---------- 真实调用（精简版韧性：超时 + 有限重试） ----------
async def chat_completion_with_usage(
    messages: list, temperature: float | None = None, max_tokens: int | None = None
) -> tuple[str, dict]:
    """非流式调用，返回 (完整文本, usage)。

    与旧版 chat_completion 的区别：**把 usage 一并返回**。
    原因：评估 Agent 的成本必须知道 token 消耗，而 OpenAI 兼容接口本来就
    返回 usage，之前丢掉它等于把成本度量能力扔掉了。
    """
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens

    if config.MOCK_MODE or not config.API_KEY:
        return await mock_chat(messages, temperature, max_tokens)

    payload = {
        "model": config.MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err: Exception | None = None
    for attempt in range(3):  # 最多重试 2 次
        try:
            resp = await get_client().post(
                f"{config.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {config.API_KEY}"},
                json=payload,
            )
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = e
            await asyncio.sleep(0.5 * (2 ** attempt))
            continue

        if resp.status_code < 400:
            data = resp.json()
            usage = data.get("usage") or {}
            return data["choices"][0]["message"]["content"], {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
        if resp.status_code >= 500:  # 5xx 重试，4xx 直接抛
            last_err = httpx.HTTPStatusError(
                f"HTTP {resp.status_code}", request=resp.request, response=resp
            )
            await asyncio.sleep(0.5 * (2 ** attempt))
            continue
        resp.raise_for_status()

    raise UpstreamUnavailableError(f"上游调用失败：{last_err}")


async def chat_completion(messages: list, temperature: float | None = None,
                          max_tokens: int | None = None) -> str:
    """兼容旧调用方：只要文本，不要 usage。"""
    text, _ = await chat_completion_with_usage(messages, temperature, max_tokens)
    return text


# ---------- 工具：解析模型输出 ----------
def json_final(content: str) -> str:
    """构造一个合法的 final JSON（Mock / 兜底用）。"""
    import json
    return json.dumps({"type": "final", "content": content}, ensure_ascii=False)
