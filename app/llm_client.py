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


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.LLM_TIMEOUT, connect=5.0)
        )
    return _client


async def close_client():
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


class UpstreamUnavailableError(Exception):
    """上游不可用 / 重试耗尽时抛出，上层据此终止 Agent 或降级。"""


# ---------- Mock 模式 ----------
async def mock_chat(messages, temperature, max_tokens) -> str:
    """无 Key 时返回一个「立即结束」的 final，让 Agent 循环能跑通（仅验证链路）。"""
    await asyncio.sleep(0.02)
    return json_final("（MOCK）未配置 API Key，Agent 直接结束。配置 SILICONFLOW_API_KEY 后重试。")


# ---------- 真实调用（精简版韧性：超时 + 有限重试） ----------
async def chat_completion(messages: list, temperature: float | None = None,
                          max_tokens: int | None = None) -> str:
    """非流式调用，返回完整文本。带超时 + 有限重试（4xx 不重试）。"""
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
            return resp.json()["choices"][0]["message"]["content"]
        if resp.status_code >= 500:  # 5xx 重试，4xx 直接抛
            last_err = httpx.HTTPStatusError(
                f"HTTP {resp.status_code}", request=resp.request, response=resp
            )
            await asyncio.sleep(0.5 * (2 ** attempt))
            continue
        resp.raise_for_status()

    raise UpstreamUnavailableError(f"上游调用失败：{last_err}")


# ---------- 工具：解析模型输出 ----------
def json_final(content: str) -> str:
    """构造一个合法的 final JSON（Mock / 兜底用）。"""
    import json
    return json.dumps({"type": "final", "content": content}, ensure_ascii=False)
