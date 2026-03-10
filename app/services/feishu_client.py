"""飞书 API 封装：Token 管理、发送文本消息、发送消息卡片"""
from __future__ import annotations

import json
from typing import Any, Optional

import httpx
from loguru import logger

from app.config import settings
from app.services.redis_client import redis

_FEISHU_BASE = "https://open.feishu.cn/open-apis"
_TOKEN_CACHE_KEY = "feishu:tenant_access_token"
_TOKEN_TTL = 7000  # 飞书 token 有效期 7200 秒, 提前 200 秒刷新


async def get_tenant_access_token() -> str:
    """获取 Tenant Access Token，优先从 Redis 缓存读取"""
    cached = await redis.get(_TOKEN_CACHE_KEY)
    if cached:
        logger.info("[Feishu] Token 从缓存获取")
        return cached

    logger.info("[Feishu] Token 缓存未命中, 请求新 Token...")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_FEISHU_BASE}/auth/v3/tenant_access_token/internal",
            json={
                "app_id": settings.feishu_app_id,
                "app_secret": settings.feishu_app_secret,
            },
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"获取飞书 Token 失败: {data}")
        token = data["tenant_access_token"]
        await redis.set(_TOKEN_CACHE_KEY, token, ex=_TOKEN_TTL)
        logger.info("[Feishu] Token 已刷新并缓存 (TTL={}s)", _TOKEN_TTL)
        return token


async def _feishu_headers() -> dict[str, str]:
    token = await get_tenant_access_token()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }


async def send_text_message(receive_id: str, text: str, receive_id_type: str = "open_id") -> dict:
    """发送纯文本消息"""
    logger.info("[Feishu] 发送文本消息: to={}, type={}, len={}",
                receive_id, receive_id_type, len(text))
    headers = await _feishu_headers()
    payload = {
        "receive_id": receive_id,
        "msg_type": "text",
        "content": json.dumps({"text": text}),
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_FEISHU_BASE}/im/v1/messages?receive_id_type={receive_id_type}",
            headers=headers,
            json=payload,
            timeout=10,
        )
        result = resp.json()
        if result.get("code") != 0:
            logger.error("[Feishu] 发送文本消息失败: {}", result)
        else:
            logger.info("[Feishu] 文本消息发送成功")
        return result


async def send_card_message(
    receive_id: str,
    card: dict[str, Any],
    receive_id_type: str = "open_id",
) -> dict:
    """发送消息卡片"""
    logger.info("[Feishu] 发送卡片消息: to={}, type={}", receive_id, receive_id_type)
    headers = await _feishu_headers()
    payload = {
        "receive_id": receive_id,
        "msg_type": "interactive",
        "content": json.dumps(card),
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_FEISHU_BASE}/im/v1/messages?receive_id_type={receive_id_type}",
            headers=headers,
            json=payload,
            timeout=10,
        )
        result = resp.json()
        if result.get("code") != 0:
            logger.error("[Feishu] 发送卡片消息失败: {}", result)
        else:
            logger.info("[Feishu] 卡片消息发送成功")
        return result


def build_analysis_card(title: str, content_md: str) -> dict[str, Any]:
    """构建 AI 分析结果的飞书消息卡片"""
    return {
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": "blue",
        },
        "elements": [
            {
                "tag": "markdown",
                "content": content_md,
            },
            {
                "tag": "hr",
            },
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": "由 AI 投研 Agent 生成，仅供参考",
                    }
                ],
            },
        ],
    }


async def download_file(message_id: str, file_key: str) -> bytes:
    """从飞书下载消息中的文件，返回文件二进制内容"""
    logger.info("[Feishu] 下载文件: msg_id={}, file_key={}", message_id, file_key)
    token = await get_tenant_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{_FEISHU_BASE}/im/v1/messages/{message_id}/resources/{file_key}?type=file"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"飞书文件下载失败: status={resp.status_code}")
        logger.info("[Feishu] 文件下载成功: size={}KB", len(resp.content) // 1024)
        return resp.content


async def reply_to_message(message_id: str, text: str) -> dict:
    """回复指定消息"""
    logger.info("[Feishu] 回复消息: msg_id={}, len={}", message_id, len(text))
    headers = await _feishu_headers()
    payload = {
        "msg_type": "text",
        "content": json.dumps({"text": text}),
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_FEISHU_BASE}/im/v1/messages/{message_id}/reply",
            headers=headers,
            json=payload,
            timeout=10,
        )
        result = resp.json()
        if result.get("code") != 0:
            logger.error("[Feishu] 回复消息失败: {}", result)
        else:
            logger.info("[Feishu] 回复消息成功")
        return result
