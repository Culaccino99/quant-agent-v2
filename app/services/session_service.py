"""Redis 短期记忆：用户会话上下文管理（基于 Redis ZSet，score=时间戳，member=消息JSON）"""
from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger

from app.services.redis_client import redis

_SESSION_KEY_PREFIX = "agent:session:"
_MAX_ROUNDS = 5  # 保留最近 5 轮对话
_MAX_MESSAGES = _MAX_ROUNDS * 2  # 每轮 = 1 user + 1 assistant = 2 条
_SESSION_TTL = 43200  # 12 小时


def _key(user_id: str) -> str:
    return f"{_SESSION_KEY_PREFIX}{user_id}"


async def _ensure_zset(key: str) -> None:
    """如果 key 存在但类型不是 zset（旧数据），先删除"""
    key_type = await redis.type(key)
    if key_type not in ("zset", "none"):
        logger.warning("[Session] key 类型不匹配({}), 删除旧数据: {}", key_type, key)
        await redis.delete(key)


async def get_chat_history(user_id: str) -> list[dict[str, Any]]:
    """从 ZSet 按 score(时间戳) 倒序取最新 5 轮对话"""
    key = _key(user_id)
    await _ensure_zset(key)

    raw_list = await redis.zrange(key, -_MAX_MESSAGES, -1)
    if not raw_list:
        return []

    history = []
    for raw in raw_list:
        try:
            history.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue

    return history


async def append_message(user_id: str, role: str, content: str) -> None:
    """追加一条消息到 ZSet，score 为当前时间戳，自动裁剪只保留最新 5 轮"""
    key = _key(user_id)
    await _ensure_zset(key)

    score = time.time()
    member = json.dumps({"role": role, "content": content, "ts": score}, ensure_ascii=False)

    await redis.zadd(key, {member: score})

    total = await redis.zcard(key)
    if total > _MAX_MESSAGES:
        remove_count = total - _MAX_MESSAGES
        await redis.zremrangebyrank(key, 0, remove_count - 1)

    await redis.expire(key, _SESSION_TTL)


async def clear_history(user_id: str) -> None:
    """清除用户会话历史"""
    await redis.delete(_key(user_id))
    logger.info("[Session] 历史已清除: user={}", user_id)
