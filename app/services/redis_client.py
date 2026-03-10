"""Redis 异步客户端封装"""
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config import settings

redis: aioredis.Redis = aioredis.from_url(
    settings.redis_url,
    decode_responses=True,
)


async def get_value(key: str) -> Optional[str]:
    return await redis.get(key)


async def set_value(key: str, value: Any, ttl: Optional[int] = None) -> None:
    if ttl:
        await redis.set(key, value, ex=ttl)
    else:
        await redis.set(key, value)


async def delete_key(key: str) -> None:
    await redis.delete(key)
