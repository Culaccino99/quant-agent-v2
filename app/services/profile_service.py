"""MySQL 长期记忆：用户画像管理"""
from __future__ import annotations

import json
from typing import Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import async_session_factory
from app.models.schemas import User, UserProfile


async def get_or_create_user(feishu_open_id: str, nickname: Optional[str] = None) -> User:
    """根据飞书 open_id 获取或创建用户"""
    async with async_session_factory() as session:
        stmt = select(User).where(User.feishu_open_id == feishu_open_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

        if user is None:
            user = User(feishu_open_id=feishu_open_id, nickname=nickname)
            session.add(user)
            await session.commit()
            await session.refresh(user)

        return user


async def get_user_profile_json(feishu_open_id: str) -> Optional[str]:
    """
    获取用户画像，返回 JSON 字符串供 Agent System Prompt 注入。
    如果用户或画像不存在，返回 None。
    """
    async with async_session_factory() as session:
        stmt = (
            select(User, UserProfile)
            .outerjoin(UserProfile, User.id == UserProfile.user_id)
            .where(User.feishu_open_id == feishu_open_id)
        )
        result = await session.execute(stmt)
        row = result.first()

        if row is None:
            return None

        user, profile = row
        if profile is None:
            return json.dumps({
                "user_id": user.id,
                "nickname": user.nickname,
                "investment_style": "unknown",
                "focus_sectors": [],
                "risk_tolerance": 5,
            }, ensure_ascii=False)

        return json.dumps({
            "user_id": user.id,
            "nickname": user.nickname,
            "investment_style": profile.investment_style,
            "focus_sectors": profile.focus_sectors or [],
            "risk_tolerance": profile.risk_tolerance,
        }, ensure_ascii=False)


async def update_user_profile(
    feishu_open_id: str,
    investment_style: Optional[str] = None,
    focus_sectors: Optional[list[str]] = None,
    risk_tolerance: Optional[int] = None,
    merge_sectors: bool = True,
) -> None:
    """
    更新用户画像（不存在则创建）。
    merge_sectors=True 时，focus_sectors 会与已有板块合并（取并集），而非覆盖。
    """
    async with async_session_factory() as session:
        user = await _get_user(session, feishu_open_id)
        if user is None:
            logger.warning("[Profile] 用户不存在: {}", feishu_open_id)
            return

        stmt = select(UserProfile).where(UserProfile.user_id == user.id)
        result = await session.execute(stmt)
        profile = result.scalar_one_or_none()

        is_new = profile is None
        if is_new:
            profile = UserProfile(user_id=user.id)
            session.add(profile)

        old_style = profile.investment_style if not is_new else None
        old_sectors = (profile.focus_sectors or []) if not is_new else []
        old_risk = profile.risk_tolerance if not is_new else None

        if investment_style is not None:
            profile.investment_style = investment_style

        if focus_sectors is not None:
            if merge_sectors and old_sectors:
                merged = list(dict.fromkeys(old_sectors + focus_sectors))
                profile.focus_sectors = merged
            else:
                profile.focus_sectors = focus_sectors

        if risk_tolerance is not None:
            profile.risk_tolerance = risk_tolerance

        await session.commit()

        logger.info("[Profile] 画像{}: user={}", "创建" if is_new else "更新", feishu_open_id)
        if investment_style and investment_style != old_style:
            logger.info("[Profile]   investment_style: {} → {}", old_style, investment_style)
        if focus_sectors:
            logger.info("[Profile]   focus_sectors: {} → {}", old_sectors, profile.focus_sectors)
        if risk_tolerance and risk_tolerance != old_risk:
            logger.info("[Profile]   risk_tolerance: {} → {}", old_risk, risk_tolerance)


async def _get_user(session: AsyncSession, feishu_open_id: str) -> Optional[User]:
    stmt = select(User).where(User.feishu_open_id == feishu_open_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
