"""预警服务：规则 CRUD + 后台行情监控 + 触发 Agent 分析推送"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger
from sqlalchemy import select, update

from app.config import settings
from app.models.database import async_session_factory
from app.models.schemas import AlertHistory, AlertRule, User

# ============================================================
# CRUD
# ============================================================

async def create_rule(
    feishu_open_id: str,
    stock_code: str,
    stock_name: Optional[str],
    rule_type: str,
    threshold: float,
    unit: str = "percent",
    cooldown_minutes: int = 60,
) -> AlertRule:
    """创建预警规则"""
    async with async_session_factory() as session:
        user = await _get_user_by_openid(session, feishu_open_id)
        if user is None:
            raise ValueError(f"用户不存在: {feishu_open_id}")

        rule = AlertRule(
            user_id=user.id,
            stock_code=stock_code,
            stock_name=stock_name,
            rule_type=rule_type,
            threshold=float(threshold),
            unit=unit,
            cooldown_minutes=cooldown_minutes,
        )
        session.add(rule)
        await session.commit()
        await session.refresh(rule)
        logger.info("[Alert] 规则创建: id={}, user={}, stock={}, type={}, threshold={}{}",
                    rule.id, feishu_open_id, stock_code, rule_type, threshold, unit)
        return rule


async def list_rules(feishu_open_id: str, only_active: bool = True) -> list[AlertRule]:
    """列出用户的预警规则"""
    async with async_session_factory() as session:
        user = await _get_user_by_openid(session, feishu_open_id)
        if user is None:
            return []
        stmt = select(AlertRule).where(AlertRule.user_id == user.id)
        if only_active:
            stmt = stmt.where(AlertRule.status == "active")
        result = await session.execute(stmt)
        rules = list(result.scalars().all())
        logger.info("[Alert] 查询规则: user={}, count={}", feishu_open_id, len(rules))
        return rules


async def update_rule(rule_id: int, **kwargs) -> None:
    """更新预警规则字段"""
    async with async_session_factory() as session:
        stmt = update(AlertRule).where(AlertRule.id == rule_id).values(**kwargs)
        await session.execute(stmt)
        await session.commit()
        logger.info("[Alert] 规则更新: id={}, fields={}", rule_id, list(kwargs.keys()))


async def delete_rule(rule_id: int) -> None:
    """软删除预警规则"""
    await update_rule(rule_id, status="deleted")
    logger.info("[Alert] 规则删除(软): id={}", rule_id)


async def pause_rule(rule_id: int) -> None:
    await update_rule(rule_id, status="paused")


async def resume_rule(rule_id: int) -> None:
    await update_rule(rule_id, status="active")


# ============================================================
# 后台行情监控协程
# ============================================================

_monitor_task: Optional[asyncio.Task] = None


def start_monitor():
    """启动后台监控协程（在 lifespan startup 中调用）"""
    global _monitor_task
    if _monitor_task is not None and not _monitor_task.done():
        logger.warning("[AlertMonitor] 监控协程已在运行, 跳过")
        return
    _monitor_task = asyncio.create_task(_monitor_loop())
    logger.info("[AlertMonitor] 后台监控协程已启动, 轮询间隔={}s", settings.alert_poll_interval)


def stop_monitor():
    """停止后台监控协程（在 lifespan shutdown 中调用）"""
    global _monitor_task
    if _monitor_task and not _monitor_task.done():
        _monitor_task.cancel()
        logger.info("[AlertMonitor] 后台监控协程已停止")


async def _monitor_loop():
    """主监控循环：轮询行情 → 比对规则 → 触发分析"""
    await asyncio.sleep(5)  # 等待服务完全启动
    logger.info("[AlertMonitor] 监控循环开始")

    while True:
        try:
            await _poll_and_check()
        except asyncio.CancelledError:
            logger.info("[AlertMonitor] 监控循环被取消")
            break
        except Exception as e:
            logger.error("[AlertMonitor] 轮询异常: {}\n{}", e, traceback.format_exc())

        await asyncio.sleep(settings.alert_poll_interval)


async def _poll_and_check():
    """单次轮询：获取所有活跃规则 → 批量查行情 → 逐条比对"""
    # 0. 时间窗口检查：仅在 09:15 - 15:00 之间推送预警（交易时段）
    now = datetime.now()
    h, m = now.hour, now.minute
    if h < 9 or (h == 9 and m < 15) or h > 15 or (h == 15 and m > 0):
        return

    # 1. 读取所有活跃规则
    async with async_session_factory() as session:
        stmt = (
            select(AlertRule, User)
            .join(User, AlertRule.user_id == User.id)
            .where(AlertRule.status == "active")
        )
        result = await session.execute(stmt)
        rows = result.all()

    if not rows:
        return

    # 2. 收集需要查询的股票代码（去重）
    stock_codes = list({rule.stock_code for rule, _ in rows})
    logger.info("[AlertMonitor] 本轮检查: {} 条规则, {} 只股票", len(rows), len(stock_codes))

    # 3. 批量获取实时行情
    quotes = await asyncio.to_thread(_fetch_realtime_quotes, stock_codes)
    if not quotes:
        logger.warning("[AlertMonitor] 行情数据为空, 跳过本轮")
        return

    logger.info("[AlertMonitor] 获取行情成功: {} 只股票", len(quotes))

    # 4. 逐条比对规则
    for rule, user in rows:
        quote = quotes.get(rule.stock_code)
        if not quote:
            continue

        # 冷却检查
        if rule.last_triggered_at:
            cooldown_end = rule.last_triggered_at + timedelta(minutes=rule.cooldown_minutes)
            if now < cooldown_end:
                continue

        triggered, reason, trigger_value = _check_rule(rule, quote)
        if triggered:
            logger.info("[AlertMonitor] 🔔 规则命中: rule_id={}, stock={}, reason={}",
                        rule.id, rule.stock_code, reason)
            asyncio.create_task(
                _on_rule_triggered(rule, user, quote, reason, trigger_value)
            )


def _fetch_realtime_quotes(stock_codes: list[str]) -> dict[str, dict[str, Any]]:
    """
    批量获取实时行情数据。
    返回 {stock_code: {"price": float, "change_pct": float, "volume": float, "turnover_rate": float, "name": str}}
    """
    for key in list(os.environ):
        if "proxy" in key.lower() and key != "GOPROXY":
            os.environ.pop(key, None)

    import requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://quote.eastmoney.com/",
    }

    quotes: dict[str, dict[str, Any]] = {}

    for code in stock_codes:
        market = 1 if code.startswith("6") else 0
        params = {
            "fltt": "2", "invt": "2",
            "fields": "f43,f57,f58,f170,f47,f168",
            "secid": f"{market}.{code}",
        }
        try:
            session = requests.Session()
            session.trust_env = False
            session.proxies = {"http": None, "https": None}
            r = session.get(
                "http://push2.eastmoney.com/api/qt/stock/get",
                params=params, headers=headers, timeout=8,
            )
            data = r.json()
            if not data or data.get("rc") != 0:
                continue
            raw = data.get("data") or {}
            quotes[code] = {
                "price": raw.get("f43", 0),
                "change_pct": raw.get("f170", 0),
                "volume": raw.get("f47", 0),
                "turnover_rate": raw.get("f168", 0),
                "name": raw.get("f58", code),
            }
        except Exception as e:
            logger.warning("[AlertMonitor] 获取 {} 行情失败: {}", code, e)
            continue

        time.sleep(0.3)

    return quotes


def _check_rule(rule: AlertRule, quote: dict[str, Any]) -> tuple[bool, str, float]:
    """
    检查单条规则是否命中。
    返回 (是否命中, 原因描述, 触发值)
    """
    change_pct = quote.get("change_pct", 0) or 0
    price = quote.get("price", 0) or 0
    volume = quote.get("volume", 0) or 0
    turnover = quote.get("turnover_rate", 0) or 0
    threshold = float(rule.threshold)

    if rule.rule_type == "price_up":
        if rule.unit == "percent" and change_pct >= threshold:
            return True, f"涨幅 {change_pct}% ≥ 阈值 {threshold}%", change_pct
        if rule.unit == "absolute" and price >= threshold:
            return True, f"价格 {price} ≥ 阈值 {threshold}", price

    elif rule.rule_type == "price_down":
        if rule.unit == "percent" and change_pct <= -threshold:
            return True, f"跌幅 {change_pct}% ≤ -{threshold}%", change_pct
        if rule.unit == "absolute" and price <= threshold:
            return True, f"价格 {price} ≤ 阈值 {threshold}", price

    elif rule.rule_type == "volume":
        if rule.unit == "times" and volume > 0:
            return True, f"成交量 {volume}", volume

    elif rule.rule_type == "turnover":
        if turnover >= threshold:
            return True, f"换手率 {turnover}% ≥ 阈值 {threshold}%", turnover

    return False, "", 0


# ============================================================
# Step 5.2: 规则命中 → Agent 分析 → 飞书推送
# ============================================================

async def _on_rule_triggered(
    rule: AlertRule, user: User, quote: dict[str, Any], reason: str, trigger_value: float,
):
    """规则命中后的完整处理链：更新触发时间 → Agent 分析 → 记录历史 → 飞书推送"""
    task_start = time.time()
    stock_name = quote.get("name", rule.stock_name or rule.stock_code)
    logger.info("[AlertTrigger] ▶ 开始处理: rule_id={}, stock={}({}), reason={}",
                rule.id, rule.stock_code, stock_name, reason)

    # 把主事件循环存入 contextvars，供工作线程中的工具调度异步操作
    from app.agent.context import main_event_loop
    main_event_loop.set(asyncio.get_running_loop())

    try:
        # 1. 更新规则的最后触发时间
        await update_rule(rule.id, last_triggered_at=datetime.now())
        logger.info("[AlertTrigger] Step1 触发时间已更新")

        # 2. 构造 Agent 查询
        query = (
            f"【预警触发】{stock_name}({rule.stock_code}) {reason}。"
            f"当前价格 {quote.get('price', '?')} 元，"
            f"涨跌幅 {quote.get('change_pct', '?')}%，"
            f"换手率 {quote.get('turnover_rate', '?')}%。"
            f"请进行深度归因分析。"
        )
        logger.info("[AlertTrigger] Step2 Agent 查询: {}", query)

        # 3. 获取用户画像和历史
        from app.services.profile_service import get_user_profile_json
        from app.services.session_service import get_chat_history
        user_profile = await get_user_profile_json(user.feishu_open_id)
        chat_history = await get_chat_history(user.feishu_open_id)

        # 4. 调用 Agent
        from app.agent.agent import run_agent
        logger.info("[AlertTrigger] Step3 调用 Agent...")
        agent_start = time.time()
        report = await asyncio.to_thread(
            run_agent, query=query, user_profile=user_profile,
            chat_history=chat_history, feishu_open_id=user.feishu_open_id,
        )
        agent_elapsed = time.time() - agent_start
        logger.info("[AlertTrigger] Step3 Agent 完成, 耗时={:.1f}s, 输出长度={}", agent_elapsed, len(report))

        # 5. 记录预警历史
        async with async_session_factory() as session:
            history = AlertHistory(
                rule_id=rule.id,
                user_id=rule.user_id,
                stock_code=rule.stock_code,
                stock_name=stock_name,
                trigger_price=quote.get("price"),
                trigger_value=trigger_value,
                trigger_reason=reason,
                analysis_result=report,
            )
            session.add(history)
            await session.commit()
            logger.info("[AlertTrigger] Step4 预警历史已记录")

        # 6. 推送飞书卡片
        from app.services.feishu_client import build_analysis_card, send_card_message
        card = build_analysis_card(
            title=f"🔔 预警：{stock_name}({rule.stock_code}) {reason}",
            content_md=report,
        )
        result = await send_card_message(
            receive_id=user.feishu_open_id, card=card, receive_id_type="open_id",
        )
        logger.info("[AlertTrigger] Step5 飞书卡片已推送: code={}", result.get("code", "?"))

        total_elapsed = time.time() - task_start
        logger.info("[AlertTrigger] ✅ 预警处理完成: rule_id={}, stock={}, 总耗时={:.1f}s",
                    rule.id, rule.stock_code, total_elapsed)

    except Exception as e:
        total_elapsed = time.time() - task_start
        logger.error("[AlertTrigger] ❌ 预警处理失败(耗时{:.1f}s): {}\n{}",
                     total_elapsed, e, traceback.format_exc())


# ============================================================
# 辅助
# ============================================================

async def _get_user_by_openid(session, feishu_open_id: str) -> Optional[User]:
    stmt = select(User).where(User.feishu_open_id == feishu_open_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
