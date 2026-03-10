"""Tool: 预警规则管理。供 Agent 通过自然语言创建/查询/删除/暂停预警规则"""
from __future__ import annotations

import asyncio
from typing import Optional

from langchain_core.tools import tool
from loguru import logger

from app.agent.context import current_user_id, run_on_main_loop


def _get_user_id() -> str:
    uid = current_user_id.get()
    if not uid:
        raise RuntimeError("未获取到当前用户身份，无法操作预警规则")
    return uid


@tool
def create_alert_rule(
    stock_code: str,
    rule_type: str,
    threshold: float,
    stock_name: Optional[str] = None,
    unit: str = "percent",
    cooldown_minutes: int = 60,
) -> str:
    """
    为当前用户创建一条股票预警规则。

    Args:
        stock_code: 6位股票代码，如 "300567"、"600519"。
        rule_type: 规则类型，可选值：
            - "price_up"：涨幅预警（涨幅达到阈值时触发）
            - "price_down"：跌幅预警（跌幅达到阈值时触发）
            - "turnover"：换手率预警（换手率达到阈值时触发）
        threshold: 阈值数字。如 rule_type="price_up", threshold=5 表示涨幅≥5%时触发。
        stock_name: 股票名称（可选），如 "精测电子"。
        unit: 单位，默认 "percent"（百分比），也可以是 "absolute"（绝对价格）。
        cooldown_minutes: 冷却时间（分钟），同一规则触发后多久内不重复触发，默认60分钟。

    Returns:
        创建结果说明文字。
    """
    user_id = _get_user_id()
    logger.info("[Tool:alert] ▶ 创建规则: user={}, stock={}, type={}, threshold={}",
                user_id, stock_code, rule_type, threshold)
    try:
        from app.services.alert_service import create_rule
        rule = run_on_main_loop(create_rule(
            feishu_open_id=user_id,
            stock_code=stock_code,
            stock_name=stock_name,
            rule_type=rule_type,
            threshold=threshold,
            unit=unit,
            cooldown_minutes=cooldown_minutes,
        ))
        type_map = {"price_up": "涨幅", "price_down": "跌幅", "turnover": "换手率", "volume": "成交量"}
        type_cn = type_map.get(rule_type, rule_type)
        name = stock_name or stock_code
        return (
            f"✅ 预警规则已创建（ID: {rule.id}）\n"
            f"股票：{name}（{stock_code}）\n"
            f"类型：{type_cn} ≥ {threshold}{unit}\n"
            f"冷却：{cooldown_minutes} 分钟\n"
            f"系统将自动监控行情，命中时推送分析报告。"
        )
    except Exception as e:
        logger.error("[Tool:alert] 创建规则失败: {}", e)
        return f"创建预警规则失败：{e}"


@tool
def list_alert_rules() -> str:
    """
    查询当前用户的所有预警规则列表。
    当用户询问"我有哪些预警"、"查看预警规则"、"我的预警列表"时调用此工具。

    Returns:
        格式化的预警规则列表文字。
    """
    user_id = _get_user_id()
    logger.info("[Tool:alert] ▶ 查询规则: user={}", user_id)
    try:
        from app.services.alert_service import list_rules
        rules = run_on_main_loop(list_rules(feishu_open_id=user_id, only_active=False))
        if not rules:
            return "你当前没有任何预警规则。可以告诉我你想监控哪只股票，我来帮你创建。"

        type_map = {"price_up": "涨幅", "price_down": "跌幅", "turnover": "换手率", "volume": "成交量", "custom": "自定义"}
        status_map = {"active": "🟢 生效中", "paused": "⏸️ 已暂停", "deleted": "🔴 已删除"}
        lines = [f"📋 你共有 {len(rules)} 条预警规则：\n"]
        for r in rules:
            type_cn = type_map.get(r.rule_type, r.rule_type)
            status_cn = status_map.get(r.status, r.status)
            name = r.stock_name or r.stock_code
            lines.append(
                f"• ID {r.id} | {name}({r.stock_code}) | {type_cn} ≥ {float(r.threshold)}{r.unit} | {status_cn}"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error("[Tool:alert] 查询规则失败: {}", e)
        return f"查询预警规则失败：{e}"


@tool
def delete_alert_rule(rule_id: int) -> str:
    """
    删除指定 ID 的预警规则。
    当用户说"删除预警规则X"、"取消规则X"时调用此工具。

    Args:
        rule_id: 要删除的预警规则 ID。

    Returns:
        删除结果说明文字。
    """
    logger.info("[Tool:alert] ▶ 删除规则: id={}", rule_id)
    try:
        from app.services.alert_service import delete_rule
        run_on_main_loop(delete_rule(rule_id))
        return f"✅ 预警规则 ID {rule_id} 已删除。"
    except Exception as e:
        logger.error("[Tool:alert] 删除规则失败: {}", e)
        return f"删除预警规则失败：{e}"


@tool
def toggle_alert_rule(rule_id: int, action: str) -> str:
    """
    暂停或恢复指定 ID 的预警规则。
    当用户说"暂停规则X"或"恢复规则X"时调用此工具。

    Args:
        rule_id: 预警规则 ID。
        action: "pause" 暂停，"resume" 恢复。

    Returns:
        操作结果说明文字。
    """
    logger.info("[Tool:alert] ▶ {}规则: id={}", action, rule_id)
    try:
        from app.services.alert_service import pause_rule, resume_rule
        if action == "pause":
            run_on_main_loop(pause_rule(rule_id))
            return f"⏸️ 预警规则 ID {rule_id} 已暂停。"
        elif action == "resume":
            run_on_main_loop(resume_rule(rule_id))
            return f"🟢 预警规则 ID {rule_id} 已恢复监控。"
        else:
            return f"未知操作: {action}，请使用 pause 或 resume。"
    except Exception as e:
        logger.error("[Tool:alert] 操作规则失败: {}", e)
        return f"操作预警规则失败：{e}"
