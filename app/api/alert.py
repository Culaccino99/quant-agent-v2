"""预警管理 REST API"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.services.alert_service import (
    create_rule,
    delete_rule,
    list_rules,
    pause_rule,
    resume_rule,
    update_rule,
)

router = APIRouter(tags=["alert"])


# ======================== Request / Response Models ========================

class CreateRuleRequest(BaseModel):
    feishu_open_id: str = Field(..., description="飞书用户 open_id")
    stock_code: str = Field(..., description="6位股票代码", min_length=6, max_length=6)
    stock_name: Optional[str] = Field(None, description="股票名称")
    rule_type: str = Field(..., description="规则类型: price_up/price_down/volume/turnover/custom")
    threshold: float = Field(..., description="阈值")
    unit: str = Field("percent", description="单位: percent/absolute/times")
    cooldown_minutes: int = Field(60, description="冷却时间(分钟)")


class UpdateRuleRequest(BaseModel):
    stock_name: Optional[str] = None
    threshold: Optional[float] = None
    unit: Optional[str] = None
    cooldown_minutes: Optional[int] = None


class RuleResponse(BaseModel):
    id: int
    stock_code: str
    stock_name: Optional[str]
    rule_type: str
    threshold: float
    unit: str
    status: str
    cooldown_minutes: int
    last_triggered_at: Optional[str] = None
    created_at: Optional[str] = None


# ======================== Routes ========================

@router.post("/rules", summary="创建预警规则")
async def api_create_rule(req: CreateRuleRequest):
    logger.info("[AlertAPI] 创建规则: user={}, stock={}, type={}, threshold={}",
                req.feishu_open_id, req.stock_code, req.rule_type, req.threshold)
    try:
        rule = await create_rule(
            feishu_open_id=req.feishu_open_id,
            stock_code=req.stock_code,
            stock_name=req.stock_name,
            rule_type=req.rule_type,
            threshold=req.threshold,
            unit=req.unit,
            cooldown_minutes=req.cooldown_minutes,
        )
        return {"code": 0, "msg": "ok", "data": {"rule_id": rule.id}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/rules", summary="查询用户的预警规则")
async def api_list_rules(feishu_open_id: str, only_active: bool = True):
    rules = await list_rules(feishu_open_id, only_active=only_active)
    data = [
        RuleResponse(
            id=r.id,
            stock_code=r.stock_code,
            stock_name=r.stock_name,
            rule_type=r.rule_type,
            threshold=float(r.threshold),
            unit=r.unit,
            status=r.status,
            cooldown_minutes=r.cooldown_minutes,
            last_triggered_at=r.last_triggered_at.isoformat() if r.last_triggered_at else None,
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in rules
    ]
    return {"code": 0, "msg": "ok", "data": data}


@router.put("/rules/{rule_id}", summary="更新预警规则")
async def api_update_rule(rule_id: int, req: UpdateRuleRequest):
    logger.info("[AlertAPI] 更新规则: id={}", rule_id)
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="无更新字段")
    await update_rule(rule_id, **updates)
    return {"code": 0, "msg": "ok"}


@router.delete("/rules/{rule_id}", summary="删除预警规则")
async def api_delete_rule(rule_id: int):
    logger.info("[AlertAPI] 删除规则: id={}", rule_id)
    await delete_rule(rule_id)
    return {"code": 0, "msg": "ok"}


@router.post("/rules/{rule_id}/pause", summary="暂停预警规则")
async def api_pause_rule(rule_id: int):
    logger.info("[AlertAPI] 暂停规则: id={}", rule_id)
    await pause_rule(rule_id)
    return {"code": 0, "msg": "ok"}


@router.post("/rules/{rule_id}/resume", summary="恢复预警规则")
async def api_resume_rule(rule_id: int):
    logger.info("[AlertAPI] 恢复规则: id={}", rule_id)
    await resume_rule(rule_id)
    return {"code": 0, "msg": "ok"}
