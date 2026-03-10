"""ORM 模型定义：与 init.sql 中的表结构对应"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    feishu_open_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    nickname: Mapped[Optional[str]] = mapped_column(String(64))
    avatar_url: Mapped[Optional[str]] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    profile: Mapped[Optional["UserProfile"]] = relationship(back_populates="user")
    alert_rules: Mapped[list["AlertRule"]] = relationship(back_populates="user")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    investment_style: Mapped[str] = mapped_column(
        Enum("conservative", "balanced", "aggressive"), default="balanced"
    )
    focus_sectors: Mapped[Optional[dict]] = mapped_column(JSON)
    risk_tolerance: Mapped[int] = mapped_column(Integer, default=5)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="profile")


class AlertRule(Base):
    __tablename__ = "alert_rules"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE")
    )
    stock_code: Mapped[str] = mapped_column(String(16), nullable=False)
    stock_name: Mapped[Optional[str]] = mapped_column(String(32))
    rule_type: Mapped[str] = mapped_column(
        Enum("price_up", "price_down", "volume", "turnover", "custom"), nullable=False
    )
    threshold: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    unit: Mapped[str] = mapped_column(
        Enum("percent", "absolute", "times"), default="percent"
    )
    status: Mapped[str] = mapped_column(
        Enum("active", "paused", "deleted"), default="active"
    )
    cooldown_minutes: Mapped[int] = mapped_column(Integer, default=60)
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="alert_rules")
    history: Mapped[list["AlertHistory"]] = relationship(back_populates="rule")


class AlertHistory(Base):
    __tablename__ = "alert_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    rule_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("alert_rules.id", ondelete="CASCADE")
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE")
    )
    stock_code: Mapped[str] = mapped_column(String(16), nullable=False)
    stock_name: Mapped[Optional[str]] = mapped_column(String(32))
    trigger_price: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    trigger_value: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    trigger_reason: Mapped[Optional[str]] = mapped_column(String(256))
    analysis_result: Mapped[Optional[str]] = mapped_column(Text)
    feishu_message_id: Mapped[Optional[str]] = mapped_column(String(64))
    sent_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    rule: Mapped["AlertRule"] = relationship(back_populates="history")
