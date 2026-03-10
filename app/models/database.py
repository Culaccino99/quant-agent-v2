"""SQLAlchemy 异步引擎 & Session 工厂"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(
    settings.mysql_dsn,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    echo=False,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncSession:
    """FastAPI 依赖注入用：每次请求获取独立 session"""
    async with async_session_factory() as session:
        yield session
