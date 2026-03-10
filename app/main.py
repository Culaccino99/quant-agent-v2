"""FastAPI 入口：lifespan 管理数据库/Redis 连接池 + 后台监控协程"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.config import settings
from app.utils.logger import setup_logger


def _setup_langsmith():
    """如果配置了 LangSmith，设置环境变量启用 tracing"""
    import os
    if settings.langchain_tracing_v2.lower() == "true" and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info("LangSmith tracing 已启用: project={}", settings.langchain_project)
    else:
        logger.info("LangSmith tracing 未启用")


@asynccontextmanager
async def lifespan(application: FastAPI):
    setup_logger(settings.log_level)
    _setup_langsmith()
    logger.info("启动服务...")

    # 初始化 MySQL 连接池
    from app.models.database import engine
    logger.info("MySQL 连接池就绪: {}", settings.mysql_host)

    # 初始化 Redis 连接池
    from app.services.redis_client import redis
    await redis.ping()
    logger.info("Redis 连接就绪: {}", settings.redis_host)

    # 启动预警监控后台协程
    from app.services.alert_service import start_monitor
    start_monitor()

    yield

    # 停止预警监控
    from app.services.alert_service import stop_monitor
    stop_monitor()

    # 关闭连接
    from app.models.database import engine
    await engine.dispose()
    from app.services.redis_client import redis
    await redis.aclose()
    logger.info("服务已关闭")


app = FastAPI(
    title="A股智能投研与预警 Agent",
    version="2.0.0",
    lifespan=lifespan,
)

# 注册路由
from app.api.health import router as health_router  # noqa: E402
from app.api.feishu import router as feishu_router   # noqa: E402
from app.api.alert import router as alert_router     # noqa: E402

app.include_router(health_router)
app.include_router(feishu_router, prefix="/api/feishu")
app.include_router(alert_router, prefix="/api/alert")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8090, reload=True)
