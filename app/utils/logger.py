"""日志配置：使用 loguru 替代标准 logging"""
import sys

from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    # INFO 使用白色，ERROR 保持红色，WARNING 保持黄色
    logger.level("INFO", color="<white>")
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
