"""LLM 客户端：封装 DeepSeek 调用（OpenAI 兼容接口）"""
from typing import Optional

from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import settings


def get_deepseek_llm(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: int = 60,
) -> ChatOpenAI:
    """构建 DeepSeek LLM 实例，参数优先，缺省从配置读取"""
    _api_key = api_key or settings.deepseek_api_key
    _base_url = base_url or settings.deepseek_base_url
    _model = model or settings.deepseek_model
    _temperature = temperature if temperature is not None else settings.agent_temperature

    if not _api_key:
        raise ValueError("未配置 DEEPSEEK_API_KEY")

    return ChatOpenAI(
        api_key=_api_key,
        base_url=_base_url.rstrip("/"),
        model=_model,
        temperature=_temperature,
        timeout=timeout,
    )
