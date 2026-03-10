"""Tool: 新闻 RAG 检索。基于 Qdrant 向量相似度召回相关新闻"""
from __future__ import annotations

import time

from langchain_core.tools import tool
from loguru import logger

from app.services.vector_store import search_news


@tool
def search_news_rag(query: str, top_k: int = 5) -> str:
    """
    在已入库的新闻/研报向量库中，按语义相似度检索与 query 最相关的片段。
    当用户询问某只股票/板块/事件的新闻、研报、舆情时调用此工具。

    Args:
        query: 检索关键词或自然语言问句，如「苯乙烯期货涨停」「精测电子 新闻」。
        top_k: 返回最多几条结果，默认 5。

    Returns:
        格式化的检索结果文本；若无结果或 Qdrant 未就绪则返回说明文字。
    """
    logger.info("[Tool:search_news_rag] ▶ 被调用: query='{}', top_k={}", query, top_k)
    start = time.time()
    try:
        results = search_news(query, top_k=top_k)
        elapsed = time.time() - start
        if not results:
            logger.info("[Tool:search_news_rag] 无结果, 耗时={:.2f}s", elapsed)
            return "未检索到相关新闻或研报，请确认向量库已写入数据且 query 与入库内容相关。"
        logger.info("[Tool:search_news_rag] 返回 {} 条结果, 最高相关度={:.4f}, 耗时={:.2f}s",
                    len(results), results[0][1] if results else 0, elapsed)
        lines = [f"【新闻 RAG 检索】query: {query}", ""]
        for i, (text, score) in enumerate(results, 1):
            lines.append(f"{i}. [相关度 {score:.4f}] {text}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("[Tool:search_news_rag] 检索失败: {}", e)
        return f"新闻 RAG 检索失败：{e}。请确认 Qdrant 已启动（默认 localhost:6333）。"
