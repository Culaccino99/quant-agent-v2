"""Tool: 文档 RAG 检索。在用户上传的文档库中按语义相似度检索相关内容"""
from __future__ import annotations

import time

from langchain_core.tools import tool
from loguru import logger

from app.services.vector_store import search_docs


@tool
def search_doc_rag(query: str, top_k: int = 5) -> str:
    """
    在用户上传的文档库（研报、分析报告等）中，按语义相似度检索与 query 最相关的段落。
    当用户提问涉及其上传的文档、研报、资料内容时调用此工具。

    Args:
        query: 检索关键词或自然语言问句，如「这份研报对半导体行业怎么看」。
        top_k: 返回最多几条结果，默认 5。

    Returns:
        格式化的检索结果文本；若无结果则返回说明文字。
    """
    logger.info("[Tool:search_doc_rag] ▶ 被调用: query='{}', top_k={}", query, top_k)
    start = time.time()
    try:
        results = search_docs(query, top_k=top_k)
        elapsed = time.time() - start
        if not results:
            logger.info("[Tool:search_doc_rag] 无结果, 耗时={:.2f}s", elapsed)
            return "文档库中未检索到相关内容。可能尚未上传文档，或 query 与文档内容不匹配。"
        logger.info("[Tool:search_doc_rag] 返回 {} 条结果, 最高相关度={:.4f}, 耗时={:.2f}s",
                    len(results), results[0][2] if results else 0, elapsed)
        lines = [f"【文档 RAG 检索】query: {query}", ""]
        for i, (text, filename, score) in enumerate(results, 1):
            lines.append(f"{i}. [相关度 {score:.4f}] [{filename}] {text}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("[Tool:search_doc_rag] 检索失败: {}", e)
        return f"文档 RAG 检索失败：{e}。请确认 Qdrant 已启动。"
