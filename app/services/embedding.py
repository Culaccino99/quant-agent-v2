"""Embedding 服务：BGE-m3 向量化"""
from __future__ import annotations

import time
from typing import List

from loguru import logger

from app.config import settings

EMBEDDING_DIM = 1024

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("[Embedding] 加载模型: {}", settings.embedding_model)
        start = time.time()
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("[Embedding] 模型加载完成, 耗时={:.1f}s", time.time() - start)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """将多段文本转为向量列表"""
    logger.info("[Embedding] 批量编码: {} 条文本", len(texts))
    start = time.time()
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    elapsed = time.time() - start
    logger.info("[Embedding] 编码完成: {} 条, dim={}, 耗时={:.2f}s",
                len(vectors), len(vectors[0]) if len(vectors) > 0 else 0, elapsed)
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> List[float]:
    """单条查询文本向量化"""
    logger.info("[Embedding] 查询编码: '{}'", query[:100])
    return embed_texts([query])[0]
