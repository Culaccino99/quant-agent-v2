"""Qdrant 向量库管理：新闻集合 + 文档集合（文档为 dense+sparse 混合检索）"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.services.embedding import EMBEDDING_DIM, embed_query, embed_texts
from app.services.bm25_sparse import get_bm25_encoder

NEWS_COLLECTION = "news_rag"
DOC_COLLECTION = "doc_rag"

# 文档集合使用 named vectors: "dense" + "sparse"，用于 hybrid RRF
DOC_DENSE_VECTOR_NAME = "dense"
DOC_SPARSE_VECTOR_NAME = "sparse"

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        logger.info("[Qdrant] 连接 Qdrant: {}:{}", settings.qdrant_host, settings.qdrant_port)
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        logger.info("[Qdrant] 连接成功")
    return _client


def _ensure_collection(name: str, client: Optional[QdrantClient] = None) -> None:
    """新闻集合：仅 dense 向量。"""
    if client is None:
        client = get_qdrant_client()
    if not client.collection_exists(name):
        logger.info("[Qdrant] 创建集合: {} (dim={}, cosine)", name, EMBEDDING_DIM)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    else:
        logger.info("[Qdrant] 集合已存在: {}", name)


def _ensure_doc_collection_hybrid(client: Optional[QdrantClient] = None) -> None:
    """文档集合：dense + sparse 双向量，用于 hybrid RRF 检索。"""
    if client is None:
        client = get_qdrant_client()
    if not client.collection_exists(DOC_COLLECTION):
        logger.info(
            "[Qdrant] 创建文档集合(hybrid): {} dense(dim={}) + sparse",
            DOC_COLLECTION,
            EMBEDDING_DIM,
        )
        client.create_collection(
            collection_name=DOC_COLLECTION,
            vectors_config={
                DOC_DENSE_VECTOR_NAME: VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                DOC_SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )
    else:
        logger.info("[Qdrant] 文档集合已存在: {}", DOC_COLLECTION)


# ==================== 新闻集合 ====================

def add_news_texts(texts: List[str], client: Optional[QdrantClient] = None) -> List[str]:
    """将多段新闻文本 Embedding 后写入 Qdrant"""
    logger.info("[Qdrant] 写入新闻: {} 条文本", len(texts))
    if client is None:
        client = get_qdrant_client()
    _ensure_collection(NEWS_COLLECTION, client)

    start = time.time()
    vectors = embed_texts(texts)
    logger.info("[Qdrant] Embedding 完成: {} 条, 耗时={:.1f}s", len(vectors), time.time() - start)

    existing = client.get_collection(NEWS_COLLECTION)
    offset = (existing.points_count or 0) if existing else 0
    points = [
        PointStruct(id=offset + i, vector=vec, payload={"text": txt})
        for i, (vec, txt) in enumerate(zip(vectors, texts))
    ]
    client.upsert(collection_name=NEWS_COLLECTION, points=points)
    logger.info("[Qdrant] 新闻写入完成: {} 条, offset={}", len(points), offset)
    return [str(offset + i) for i in range(len(texts))]


def search_news(query: str, top_k: int = 5, client: Optional[QdrantClient] = None) -> List[Tuple[str, float]]:
    logger.info("[Qdrant] 检索新闻: query='{}', top_k={}", query[:100], top_k)
    if client is None:
        client = get_qdrant_client()
    if not client.collection_exists(NEWS_COLLECTION):
        logger.info("[Qdrant] 新闻集合不存在, 返回空")
        return []

    start = time.time()
    q_vec = embed_query(query)
    response = client.query_points(collection_name=NEWS_COLLECTION, query=q_vec, limit=top_k)
    elapsed = time.time() - start
    logger.info("[Qdrant] 新闻检索完成: {} 条结果, 耗时={:.2f}s", len(response.points), elapsed)
    return [((hit.payload or {}).get("text", ""), hit.score) for hit in response.points]


# ==================== 文档集合（hybrid: dense + BM25 sparse）====================
# 若清空 Qdrant 文档集合，建议同时删除 data/bm25_doc_rag.pkl，使 BM25 词表与库一致。

def add_doc_chunks(
    chunks: List[str],
    filename: str,
    user_id: str,
    client: Optional[QdrantClient] = None,
) -> int:
    """将文档切片做 Dense + Sparse 双向量后写入 Qdrant 文档集合，返回写入数量。"""
    if not chunks:
        return 0
    logger.info("[Qdrant] 写入文档切片(hybrid): file={}, user={}, chunks={}", filename, user_id, len(chunks))
    if client is None:
        client = get_qdrant_client()
    _ensure_doc_collection_hybrid(client)

    bm25 = get_bm25_encoder()
    bm25.add_documents(chunks)

    start = time.time()
    vectors = embed_texts(chunks)
    logger.info("[Qdrant] 文档 Dense Embedding 完成: {} 条, 耗时={:.1f}s", len(vectors), time.time() - start)

    sparse_pairs = bm25.encode_texts(chunks)

    existing = client.get_collection(DOC_COLLECTION)
    offset = (existing.points_count or 0) if existing else 0
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        indices, values = sparse_pairs[i] if i < len(sparse_pairs) else ([], [])
        if not indices:
            indices, values = [0], [0.0]
        points.append(
            PointStruct(
                id=offset + i,
                vector={
                    DOC_DENSE_VECTOR_NAME: vec,
                    DOC_SPARSE_VECTOR_NAME: SparseVector(indices=indices, values=values),
                },
                payload={"text": chunk, "filename": filename, "user_id": user_id},
            )
        )
    client.upsert(collection_name=DOC_COLLECTION, points=points)
    logger.info("[Qdrant] 文档(hybrid)写入完成: {} 条, offset={}", len(points), offset)
    return len(points)


def search_docs(
    query: str,
    top_k: int = 5,
    user_id: Optional[str] = None,
    client: Optional[QdrantClient] = None,
) -> List[Tuple[str, str, float]]:
    """Hybrid 检索文档集合（dense + sparse RRF 融合），返回 (文本, 文件名, score) 列表。"""
    logger.info("[Qdrant] 检索文档(hybrid): query='{}', top_k={}, user_id={}",
                query[:100], top_k, user_id or "(全部)")
    if client is None:
        client = get_qdrant_client()
    if not client.collection_exists(DOC_COLLECTION):
        logger.info("[Qdrant] 文档集合不存在, 返回空")
        return []

    start = time.time()
    q_vec = embed_query(query)
    bm25 = get_bm25_encoder()
    q_indices, q_values = bm25.encode_text(query, use_tf=False)
    if not q_indices:
        q_indices, q_values = [0], [0.0]
    query_sparse = SparseVector(indices=q_indices, values=q_values)

    query_filter = None
    if user_id:
        query_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )

    prefetch_k = max(top_k * 2, 20)
    response = client.query_points(
        collection_name=DOC_COLLECTION,
        prefetch=[
            Prefetch(
                query=query_sparse,
                using=DOC_SPARSE_VECTOR_NAME,
                limit=prefetch_k,
            ),
            Prefetch(
                query=q_vec,
                using=DOC_DENSE_VECTOR_NAME,
                limit=prefetch_k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=top_k,
    )
    elapsed = time.time() - start
    logger.info("[Qdrant] 文档(hybrid)检索完成: {} 条结果, 耗时={:.2f}s", len(response.points), elapsed)
    return [
        (
            (hit.payload or {}).get("text", ""),
            (hit.payload or {}).get("filename", ""),
            hit.score,
        )
        for hit in response.points
    ]
