"""BM25 稀疏向量编码：中文 jieba 分词 + rank_bm25，用于混合检索的 sparse 一路。"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import jieba
from loguru import logger
from rank_bm25 import BM25Okapi

# 持久化路径：词表 + IDF，供查询时编码 query 用
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_BM25_STATE_PATH = _DATA_DIR / "bm25_doc_rag.pkl"


def _tokenize(text: str) -> List[str]:
    """中文分词，去掉空白与过短 token。"""
    tokens = [t.strip() for t in jieba.lcut(text) if t.strip() and len(t.strip()) > 1]
    return tokens


def _ensure_data_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _compute_idf(tokenized_corpus: List[List[str]]) -> Tuple[List[str], List[float]]:
    """根据 tokenized_corpus 计算词表与 IDF，返回 (vocab, idf_list)。"""
    from math import log

    N = len(tokenized_corpus)
    if N == 0:
        return [], []

    doc_freq: dict[str, int] = {}
    for doc in tokenized_corpus:
        seen = set()
        for t in doc:
            if t not in seen:
                seen.add(t)
                doc_freq[t] = doc_freq.get(t, 0) + 1

    vocab = sorted(doc_freq.keys())
    idf_list = []
    for t in vocab:
        df = doc_freq[t]
        # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        idf = log((N - df + 0.5) / (df + 0.5) + 1.0)
        idf_list.append(idf)

    return vocab, idf_list


def _encode_with_vocab_idf(
    tokens: List[str],
    vocab: List[str],
    idf_list: List[float],
    use_tf: bool = True,
) -> Tuple[List[int], List[float]]:
    """用词表与 IDF 将 token 列表编码为 sparse (indices, values)。"""
    from math import log

    vocab_index = {t: i for i, t in enumerate(vocab)}
    tf: dict[int, int] = {}
    for t in tokens:
        if t in vocab_index:
            idx = vocab_index[t]
            tf[idx] = tf.get(idx, 0) + 1

    if not tf:
        return [], []

    indices = []
    values = []
    for idx, cnt in tf.items():
        # 文档侧/查询侧权重：idf * (1 + log(tf))
        w = idf_list[idx] * (1.0 + log(cnt)) if use_tf else idf_list[idx]
        indices.append(idx)
        values.append(float(w))

    return indices, values


class BM25SparseEncoder:
    """维护 BM25 词表与 IDF，支持增量添加文档并编码文本为 sparse 向量。"""

    def __init__(self) -> None:
        self._tokenized_corpus: List[List[str]] = []
        self._vocab: List[str] = []
        self._idf_list: List[float] = []
        self._bm25: BM25Okapi | None = None

    def load(self) -> bool:
        """从磁盘加载状态，返回是否成功。"""
        _ensure_data_dir()
        if not _BM25_STATE_PATH.exists():
            return False
        try:
            with open(_BM25_STATE_PATH, "rb") as f:
                data = pickle.load(f)
            self._tokenized_corpus = data.get("tokenized_corpus", [])
            self._vocab = data.get("vocab", [])
            self._idf_list = data.get("idf_list", [])
            if self._tokenized_corpus and self._vocab:
                self._bm25 = BM25Okapi(self._tokenized_corpus)
            else:
                self._bm25 = None
            logger.info("[BM25] 已加载状态: {} 文档, {} 词", len(self._tokenized_corpus), len(self._vocab))
            return True
        except Exception as e:
            logger.warning("[BM25] 加载状态失败: {}", e)
            return False

    def save(self) -> None:
        """将状态持久化到磁盘。"""
        _ensure_data_dir()
        try:
            with open(_BM25_STATE_PATH, "wb") as f:
                pickle.dump(
                    {
                        "tokenized_corpus": self._tokenized_corpus,
                        "vocab": self._vocab,
                        "idf_list": self._idf_list,
                    },
                    f,
                )
            logger.debug("[BM25] 状态已保存")
        except Exception as e:
            logger.warning("[BM25] 保存状态失败: {}", e)

    def add_documents(self, chunks: List[str]) -> None:
        """将新文档块加入语料并更新词表与 IDF。"""
        if not chunks:
            return
        new_tokenized = [_tokenize(c) for c in chunks]
        self._tokenized_corpus.extend(new_tokenized)
        self._vocab, self._idf_list = _compute_idf(self._tokenized_corpus)
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        self.save()
        logger.info("[BM25] 已加入 {} 条文档, 词表大小 {}", len(chunks), len(self._vocab))

    def encode_text(self, text: str, use_tf: bool = True) -> Tuple[List[int], List[float]]:
        """将单段文本编码为 sparse (indices, values)。若无词表则返回空向量。"""
        if not self._vocab:
            return [], []
        tokens = _tokenize(text)
        return _encode_with_vocab_idf(tokens, self._vocab, self._idf_list, use_tf=use_tf)

    def encode_texts(self, texts: List[str], use_tf: bool = True) -> List[Tuple[List[int], List[float]]]:
        """批量编码，返回 [(indices, values), ...]。"""
        return [self.encode_text(t, use_tf=use_tf) for t in texts]


# 单例，供 vector_store 使用
_encoder: BM25SparseEncoder | None = None


def get_bm25_encoder() -> BM25SparseEncoder:
    global _encoder
    if _encoder is None:
        _encoder = BM25SparseEncoder()
        _encoder.load()
    return _encoder
