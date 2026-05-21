"""检索模块 — 基于 FAISS 的相似度搜索"""

import numpy as np

from config import SearchResult
from vectorstore import load_vectorstore


def search(query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
    """根据查询向量检索最相关的文本块。

    Args:
        query_embedding: 查询文本的向量
        top_k: 返回结果数量

    Returns:
        SearchResult 列表，按相关度降序排列
    """
    index, chunks = load_vectorstore()
    query_vec = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        meta = chunks[idx]
        # FAISS L2 距离，越小越相关，转为分数
        score = 1.0 / (1.0 + dist)
        results.append(SearchResult(
            text=meta["text"],
            page=meta.get("page", 0),
            filename=meta.get("filename", ""),
            score=score,
        ))

    return results


def search_by_text(query: str, embed_func, top_k: int = 3) -> list[SearchResult]:
    """用文本直接检索（内部自动嵌入）。

    Args:
        query: 查询文本
        embed_func: 嵌入函数，接收 str 返回 list[float]
        top_k: 返回结果数量
    """
    query_embedding = embed_func(query)
    return search(query_embedding, top_k)
