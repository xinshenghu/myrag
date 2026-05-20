"""RAG 问答链 — 串联检索 + 生成"""

from config import SearchResult
from retriever import search
from llm import build_context, answer, QAConfig


def ask(query: str, query_embedding: list[float], top_k: int = 3) -> tuple[str, list[SearchResult]]:
    """完整 RAG 问答流程。

    1. 用 query_embedding 检索相关 chunk
    2. 拼接上下文
    3. 调用 LLM 生成回答

    Returns:
        (回答文本, 来源列表)
    """
    results = search(query_embedding, top_k=top_k)
    if not results:
        return "未找到相关文档内容。", []

    context = build_context(results)
    reply = answer(query, context)
    return reply, results
