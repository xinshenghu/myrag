"""RAG 交互式问答"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from retriever import search_by_text
from embedder import embed_text
from llm import build_context, answer


def chat_loop():
    """交互式问答循环。"""
    print("[OK] RAG 问答系统已就绪。输入问题开始提问，输入 'quit' 退出。")
    print("=" * 60)

    while True:
        query = input("\n>>> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not query:
            continue

        results = search_by_text(query, embed_text, top_k=5)
        if not results:
            print("[WARN] 未找到相关内容。")
            continue

        print(f"\n[找到 {len(results)} 个相关片段]")
        for r in results:
            print(f"    [{r.filename}] 页码:{r.page} (相似度: {r.score:.3f})")

        context = build_context(results)
        reply = answer(query, context)
        print(f"\n[回答]\n{reply}")

        # 标注来源
        sources = sorted(set((r.filename, r.page) for r in results))
        print(f"\n[来源] {', '.join(f'{f}(p{p})' for f, p in sources)}")
        print("-" * 60)


if __name__ == "__main__":
    chat_loop()
