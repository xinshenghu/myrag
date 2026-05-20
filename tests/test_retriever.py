"""测试检索模块"""

import os
import sys
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def setup_test_index():
    """准备测试用的向量索引。"""
    from vectorstore import save_vectorstore

    mock_vectors = [[0.1 * j for j in range(128)] for i in range(5)]
    mock_chunks = [
        ("张然 报考 首都经济贸易大学 计算机科学与技术", 1),
        ("考试时间为12月20日至21日", 1),
        ("考场位于博学楼5层510教室", 2),
        ("思想政治理论 英语一 数学一 408", 1),
        ("考生须携带准考证和身份证", 2),
    ]
    save_vectorstore(mock_vectors, mock_chunks)


def test_search():
    """测试检索功能。"""
    import numpy as np
    from retriever import search

    # 用接近 chunk 0 的向量搜索
    query_vec = [0.1 * j + 0.005 for j in range(128)]
    results = search(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].score > 0  # 分数应该为正
    print(f"    找到 {len(results)} 个结果:")
    for r in results:
        print(f"    [页{r.page}] (分数:{r.score:.4f}) {r.text[:30]}...")
    print("[PASS] test_search")


def test_search_by_text():
    """测试文本检索（需要 API Key）。"""
    api_key = os.environ.get("AGI_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        print("[SKIP] 未设置 AGI_API_KEY，跳过")
        return

    from embedder import embed_text
    from retriever import search_by_text

    results = search_by_text("张然 考研", embed_text, top_k=2)
    assert len(results) >= 1
    print(f"    搜索 '张然 考研' 找到 {len(results)} 个结果:")
    for r in results:
        print(f"    [页{r.page}] (分数:{r.score:.4f}) {r.text}")
    print("[PASS] test_search_by_text")


def cleanup():
    if os.path.exists("./data"):
        shutil.rmtree("./data")


if __name__ == "__main__":
    setup_test_index()
    test_search()
    test_search_by_text()
    cleanup()
    print("\n[ALL PASS] retriever 所有测试通过！")
