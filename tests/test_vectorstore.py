"""测试向量存储模块"""

import os
import sys
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_save_and_load():
    """测试保存和加载向量数据库。"""
    import faiss
    import numpy as np
    from vectorstore import save_vectorstore, load_vectorstore

    # 用 mock 数据
    mock_vectors = [[0.1 * j for j in range(128)] for i in range(5)]
    mock_chunks = [
        ("chunk 1 text", 1),
        ("chunk 2 text", 1),
        ("chunk 3 text", 2),
        ("chunk 4 text", 2),
        ("chunk 5 text", 3),
    ]

    save_vectorstore(mock_vectors, mock_chunks)

    # 验证文件存在
    assert os.path.exists("./data/faiss_index")
    assert os.path.exists("./data/chunks.json")

    # 加载验证
    index, chunks = load_vectorstore()
    assert index.ntotal == 5
    assert len(chunks) == 5
    assert chunks[0]["text"] == "chunk 1 text"
    assert chunks[0]["page"] == 1
    assert chunks[2]["page"] == 2

    print(f"    索引条目: {index.ntotal}, 维度: {index.d}")
    for c in chunks:
        print(f"    [{c['page']}] {c['text']}")
    print("[PASS] test_save_and_load")


def test_faiss_search():
    """测试 FAISS 搜索功能。"""
    import numpy as np
    import faiss
    from vectorstore import load_vectorstore

    index, chunks = load_vectorstore()

    # 用 chunk 1 的向量（近似）搜索，应该返回 chunk 1
    query_vec = np.array([[0.1 * j + 0.01 for j in range(128)]], dtype=np.float32)
    distances, indices = index.search(query_vec, 1)

    assert indices[0][0] == 0  # 最接近的应该是 chunk 0
    print(f"    搜索结果: 索引={indices[0][0]}, 距离={distances[0][0]:.4f}")
    print("[PASS] test_faiss_search")


def cleanup():
    """清理测试数据。"""
    if os.path.exists("./data"):
        shutil.rmtree("./data")
    print("    测试数据已清理")


if __name__ == "__main__":
    test_save_and_load()
    test_faiss_search()
    cleanup()
    print("\n[ALL PASS] vectorstore 所有测试通过！")
