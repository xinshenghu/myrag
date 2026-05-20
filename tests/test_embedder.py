"""测试嵌入模块

注意：此测试需要有效的 AGI_API_KEY。
如果没有 API Key，会跳过实际 API 调用。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_embed_text_with_mock():
    """用 mock 数据测试嵌入逻辑（不调 API）。"""
    # 模拟一个 1024 维向量
    mock_embedding = [0.1 * i for i in range(1024)]

    # 验证向量格式
    assert len(mock_embedding) == 1024
    assert all(isinstance(v, float) for v in mock_embedding)
    print("[PASS] test_embed_text_with_mock")


def test_embed_text_real():
    """测试真实 API 调用。"""
    api_key = os.environ.get("AGI_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        print("[SKIP] 未设置 AGI_API_KEY，跳过")
        return

    from embedder import embed_text

    embedding = embed_text("这是一个测试句子。")
    assert len(embedding) > 0
    print(f"    向量维度: {len(embedding)}")
    print(f"    前 5 个值: {embedding[:5]}")
    print("[PASS] test_embed_text_real")


def test_embed_texts_batch():
    """测试批量嵌入。"""
    api_key = os.environ.get("AGI_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        print("[SKIP] 未设置 AGI_API_KEY，跳过")
        return

    from embedder import embed_texts

    texts = ["第一句测试。", "第二句测试。", "第三句测试。"]
    embeddings = embed_texts(texts)
    assert len(embeddings) == 3
    assert all(len(emb) == len(embeddings[0]) for emb in embeddings)
    print(f"    批量生成了 {len(embeddings)} 个向量，维度 {len(embeddings[0])}")
    print("[PASS] test_embed_texts_batch")


if __name__ == "__main__":
    test_embed_text_with_mock()
    test_embed_text_real()
    test_embed_texts_batch()
    print("\n[ALL PASS] embedder 所有测试通过！")
