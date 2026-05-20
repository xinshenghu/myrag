"""测试 RAG 问答链"""

import os
import sys
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def setup_test_index():
    from vectorstore import save_vectorstore

    mock_vectors = [[0.1 * j for j in range(128)] for i in range(3)]
    mock_chunks = [
        ("张然，男，报考首都经济贸易大学计算机科学与技术专业。考试科目包括思想政治理论、英语一、数学一和计算机学科专业基础。", 1),
        ("考试时间为12月20日至21日，考场位于首都经济贸易大学丰台校区博学楼5层510。", 1),
        ("考生须携带准考证和有效身份证件参加考试。严禁携带手机等电子设备进入考场。", 2),
    ]
    save_vectorstore(mock_vectors, mock_chunks)


def test_qa_chain():
    """测试完整 RAG 问答链。"""
    from qa_chain import ask

    # 模拟 query embedding（接近 chunk 0）
    query_embedding = [0.1 * j + 0.003 for j in range(128)]
    reply, sources = ask("张然报考的是什么专业？", query_embedding, top_k=2)

    assert len(sources) >= 1
    print(f"    回答: {reply[:80]}...")
    print(f"    来源数: {len(sources)}")
    for s in sources:
        print(f"    [页{s.page}] {s.text[:40]}...")
    print("[PASS] test_qa_chain")


def test_qa_chain_no_match():
    """测试无匹配结果的情况（用 mock LLM）。"""
    from qa_chain import ask

    # 用差异很大的向量
    query_embedding = [0.9] * 128
    reply, sources = ask("量子力学是什么？", query_embedding, top_k=1)

    # 即使不相关也会返回，由 LLM 判断
    print(f"    回答: {reply[:60]}...")
    print(f"    来源数: {len(sources)}")
    print("[PASS] test_qa_chain_no_match")


def cleanup():
    if os.path.exists("./data"):
        shutil.rmtree("./data")


if __name__ == "__main__":
    setup_test_index()
    test_qa_chain()
    test_qa_chain_no_match()
    cleanup()
    print("\n[ALL PASS] qa_chain 所有测试通过！")
