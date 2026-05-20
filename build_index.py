"""构建索引 — 从 PDF 文件夹生成向量数据库"""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(__file__))

from pdf_loader import extract_text_with_pages, build_char_page_map
from chunker import chunk_with_pages
from embedder import embed_texts
from vectorstore import save_vectorstore


def build_index_from_folder(pdf_folder: str):
    """从一个文件夹中读取所有 PDF，构建统一向量索引。"""
    pdf_files = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    if not pdf_files:
        print(f"[ERROR] 在 {pdf_folder} 中没有找到 PDF 文件")
        sys.exit(1)

    print(f"[1] 找到 {len(pdf_files)} 个 PDF 文件:")
    for f in pdf_files:
        print(f"    - {os.path.basename(f)}")

    # 提取所有 PDF 的文本
    all_pages = []
    total_chars = 0
    for pdf_path in pdf_files:
        print(f"\n    读取: {os.path.basename(pdf_path)}")
        pages = extract_text_with_pages(pdf_path)
        for text, page_num, filename in pages:
            all_pages.append((text, page_num, filename))
        page_chars = sum(len(t) for t, _, _ in pages)
        total_chars += page_chars
        print(f"      {len(pages)} 页, {page_chars} 字符")

    print(f"\n    合计: {len(all_pages)} 页, {total_chars} 字符")

    # 构建字符-页码映射
    char_page_map = build_char_page_map(all_pages)
    print(f"    字符-页码映射: {len(char_page_map)} 个条目")

    # 分块
    print("\n[2] 正在分块...")
    chunks = chunk_with_pages(all_pages)
    print(f"    共 {len(chunks)} 个 chunk")

    # 生成向量
    print("\n[3] 正在生成向量嵌入...")
    chunk_texts = [t for t, _, _ in chunks]
    vectors = embed_texts(chunk_texts)
    print(f"    向量维度: {len(vectors[0])}")

    # 保存
    print("\n[4] 正在保存向量数据库...")
    save_vectorstore(vectors, chunks)
    print("\n[OK] 索引构建完成！\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python build_index.py <pdf文件夹路径>")
        print("示例: python build_index.py ./pdf")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        build_index_from_folder(target)
    elif os.path.isfile(target):
        build_index_from_folder(os.path.dirname(target))
    else:
        print(f"[ERROR] 找不到: {target}")
        sys.exit(1)
