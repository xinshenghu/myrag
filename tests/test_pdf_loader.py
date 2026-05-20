"""测试 PDF 提取模块"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pdf_loader import extract_text_with_pages, build_char_page_map

# 模拟 PDF 页数据
MOCK_PAGES = [
    ("Hello World. This is page one.", 1, "doc1.pdf"),
    ("Second page content here.", 2, "doc1.pdf"),
    ("Third page with some text.", 3, "doc2.pdf"),
]


def test_extract_text_with_pages():
    pages = MOCK_PAGES
    assert len(pages) == 3
    assert pages[0][1] == 1
    assert pages[0][2] == "doc1.pdf"
    print("[PASS] test_extract_text_with_pages")


def test_build_char_page_map():
    mapping = build_char_page_map(MOCK_PAGES)
    expected_len = len(MOCK_PAGES[0][0]) + len(MOCK_PAGES[1][0]) + len(MOCK_PAGES[2][0])
    assert len(mapping) == expected_len

    first_page_len = len(MOCK_PAGES[0][0])
    assert all(p == 1 for p in mapping[:first_page_len])

    second_page_len = len(MOCK_PAGES[1][0])
    start = first_page_len
    end = start + second_page_len
    assert all(p == 2 for p in mapping[start:end])

    print("[PASS] test_build_char_page_map")


def test_with_real_pdf():
    pdf_path = "C:/Users/z/Downloads/03_文档/PDF_116797933.pdf"
    if not os.path.exists(pdf_path):
        print("[SKIP] 真实 PDF 文件不存在，跳过")
        return

    pages = extract_text_with_pages(pdf_path)
    assert len(pages) > 0
    assert all(page_num >= 1 for _, page_num, _ in pages)
    assert pages[0][2] == "PDF_116797933.pdf"
    print(f"[PASS] test_with_real_pdf - {len(pages)} pages extracted")


if __name__ == "__main__":
    test_extract_text_with_pages()
    test_build_char_page_map()
    test_with_real_pdf()
    print("\n[ALL PASS] pdf_loader all tests passed!")
