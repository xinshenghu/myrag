"""PDF 文本提取模块 — 提取纯文本并记录每个字符的页码和来源文件"""

import os
import fitz  # PyMuPDF


def extract_text_with_pages(pdf_path: str) -> list[tuple[str, int, str]]:
    """从 PDF 提取文本，返回 [(文本内容, 页码, 文件名), ...]

    Args:
        pdf_path: PDF 文件路径

    Returns:
        每页的 (文本, 页码, 文件名) 列表。页码从 1 开始。
    """
    filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append((text, i + 1, filename))
    doc.close()
    return pages


def build_char_page_map(pages: list[tuple[str, int, str]]) -> list[int]:
    """构建字符到页码的映射表。"""
    mapping = []
    for text, page_num, _ in pages:
        mapping.extend([page_num] * len(text))
    return mapping
