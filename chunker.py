"""文本分块模块 — 将全文切分为小 chunk"""

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(full_text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """将全文按字符数切分为 chunk，相邻 chunk 之间有重叠。

    Args:
        full_text: 完整文本
        chunk_size: 每个 chunk 的最大字符数
        overlap: 相邻 chunk 的重叠字符数

    Returns:
        chunk 列表
    """
    chunks = []
    start = 0
    text_len = len(full_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(full_text[start:end])
        start += chunk_size - overlap

    return chunks


def chunk_with_pages(
    pages: list[tuple[str, int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, int, str]]:
    """分块，同时返回每个 chunk 对应的页码和文件名。

    对每页文本分别分块，chunk 继承其来源页码和文件名。

    Returns:
        [(chunk_text, page_number, filename), ...]
    """
    result = []
    for page_text, page_num, filename in pages:
        if not page_text.strip():
            continue
        chunks = chunk_text(page_text, chunk_size, overlap)
        for chunk in chunks:
            result.append((chunk, page_num, filename))
    return result
