"""测试分块模块"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chunker import chunk_text, chunk_with_pages


def test_chunk_text_basic():
    text = "ABCDEFGHIJ"
    chunks = chunk_text(text, chunk_size=4, overlap=1)

    assert len(chunks) > 0
    assert all(len(c) <= 4 for c in chunks)
    for i in range(len(chunks) - 1):
        assert chunks[i][-1] == chunks[i + 1][0]

    print(f"    chunks: {chunks}")
    print("[PASS] test_chunk_text_basic")


def test_chunk_text_empty():
    chunks = chunk_text("", chunk_size=10)
    assert chunks == []
    print("[PASS] test_chunk_text_empty")


def test_chunk_text_short():
    chunks = chunk_text("Hi", chunk_size=10)
    assert len(chunks) == 1
    assert chunks[0] == "Hi"
    print("[PASS] test_chunk_text_short")


def test_chunk_with_pages():
    pages = [
        ("AAAAA BBBBB CCCCC", 1, "doc1.pdf"),
        ("DDDDD EEEEE", 2, "doc2.pdf"),
    ]
    result = chunk_with_pages(pages, chunk_size=8, overlap=2)

    assert all(isinstance(chunk, str) and isinstance(page, int) and isinstance(fn, str)
               for chunk, page, fn in result)

    for text, page, fn in result:
        has_abc = any(c in "ABC" for c in text)
        has_de = any(c in "DE" for c in text)
        if has_abc and not has_de:
            assert page == 1
            assert fn == "doc1.pdf"
        elif has_de and not has_abc:
            assert page == 2
            assert fn == "doc2.pdf"

    print(f"    {len(result)} chunks generated")
    print("[PASS] test_chunk_with_pages")


if __name__ == "__main__":
    test_chunk_text_basic()
    test_chunk_text_empty()
    test_chunk_text_short()
    test_chunk_with_pages()
    print("\n[ALL PASS] chunker all tests passed!")
