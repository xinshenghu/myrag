"""FAISS 向量数据库模块 — 保存/加载向量索引和元数据"""

import json
import os
import faiss
import numpy as np

from config import FAISS_INDEX_PATH, CHUNKS_JSON_PATH


def save_vectorstore(vectors: list[list[float]], chunks_with_meta: list[tuple[str, int, str]]):
    """将向量和元数据保存到磁盘。

    Args:
        vectors: N 个向量 (list of list of float)
        chunks_with_meta: [(chunk_text, page_number, filename), ...]
    """
    index_dir = os.path.dirname(FAISS_INDEX_PATH)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)

    # 构建 FAISS 索引 (L2 距离)
    arr = np.array(vectors, dtype=np.float32)
    dimension = arr.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(arr)
    faiss.write_index(index, FAISS_INDEX_PATH)

    # 保存 chunk 文本、页码和文件名
    data = {
        "chunks": [{"text": t, "page": p, "filename": f} for t, p, f in chunks_with_meta],
        "total": len(chunks_with_meta),
    }
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    print(f"[OK] 已保存 {len(chunks_with_meta)} 个 chunk，向量维度={dimension}")


def load_vectorstore() -> tuple[faiss.IndexFlatL2, list[dict]]:
    """加载 FAISS 索引和 chunk 元数据。

    Returns:
        (faiss_index, chunks_metadata)
    """
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return index, data["chunks"]
