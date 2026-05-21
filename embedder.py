"""向量嵌入模块 — 调用 OpenAI 兼容的 embedding API"""

import requests
import time
from config import EMBEDDING_MODEL, get_api_key, get_base_url


def embed_text(text: str) -> list[float]:
    url = f"{get_base_url()}/embeddings"
    headers = {"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"}
    data = {"model": EMBEDDING_MODEL, "input": text}

    resp = requests.post(url, headers=headers, json=data, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding failed: {resp.status_code} - {resp.text}")

    result = resp.json()
    if "data" not in result:
        raise RuntimeError(f"Unexpected API response: {result}")

    return result["data"][0]["embedding"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    batch_size = 25

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        url = f"{get_base_url()}/embeddings"
        headers = {"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"}
        data = {"model": EMBEDDING_MODEL, "input": batch}

        resp = requests.post(url, headers=headers, json=data, timeout=120)

        if resp.status_code == 200 and "data" in resp.json():
            result = resp.json()
            for item in result["data"]:
                all_embeddings.append(item["embedding"])
            print(f"    批量处理 [{i}-{i+len(batch)}] / {len(texts)}")
        else:
            print(f"    [WARN] 批量请求失败 (status={resp.status_code}), 降级为逐条处理...")
            for j, text in enumerate(batch):
                emb = embed_text(text)
                all_embeddings.append(emb)
                if (j + 1) % 10 == 0:
                    print(f"    逐条处理 [{i+j+1}] / {len(texts)}")
                time.sleep(0.1)

    return all_embeddings
