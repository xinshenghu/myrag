"""LLM 模块 — 调用 OpenAI 兼容接口"""

import requests
from config import LLM_MODEL, get_api_key, get_base_url, QAConfig


def build_context(results) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[来源 {r.filename} 页码:{r.page}] {r.text}")
    return "\n\n---\n\n".join(parts)


def answer(query: str, context: str, config: QAConfig = None) -> str:
    config = config or QAConfig()
    prompt = f"""{config.system_prompt}

【上下文】
{context}

【问题】
{query}

请回答："""

    url = f"{get_base_url()}/chat/completions"
    headers = {"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"}
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"LLM 调用失败: {resp.status_code} - {resp.text}")
