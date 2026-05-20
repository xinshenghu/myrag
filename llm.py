"""LLM 模块 — 调用 OpenAI 兼容接口"""

import requests
from config import LLM_MODEL, AGI_API_KEY, AGI_BASE_URL, QAConfig


def build_context(results) -> str:
    """将检索结果拼成上下文。

    Args:
        results: SearchResult 列表
    """
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[来源 {r.filename} 页码:{r.page}] {r.text}")
    return "\n\n---\n\n".join(parts)


def answer(query: str, context: str, config: QAConfig = None) -> str:
    """调用 LLM 生成回答。

    Args:
        query: 用户问题
        context: 检索到的上下文
        config: 问答配置
    """
    config = config or QAConfig()
    prompt = f"""{config.system_prompt}

【上下文】
{context}

【问题】
{query}

请回答："""

    url = f"{AGI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {AGI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"LLM 调用失败: {resp.status_code} - {resp.text}")
