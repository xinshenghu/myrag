"""RAG 项目配置"""

import os
from dataclasses import dataclass

# 加载本地 .env 文件（仅本地开发用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 懒加载 API key（Streamlit Cloud 上 st.secrets 在模块导入时还不可用）
_agi_api_key = None
_agi_base_url = None


def get_api_key():
    global _agi_api_key
    if _agi_api_key is not None:
        return _agi_api_key

    # 优先环境变量
    val = os.environ.get("AGI_API_KEY")
    if val:
        _agi_api_key = val
        return val

    # 尝试 Streamlit Cloud Secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            val = st.secrets.get("AGI_API_KEY")
            if val:
                _agi_api_key = val
                return val
    except Exception:
        pass

    # 最后尝试 os.environ（可能 secrets 已注入环境变量）
    val = os.environ.get("AGI_API_KEY")
    if val:
        _agi_api_key = val
        return val

    return ""


def get_base_url():
    global _agi_base_url
    if _agi_base_url is not None:
        return _agi_base_url

    val = os.environ.get("AGI_BASE_URL")
    if val:
        _agi_base_url = val
        return val

    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            val = st.secrets.get("AGI_BASE_URL")
            if val:
                _agi_base_url = val
                return val
    except Exception:
        pass

    _agi_base_url = "https://api.agicto.cn/v1"
    return _agi_base_url


# 兼容旧代码的变量（会在首次调用时懒加载）
def __getattr__(name):
    if name == "AGI_API_KEY":
        return get_api_key()
    if name == "AGI_BASE_URL":
        return get_base_url()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 向量模型
EMBEDDING_MODEL = "text-embedding-v3"
LLM_MODEL = "qwen-turbo"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
NOTE_DIR = r"C:\13683246141@163.com_2026-05-14-12-18"
INDEX_DIR = os.path.join(NOTE_DIR, ".rag_index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index")
CHUNKS_JSON_PATH = os.path.join(INDEX_DIR, "chunks.json")


@dataclass
class SearchResult:
    text: str
    page: int
    filename: str
    score: float


@dataclass
class QAConfig:
    system_prompt: str = """你是一个基于文档的问答助手。请根据提供的上下文回答问题。
如果上下文中没有足够的信息来回答问题，请直接说"根据提供的文档，无法回答此问题"。
回答时请标注引用来源的页码。"""
