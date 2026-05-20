"""RAG 项目配置"""

import os
from dataclasses import dataclass

# 加载本地 .env 文件（Streamlit Cloud 通过 Secrets 设置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API 配置 — 兼容本地 .env 和 Streamlit Cloud Secrets
def _get_api_key():
    # 优先环境变量
    val = os.environ.get("AGI_API_KEY")
    if val:
        return val
    # 尝试 Streamlit Cloud Secrets
    try:
        import streamlit as st
        return st.secrets.get("AGI_API_KEY", "")
    except Exception:
        return ""

def _get_base_url():
    val = os.environ.get("AGI_BASE_URL")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get("AGI_BASE_URL", "https://api.agicto.cn/v1")
    except Exception:
        return "https://api.agicto.cn/v1"

AGI_API_KEY = _get_api_key()
AGI_BASE_URL = _get_base_url()

# 向量模型
EMBEDDING_MODEL = "text-embedding-v3"

# LLM 模型
LLM_MODEL = "qwen-turbo"

# Chunk 配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# FAISS 存储路径
FAISS_INDEX_PATH = "./data/faiss_index"
CHUNKS_JSON_PATH = "./data/chunks.json"


@dataclass
class SearchResult:
    """检索结果"""
    text: str
    page: int
    filename: str
    score: float


@dataclass
class QAConfig:
    """问答配置"""
    system_prompt: str = """你是一个基于文档的问答助手。请根据提供的上下文回答问题。
如果上下文中没有足够的信息来回答问题，请直接说"根据提供的文档，无法回答此问题"。
回答时请标注引用来源的页码。"""
