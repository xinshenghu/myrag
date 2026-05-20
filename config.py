"""RAG 项目配置"""

import os
from dataclasses import dataclass

# 加载本地 .env 文件（Streamlit Cloud 通过 Secrets 设置环境变量，优先级更高）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API 配置 — OpenAI 兼容接口
AGI_API_KEY = os.environ.get("AGI_API_KEY", "")
AGI_BASE_URL = os.environ.get("AGI_BASE_URL", "https://api.agicto.cn/v1")

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
