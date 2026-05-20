"""RAG 项目配置"""

import os
from dataclasses import dataclass, field

# API 配置 — OpenAI 兼容接口
AGI_API_KEY = "sk-tZRtICqTySOZXZ4KCUEYg3pvJ9VtHoaTgDJ5iJRwhBN9wF2f"
AGI_BASE_URL = "https://api.agicto.cn/v1"

# 向量模型
EMBEDDING_MODEL = "text-embedding-v3"

# LLM 模型
LLM_MODEL = "qwen-turbo"

# Chunk 配置
CHUNK_SIZE = 500          # 每个 chunk 的字符数
CHUNK_OVERLAP = 50        # chunk 之间的重叠字符数

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
