# MyRAG

一个基于 RAG 技术的文档智能问答系统。上传 PDF 文档，即可与文档内容对话，回答自带页码和文件名来源标注。

在线体验：https://xinshenghu-myrag.streamlit.app

## 架构

```
┌─────────────┐     ┌─────────────┐     ──────────────┐     ┌─────────────┐
│  PDF 文件    │────▶│  pdf_loader │────▶│   chunker    │────▶│  embedder   │
│  (上传/解析) │     │  (文本提取)  │     │  (文本分块)   │     │ (向量生成)   │
└─────────────┘     ─────────────┘     └──────────────┘     └──────┬──────┘
                                                                     │
─────────────┐     ┌─────────────┐     ┌──────────────┐            │
│  Streamlit  │────▶│  retriever  │◀────│ vectorstore  │◀───────────┘
│  (前端对话)  │     │  (相似度检索)│     │  (FAISS索引)  │
└──────┬──────┘     └──────┬──────┘     └──────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌──────────────┐
│  用户对话    │     │  LLM (问答)  │
│  (查询/回答) │◀────│  (答案生成)   │
└─────────────┘     └──────────────┘
```

### 模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| **PDF 提取** | `pdf_loader.py` | 用 PyMuPDF 提取 PDF 文本，记录每页字符级页码映射 |
| **文本分块** | `chunker.py` | 按字符数切分文本，支持 overlap 重叠分块 |
| **向量嵌入** | `embedder.py` | 调用 OpenAI 兼容 API 生成文本向量 (text-embedding-v3) |
| **向量存储** | `vectorstore.py` | FAISS L2 索引的保存与加载，chunk 元数据 JSON 存储 |
| **相似度检索** | `retriever.py` | 基于向量距离的 Top-K 相似度搜索 |
| **LLM 生成** | `llm.py` | 调用 OpenAI 兼容 API 生成回答 (qwen-turbo) |
| **问答链** | `qa_chain.py` | 检索 + 上下文拼接 + LLM 生成的完整流程 |
| **Web 前端** | `app.py` | Streamlit 界面：文件上传、增量索引、对话、PDF 预览、删除 |
| **命令行工具** | `build_index.py` / `chat.py` | 独立的索引构建和终端对话脚本 |

### 数据流向

```
用户上传 PDF
  → 物理文件存入 pdf/ 目录
  → pdf_loader 提取文本 + 页码 + 文件名
  → chunker 按 500 字符/块切分（overlap 50）
  → embedder 批量调用 embedding API 生成 1024 维向量
  → vectorstore 保存 FAISS 索引 + chunks.json 元数据
```

查询时：
```
用户提问
  → embedder 生成查询向量
  → retriever 在 FAISS 中 L2 搜索 Top-5
  → llm 将检索结果拼成上下文，调用 LLM 生成回答
  → 返回答案 + 来源（文件名 + 页码）
```

## 开发思路

**第一步 — 核心链路跑通（MVP）**  
先不管 UI，用最少的代码让 "PDF → 文本 → 分块 → 向量 → 搜索 → LLM 回答" 这条主线跑通。每个模块独立实现、独立测试，确保链路中任何一环出问题都能快速定位。

**第二步 — 拆成可测试的模块**  
每个模块一个文件，只暴露必要的公开函数。依赖关系单向：`pdf_loader → chunker → embedder → vectorstore → retriever → llm → qa_chain`。这样单元测试不需要 mock 整条链路，测 chunker 时只需要给一段字符串，不需要真实 PDF。

**第三步 — 增量索引**  
第一次全量构建，后续上传新文件时只处理未索引的部分。用 `indexed_files.txt` 记录已处理文件名，避免重复生成向量（每次调用 embedding API 都有成本）。

**第四步 — 文件名来源追溯**  
最初只有页码作为来源，多文档场景下不够用。在 `SearchResult` 和整个数据链路上增加 `filename` 字段，让用户知道回答来自哪个文档的哪一页。

**第五步 — Web 界面**  
用 Streamlit 快速搭建，因为后端全是 Python，无需额外写 API 层。上传、索引、对话、PDF 预览集成到一个页面，删除操作通过重建索引实现彻底清除。

## 快速开始

### 本地运行

```bash
git clone https://github.com/xinshenghu/myrag.git
cd myrag
pip install -r requirements.txt

# 方式一：Web 界面
streamlit run app.py

# 方式二：命令行
python build_index.py ./pdf
python chat.py
```

### 配置

API key 和 base URL 在 `config.py` 中硬编码。目前使用 OpenAI 兼容接口，支持 DashScope、AGICTO 等后端。

## 技术栈

- **前端**：Streamlit
- **向量检索**：FAISS (L2)
- **PDF 解析**：PyMuPDF (fitz)
- **嵌入模型**：text-embedding-v3 (1024 维)
- **大模型**：qwen-turbo (OpenAI 兼容接口)
- **存储**：本地 FAISS 索引文件 + JSON

## 目录结构

```
myrag/
├── app.py              # Streamlit 前端
├── build_index.py      # 命令行索引构建
├── chat.py             # 命令行对话
├── config.py           # 全局配置
├── pdf_loader.py       # PDF 文本提取
├── chunker.py          # 文本分块
├── embedder.py         # 向量嵌入
├── vectorstore.py      # FAISS 存储
├── retriever.py        # 相似度检索
├── llm.py              # LLM 调用
├── qa_chain.py         # 问答链
├── requirements.txt
├── .gitignore
├── README.md
├── pdf/                # PDF 存放目录
├── data/               # FAISS 索引（已 gitignore）
└── tests/              # 单元测试
```
