"""RAG Streamlit 前端 — PDF 上传、对话、PDF 浏览、删除"""

import os
import sys
import glob
import json

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from pdf_loader import extract_text_with_pages
from chunker import chunk_with_pages
from embedder import embed_texts, embed_text
from vectorstore import save_vectorstore, load_vectorstore
from retriever import search
from llm import build_context, answer


# === 配置 ===
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "pdf")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)

INDEX_FILE = os.path.join(os.path.dirname(__file__), "data", "indexed_files.txt")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "faiss_index")

st.set_page_config(page_title="RAG 文档问答", layout="wide")


def get_indexed_files():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_indexed_files(files):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        for fp in sorted(files):
            f.write(fp + "\n")


def index_exists():
    return os.path.exists(FAISS_INDEX_PATH)


def rebuild_index_from_folder(pdf_folder):
    """从文件夹的所有 PDF 重新构建索引"""
    pdf_files = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))

    if not pdf_files:
        save_indexed_files(set())
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(os.path.join(os.path.dirname(__file__), "data", "chunks.json")):
            os.remove(os.path.join(os.path.dirname(__file__), "data", "chunks.json"))
        return True, 0

    all_pages = []
    for pdf_path in pdf_files:
        pages = extract_text_with_pages(pdf_path)
        filename = os.path.basename(pdf_path)
        for text, page_num, _ in pages:
            all_pages.append((text, page_num, filename))

    chunks = chunk_with_pages(all_pages)
    chunk_texts = [t for t, _, _ in chunks]
    vectors = embed_texts(chunk_texts)
    save_vectorstore(vectors, chunks)

    all_names = set(os.path.basename(f) for f in pdf_files)
    save_indexed_files(all_names)

    return True, len(pdf_files)


def delete_document(filename):
    """彻底删除文档：删文件 + 重建索引"""
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(pdf_path):
        return False, f"文件不存在: {filename}"

    os.remove(pdf_path)
    return rebuild_index_from_folder(UPLOAD_DIR)


def index_new_files(pdf_folder):
    """增量索引新 PDF"""
    existing_files = get_indexed_files()
    pdf_files = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    new_files = [f for f in pdf_files if os.path.basename(f) not in existing_files]
    if not new_files:
        return False, 0

    all_pages = []
    for pdf_path in new_files:
        pages = extract_text_with_pages(pdf_path)
        filename = os.path.basename(pdf_path)
        for text, page_num, _ in pages:
            all_pages.append((text, page_num, filename))

    chunks = chunk_with_pages(all_pages)
    chunk_texts = [t for t, _, _ in chunks]
    vectors = embed_texts(chunk_texts)
    save_vectorstore(vectors, chunks)

    existing_files.update(os.path.basename(f) for f in new_files)
    save_indexed_files(existing_files)
    return True, len(new_files)


# === 侧边栏 ===
st.sidebar.title("RAG 文档问答")

# 上传
st.sidebar.subheader("上传 PDF")
uploaded = st.sidebar.file_uploader("选择 PDF 文件", type=["pdf"], accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        with open(os.path.join(UPLOAD_DIR, f.name), "wb") as fh:
            fh.write(f.read())
    st.sidebar.success(f"已上传 {len(uploaded)} 个文件")

# 索引管理
st.sidebar.subheader("索引管理")
col1, col2 = st.sidebar.columns(2)
if col1.button("增量索引", use_container_width=True):
    ok, count = index_new_files(UPLOAD_DIR)
    if ok:
        st.sidebar.success(f"新索引 {count} 个文件")
    else:
        st.sidebar.info("没有新文件需要索引")

if col2.button("全量重建", use_container_width=True):
    ok, count = rebuild_index_from_folder(UPLOAD_DIR)
    if ok:
        st.sidebar.success(f"已重建 {count} 个文件的索引")

# 文件列表 + 删除（用 form 避免重复触发）
st.sidebar.subheader("已上传文件")
pdf_list = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.pdf")))
indexed = get_indexed_files()

# 用 session_state 管理待删除文件
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None

for fp in pdf_list:
    name = os.path.basename(fp)
    status = "已索引" if name in indexed else "未索引"
    st.sidebar.text(f"[{status}] {name}")

# 删除确认区域（放在列表下方，避免和按钮 key 冲突）
if st.session_state.pending_delete:
    name = st.session_state.pending_delete
    st.sidebar.warning(f"确定删除 {name}？此操作不可撤销。")
    c_yes, c_no = st.sidebar.columns(2)
    if c_yes.button("确定删除", key="confirm_yes", use_container_width=True):
        ok, msg = delete_document(name)
        if ok:
            st.sidebar.success(f"已删除 {name}")
        else:
            st.sidebar.error(msg)
        st.session_state.pending_delete = None
        st.rerun()
    if c_no.button("取消", key="confirm_no", use_container_width=True):
        st.session_state.pending_delete = None
        st.rerun()

# PDF 预览
st.sidebar.subheader("PDF 预览")
current_names = [os.path.basename(f) for f in glob.glob(os.path.join(UPLOAD_DIR, "*.pdf"))]
selected_pdf = st.sidebar.selectbox("选择文件", ["-- 请选择 --"] + current_names)


# === 主区域 ===
st.title("文档智能问答")

if not index_exists():
    st.info("还没有索引。请先在左侧上传 PDF 并点击「增量索引」或「全量重建」。")
else:
    # PDF 预览
    if selected_pdf and selected_pdf != "-- 请选择 --":
        pdf_path = os.path.join(UPLOAD_DIR, selected_pdf)
        if os.path.exists(pdf_path):
            st.subheader(f"预览: {selected_pdf}")
            try:
                from streamlit_pdf_viewer import pdf_viewer
                pdf_viewer(pdf_path, width=700)
            except ImportError:
                import base64
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                pdf_base64 = base64.b64encode(pdf_bytes).decode()
                st.components.v1.html(
                    f'<iframe src="data:application/pdf;base64,{pdf_base64}" '
                    f'width="700" height="800" style="border:none;"></iframe>',
                    height=820,
                )

    # 对话
    st.divider()
    st.subheader("对话")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                st.caption(msg["sources"])

    if prompt := st.chat_input("输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在检索文档..."):
                try:
                    query_embedding = embed_text(prompt)
                    results = search(query_embedding, top_k=5)

                    if not results:
                        reply = "未找到相关文档内容。"
                        sources_text = ""
                    else:
                        context = build_context(results)
                        reply = answer(prompt, context)
                        sources = sorted(set((r.filename, r.page) for r in results))
                        sources_text = f"来源: {', '.join(f'{f}(p{p})' for f, p in sources)}"

                    st.markdown(reply)
                    if sources_text:
                        st.caption(sources_text)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "sources": sources_text,
                    })
                except Exception as e:
                    st.error(f"出错了: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"出错了: {e}",
                    })
