import sys
sys.path.insert(0, r'E:/mycode/test/myrag')
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
from retriever import search_by_text
from embedder import embed_text
from llm import build_context, answer

q = sys.argv[1] if len(sys.argv) > 1 else 'RAG是什么'
results = search_by_text(q, embed_text, top_k=3)
print(f'Found {len(results)} results')
for r in results:
    safe_text = r.text[:300].replace('\n', ' ')
    try:
        safe_text.encode('gbk')
    except UnicodeEncodeError:
        safe_text = safe_text.encode('gbk', errors='ignore').decode('gbk')
    print(f'--- {r.filename} page {r.page} score: {r.score:.3f}')
    print(safe_text)
    print()
context = build_context(results)
reply = answer(q, context)
print('=== ANSWER ===')
print(reply)
