# smoke_bm25.py
from app.search_core import Searcher

s = Searcher("./cache/vectors.faiss", "./cache/meta.jsonl")
pairs = s.bm25_topn("wheel speed sensor", topn=5)

# 打印前 5 个 BM25 结果及标题，确认 BM25 正常返回
for idx, score in pairs:
    m = __import__("json").loads(s.meta_lines[idx])
    print(idx, round(score, 2), m.get("title", "")[:80])
