# smoke_hybrid.py
import json
from app.search_core import Searcher

s = Searcher("./cache/vectors.faiss", "./cache/meta.jsonl")

print("=== BASE (vector only) ===")
base = s.search("wheel speed sensor", topk=5, section="both")
for r in base:
    print(round(r["score"], 3), r["title"][:80])

print("\n=== HYBRID (bm25+vector) ===")
hyb = s.search_hybrid("wheel speed sensor", topk=5, bm25_weight=0.5, section="both")
for r in hyb:
    print(round(r["score"], 3), r["title"][:80])
