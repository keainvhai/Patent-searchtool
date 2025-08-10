# app/smoke_local.py
import os, json, argparse
import numpy as np

# 强制离线
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def l2_normalize(x):
    n = (x ** 2).sum(axis=1, keepdims=True) ** 0.5 + 1e-12
    return x / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss", default="./cache/vectors.faiss")
    ap.add_argument("--meta", default="./cache/meta.jsonl")
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import faiss

    # 加载本地模型（已缓存）
    model = SentenceTransformer(EMB_MODEL)

    # 编码查询
    q = model.encode([args.query], convert_to_numpy=True).astype("float32")
    q = l2_normalize(q)

    # 读取索引
    index = faiss.read_index(args.faiss)

    # 搜索
    scores, idxs = index.search(q, args.topk)
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    # 读取元信息（与索引顺序对应）
    with open(args.meta, "r", encoding="utf-8") as f:
        meta_lines = f.readlines()

    print(f"\n[Smoke Test - Offline]\nQuery: {args.query!r}\n")
    for rank, (s, i) in enumerate(zip(scores, idxs), start=1):
        if i < 0: 
            continue
        m = json.loads(meta_lines[i])
        snippet = m["text"][:200].replace("\n", " ")
        print(f"{rank:>2}. score={s:.4f} | [{m['section']}] {m['title']} "
              f"(#{m['doc_number']}, cpc={m['classification']}, idx={m['idx']})")
        print(f"    {snippet}...")
    print()

if __name__ == "__main__":
    main()
