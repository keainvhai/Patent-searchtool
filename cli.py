# cli.py
import os, json, argparse
import numpy as np

# 建议：离线优先，确保与建索引时用同一份缓存模型
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def l2_normalize(x):
    n = (x ** 2).sum(axis=1, keepdims=True) ** 0.5 + 1e-12
    return x / n

def load_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def search_once(faiss_path, meta, query, topk=5, section="both"):
    from sentence_transformers import SentenceTransformer
    import faiss

    # 1) 编码查询
    model = SentenceTransformer(EMB_MODEL)
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    q = l2_normalize(q)

    # 2) 加载索引 & 取较多候选
    index = faiss.read_index(faiss_path)
    K = max(topk * 5, topk)  # 先多取一些，后面再过滤
    scores, idxs = index.search(q, K)
    scores, idxs = scores[0].tolist(), idxs[0].tolist()

    # 3) 过滤 section 并组装结果
    out = []
    for s, i in zip(scores, idxs):
        if i < 0: 
            continue
        m = meta[i]
        if section in ("claim", "desc") and m["section"] != section:
            continue
        out.append((s, i, m))
        if len(out) >= topk:
            break
    return out

def main():
    ap = argparse.ArgumentParser(description="Patent paragraph search (Part 1)")
    ap.add_argument("--query", required=True, help="自然语言或一段 claim 文本")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--section", choices=["both","claim","desc"], default="both")
    ap.add_argument("--faiss", default="./cache/vectors.faiss")
    ap.add_argument("--meta", default="./cache/meta.jsonl")
    args = ap.parse_args()

    meta = load_meta(args.meta)
    results = search_once(args.faiss, meta, args.query, args.topk, args.section)

    print(f"\nQuery: {args.query!r} | section={args.section} | topk={args.topk}\n")
    for rank, (s, i, m) in enumerate(results, start=1):
        head = f"{rank:>2}. score={s:.4f} | [{m['section']}] {m['title']} (#{m['doc_number']}, cpc={m['classification']}, idx={m['idx']})"
        snippet = m["text"][:200].replace("\n", " ")
        print(head)
        print(f"    {snippet}...\n")

if __name__ == "__main__":
    main()
