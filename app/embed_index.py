# app/embed_index.py
import os, json, argparse, time
import numpy as np
import pandas as pd

# 1) 选一个轻量句向量模型（速度优先，够用）
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def build_and_save_index(units_path: str, faiss_path: str, meta_path: str, batch_size: int = 128):
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    df = pd.read_parquet(units_path)
    texts = df["text"].astype(str).tolist()

    print(f"[1/4] Load units: {len(texts)} rows from {units_path}")

    # 2) 批量向量化
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False
    ).astype("float32")
    emb = l2_normalize(emb)  # 归一化，后面用内积≈余弦

    print(f"[2/4] Encoded: shape={emb.shape} (rows, dim)")

    # 3) 建索引（FAISS 内积索引）
    try:
        import faiss
    except Exception as e:
        raise RuntimeError(
            "FAISS 未安装或不可用。请尝试 `pip install faiss-cpu`。如果仍失败，告诉我我给你 sklearn 版替代索引。"
        ) from e

    index = faiss.IndexFlatIP(emb.shape[1])  # 内积（配合归一化 = 余弦）
    index.add(emb)
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
    faiss.write_index(index, faiss_path)

    # 4) 保存与向量同序的元信息（检索命中后要显示）
    keep_cols = ["doc_number", "title", "classification", "section", "idx", "text"]
    meta = df[keep_cols].to_dict(orient="records")
    with open(meta_path, "w", encoding="utf-8") as w:
        for m in meta:
            w.write(json.dumps(m, ensure_ascii=False) + "\n")

    t1 = time.perf_counter()
    print(f"[3/4] Saved index → {faiss_path}")
    print(f"[4/4] Saved meta  → {meta_path}")
    print(f"[OK] Build time: {t1 - t0:.2f}s | vectors={emb.shape[0]} | dim={emb.shape[1]}")

def smoke_test(faiss_path: str, meta_path: str, query: str, topk: int = 5):
    """小冒烟测试：给一句查询，看能否返回前几条结果"""
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer(EMB_MODEL)
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    q = l2_normalize(q)

    index = faiss.read_index(faiss_path)

    scores, idxs = index.search(q, topk)
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    # 读取前 topk 行元信息
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        # 简单读到列表（小数据足够；大数据可优化为随机访问）
        meta_lines = f.readlines()

    print("\n[Smoke Test]")
    print(f"Query: {query!r}")
    for rank, (s, i) in enumerate(zip(scores, idxs), start=1):
        if i < 0:
            continue
        m = json.loads(meta_lines[i])
        snippet = m["text"][:200].replace("\n", " ")
        print(f"{rank:>2}. score={s:.4f} | [{m['section']}] {m['title']} (#{m['doc_number']}, cpc={m['classification']}, idx={m['idx']})")
        print(f"    {snippet}...")
    print()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="./cache/units.parquet", help="A 步产物")
    ap.add_argument("--faiss", default="./cache/vectors.faiss", help="索引输出路径")
    ap.add_argument("--meta", default="./cache/meta.jsonl", help="元信息输出路径（与索引同序）")
    args = ap.parse_args()

    build_and_save_index(args.in_path, args.faiss, args.meta)

    