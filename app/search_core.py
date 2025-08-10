# app/search_core.py
import os, json
import numpy as np

# 为了结果稳定，默认优先离线用本地缓存的模型（如果想联网，删掉下面两行）
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = (x ** 2).sum(axis=1, keepdims=True) ** 0.5 + 1e-12
    return x / n

class Searcher:
    """
    负责：加载模型/索引/元数据 & 执行一次查询
    用法：
        s = Searcher("./cache/vectors.faiss", "./cache/meta.jsonl")
        results = s.search("wheel speed sensor", topk=5, section="both")
    """
    def __init__(self, faiss_path: str, meta_path: str, model_name: str = EMB_MODEL):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(faiss_path)

        # 读 meta.jsonl（与向量顺序一一对应）
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta_lines = f.readlines()  # 31k 行体量不大，直接放内存

    def encode_query(self, query: str) -> np.ndarray:
        v = self.model.encode([query], convert_to_numpy=True).astype("float32")
        return l2_normalize(v)

    def search(self, query: str, topk: int = 5, section: str = "both", cpc_prefix: str | None = None):
        """
        返回：list[dict]，每条包含 score/section/title/doc_number/classification/idx/snippet
        """
        q = self.encode_query(query)

        # 先多取一些候选，后面再过滤（保证有足够结果）
        K = max(topk * 5, topk)
        scores, idxs = self.index.search(q, K)
        scores, idxs = scores[0].tolist(), idxs[0].tolist()

        out = []
        for s, i in zip(scores, idxs):
            if i < 0:
                continue
            m = json.loads(self.meta_lines[i])

            # section 过滤
            if section in ("claim", "desc") and m["section"] != section:
                continue

            # cpc 前缀过滤（可选）
            if cpc_prefix:
                cpc = (m.get("classification") or "").upper()
                if not cpc.startswith(cpc_prefix.upper()):
                    continue

            snippet = (m["text"] or "")[:200].replace("\n", " ")
            out.append({
                "score": float(s),
                "section": m["section"],
                "title": m["title"],
                "doc_number": m["doc_number"],
                "classification": m.get("classification", ""),
                "idx": int(m["idx"]),
                "snippet": snippet
            })
            if len(out) >= topk:
                break

            # 如果过滤太严格导致不够 topk，可以在这里考虑“补齐”逻辑（先不做，保持简单）
        return out
