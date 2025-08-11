# app/search_core.py
import os, json
import numpy as np
import re
from rank_bm25 import BM25Okapi


# 为了结果稳定，默认优先离线用本地缓存的模型（如果想联网，删掉下面两行）
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_WORD = re.compile(r"\w+", re.U)
def _tok(s: str):
    # 简单英文 token 切分；后面如果含中文再换分词器
    return _WORD.findall((s or "").lower())

def _minmax(a: np.ndarray):
    if a.size == 0: return a
    mn, mx = float(a.min()), float(a.max())
    return (a - mn) / (mx - mn + 1e-9) if mx > mn else np.zeros_like(a)

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

  # 👇 新增：记录是否是 Inner Product 索引
        self._is_ip = hasattr(self.index, "metric_type") and \
                    (self.index.metric_type == faiss.METRIC_INNER_PRODUCT)

        # 读 meta.jsonl（与向量顺序一一对应）
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta_lines = f.readlines()  # 31k 行体量不大，直接放内存
        
        # ====== 新增：把每条文档的可检索文本取出来（用于 BM25）======
        # 用 text 字段
        self._docs_texts = []
        for line in self.meta_lines:
            m = json.loads(line)
            # 这里用你已有的字段，保持和向量索引时的一致性
            self._docs_texts.append((m.get("text") or ""))

        # ====== 新增：初始化 BM25 ======
        self._doc_tokens = [_tok(t) for t in self._docs_texts]
        self._bm25 = BM25Okapi(self._doc_tokens)   # 默认参数 k1/b 即可


    def encode_query(self, query: str) -> np.ndarray:
        v = self.model.encode([query], convert_to_numpy=True).astype("float32")
        return l2_normalize(v)
    
    def vector_topn(self, query: str, topn: int = 200):
        """
        返回 [(doc_idx, sim)]，sim 越大越相关
        """
        q = self.encode_query(query)
        D, I = self.index.search(q, topn)  # D: 距离或相似度；I: 索引
        D, I = D[0], I[0]

        # 将 D 统一转为“越大越好”
        # 经验：若是 L2 索引 → 值越小越近；转换为 sim = 1/(1+d)
        # 若是 IP 索引 → 已经是相似度；此转换也可用，不会出错（只是单调映射）
        # 👇区分两种度量
        if self._is_ip:
            # IP：D 本身就是“越大越好”的相似度
            sims = D
        else:
            # L2：D 是距离，越小越近 → 转成“越大越好”的相似度
            sims = 1.0 / (1.0 + D)
        return [(int(i), float(s)) for i, s in zip(I, sims) if int(i) >= 0]

    def bm25_topn(self, query: str, candidate_ids: list[int] | None = None, topn: int = 200):
            """
            仅 BM25 的检索：返回 [(doc_idx, bm25_score)]，score 越大越相关
            - candidate_ids: 可选的候选集合（传入则只在该集合里打分；不传则全量）
            """
            assert self._bm25 is not None, "BM25 is not initialized."
            qtok = _tok(query)

            scores_full = self._bm25.get_scores(qtok)  # 对全量文档打分
            if candidate_ids is None:
                pairs = [(i, float(scores_full[i])) for i in range(len(scores_full))]
            else:
                pairs = [(i, float(scores_full[i])) for i in candidate_ids]

            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:topn]

    def hybrid_topk(self, query: str, topk: int = 10, bm25_weight: float = 0.5,
                    candidate_ids: list[int] | None = None,
                    bm25_M: int = 200, vec_N: int = 200):
        """
        混合检索：线性融合  final = alpha * bm25_norm + (1-alpha) * vec_norm
        返回 [(idx, final_score)]
        """
        # 1) 两路分数
        bm25_pairs = self.bm25_topn(query, candidate_ids=candidate_ids, topn=bm25_M)
        vec_pairs  = self.vector_topn(query, topn=vec_N)

        bm25_map = {i: s for i, s in bm25_pairs}
        vec_map  = {i: s for i, s in vec_pairs}

        # 2) 候选集合 = 两路并集
        cand = list(set(bm25_map.keys()) | set(vec_map.keys()))
        if not cand:
            return []

        bm_vals = np.array([bm25_map.get(i, 0.0) for i in cand], dtype=float)
        vc_vals = np.array([vec_map.get(i, 0.0) for i in cand], dtype=float)

        bm_n = _minmax(bm_vals)
        vc_n = _minmax(vc_vals)

        alpha = float(bm25_weight)
        final = alpha * bm_n + (1.0 - alpha) * vc_n

        ranked = sorted(zip(cand, final), key=lambda x: x[1], reverse=True)[:topk]
        return ranked


    def _format_by_idxs(self, idx_score_pairs):
        out = []
        for i, s in idx_score_pairs:
            m = json.loads(self.meta_lines[i])
            text_full = (m.get("text") or "")           # ✅ 先取全文
            snippet = text_full[:200].replace("\n", " ")# ✅ 再做摘要
            out.append({
                "score": float(s),
                "section": m["section"],
                "title": m["title"],
                "doc_number": m["doc_number"],
                "classification": m.get("classification", ""),
                "idx": int(m["idx"]),
                "snippet": snippet,
                "text_full": text_full,                 # ✅ 现在变量已存在
            })
        return out

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

            text_full = (m.get("text") or "")
            snippet = text_full[:200].replace("\n", " ")
            out.append({
                "score": float(s),
                "section": m["section"],
                "title": m["title"],
                "doc_number": m["doc_number"],
                "classification": m.get("classification", ""),
                "idx": int(m["idx"]),
                "snippet": snippet,
                "text_full": text_full,   # ✅ 新增
            })
            if len(out) >= topk:
                break

            # 如果过滤太严格导致不够 topk，可以在这里考虑“补齐”逻辑（先不做，保持简单）
        return out
  
    def search_hybrid(self, query: str, topk: int = 5, bm25_weight: float = 0.5,
                      section: str = "both", cpc_prefix: str | None = None):
        """
        混合检索 + CPC/section 过滤
        返回与 self.search 相同结构的列表
        """
        # 1) 构建候选集合（先按 CPC/section 过滤）
        candidate_ids = []
        prefix = (cpc_prefix or "").upper()
        for i, line in enumerate(self.meta_lines):
            m = json.loads(line)

            # section 过滤
            if section in ("claim", "desc") and m["section"] != section:
                continue

            # CPC 前缀过滤
            if prefix:
                cpc = (m.get("classification") or "").upper()
                if not cpc.startswith(prefix):
                    continue

            candidate_ids.append(i)

        if not candidate_ids:
            # 没有候选则回退全量
            candidate_ids = None

        # 2) 混合融合
        ranked = self.hybrid_topk(query, topk=topk, bm25_weight=bm25_weight,
                                  candidate_ids=candidate_ids)

        # 3) 格式化输出
        return self._format_by_idxs(ranked)


