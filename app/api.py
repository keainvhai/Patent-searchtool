

# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Literal
from app.search_core import Searcher

# ---- 启动时全局加载（只加载一次，后续复用）----
FAISS_PATH = "./cache/vectors.faiss"
META_PATH  = "./cache/meta.jsonl"
SEARCHER = Searcher(FAISS_PATH, META_PATH)

app = FastAPI(title="Patent Paragraph Search (Part 1)")

from fastapi.middleware.cors import CORSMiddleware


# 仅开发环境：放开跨域，方便本地网页调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 生产环境请改成具体来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., description="自然语言或一段 claim 文本")
    topk: int = Field(5, ge=1, le=50, description="返回条数（1-50）")
    section: Literal["both","claim","desc"] = Field("both", description="检索范围")
    cpc_prefix: Optional[str] = Field(None, description="可选：CPC 分类前缀过滤，如 B60B")
    use_hybrid: bool = Field(False, description="是否启用 BM25+向量融合")
    bm25_weight: float = Field(0.5, ge=0.0, le=1.0, description="融合权重，越大越偏向关键词")


class SearchResult(BaseModel):
    rank: int
    score: float
    section: Literal["claim","desc"]
    title: str
    doc_number: str
    classification: str
    idx: int
    snippet: str
    text_full: Optional[str] = None   # 👈 新增


@app.post("/search", response_model=list[SearchResult])
def search(req: SearchRequest):
    # 1) 根据开关选择基础向量 or 混合检索
    if req.use_hybrid:
        raw = SEARCHER.search_hybrid(
            query=req.query,
            topk=req.topk,
            bm25_weight=req.bm25_weight,
            section=req.section,
            cpc_prefix=req.cpc_prefix
        )
    else:
        raw = SEARCHER.search(
            query=req.query,
            topk=req.topk,
            section=req.section,
            cpc_prefix=req.cpc_prefix
        )

    # 2) 统一加上 rank，并按你原来的响应模型返回
    return [
        SearchResult(rank=i+1, **r).dict()
        for i, r in enumerate(raw)
    ]

@app.get("/")
def root():
    return {"ok": True, "message": "Go to /docs to try the API."}
