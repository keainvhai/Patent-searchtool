

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

app = FastAPI(title="Patent Paragraph Search (Part 1)")

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

class SearchResult(BaseModel):
    rank: int
    score: float
    section: Literal["claim","desc"]
    title: str
    doc_number: str
    classification: str
    idx: int
    snippet: str

@app.post("/search", response_model=list[SearchResult])
def search(req: SearchRequest):
    results = SEARCHER.search(
        query=req.query,
        topk=req.topk,
        section=req.section,
        cpc_prefix=req.cpc_prefix
    )
    # 加上 rank 字段
    return [
        SearchResult(rank=i+1, **r).dict()
        for i, r in enumerate(results)
    ]

@app.get("/")
def root():
    return {"ok": True, "message": "Go to /docs to try the API."}
