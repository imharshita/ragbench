"""Optional FastAPI surface for ragbench.

Run with: uvicorn ragbench.server:app --reload
Requires the `serve` extra: pip install -e ".[serve]"
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .answer import extractive_answer
from .chunking import chunk_corpus, load_corpus
from .index import BM25Index


class AskRequest(BaseModel):
    question: str
    k: int = 4


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    text: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    support: float
    citations: list[Citation]


def create_app(corpus_dir: str = "data/corpus") -> FastAPI:
    app = FastAPI(title="ragbench", version="0.1.0")
    chunks = chunk_corpus(load_corpus(corpus_dir))
    index = BM25Index(chunks)
    by_id = {chunk.id: chunk for chunk in chunks}

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "chunks": len(chunks)}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        retrieved = index.search(request.question, k=request.k)
        answer = extractive_answer(request.question, retrieved)
        citations = [
            Citation(chunk_id=cid, doc_id=by_id[cid].doc_id, text=by_id[cid].text)
            for cid in answer.citations
            if cid in by_id
        ]
        return AskResponse(
            answer=answer.text,
            abstained=answer.abstained,
            support=answer.support,
            citations=citations,
        )

    return app


app = create_app()
