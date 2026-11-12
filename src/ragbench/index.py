"""Retrieval: BM25 by default, with optional dense retrieval and hybrid fusion."""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from rank_bm25 import BM25Okapi

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens shared by retrieval and answer scoring."""
    return _TOKEN_RE.findall(text.lower())


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]: ...


class BM25Index:
    """Lexical retriever over a fixed chunk list."""

    def __init__(self, chunks: Sequence[Chunk]):
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self.chunks: list[Chunk] = list(chunks)
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            ((self.chunks[i], float(scores[i])) for i in range(len(self.chunks))),
            key=lambda item: item[1],
            reverse=True,
        )
        return [item for item in ranked[:k] if item[1] > 0]


class DenseIndex:
    """Optional dense retriever. Requires the `dense` extra (sentence-transformers)."""

    def __init__(self, chunks: Sequence[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional path
            raise ImportError(
                "dense retrieval requires sentence-transformers: pip install -e '.[dense]'"
            ) from exc

        if not chunks:
            raise ValueError("DenseIndex needs at least one chunk")
        self.chunks = list(chunks)
        self._model = SentenceTransformer(model_name)
        self._vectors = self._model.encode(
            [chunk.text for chunk in self.chunks], normalize_embeddings=True
        )

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        query_vector = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._vectors @ query_vector
        order = scores.argsort()[::-1][:k]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in order]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[Chunk, float]]], k: int = 60
) -> list[tuple[Chunk, float]]:
    """Fuse ranked lists: score = sum(1 / (k + rank)) over lists (1-based ranks)."""
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (k + rank)
            by_id[chunk.id] = chunk
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [(by_id[chunk_id], score) for chunk_id, score in ordered]


def hybrid_search(
    lexical: Retriever, dense: Retriever, query: str, k: int = 5, pool: int = 20
) -> list[tuple[Chunk, float]]:
    """Retrieve from both backends, fuse with RRF, return the top k."""
    return reciprocal_rank_fusion(
        [lexical.search(query, pool), dense.search(query, pool)], k=k
    )[:k]
