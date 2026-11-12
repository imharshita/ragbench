from pathlib import Path

import pytest

from ragbench.chunking import chunk_corpus, load_corpus
from ragbench.index import BM25Index

DATA = Path(__file__).resolve().parents[1] / "data" / "corpus"


def make_index() -> BM25Index:
    return BM25Index(chunk_corpus(load_corpus(DATA), strategy="fixed"))


def test_retrieval_ranks_gold_document_first() -> None:
    results = make_index().search("reciprocal rank fusion", k=3)
    assert results
    assert results[0][0].doc_id == "production"


def test_unrelated_query_returns_nothing() -> None:
    assert make_index().search("xylophone quasar zzzz", k=5) == []


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        make_index().search("rag", k=0)
