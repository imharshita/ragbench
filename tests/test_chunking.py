from pathlib import Path

import pytest

from ragbench.chunking import (
    chunk_corpus,
    chunk_fixed,
    chunk_paragraphs,
    load_corpus,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "corpus"


def test_load_corpus_finds_documents() -> None:
    documents = load_corpus(DATA)
    assert len(documents) == 4
    ids = {doc.doc_id for doc in documents}
    assert "evaluation" in ids and "production" in ids


def test_fixed_chunks_are_bounded_and_unique() -> None:
    chunks = chunk_corpus(load_corpus(DATA), strategy="fixed", size=400, overlap=80)
    assert chunks
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert all(len(chunk.text) <= 400 for chunk in chunks)


def test_overlap_does_not_reduce_chunk_count() -> None:
    document = load_corpus(DATA)[0]
    without = chunk_fixed(document, size=200, overlap=0)
    with_overlap = chunk_fixed(document, size=200, overlap=80)
    assert len(with_overlap) >= len(without)


def test_overlap_must_be_smaller_than_size() -> None:
    document = load_corpus(DATA)[0]
    with pytest.raises(ValueError):
        chunk_fixed(document, size=100, overlap=100)


def test_paragraph_chunking_keeps_content() -> None:
    document = next(doc for doc in load_corpus(DATA) if doc.doc_id == "retrieval-augmented-generation")
    chunks = chunk_paragraphs(document, max_chars=300)
    assert chunks
    assert "RAG" in " ".join(chunk.text for chunk in chunks)
