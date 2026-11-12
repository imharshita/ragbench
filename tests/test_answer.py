from pathlib import Path

from ragbench.answer import ABSTAIN_TEXT, extractive_answer
from ragbench.chunking import chunk_corpus, load_corpus
from ragbench.index import BM25Index

DATA = Path(__file__).resolve().parents[1] / "data" / "corpus"


def make_index() -> BM25Index:
    return BM25Index(chunk_corpus(load_corpus(DATA), strategy="fixed"))


def test_answer_is_cited_from_gold_document() -> None:
    index = make_index()
    retrieved = index.search("How are hybrid ranked lists fused?", k=4)
    answer = extractive_answer("How are hybrid ranked lists fused?", retrieved)
    assert not answer.abstained
    assert answer.citations
    assert any(citation.startswith("production") for citation in answer.citations)
    assert answer.support >= 0.3


def test_abstains_when_nothing_relevant_is_retrieved() -> None:
    index = make_index()
    retrieved = index.search("What is the capital of France?", k=4)
    answer = extractive_answer("What is the capital of France?", retrieved)
    assert answer.abstained
    assert answer.citations == ()
    assert answer.text == ABSTAIN_TEXT


def test_abstains_when_support_is_weak() -> None:
    index = make_index()
    retrieved = index.search("Which cloud provider hosts the corpus?", k=4)
    answer = extractive_answer("Which cloud provider hosts the corpus?", retrieved)
    assert answer.abstained
