"""Extractive, citation-first answering.

The baseline answerer never invents text: it selects sentences from the retrieved
chunks, records exactly which chunks it used, and abstains when support is weak.
An LLM generator can later sit behind the same `Answer` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .chunking import Chunk
from .index import tokenize

ABSTAIN_TEXT = "I don't know based on the provided corpus."

STOPWORDS: frozenset[str] = frozenset(
    """
    a an the is are was were be been being do does did done have has had having
    what which who whom whose when where why how this that these those it its
    of in on at to for with from by as and or not no nor but if then than so
    should must can could would may might will shall about over under between
    into out up down more most many much some any all both each few other
    also there here their our your his her they we you i one two
    """.split()
)


def content_tokens(text: str) -> list[str]:
    """Word tokens that carry topical meaning (no stopwords, no single characters)."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without pulling in an NLP dependency."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[str, ...]
    abstained: bool
    support: float


def _sentence_candidates(
    retrieved: Sequence[tuple[Chunk, float]],
) -> list[tuple[str, Chunk]]:
    candidates: list[tuple[str, Chunk]] = []
    for chunk, _score in retrieved:
        for sentence in split_sentences(chunk.text):
            candidates.append((sentence, chunk))
    return candidates


def extractive_answer(
    question: str,
    retrieved: Sequence[tuple[Chunk, float]],
    max_sentences: int = 3,
    min_support: float = 0.3,
) -> Answer:
    """Answer by selecting retrieved sentences that cover the question's terms.

    Support is the fraction of the question's content tokens present in a selected
    sentence. Below `min_support` the answerer abstains rather than guessing.
    """
    if not retrieved:
        return Answer(ABSTAIN_TEXT, (), True, 0.0)

    question_tokens = set(content_tokens(question))
    if not question_tokens:
        return Answer(ABSTAIN_TEXT, (), True, 0.0)

    scored: list[tuple[float, str, Chunk]] = []
    for sentence, chunk in _sentence_candidates(retrieved):
        sentence_tokens = set(content_tokens(sentence))
        support = len(question_tokens & sentence_tokens) / len(question_tokens)
        if support > 0:
            scored.append((support, sentence, chunk))

    if not scored:
        return Answer(ABSTAIN_TEXT, (), True, 0.0)

    scored.sort(key=lambda item: item[0], reverse=True)
    best_support = scored[0][0]
    if best_support < min_support:
        return Answer(ABSTAIN_TEXT, (), True, best_support)

    selected: list[str] = []
    citations: list[str] = []
    for support, sentence, chunk in scored:
        if support < min_support or len(selected) >= max_sentences:
            break
        if sentence in selected:
            continue
        selected.append(sentence)
        if chunk.id not in citations:
            citations.append(chunk.id)

    return Answer(" ".join(selected), tuple(citations), False, best_support)
