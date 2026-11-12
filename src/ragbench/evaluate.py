"""Evaluation harness: metrics, configuration sweep, and a leaderboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from .answer import ABSTAIN_TEXT, content_tokens, extractive_answer
from .chunking import Chunk, Document, chunk_corpus, load_corpus
from .index import BM25Index


@dataclass(frozen=True)
class EvalRow:
    strategy: str
    k: int
    recall_at_k: float
    mrr_at_k: float
    citation_precision: float
    citation_recall: float
    answer_f1: float
    abstention_accuracy: float
    grounded_score: float
    answered: int
    total: int
    config: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_qa(path: str | Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        raise ValueError(f"no QA items in {path}")
    return items


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-content-words F1 between an answer and a reference answer."""
    pred = content_tokens(prediction)
    ref = content_tokens(reference)
    if not pred or not ref:
        return 0.0
    overlap = sum(min(pred.count(token), ref.count(token)) for token in set(pred))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _chunk_doc_map(chunks: Sequence[Chunk]) -> dict[str, str]:
    return {chunk.id: chunk.doc_id for chunk in chunks}


def evaluate_config(
    documents: Sequence[Document],
    qa: Sequence[dict[str, object]],
    strategy: str = "fixed",
    k: int = 5,
    **chunk_kwargs: int,
) -> EvalRow:
    chunks = chunk_corpus(list(documents), strategy=strategy, **chunk_kwargs)
    index = BM25Index(chunks)
    doc_of = _chunk_doc_map(chunks)

    recall_hits = 0
    reciprocal_ranks: list[float] = []
    citation_precisions: list[float] = []
    citation_recalls: list[float] = []
    f1_scores: list[float] = []
    abstain_hits = 0
    abstain_total = 0
    answered_answerable = 0

    for item in qa:
        question = str(item["question"])
        answers = [str(a) for a in item.get("answers", [])]  # type: ignore[union-attr]
        gold_docs = {str(d) for d in item.get("gold_docs", [])}  # type: ignore[union-attr]
        expected = str(item.get("expected_behavior", "answer"))

        retrieved = index.search(question, k=k)
        answer = extractive_answer(question, retrieved)
        retrieved_docs = [chunk.doc_id for chunk, _ in retrieved]

        if expected == "answer":
            hit = any(doc in gold_docs for doc in retrieved_docs)
            recall_hits += int(hit)
            rank = next(
                (i for i, doc in enumerate(retrieved_docs, start=1) if doc in gold_docs),
                None,
            )
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)

            if not answer.abstained:
                answered_answerable += 1
                cited_docs = [doc_of[cid] for cid in answer.citations if cid in doc_of]
                citation_recalls.append(1.0 if any(d in gold_docs for d in cited_docs) else 0.0)
                citation_precisions.append(
                    sum(1 for d in cited_docs if d in gold_docs) / len(cited_docs)
                    if cited_docs
                    else 0.0
                )
                reference = answers[0] if answers else ""
                f1_scores.append(token_f1(answer.text, reference))
            else:
                citation_recalls.append(0.0)
                citation_precisions.append(0.0)
                f1_scores.append(0.0)
        else:
            abstain_total += 1
            abstain_hits += int(answer.abstained)

    answerable_total = sum(1 for item in qa if item.get("expected_behavior") == "answer")
    mean = lambda values: sum(values) / len(values) if values else 0.0  # noqa: E731

    recall = recall_hits / answerable_total if answerable_total else 0.0
    mrr = mean(reciprocal_ranks)
    citation_precision = mean(citation_precisions)
    citation_recall = mean(citation_recalls)
    answer_f1 = mean(f1_scores)
    abstention_accuracy = abstain_hits / abstain_total if abstain_total else 0.0
    grounded_score = mean(
        [
            recall,
            mrr,
            citation_precision,
            citation_recall,
            answer_f1,
            abstention_accuracy,
        ]
    )

    config_label = ", ".join(f"{key}={value}" for key, value in sorted(chunk_kwargs.items()))
    return EvalRow(
        strategy=strategy,
        k=k,
        recall_at_k=round(recall, 4),
        mrr_at_k=round(mrr, 4),
        citation_precision=round(citation_precision, 4),
        citation_recall=round(citation_recall, 4),
        answer_f1=round(answer_f1, 4),
        abstention_accuracy=round(abstention_accuracy, 4),
        grounded_score=round(grounded_score, 4),
        answered=answered_answerable,
        total=len(qa),
        config=config_label,
    )


DEFAULT_SWEEP: tuple[tuple[str, dict[str, int]], ...] = (
    ("fixed", {"size": 400, "overlap": 80}),
    ("fixed", {"size": 250, "overlap": 60}),
    ("paragraph", {"max_chars": 600}),
)


def run_sweep(
    corpus_dir: str | Path,
    qa_path: str | Path,
    configs: Iterable[tuple[str, dict[str, int]]] = DEFAULT_SWEEP,
    ks: Sequence[int] = (3, 5),
) -> list[EvalRow]:
    documents = load_corpus(corpus_dir)
    qa = load_qa(qa_path)
    rows: list[EvalRow] = []
    for strategy, kwargs in configs:
        for k in ks:
            rows.append(evaluate_config(documents, qa, strategy=strategy, k=k, **kwargs))
    return rows


def leaderboard(rows: Sequence[EvalRow]) -> str:
    """Render rows as a Markdown leaderboard sorted by grounded score."""
    ordered = sorted(rows, key=lambda row: row.grounded_score, reverse=True)
    header = (
        "| chunking | config | k | recall@k | MRR@k | citation P | citation R | answer F1 | abstain | grounded |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for row in ordered:
        lines.append(
            f"| {row.strategy} | {row.config or '-'} | {row.k} | {row.recall_at_k:.3f} | {row.mrr_at_k:.3f} | "
            f"{row.citation_precision:.3f} | {row.citation_recall:.3f} | "
            f"{row.answer_f1:.3f} | {row.abstention_accuracy:.3f} | {row.grounded_score:.3f} |"
        )
    return "\n".join(lines)


def write_report(rows: Sequence[EvalRow], out_dir: str | Path) -> Path:
    """Persist the sweep as JSON + Markdown so results are reviewable in the repo."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"rows": [row.as_dict() for row in rows[:]]}
    (directory / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = "# ragbench evaluation\n\n" + leaderboard(rows) + "\n"
    path = directory / "results.md"
    path.write_text(report, encoding="utf-8")
    return path


__all__ = [
    "ABSTAIN_TEXT",
    "EvalRow",
    "evaluate_config",
    "leaderboard",
    "load_qa",
    "run_sweep",
    "token_f1",
    "write_report",
]
