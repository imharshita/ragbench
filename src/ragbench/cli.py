"""Command-line interface for ragbench."""

from __future__ import annotations

import argparse

from .answer import extractive_answer
from .chunking import chunk_corpus, load_corpus
from .evaluate import leaderboard, run_sweep, write_report
from .index import BM25Index

DEFAULT_CORPUS = "data/corpus"
DEFAULT_QA = "data/qa.jsonl"
DEFAULT_OUT = "eval"


def _load_index(corpus_dir: str, size: int = 400, overlap: int = 80):
    documents = load_corpus(corpus_dir)
    chunks = chunk_corpus(documents, strategy="fixed", size=size, overlap=overlap)
    return chunks, BM25Index(chunks)


def cmd_ask(args: argparse.Namespace) -> int:
    chunks, index = _load_index(args.corpus, size=args.size, overlap=args.overlap)
    retrieved = index.search(args.question, k=args.k)
    answer = extractive_answer(args.question, retrieved)

    print(answer.text)
    if answer.citations:
        print()
        by_id = {chunk.id: chunk for chunk in chunks}
        for position, citation in enumerate(answer.citations, start=1):
            chunk = by_id[citation]
            print(f"[{position}] {chunk.doc_id}: {chunk.text}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    rows = run_sweep(args.corpus, args.qa, ks=tuple(args.k))
    print(leaderboard(rows))
    print()
    path = write_report(rows, args.out)
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragbench",
        description="Ground answers in a corpus, cite the passages, and measure it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask one question against the corpus")
    ask.add_argument("question")
    ask.add_argument("--corpus", default=DEFAULT_CORPUS)
    ask.add_argument("--k", type=int, default=4)
    ask.add_argument("--size", type=int, default=400)
    ask.add_argument("--overlap", type=int, default=80)
    ask.set_defaults(func=cmd_ask)

    evaluate = sub.add_parser("eval", help="sweep configs and print a leaderboard")
    evaluate.add_argument("--corpus", default=DEFAULT_CORPUS)
    evaluate.add_argument("--qa", default=DEFAULT_QA)
    evaluate.add_argument("--out", default=DEFAULT_OUT)
    evaluate.add_argument("--k", type=int, nargs="+", default=[3, 5])
    evaluate.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
