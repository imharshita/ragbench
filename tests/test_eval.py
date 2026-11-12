from pathlib import Path

from ragbench.chunking import load_corpus
from ragbench.evaluate import (
    evaluate_config,
    leaderboard,
    load_qa,
    run_sweep,
    token_f1,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_token_f1_exact_and_disjoint() -> None:
    assert token_f1("a retrieval system with a language model", "a retrieval system with a language model") == 1.0
    assert token_f1("completely different words", "a retrieval system") == 0.0


def test_metrics_are_sane_on_the_sample_set() -> None:
    documents = load_corpus(DATA / "corpus")
    qa = load_qa(DATA / "qa.jsonl")
    row = evaluate_config(documents, qa, strategy="fixed", k=5, size=400, overlap=80)

    assert 0.0 <= row.recall_at_k <= 1.0
    assert 0.0 <= row.grounded_score <= 1.0
    assert row.recall_at_k >= 0.5
    assert row.abstention_accuracy == 1.0


def test_sweep_produces_leaderboard_and_report(tmp_path: Path) -> None:
    rows = run_sweep(DATA / "corpus", DATA / "qa.jsonl")
    assert len(rows) >= 2
    markdown = leaderboard(rows)
    assert "| chunking |" in markdown

    path = write_report(rows, tmp_path)
    assert path.exists()
    assert (tmp_path / "results.json").exists()
