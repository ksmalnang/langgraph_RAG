from __future__ import annotations

import json
from pathlib import Path

from app.eval.retrieval_eval import evaluate_from_predictions, load_benchmark_cases


FIXTURES_DIR = Path("tests/fixtures")
DATASET_PATH = FIXTURES_DIR / "retrieval_eval.json"
PREDICTIONS_PATH = FIXTURES_DIR / "retrieval_eval_predictions.json"
BASELINE_PATH = FIXTURES_DIR / "retrieval_eval_baseline.json"


def test_retrieval_eval_dataset_is_loadable():
    cases = load_benchmark_cases(DATASET_PATH)
    assert len(cases) >= 20
    assert all(case.query for case in cases)


def test_retrieval_eval_metrics_match_baseline_snapshot():
    report = evaluate_from_predictions(
        dataset_path=DATASET_PATH,
        predictions_path=PREDICTIONS_PATH,
        k_eval=5,
        evaluation_date="2026-04-08",
    )
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert report["dataset_id"] == baseline["dataset_id"]
    assert report["evaluation_date"] == baseline["evaluation_date"]
    assert report["retrieval_config"] == baseline["retrieval_config"]
    assert report["dependencies"] == baseline["dependencies"]
    assert report["stages"]["pre_rerank"]["metrics"] == baseline["stages"]["pre_rerank"]["metrics"]
    assert report["stages"]["post_rerank"]["metrics"] == baseline["stages"]["post_rerank"]["metrics"]


def test_retrieval_eval_uses_comparable_depth():
    report = evaluate_from_predictions(
        dataset_path=DATASET_PATH,
        predictions_path=PREDICTIONS_PATH,
        k_eval=5,
        evaluation_date="2026-04-08",
    )

    pre = report["stages"]["pre_rerank"]
    post = report["stages"]["post_rerank"]

    assert pre["k_eval"] == 5
    assert post["k_eval"] == 5
    assert pre["query_count"] == post["query_count"]

