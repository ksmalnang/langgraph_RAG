from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.eval import retrieval_eval
from app.eval.retrieval_eval import evaluate_from_predictions, load_benchmark_cases
from app.utils.helpers import generate_doc_id


FIXTURES_DIR = Path("tests/fixtures")
DATASET_PATH = FIXTURES_DIR / "retrieval_eval.json"
PREDICTIONS_PATH = FIXTURES_DIR / "retrieval_eval_predictions.json"
BASELINE_PATH = FIXTURES_DIR / "retrieval_eval_baseline.json"
LIVE_CURRENT_BASELINE_PATH = FIXTURES_DIR / "retrieval_eval_live_current_corpus_baseline.json"
LIVE_HARD_BASELINE_PATH = FIXTURES_DIR / "retrieval_eval_live_current_corpus_hard_baseline.json"


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


def test_live_current_corpus_baseline_snapshot_is_stable():
    baseline = json.loads(LIVE_CURRENT_BASELINE_PATH.read_text(encoding="utf-8"))

    assert baseline["dataset_id"] == "retrieval_eval_live_current_corpus.json"
    assert baseline["evaluation_mode"] == "live_services"
    assert baseline["evaluation_date"] == "2026-04-09"
    assert baseline["stages"]["pre_rerank"]["metrics"] == {
        "hit@1": 0.75,
        "hit@3": 0.9167,
        "hit@5": 1.0,
        "mrr": 0.8542,
        "recall@5": 1.0,
    }
    assert baseline["stages"]["post_rerank"]["metrics"] == {
        "hit@1": 0.6667,
        "hit@3": 0.8333,
        "hit@5": 1.0,
        "mrr": 0.7833,
        "recall@5": 1.0,
    }


def test_live_hard_corpus_baseline_snapshot_is_stable():
    baseline = json.loads(LIVE_HARD_BASELINE_PATH.read_text(encoding="utf-8"))

    assert baseline["dataset_id"] == "retrieval_eval_live_current_corpus_hard.json"
    assert baseline["evaluation_mode"] == "live_services"
    assert baseline["evaluation_date"] == "2026-04-09"
    assert baseline["stages"]["pre_rerank"]["metrics"] == {
        "hit@1": 0.7692,
        "hit@3": 1.0,
        "hit@5": 1.0,
        "mrr": 0.8846,
        "recall@5": 1.0,
    }
    assert baseline["stages"]["post_rerank"]["metrics"] == {
        "hit@1": 0.9231,
        "hit@3": 1.0,
        "hit@5": 1.0,
        "mrr": 0.9615,
        "recall@5": 1.0,
    }


def test_retrieval_eval_normalizes_filename_path_case_and_chunk_index():
    cases = [
        retrieval_eval.BenchmarkCase(
            name="q1",
            query="dummy",
            k_target=1,
            targets=(
                retrieval_eval.TargetSpec(
                    expected_filename="registrasi_mahasiswa_baru.pdf",
                    expected_chunk_index=2,
                ),
            ),
        )
    ]
    stage_results = {
        "q1": [
            {
                "filename": "KB/REGISTRASI_MAHASISWA_BARU.PDF",
                "chunk_index": "2",
            }
        ]
    }

    scored = retrieval_eval.score_stage(cases, stage_results, k_eval=5)
    assert scored["metrics"]["hit@1"] == 1.0
    assert scored["metrics"]["mrr"] == 1.0


@pytest.mark.asyncio
async def test_live_eval_matches_expected_filename_by_derived_doc_id_and_persists_debug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "name": "q1",
                    "query": "where is schedule",
                    "expected_filename": "registrasi_mahasiswa_baru.pdf",
                    "k_target": 5,
                }
            ]
        ),
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        retrieval_top_k=10,
        rerank_top_n=5,
        relevance_threshold=0.17,
        embedding_model="unit-test-embed",
        jina_reranker_model="unit-test-reranker",
    )

    expected_doc_id = generate_doc_id("registrasi_mahasiswa_baru.pdf")

    async def fake_embed_query(_query: str) -> list[float]:
        return [0.1, 0.2]

    async def fake_hybrid_search(**_kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "doc_id": expected_doc_id,
                "chunk_index": 0,
                "score": 0.97,
                "source": "kb/REGISTRASI_MAHASISWA_BARU.PDF",
            }
        ]

    async def fake_rerank(
        query: str,
        documents: list[dict[str, object]],
        top_n: int | None = None,
    ) -> list[dict[str, object]]:
        assert query == "where is schedule"
        assert top_n == 5
        return documents

    monkeypatch.setattr(retrieval_eval, "get_settings", lambda: settings)
    monkeypatch.setattr(retrieval_eval, "embed_query", fake_embed_query)
    monkeypatch.setattr(retrieval_eval, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(retrieval_eval, "rerank", fake_rerank)

    report = await retrieval_eval.evaluate_live(
        dataset_path=dataset_path,
        k_eval=5,
        evaluation_date="2026-04-09",
    )

    assert report["stages"]["pre_rerank"]["metrics"]["hit@1"] == 1.0
    assert report["stages"]["post_rerank"]["metrics"]["hit@1"] == 1.0
    assert report["debug_candidates"]["pre_rerank"]["q1"][0]["doc_id"] == expected_doc_id
    assert (
        report["debug_candidates"]["pre_rerank"]["q1"][0]["source"]
        == "kb/REGISTRASI_MAHASISWA_BARU.PDF"
    )

