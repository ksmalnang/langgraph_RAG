"""Retrieval-only benchmark evaluator for the public/admin path."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.embeddings import embed_query
from app.services.reranker import rerank
from app.services.vectorstore import hybrid_search


@dataclass(frozen=True)
class TargetSpec:
    """Expected supporting target for one benchmark query."""

    expected_filename: str | None = None
    expected_doc_id: str | None = None
    expected_chunk_index: int | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    """Single retrieval benchmark case."""

    name: str
    query: str
    k_target: int
    targets: tuple[TargetSpec, ...]
    notes: str | None = None


def _parse_target(data: dict[str, Any]) -> TargetSpec:
    return TargetSpec(
        expected_filename=data.get("expected_filename"),
        expected_doc_id=data.get("expected_doc_id"),
        expected_chunk_index=data.get("expected_chunk_index"),
    )


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    """Load benchmark dataset entries from JSON."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []

    for entry in entries:
        targets_data = entry.get("expected_targets")
        if targets_data:
            targets = tuple(_parse_target(t) for t in targets_data)
        else:
            targets = (_parse_target(entry),)

        if not any(t.expected_filename or t.expected_doc_id for t in targets):
            raise ValueError(
                f"Case '{entry.get('name', '<unnamed>')}' must include expected_filename or expected_doc_id"
            )

        cases.append(
            BenchmarkCase(
                name=entry["name"],
                query=entry["query"],
                k_target=int(entry.get("k_target", 5)),
                targets=targets,
                notes=entry.get("notes"),
            )
        )

    return cases


def load_stage_predictions(path: str | Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load staged candidate lists keyed by benchmark case name."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pre = payload.get("pre_rerank", {})
    post = payload.get("post_rerank", {})
    if not isinstance(pre, dict) or not isinstance(post, dict):
        raise ValueError("Prediction file must contain pre_rerank and post_rerank maps")
    return {"pre_rerank": pre, "post_rerank": post}


def _matches_target(candidate: dict[str, Any], target: TargetSpec) -> bool:
    if target.expected_filename is not None:
        if candidate.get("filename") != target.expected_filename:
            return False
    if target.expected_doc_id is not None:
        if candidate.get("doc_id") != target.expected_doc_id:
            return False
    if target.expected_chunk_index is not None:
        if candidate.get("chunk_index") != target.expected_chunk_index:
            return False
    return True


def _first_relevant_rank(
    candidates: list[dict[str, Any]],
    targets: tuple[TargetSpec, ...],
) -> int | None:
    for idx, candidate in enumerate(candidates, start=1):
        if any(_matches_target(candidate, target) for target in targets):
            return idx
    return None


def _recall_at_k(
    candidates: list[dict[str, Any]],
    targets: tuple[TargetSpec, ...],
    k: int,
) -> float:
    if not targets:
        return 0.0

    seen: set[int] = set()
    for candidate in candidates[:k]:
        for idx, target in enumerate(targets):
            if idx in seen:
                continue
            if _matches_target(candidate, target):
                seen.add(idx)

    return len(seen) / len(targets)


def _round_metrics(metrics: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {key: round(val, digits) for key, val in metrics.items()}


def score_stage(
    cases: list[BenchmarkCase],
    stage_results: dict[str, list[dict[str, Any]]],
    k_eval: int = 5,
) -> dict[str, Any]:
    """Compute retrieval metrics for one stage."""
    if k_eval < 1:
        raise ValueError("k_eval must be >= 1")

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_sum = 0.0
    recall_at_5_sum = 0.0
    details: list[dict[str, Any]] = []

    for case in cases:
        candidates = stage_results.get(case.name, [])[:k_eval]
        rank = _first_relevant_rank(candidates, case.targets)

        hit1 = 1.0 if rank is not None and rank <= 1 else 0.0
        hit3 = 1.0 if rank is not None and rank <= 3 else 0.0
        hit5 = 1.0 if rank is not None and rank <= 5 else 0.0
        mrr = (1.0 / rank) if rank is not None else 0.0
        recall5 = _recall_at_k(candidates, case.targets, k=5)

        hits_at_1 += hit1
        hits_at_3 += hit3
        hits_at_5 += hit5
        reciprocal_sum += mrr
        recall_at_5_sum += recall5

        details.append(
            {
                "name": case.name,
                "k_target": case.k_target,
                "rank": rank,
                "hit@1": hit1,
                "hit@3": hit3,
                "hit@5": hit5,
                "mrr": round(mrr, 4),
                "recall@5": round(recall5, 4),
            }
        )

    count = len(cases)
    metrics = {
        "hit@1": hits_at_1 / count if count else 0.0,
        "hit@3": hits_at_3 / count if count else 0.0,
        "hit@5": hits_at_5 / count if count else 0.0,
        "mrr": reciprocal_sum / count if count else 0.0,
        "recall@5": recall_at_5_sum / count if count else 0.0,
    }

    return {
        "query_count": count,
        "k_eval": k_eval,
        "metrics": _round_metrics(metrics),
        "per_query": details,
    }


def evaluate_from_predictions(
    dataset_path: str | Path,
    predictions_path: str | Path,
    *,
    k_eval: int = 5,
    evaluation_date: str | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval stages from prepared candidate lists."""
    settings = get_settings()
    cases = load_benchmark_cases(dataset_path)
    stages = load_stage_predictions(predictions_path)

    stage_scores = {
        "pre_rerank": score_stage(cases, stages["pre_rerank"], k_eval=k_eval),
        "post_rerank": score_stage(cases, stages["post_rerank"], k_eval=k_eval),
    }

    date_value = evaluation_date or datetime.now(UTC).date().isoformat()
    return {
        "dataset_id": Path(dataset_path).name,
        "evaluation_date": date_value,
        "corpus_id": "fixture-admin-kb-v1",
        "evaluation_mode": "fixture_predictions",
        "retrieval_config": {
            "k_eval": k_eval,
            "retrieval_top_k": max(k_eval, settings.retrieval_top_k),
            "rerank_top_n": max(k_eval, settings.rerank_top_n),
            "relevance_threshold": settings.relevance_threshold,
        },
        "dependencies": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.jina_reranker_model,
        },
        "stages": stage_scores,
    }


async def evaluate_live(
    dataset_path: str | Path,
    *,
    k_eval: int = 5,
    evaluation_date: str | None = None,
) -> dict[str, Any]:
    """Run a live retrieval benchmark against configured services."""
    settings = get_settings()
    cases = load_benchmark_cases(dataset_path)

    retrieval_top_k = max(k_eval, settings.retrieval_top_k)
    rerank_top_n = max(k_eval, settings.rerank_top_n)

    pre_results: dict[str, list[dict[str, Any]]] = {}
    post_results: dict[str, list[dict[str, Any]]] = {}

    for case in cases:
        vector = await embed_query(case.query)
        pre_docs = await hybrid_search(
            query_text=case.query,
            query_vector=vector,
            top_k=retrieval_top_k,
        )
        reranked_docs = await rerank(
            query=case.query,
            documents=pre_docs,
            top_n=rerank_top_n,
        )
        pre_results[case.name] = pre_docs[:k_eval]
        post_results[case.name] = reranked_docs[:k_eval]

    stage_scores = {
        "pre_rerank": score_stage(cases, pre_results, k_eval=k_eval),
        "post_rerank": score_stage(cases, post_results, k_eval=k_eval),
    }

    date_value = evaluation_date or datetime.now(UTC).date().isoformat()
    return {
        "dataset_id": Path(dataset_path).name,
        "evaluation_date": date_value,
        "corpus_id": "live-admin-kb",
        "evaluation_mode": "live_services",
        "retrieval_config": {
            "k_eval": k_eval,
            "retrieval_top_k": retrieval_top_k,
            "rerank_top_n": rerank_top_n,
            "relevance_threshold": settings.relevance_threshold,
        },
        "dependencies": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.jina_reranker_model,
        },
        "stages": stage_scores,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only benchmark")
    parser.add_argument("--dataset", required=True, help="Path to benchmark dataset JSON")
    parser.add_argument("--predictions", help="Path to fixture prediction JSON")
    parser.add_argument("--output", help="Path to write JSON report")
    parser.add_argument("--k-eval", type=int, default=5, help="Comparable eval depth")
    parser.add_argument(
        "--mode",
        choices=["fixture", "live"],
        default="fixture",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--evaluation-date",
        default=None,
        help="Optional YYYY-MM-DD override for deterministic snapshots",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "live":
        report = asyncio.run(
            evaluate_live(
                dataset_path=args.dataset,
                k_eval=args.k_eval,
                evaluation_date=args.evaluation_date,
            )
        )
    else:
        if not args.predictions:
            raise SystemExit("--predictions is required in fixture mode")
        report = evaluate_from_predictions(
            dataset_path=args.dataset,
            predictions_path=args.predictions,
            k_eval=args.k_eval,
            evaluation_date=args.evaluation_date,
        )

    rendered = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

