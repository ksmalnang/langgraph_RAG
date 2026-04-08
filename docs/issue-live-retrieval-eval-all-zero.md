# Issue: Live retrieval eval returns all-zero metrics

## Summary

Running retrieval eval in `live` mode produces `0.0` for all metrics in both stages (`pre_rerank` and `post_rerank`) across all 20 queries.

## Environment

- Repo: `langgraph_agent_ai`
- Date observed: `2026-04-09`
- Evaluator: `app/eval/retrieval_eval.py`
- Dataset: `tests/fixtures/retrieval_eval.json`
- Output: `tests/fixtures/retrieval_eval_live.json`

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m app.eval.retrieval_eval `
  --mode live `
  --dataset tests/fixtures/retrieval_eval.json `
  --output tests/fixtures/retrieval_eval_live.json `
  --k-eval 5
```

## Actual Result

- `tests/fixtures/retrieval_eval_live.json` shows:
  - `pre_rerank.metrics`: all `0.0`
  - `post_rerank.metrics`: all `0.0`
  - every `per_query[*].rank` is `null`

## Expected Result

- Live metrics should be non-zero when retrieval pipeline is healthy and indexed KB content corresponds to the benchmark targets.
- At minimum, some queries should resolve to expected targets within top-5.

## Suspected Root Cause

`_matches_target` in `app/eval/retrieval_eval.py` requires exact equality for `filename` and/or `doc_id`.

If live-indexed metadata naming/identity differs from fixture expectations (for example, normalized `doc_id` changed, filename casing/path differences, or missing metadata fields), all candidates are treated as misses even if semantically correct.

## Evidence

- Fixture mode with fixed predictions is healthy and reproducible against baseline:
  - `pre_rerank`: `hit@1=0.5`, `hit@3=0.85`, `hit@5=0.95`, `mrr=0.6833`, `recall@5=0.95`
  - `post_rerank`: `hit@1=0.85`, `hit@3=1.0`, `hit@5=1.0`, `mrr=0.9167`, `recall@5=1.0`
- Live snapshot shows complete failure only in `live` mode.

## Proposed Fix Direction

1. Add debug logging in `evaluate_live` to persist top-k candidate metadata (`doc_id`, `filename`, `chunk_index`) per query when `--mode live`.
2. Validate ingestion identity consistency with the documented rules in `docs/retrieval-eval.md`:
   - stable normalized `doc_id`
   - deterministic chunk identity
   - replace-on-reingestion behavior for same `doc_id`
3. Consider target-matching normalization guardrails:
   - normalize filename before compare (case/path/extension handling)
   - allow optional alias mapping if canonical identity changed intentionally
4. Add a regression test that fails when live-like candidate payload has semantically correct target but different raw formatting.

## Acceptance Criteria

- Re-running live eval with current corpus no longer yields all-zero metrics.
- At least one query produces non-null rank in `pre_rerank` and `post_rerank`.
- Root cause and final fix documented in `docs/retrieval-eval.md`.
