## Retrieval Eval (Public/Admin Path)

This benchmark evaluates retrieval quality only for the public/admin knowledge path.
It does not score final answer generation and does not include student/SIAKAD flows.

### Dataset

- Benchmark dataset: `tests/fixtures/retrieval_eval.json`
- Fixture predictions: `tests/fixtures/retrieval_eval_predictions.json`
- Baseline snapshot: `tests/fixtures/retrieval_eval_baseline.json`

Each benchmark case includes:
- `name`
- `query`
- `expected_filename` or `expected_doc_id`
- `k_target`
- optional `expected_targets` for multi-target recall checks

### Metrics

The evaluator reports:
- `Hit@1`
- `Hit@3`
- `Hit@5`
- `MRR`
- `Recall@5`

Metrics are reported for both stages:
- `pre_rerank` (raw hybrid retrieval output)
- `post_rerank` (after reranker)

### Comparable Depth Rule

Both stages are evaluated with the same fixed depth: `k_eval=5`.
This keeps pre-rerank vs post-rerank comparisons meaningful.

### Run Commands

Fixture-mode (deterministic baseline):

```powershell
.\.venv\Scripts\python.exe -m app.eval.retrieval_eval `
  --mode fixture `
  --dataset tests/fixtures/retrieval_eval.json `
  --predictions tests/fixtures/retrieval_eval_predictions.json `
  --output tests/fixtures/retrieval_eval_baseline.json `
  --k-eval 5 `
  --evaluation-date 2026-04-08
```

Live-mode (real services + current corpus):

```powershell
.\.venv\Scripts\python.exe -m app.eval.retrieval_eval `
  --mode live `
  --dataset tests/fixtures/retrieval_eval.json `
  --output tests/fixtures/retrieval_eval_live.json `
  --k-eval 5
```

