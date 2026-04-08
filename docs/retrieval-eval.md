## Retrieval Eval (Public/Admin Path)

This benchmark evaluates retrieval quality only for the public/admin knowledge path.
It does not score final answer generation and does not include student/SIAKAD flows.

### Dataset

- Benchmark dataset: `tests/fixtures/retrieval_eval.json`
- Alternate live dataset for current 3-doc corpus: `tests/fixtures/retrieval_eval_live_current_corpus.json`
- Harder paraphrase live dataset for current 3-doc corpus: `tests/fixtures/retrieval_eval_live_current_corpus_hard.json`
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

Live-mode (current 3-doc corpus dataset):

```powershell
.\.venv\Scripts\python.exe -m app.eval.retrieval_eval `
  --mode live `
  --dataset tests/fixtures/retrieval_eval_live_current_corpus.json `
  --output tests/fixtures/retrieval_eval_live_current_corpus_report.json `
  --k-eval 5
```

Live baseline snapshot (current 3-doc corpus dataset):

```powershell
.\.venv\Scripts\python.exe -m app.eval.retrieval_eval `
  --mode live `
  --dataset tests/fixtures/retrieval_eval_live_current_corpus.json `
  --output tests/fixtures/retrieval_eval_live_current_corpus_baseline.json `
  --k-eval 5 `
  --evaluation-date 2026-04-09
```

Live baseline snapshot (hard paraphrase dataset):

```powershell
.\.venv\Scripts\python.exe -m app.eval.retrieval_eval `
  --mode live `
  --dataset tests/fixtures/retrieval_eval_live_current_corpus_hard.json `
  --output tests/fixtures/retrieval_eval_live_current_corpus_hard_baseline.json `
  --k-eval 5 `
  --evaluation-date 2026-04-09
```

Live-mode output also includes `debug_candidates` with compact top-`k_eval` metadata
per query and stage (`doc_id`, `filename`, `chunk_index`, score fields). This is
used to diagnose identity/matching issues without re-running retrieval traces.

### Target Matching Rules

Matching in eval is identity-aware and normalization-safe:
- filename match is normalized by basename and lowercase
- doc_id match accepts exact `doc_id` and filename-derived `doc_id`
- chunk index is compared after integer normalization

This allows stable scoring when live payload formatting differs (for example
`kb/FILE.PDF` vs `file.pdf`) while still enforcing deterministic identity checks.

### Re-ingestion Identity Rules

To keep benchmark comparisons stable:
- same document identity: normalized filename-derived `doc_id`
- same chunk identity: deterministic hash of `doc_id + chunk_index`
- re-ingestion behavior: replace prior chunks for the same `doc_id` before upsert

This prevents silent duplicate accumulation when the same source file is ingested again.
