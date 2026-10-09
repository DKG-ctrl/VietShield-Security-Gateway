# Current execution status — 2026-10-09

This file distinguishes code/package verification from experiment results reported from the completed local/Colab runs.

## Engineering status

- Project architecture: complete for demo v0.1.
- Normalization, heuristics, risk engine, `scan()`, and batch scanning: implemented.
- Model A training/inference/save/load: implemented and trained.
- Model B multilingual fine-tuning: implemented and trained on Colab GPU.
- Leakage-safe freeze/external-evaluation workflow: implemented.
- `scripts.runtime` package import regression fixed (`from scripts.runtime import load_frozen_gateway`).
- One-command live demo added: `scripts/demo.py`.
- Unit tests: run from the packaged repository after the import fix; see the latest test command in the submission checklist.

## RMIT 2025 internal validation

Same stratified split for both models: `3200 train / 800 validation`, seed `42`.

### Model A — TF-IDF + Logistic Regression

- status: trained
- threshold: `0.48`
- accuracy: `0.89125`
- precision (jailbreak): `0.8319088319`
- recall (jailbreak): `0.9125`
- F1 (jailbreak): `0.8703427720`
- FPR: `0.1229166667`
- FNR: `0.0875`
- ROC-AUC: `0.95830078125`
- validation confusion matrix: `[[421, 59], [28, 292]]`
- training time: ~`2.75 s`
- average latency: ~`1.70 ms/document`

### Model B — multilingual DistilBERT

- model: `distilbert/distilbert-base-multilingual-cased`
- status: trained (`fine_tuned_transformer`)
- threshold: `0.21`
- accuracy: `0.89125`
- precision (jailbreak): `0.8652037618`
- recall (jailbreak): `0.8625`
- F1 (jailbreak): `0.8638497653`
- FPR: `0.0895833333`
- FNR: `0.1375`
- ROC-AUC: `0.9571907552`
- validation confusion matrix: `[[437, 43], [44, 276]]`
- training time: ~`111.49 s`
- average latency: ~`6.15 ms/document`
- early stopping occurred after validation loss worsened; the trainer loaded the best checkpoint at end.

Under the repository's current selection utility (`recall + 0.5*F1 - 0.35*FPR`), Model A scores approximately `1.3047` vs Model B `1.2631`; therefore the default freeze policy selects Model A.

## Preliminary multilingual sanity check

These are development probes only, not a final benchmark.

- English malicious — Model A `0.9984`; Model B `0.9896`.
- Vietnamese equivalent — Model A `0.3828` (gateway allowed in the observed run); Model B `0.8671` (jailbreak at threshold `0.21`).
- EN–VI code-switch — Model B `0.9703`.
- Vietnamese benign — Model B `0.1696` (benign at threshold `0.21`).

Interpretation: the lexical baseline showed a clear cross-lingual failure on one paired prompt, while the multilingual encoder recovered a high malicious score. This is motivation for systematic multilingual evaluation, not proof that Model B solves Vietnamese security.

## Known non-blocking issue

The Colab run emitted a tokenizer warning mentioning `fix_mistral_regex=True` even though the model is multilingual DistilBERT. Training and inference completed successfully. Do not change tokenizer/library behavior immediately before the demo; pin/fix dependencies and retrain in the post-demo research iteration if needed.

## Still intentionally not completed

- External BIPIA/Test60 final evaluation.
- Test60-driven threshold/rule tuning (must never be done).
- A+B ensemble/risk-weight tuning.
- English-proxy cross-lingual verification layer.
- Full RAG application integration and output grounding validator.

These are post-demo milestones. The current deliverable is **Architecture Validation / Proof of Concept**, not a production security gateway.

## Artifact handling

The Git repository intentionally ignores large/generated artifacts and datasets. Before a live demo, restore locally:

```text
models/model_a.joblib
models/model_b_transformer/
models/frozen_manifest.json   # only if demonstrating the frozen primary gateway
```

Keep training data and model checkpoints in Google Drive or another artifact store rather than committing them to GitHub.
