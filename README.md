# VietShield Security Gateway

> A proof-of-concept security gateway trained using the RMIT Hackathon 2025 dataset to validate the proposed security architecture.

This repository is an architecture-validation demo for a RAG system. It scans an **untrusted retrieved document/chunk** before that text is allowed into an LLM context. It is not a production security gateway and does not claim complete multilingual or Vietnamese coverage.

## Purpose and boundary

The module owns only the Security gateway and its integration contract:

```python
scan(text: str, source_metadata: dict | None = None) -> dict
```

It does not own retrieval, the application LLM, or the rest of the team's RAG stack. The supplied RMIT Hackathon 2025 data is used only to establish a binary `benign`/`jailbreak` baseline and to validate the pipeline.

## Architecture

```text
Retrieved Document / Chunk (untrusted)
        ↓
NFC Unicode and Text Normalization
        ↓
Primary ML Detector
        ↓
Indirect Prompt-Injection Heuristics
        ↓
Unicode/Text Anomaly Signal
        ↓
Frozen Risk Aggregation
        ↓
allow | warn_sanitize | block
        ↓
Only safe/wrapped context may be passed onward
```

The model boundary is pluggable: a future Vietnamese/multilingual security classifier can replace the RMIT classifier without changing normalization, policy, batch scanning, or the API.

## Models

- **Model A:** word `(1,2)` TF-IDF + character-within-word `(3,5)` TF-IDF + class-balanced Logistic Regression. Character features improve tolerance to simple spacing, typo, fragmentation, and no-diacritic variation while retaining an explainable linear baseline.
- **Model B:** fine-tuned `distilbert-base-multilingual-cased` binary classifier when CUDA and the Transformer stack are available. `auto` mode uses frozen multilingual MiniLM embeddings plus Logistic Regression when practical fine-tuning is unavailable. If neither dependency path is available, the run records Model B as `unavailable`; it never fabricates a result.

Model selection uses internal-validation security utility:

```text
recall(jailbreak) + 0.5 × F1(jailbreak) − 0.35 × false-positive-rate
```

This deliberately does not choose by accuracy alone. Model A can remain primary when Model B is unavailable or overfits.

## Normalization choice

The user-visible `normalized_text` uses **NFC**, preserving Vietnamese diacritics and canonical text meaning. A separate comparison-only form uses NFKC + case folding to improve matching of compatibility variants. Raw input is always preserved. Zero-width characters, controls, unusual whitespace, dense combining marks, and mixed Latin/Cyrillic/Greek signals are reported; Vietnamese diacritics are never globally stripped.

## Heuristic design

Rules are grouped by behavior, not single keywords:

- instruction override;
- role manipulation;
- hidden/system prompt extraction;
- context-priority takeover;
- RAG/AI-reader targeting;
- response control;
- concealment;
- secret/internal-context exfiltration.

A generic term such as “API key” or “ignore” cannot block by itself. Independent rule families increase confidence. For `warn_sanitize`, sanitization is conservative: unsafe format characters are removed and the content is wrapped as untrusted context; the document is not heuristically rewritten.

## Dataset constraints and leakage prevention

Only `train.csv` from the RMIT Hackathon 2025 ZIP is used for fitting. `test.csv` remains unlabeled and is never assigned inferred ground truth.

The split is created once with:

```python
train_test_split(..., stratify=y, random_state=42)
```

and stored in `outputs/internal_split.csv`, so both models use the same internal holdout. Vectorizers/tokenizers fit only on the training partition.

The external protocol is enforced structurally:

1. train/evaluate both model levels internally;
2. select the primary model;
3. freeze model, tokenizer/vectorizer, rules, config, weights, and thresholds with SHA-256 hashes;
4. print `MODEL FROZEN` and `THRESHOLDS FROZEN`;
5. run `run_external_test.py`, which intentionally has **no ground-truth argument**;
6. save predictions and a hash receipt, then print `EXTERNAL TEST PREDICTION COMPLETE`;
7. only `evaluate_external_test.py` can open ground truth, and only after verifying the prediction/input receipt; it then prints `GROUND TRUTH UNLOCKED`.

The BIPIA Test60 data is never used to tune hyperparameters, tokenization, rules, thresholds, or risk weights.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

## End-to-end run

```bash
python scripts/run_pipeline.py \
  --rmit-zip /path/to/rmit-hackathon-2025.zip \
  --external-inputs /path/to/security_test_inputs.csv \
  --external-ground-truth /path/to/security_test_ground_truth.csv \
  --model-b-mode auto
```

Explicit stages:

```bash
python scripts/inspect_data.py --zip /path/to/rmit-hackathon-2025.zip
python scripts/train_baseline.py --train-csv data/rmit2025/train.csv
python scripts/train_multilingual.py --train-csv data/rmit2025/train.csv --mode auto
python scripts/freeze.py
python scripts/validate.py
python scripts/run_external_test.py --inputs /path/to/security_test_inputs.csv
python scripts/evaluate_external_test.py --inputs /path/to/security_test_inputs.csv --ground-truth /path/to/security_test_ground_truth.csv
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## API usage

```python
from scripts.runtime import load_frozen_gateway

gateway, manifest = load_frozen_gateway()
result = gateway.scan(
    "Retrieved document text",
    {"document_id": "doc-17", "retriever": "vector-store"},
)
```

Important fields include `risk_score`, `decision`, nested ML/heuristic/anomaly results, reasons, both raw and normalized text, optional sanitized text, and source metadata.

## Outputs

- `dataset_summary.json`: schemas, sizes, label distribution, duplicate count, ZIP hash;
- `model_a_validation_metrics.json` and `model_b_validation_metrics.json`;
- `validation_metrics.json`: side-by-side comparison and selected model;
- `security_test_predictions.csv`;
- `external_prediction_receipt.json`: hashes and full-pipeline latency;
- `external_test_metrics.json`: accuracy, precision, recall, F1, ROC-AUC, FPR, FNR, and confusion matrix;
- `ablation_metrics.json`: ML only, heuristic only, ML + heuristic, and full pipeline;
- `error_analysis.csv`: false positives and false negatives with component scores.

Category breakdown is produced only when the provided ground truth contains an actual category field. No attack categories are invented.

## Reproducibility

- seed: `42` for Python, NumPy, scikit-learn, PyTorch, and Hugging Face training where supported;
- exact split persisted by `Id`;
- config, rules, normalization, and model artifacts hashed at freeze time;
- dependency ranges recorded in `requirements.txt`;
- training configuration and thresholds written into validation reports and the frozen manifest.

## Mandatory limitations

1. The RMIT Hackathon 2025 dataset is not the dataset for the current hackathon.
2. Its supervised labels are only `benign` and `jailbreak`. The model is not directly supervised across the full space of indirect prompt injection, RAG poisoning, multilingual jailbreak, Vietnamese obfuscation, code-switch attacks, or Unicode attacks.
3. Heuristics add useful security signals but do not replace a real multilingual prompt-injection model.
4. External BIPIA Test60 is used only to assess generalization after the system is frozen.
5. This is **Architecture Validation / Proof of Concept**, not a production security gateway.
6. Pattern-based sanitization cannot make hostile content intrinsically trustworthy. Downstream prompts must continue treating retrieved text as data rather than instructions.

## Recommended next step for the actual hackathon

Keep the integration contract stable, then replace the current binary detector with a Vietnamese/multilingual security classifier trained on competition-approved data. Build an evaluation suite covering English–Vietnamese code switching, no-diacritic Vietnamese, Unicode/homoglyph attacks, document-level RAG injections, and clean Vietnamese domain documents. Track attack recall/FNR alongside false-positive rate, latency, clean-task utility, and cross-lingual consistency; add output safety/grounding validation only as a separate downstream layer.
