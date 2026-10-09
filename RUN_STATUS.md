# Current execution status — 2026-10-09

This file distinguishes verified engineering checks from dataset-dependent experiment results.

## Verified in this environment

- Python compilation: passed for all `src/`, `scripts/`, and `tests/` modules.
- Unit tests: **10/10 passed**.
- Model A integration smoke test: fit, probability inference, artifact save, and artifact reload passed with scikit-learn 1.9.1 and joblib 1.6.0 installed in a temporary workspace-only dependency directory.
- Leakage-safe end-to-end smoke pipeline: passed in a separate `work/` copy using clearly synthetic fixtures. The observed log order was:
  1. `MODEL FROZEN`
  2. `THRESHOLDS FROZEN`
  3. `EXTERNAL TEST PREDICTION COMPLETE`
  4. `GROUND TRUTH UNLOCKED`
- GPU detected: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB VRAM; the fine-tuning configuration therefore uses batch size 4, FP16 when CUDA is available, and two-step gradient accumulation.

Synthetic smoke metrics are intentionally excluded from this deliverable because they are not evidence about RMIT or BIPIA performance.

## Blocked dataset-dependent steps

The following requested paths do not exist on this Windows host, were not present in the current attachment bundle, and were not found elsewhere under the Codex workspace:

```text
/workspace/scratch/4cf8c1c2b019/upload/rmit-hackathon-2025.zip
/workspace/scratch/4cf8c1c2b019/VietShield_BIPIA_Test60/security_test_inputs.csv
/workspace/scratch/4cf8c1c2b019/VietShield_BIPIA_Test60/security_test_ground_truth.csv
```

Consequently, no real dataset summary, Model A validation metrics, Model B validation metrics, selected primary model, frozen production artifact, external Test60 metrics, ablation result, error examples, or real latency benchmark is claimed.

The default Python environment also lacked PyTorch, Transformers, and Sentence Transformers. Model B code is complete, but it was not fine-tuned. Installing those packages before the actual run is required; the code records an honest `unavailable` status if both fine-tuning and frozen-embedding fallback are unavailable.

## Exact next action

Attach or copy the three data artifacts to accessible Windows paths, install `requirements.txt`, then run:

```powershell
python scripts/run_pipeline.py `
  --rmit-zip "C:\path\to\rmit-hackathon-2025.zip" `
  --external-inputs "C:\path\to\security_test_inputs.csv" `
  --external-ground-truth "C:\path\to\security_test_ground_truth.csv" `
  --model-b-mode auto
```

The pipeline will create real reports only after observing the required freeze and ground-truth unlock protocol.
