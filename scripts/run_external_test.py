from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from common import PROJECT_ROOT, sha256_file, write_json
from runtime import load_frozen_gateway


def percentile(values, q: float) -> float:
    import numpy as np
    return float(np.percentile(values, q))


def verify_frozen_artifacts(manifest: dict) -> None:
    artifact = PROJECT_ROOT / manifest["model"]["artifact"]
    hashes = manifest["model"]["artifact_hashes"]
    for relative, expected in hashes.items():
        path = artifact / relative if artifact.is_dir() else artifact
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen model artifact changed: {path}")
    fixed = {
        PROJECT_ROOT / "src" / "heuristic_detector.py": manifest["heuristic_rules_sha256"],
        PROJECT_ROOT / "src" / "normalization.py": manifest["normalization_sha256"],
        PROJECT_ROOT / "config.yaml": manifest["config_sha256"],
    }
    for path, expected in fixed.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen code/config changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict external inputs. This script has no ground-truth argument by design.")
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "security_test_predictions.csv")
    args = parser.parse_args()
    import pandas as pd

    gateway, manifest = load_frozen_gateway()
    verify_frozen_artifacts(manifest)
    frame = pd.read_csv(args.inputs, encoding="utf-8")
    if "text" not in frame.columns:
        raise ValueError("external input must contain a text column")
    if "Id" not in frame.columns:
        raise ValueError("external input must contain an Id column")
    rows, timings = [], []
    for _, source in frame.iterrows():
        started = perf_counter()
        result = gateway.scan(str(source["text"]), {"Id": source["Id"]})
        timings.append((perf_counter() - started) * 1000.0)
        rows.append({
            "Id": source["Id"],
            "risk_score": result["risk_score"],
            "ml_score": result["ml"]["jailbreak_probability"],
            "heuristic_score": result["prompt_injection"]["score"],
            "unicode_score": result["text_anomalies"]["score"],
            "decision": result["decision"],
            "reasons": json.dumps(result["reasons"], ensure_ascii=False),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8")
    receipt = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.inputs),
        "input_sha256": sha256_file(args.inputs),
        "prediction_path": str(args.output),
        "prediction_sha256": sha256_file(args.output),
        "rows": len(rows),
        "frozen_manifest_sha256": sha256_file(PROJECT_ROOT / "models" / "frozen_manifest.json"),
        "full_pipeline_latency": {
            "documents": len(timings),
            "average_ms_per_document": sum(timings) / len(timings) if timings else None,
            "p50_ms": percentile(timings, 50) if timings else None,
            "p95_ms": percentile(timings, 95) if timings else None,
        },
    }
    write_json(PROJECT_ROOT / "outputs" / "external_prediction_receipt.json", receipt)
    print("EXTERNAL TEST PREDICTION COMPLETE")
    print(receipt)


if __name__ == "__main__":
    main()
