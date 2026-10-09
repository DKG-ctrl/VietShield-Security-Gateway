from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from common import PROJECT_ROOT, load_config, sha256_file, write_json


def utility(report: dict) -> float:
    metrics = report["metrics"]
    return metrics["recall_jailbreak"] + 0.5 * metrics["f1_jailbreak"] - 0.35 * metrics["false_positive_rate"]


def main() -> None:
    config = load_config()
    a_path = PROJECT_ROOT / "outputs" / "model_a_validation_metrics.json"
    if not a_path.exists():
        raise FileNotFoundError("Train Model A before freezing")
    model_a = json.loads(a_path.read_text(encoding="utf-8"))
    candidates = [("model_a", model_a)]
    b_path = PROJECT_ROOT / "outputs" / "model_b_validation_metrics.json"
    model_b = json.loads(b_path.read_text(encoding="utf-8")) if b_path.exists() else {"status": "not_attempted"}
    if model_b.get("status") == "trained":
        candidates.append(("model_b", model_b))
    selected_name, selected = max(candidates, key=lambda pair: utility(pair[1]))
    artifact = PROJECT_ROOT / selected["artifact"]
    if not artifact.exists():
        raise FileNotFoundError(f"selected artifact missing: {artifact}")
    if artifact.is_dir():
        artifact_hashes = {str(p.relative_to(artifact)): sha256_file(p) for p in sorted(artifact.rglob("*")) if p.is_file()}
    else:
        artifact_hashes = {artifact.name: sha256_file(artifact)}
    manifest = {
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary": selected_name,
        "selection_rationale": "highest internal-validation security utility = recall + 0.5*F1 - 0.35*FPR; external Test60 not consulted",
        "model": {
            "name": selected["model"],
            "artifact": selected["artifact"],
            "artifact_type": selected.get("artifact_type", "baseline_joblib"),
            "threshold": selected["threshold"],
            "artifact_hashes": artifact_hashes,
        },
        "validation": {"model_a": model_a, "model_b": model_b},
        "risk": config["risk"],
        "evaluation": config["evaluation"],
        "heuristic_rules_sha256": sha256_file(PROJECT_ROOT / "src" / "heuristic_detector.py"),
        "normalization_sha256": sha256_file(PROJECT_ROOT / "src" / "normalization.py"),
        "config_sha256": sha256_file(PROJECT_ROOT / "config.yaml"),
    }
    write_json(PROJECT_ROOT / "models" / "frozen_manifest.json", manifest)
    print("MODEL FROZEN")
    print("THRESHOLDS FROZEN")
    print(f"PRIMARY_ML_DETECTOR={selected_name}:{selected['model']}")


if __name__ == "__main__":
    main()
