from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import PROJECT_ROOT, read_label_column, sha256_file, write_json
from src.metrics import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Unlock ground truth only after prediction receipt validation.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "outputs" / "security_test_predictions.csv")
    args = parser.parse_args()
    import numpy as np
    import pandas as pd

    receipt_path = PROJECT_ROOT / "outputs" / "external_prediction_receipt.json"
    if not args.predictions.exists() or not receipt_path.exists():
        raise RuntimeError("predictions and receipt must exist before ground truth can be unlocked")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if sha256_file(args.predictions) != receipt["prediction_sha256"]:
        raise RuntimeError("prediction file changed after completion")
    if sha256_file(args.inputs) != receipt["input_sha256"]:
        raise RuntimeError("external input does not match prediction receipt")

    predictions = pd.read_csv(args.predictions, encoding="utf-8")
    inputs = pd.read_csv(args.inputs, encoding="utf-8")
    # First and only ground-truth read in the protocol.
    truth = pd.read_csv(args.ground_truth, encoding="utf-8")
    print("GROUND TRUTH UNLOCKED")
    label_column, labels = read_label_column(truth)
    truth = truth.assign(_target=labels)
    if "Id" not in truth.columns:
        raise ValueError("ground truth must contain Id")
    merged = predictions.merge(truth[["Id", "_target"]], on="Id", how="inner", validate="one_to_one")
    if len(merged) != len(predictions) or len(merged) != len(truth):
        raise ValueError("Id mismatch between predictions and ground truth")
    predicted = (merged["decision"] != "allow").astype(int)
    metrics = binary_metrics(merged["_target"], predicted, merged["risk_score"])

    manifest = json.loads((PROJECT_ROOT / "models" / "frozen_manifest.json").read_text(encoding="utf-8"))
    risk = manifest["risk"]
    ml_threshold = float(manifest["model"]["threshold"])
    medium = float(risk["medium_threshold"])
    ml_only = (merged["ml_score"] >= ml_threshold).astype(int)
    heuristic_only = (merged["heuristic_score"] >= float(manifest["evaluation"]["heuristic_positive_threshold"])).astype(int)
    combined_no_unicode_score = (
        (risk["ml_weight"] * merged["ml_score"] + risk["injection_weight"] * merged["heuristic_score"])
        / (risk["ml_weight"] + risk["injection_weight"])
    )
    combined_no_unicode = (combined_no_unicode_score >= medium).astype(int)
    ablation = {
        "A_ml_only": binary_metrics(merged["_target"], ml_only, merged["ml_score"]),
        "B_heuristic_only": binary_metrics(merged["_target"], heuristic_only, merged["heuristic_score"]),
        "C_ml_plus_heuristic": binary_metrics(merged["_target"], combined_no_unicode, combined_no_unicode_score),
        "D_full_pipeline": metrics,
        "note": "All thresholds and weights were frozen before external ground truth was read.",
    }
    enriched = merged.merge(inputs[["Id", "text"]], on="Id", how="left", validate="one_to_one")
    enriched["prediction"] = predicted
    errors = enriched.loc[enriched["prediction"] != enriched["_target"], [
        "Id", "text", "_target", "prediction", "decision", "risk_score", "ml_score", "heuristic_score", "unicode_score", "reasons"
    ]].copy()
    errors["error_type"] = np.where(errors["_target"] == 1, "false_negative", "false_positive")
    errors.to_csv(PROJECT_ROOT / "outputs" / "error_analysis.csv", index=False, encoding="utf-8")

    report = {
        "label_column": label_column,
        "rows": len(merged),
        "binary_policy": "warn_sanitize or block = predicted positive",
        "metrics": metrics,
        "ablation": ablation,
        "false_positive_examples": errors.loc[errors["error_type"] == "false_positive"].head(5).to_dict("records"),
        "false_negative_examples": errors.loc[errors["error_type"] == "false_negative"].head(5).to_dict("records"),
    }
    for category_column in ("category", "attack_category", "type"):
        if category_column in truth.columns:
            categorized = predictions.merge(truth[["Id", "_target", category_column]], on="Id", validate="one_to_one")
            report["category_breakdown"] = {}
            for category, group in categorized.groupby(category_column):
                group_predicted = (group["decision"] != "allow").astype(int)
                report["category_breakdown"][str(category)] = binary_metrics(group["_target"], group_predicted, group["risk_score"])
            break
    write_json(PROJECT_ROOT / "outputs" / "external_test_metrics.json", report)
    write_json(PROJECT_ROOT / "outputs" / "ablation_metrics.json", ablation)
    print(report)


if __name__ == "__main__":
    main()
