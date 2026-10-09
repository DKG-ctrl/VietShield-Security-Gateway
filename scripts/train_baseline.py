from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from common import PROJECT_ROOT, load_config, load_training_frame, make_or_load_split, set_seed, write_json
from src.metrics import benchmark, binary_metrics, choose_threshold
from src.models.baseline import new_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    args = parser.parse_args()
    config = load_config()
    seed = int(config["seed"])
    set_seed(seed)
    frame = load_training_frame(args.train_csv)
    training, validation = make_or_load_split(
        frame,
        PROJECT_ROOT / "outputs" / "internal_split.csv",
        float(config["validation_size"]),
        seed,
    )

    detector = new_baseline(seed)
    started = perf_counter()
    detector.fit(training["text"], training["target"])
    training_seconds = perf_counter() - started
    validation_scores = detector.predict_proba(validation["text"])
    selected = choose_threshold(validation["target"].to_numpy(), validation_scores)
    detector.threshold = float(selected["threshold"])
    predictions = (validation_scores >= detector.threshold).astype(int)
    metrics = binary_metrics(validation["target"], predictions, validation_scores)
    latency = benchmark(detector.score_text, validation["text"].head(min(100, len(validation))))
    artifact = PROJECT_ROOT / "models" / "model_a.joblib"
    detector.save(artifact)

    validation_output = validation[["Id", "text", "target"]].copy()
    validation_output["score"] = validation_scores
    validation_output["prediction"] = predictions
    validation_output.to_csv(PROJECT_ROOT / "outputs" / "model_a_validation_predictions.csv", index=False, encoding="utf-8")
    report = {
        "status": "trained",
        "model": detector.model_name,
        "train_rows": int(len(training)),
        "validation_rows": int(len(validation)),
        "threshold": detector.threshold,
        "threshold_selection": selected["selection_rule"],
        "metrics": metrics,
        "training_seconds": training_seconds,
        "latency": latency,
        "artifact": str(artifact.relative_to(PROJECT_ROOT)),
    }
    write_json(PROJECT_ROOT / "outputs" / "model_a_validation_metrics.json", report)
    print(report)


if __name__ == "__main__":
    main()
