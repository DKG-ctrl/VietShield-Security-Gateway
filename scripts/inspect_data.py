from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, load_training_frame, safe_extract_zip, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and inspect RMIT 2025 data without reading external ground truth.")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--extract-to", type=Path, default=PROJECT_ROOT / "data" / "rmit2025")
    args = parser.parse_args()
    safe_extract_zip(args.zip, args.extract_to)
    required = {"train.csv", "test.csv", "sample_submission.csv"}
    found = {path.name: path for path in args.extract_to.rglob("*.csv")}
    missing = required - set(found)
    if missing:
        raise FileNotFoundError(f"ZIP is missing: {sorted(missing)}")

    frame = load_training_frame(found["train.csv"])
    import pandas as pd
    test = pd.read_csv(found["test.csv"], encoding="utf-8")
    submission = pd.read_csv(found["sample_submission.csv"], encoding="utf-8")
    if list(test.columns) != ["Id", "text"]:
        raise ValueError(f"test.csv schema must be Id,text; found {list(test.columns)}")
    if list(submission.columns) != ["Id", "TARGET"]:
        raise ValueError(f"sample_submission.csv schema must be Id,TARGET; found {list(submission.columns)}")
    summary = {
        "zip_sha256": sha256_file(args.zip),
        "train_rows": int(len(frame)),
        "test_rows_unlabeled": int(len(test)),
        "label_distribution": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        "target_distribution": {str(k): int(v) for k, v in frame["target"].value_counts().items()},
        "duplicate_texts": int(frame["text"].duplicated().sum()),
        "schemas": {
            "train.csv": list(frame[["Id", "text", "label"]].columns),
            "test.csv": list(test.columns),
            "sample_submission.csv": list(submission.columns),
        },
    }
    write_json(PROJECT_ROOT / "outputs" / "dataset_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
