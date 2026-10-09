from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from common import PROJECT_ROOT


def run(script: str, *args: object) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *map(str, args)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe end-to-end runner")
    parser.add_argument("--rmit-zip", required=True, type=Path)
    parser.add_argument("--external-inputs", required=True, type=Path)
    parser.add_argument("--external-ground-truth", required=True, type=Path)
    parser.add_argument("--model-b-mode", choices=["auto", "finetune", "frozen-embeddings"], default="auto")
    args = parser.parse_args()
    extract_to = PROJECT_ROOT / "data" / "rmit2025"
    run("inspect_data.py", "--zip", args.rmit_zip, "--extract-to", extract_to)
    candidates = list(extract_to.rglob("train.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one train.csv after extraction; found {len(candidates)}")
    train_csv = candidates[0]
    run("train_baseline.py", "--train-csv", train_csv)
    run("train_multilingual.py", "--train-csv", train_csv, "--mode", args.model_b_mode)
    run("freeze.py")
    run("validate.py")
    # No process above this line reads the ground-truth file contents.
    run("run_external_test.py", "--inputs", args.external_inputs)
    # Only now is the evaluation process allowed to open ground truth.
    run("evaluate_external_test.py", "--ground-truth", args.external_ground_truth, "--inputs", args.external_inputs)


if __name__ == "__main__":
    main()
