from __future__ import annotations

import json

from common import PROJECT_ROOT, write_json


def main() -> None:
    a = json.loads((PROJECT_ROOT / "outputs" / "model_a_validation_metrics.json").read_text(encoding="utf-8"))
    b_path = PROJECT_ROOT / "outputs" / "model_b_validation_metrics.json"
    b = json.loads(b_path.read_text(encoding="utf-8")) if b_path.exists() else {"status": "not_attempted"}
    manifest_path = PROJECT_ROOT / "models" / "frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    report = {
        "model_a": a,
        "model_b": b,
        "selected_primary": manifest["primary"] if manifest else None,
        "selection_rationale": manifest["selection_rationale"] if manifest else "run freeze.py after validation",
    }
    write_json(PROJECT_ROOT / "outputs" / "validation_metrics.json", report)
    print(report)


if __name__ == "__main__":
    main()
