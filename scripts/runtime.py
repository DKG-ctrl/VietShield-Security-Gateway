from __future__ import annotations

import json

try:
    from .common import PROJECT_ROOT
except ImportError:  # supports direct script-style execution contexts
    from common import PROJECT_ROOT
from src.gateway import SecurityGateway
from src.risk_engine import RiskConfig


def load_frozen_gateway() -> tuple[SecurityGateway, dict]:
    manifest_path = PROJECT_ROOT / "models" / "frozen_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("models/frozen_manifest.json does not exist; run freeze.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise RuntimeError("model manifest is not frozen")
    model_info = manifest["model"]
    artifact = PROJECT_ROOT / model_info["artifact"]
    artifact_type = model_info["artifact_type"]
    if artifact_type == "baseline_joblib":
        from src.models.baseline import BaselineDetector
        detector = BaselineDetector.load(artifact)
    elif artifact_type == "transformer":
        from src.models.multilingual_encoder import MultilingualDetector
        detector = MultilingualDetector(artifact, float(model_info["threshold"]), model_info["name"])
    elif artifact_type == "frozen_embeddings":
        from src.models.multilingual_encoder import FrozenEmbeddingDetector
        detector = FrozenEmbeddingDetector.load(artifact)
    else:
        raise ValueError(f"unknown artifact type: {artifact_type}")
    detector.threshold = float(model_info["threshold"])
    risk = RiskConfig(**manifest["risk"])
    return SecurityGateway(detector, risk), manifest
