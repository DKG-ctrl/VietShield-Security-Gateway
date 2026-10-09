from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Protocol

from .heuristic_detector import detect_prompt_injection
from .normalization import normalize_text, remove_unsafe_formatting
from .risk_engine import RiskConfig, aggregate_risk


class MLDetector(Protocol):
    model_name: str

    def score_text(self, text: str) -> float: ...


class SecurityGateway:
    def __init__(self, detector: MLDetector, risk_config: RiskConfig | None = None):
        self.detector = detector
        self.risk_config = risk_config or RiskConfig()

    def scan(self, text: str, source_metadata: dict | None = None) -> dict:
        normalized = normalize_text(text)
        ml_score = float(self.detector.score_text(normalized.normalized_text))
        injection = detect_prompt_injection(normalized.comparison_text)
        risk = aggregate_risk(ml_score, injection["score"], normalized.anomaly_score, self.risk_config)
        reasons = list(injection["reasons"])
        reasons.extend(signal.replace("_", " ") for signal in normalized.signals)
        if ml_score >= 0.5:
            reasons.insert(0, "possible jailbreak intent")
        if not reasons:
            reasons.append("no material security signal detected")

        sanitized = None
        if risk["decision"] == "warn_sanitize":
            clean = remove_unsafe_formatting(text)
            sanitized = "[UNTRUSTED RETRIEVED CONTEXT]\n" + clean + "\n[END UNTRUSTED CONTEXT]"

        return {
            "decision": risk["decision"],
            "risk_level": risk["risk_level"],
            "risk_score": risk["risk_score"],
            "ml": {
                "model": self.detector.model_name,
                "jailbreak_probability": round(ml_score, 6),
            },
            "prompt_injection": {
                "score": injection["score"],
                "matched_rules": injection["matched_rules"],
            },
            "text_anomalies": {
                "score": normalized.anomaly_score,
                "signals": normalized.signals,
            },
            "reasons": reasons,
            "raw_text": normalized.raw_text,
            "normalized_text": normalized.normalized_text,
            "sanitized_text": sanitized,
            "source_metadata": source_metadata,
        }

    def scan_batch(self, input_csv: str | Path, output_csv: str | Path, text_column: str = "text", id_column: str = "Id") -> None:
        input_csv, output_csv = Path(input_csv), Path(output_csv)
        with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or text_column not in rows[0]:
            raise ValueError(f"input CSV must contain {text_column!r}")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [id_column, "risk_score", "ml_score", "heuristic_score", "unicode_score", "decision", "reasons"]
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(rows):
                result = self.scan(row[text_column], {"row_index": index})
                writer.writerow({
                    id_column: row.get(id_column, index),
                    "risk_score": result["risk_score"],
                    "ml_score": result["ml"]["jailbreak_probability"],
                    "heuristic_score": result["prompt_injection"]["score"],
                    "unicode_score": result["text_anomalies"]["score"],
                    "decision": result["decision"],
                    "reasons": json.dumps(result["reasons"], ensure_ascii=False),
                })


_DEFAULT_GATEWAY: SecurityGateway | None = None


def configure_default_gateway(gateway: SecurityGateway) -> None:
    global _DEFAULT_GATEWAY
    _DEFAULT_GATEWAY = gateway


def scan(text: str, source_metadata: dict | None = None) -> dict:
    if _DEFAULT_GATEWAY is None:
        raise RuntimeError("No trained model loaded. Call configure_default_gateway(SecurityGateway(detector)) first.")
    return _DEFAULT_GATEWAY.scan(text, source_metadata)


def scan_batch(input_csv: str | Path, output_csv: str | Path, text_column: str = "text", id_column: str = "Id") -> None:
    if _DEFAULT_GATEWAY is None:
        raise RuntimeError("No trained model loaded. Configure the default gateway first.")
    _DEFAULT_GATEWAY.scan_batch(input_csv, output_csv, text_column, id_column)
