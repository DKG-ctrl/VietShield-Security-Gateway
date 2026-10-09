from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    ml_weight: float = 0.62
    injection_weight: float = 0.30
    unicode_weight: float = 0.08
    medium_threshold: float = 0.45
    high_threshold: float = 0.72

    def validate(self) -> None:
        total = self.ml_weight + self.injection_weight + self.unicode_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("risk weights must sum to 1.0")
        if not 0 <= self.medium_threshold < self.high_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= medium < high <= 1")


def aggregate_risk(ml_score: float, injection_score: float, unicode_score: float, config: RiskConfig) -> dict:
    config.validate()
    values = (ml_score, injection_score, unicode_score)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("component scores must be in [0, 1]")
    risk = (
        config.ml_weight * ml_score
        + config.injection_weight * injection_score
        + config.unicode_weight * unicode_score
    )
    if risk >= config.high_threshold:
        level, decision = "HIGH RISK", "block"
    elif risk >= config.medium_threshold:
        level, decision = "MEDIUM RISK", "warn_sanitize"
    else:
        level, decision = "LOW RISK", "allow"
    return {"risk_score": round(risk, 6), "risk_level": level, "decision": decision}
