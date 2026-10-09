from __future__ import annotations

from time import perf_counter
from typing import Callable, Iterable


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
    except ImportError as exc:
        raise RuntimeError("Metrics require scikit-learn") from exc

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_jailbreak": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_jailbreak": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_jailbreak": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support": {"benign": tn + fp, "jailbreak": fn + tp},
    }
    if y_score is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            result["roc_auc"] = None
    return result


def choose_threshold(y_true, scores) -> dict:
    """Validation-only selection: maximize F1, prefer recall, then lower FPR."""
    candidates = [round(value / 100, 2) for value in range(20, 81)]
    ranked = []
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        metrics = binary_metrics(y_true, predicted, scores)
        ranked.append((metrics["f1_jailbreak"], metrics["recall_jailbreak"], -metrics["false_positive_rate"], threshold, metrics))
    best = max(ranked, key=lambda item: item[:4])
    return {"threshold": best[3], "metrics": best[4], "selection_rule": "max_f1_then_recall_then_lower_fpr_on_internal_validation"}


def benchmark(callable_: Callable[[str], object], texts: Iterable[str], warmup: int = 3) -> dict:
    import numpy as np

    items = list(texts)
    if not items:
        raise ValueError("benchmark needs at least one text")
    for text in items[:warmup]:
        callable_(text)
    samples = []
    for text in items:
        start = perf_counter()
        callable_(text)
        samples.append((perf_counter() - start) * 1000.0)
    return {
        "documents": len(samples),
        "average_ms_per_document": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
    }
