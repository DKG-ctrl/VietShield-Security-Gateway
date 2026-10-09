from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def build_pipeline(seed: int = 42):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("Model A requires scikit-learn; install requirements.txt") from exc

    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.995,
            sublinear_tf=True, max_features=80_000, strip_accents=None,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, max_features=120_000, strip_accents=None,
        )),
    ])
    classifier = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
        solver="liblinear",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


@dataclass
class BaselineDetector:
    pipeline: object
    threshold: float = 0.5
    model_name: str = "word_char_tfidf_logistic_regression"

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> "BaselineDetector":
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts: Iterable[str]):
        probabilities = self.pipeline.predict_proba(list(texts))
        return probabilities[:, 1]

    def score_text(self, text: str) -> float:
        return float(self.predict_proba([text])[0])

    def predict(self, texts: Iterable[str]):
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    def save(self, path: str | Path) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Saving Model A requires joblib") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineDetector":
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Loading Model A requires joblib") from exc
        payload = joblib.load(path)
        return cls(**payload)


def new_baseline(seed: int = 42) -> BaselineDetector:
    return BaselineDetector(build_pipeline(seed))
