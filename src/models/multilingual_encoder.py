from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL = "distilbert/distilbert-base-multilingual-cased"


@dataclass
class MultilingualDetector:
    model_dir: str | Path
    threshold: float = 0.5
    model_name: str = DEFAULT_MODEL
    max_length: int = 256

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Model B requires torch and transformers") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()

    def predict_proba(self, texts: Iterable[str]):
        import numpy as np

        values = list(texts)
        scores: list[float] = []
        batch_size = 16 if self._device.type == "cuda" else 4
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
                probs = self._torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.detach().cpu().tolist())
        return np.asarray(scores)

    def score_text(self, text: str) -> float:
        return float(self.predict_proba([text])[0])


@dataclass
class FrozenEmbeddingDetector:
    encoder_name: str
    classifier: object
    threshold: float = 0.5
    model_name: str = "frozen_multilingual_embeddings_logistic_regression"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Frozen embedding fallback requires sentence-transformers") from exc
        self._encoder = SentenceTransformer(self.encoder_name)

    def predict_proba(self, texts: Iterable[str]):
        embeddings = self._encoder.encode(list(texts), normalize_embeddings=True)
        return self.classifier.predict_proba(embeddings)[:, 1]

    def score_text(self, text: str) -> float:
        return float(self.predict_proba([text])[0])

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "encoder_name": self.encoder_name,
            "classifier": self.classifier,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenEmbeddingDetector":
        import joblib

        return cls(**joblib.load(path))
