from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import traceback

from common import PROJECT_ROOT, load_config, load_training_frame, make_or_load_split, set_seed, write_json
from src.metrics import benchmark, binary_metrics, choose_threshold


def train_frozen_embeddings(training, validation, config: dict) -> dict:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from src.models.multilingual_encoder import FrozenEmbeddingDetector

    encoder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    encoder = SentenceTransformer(encoder_name)
    started = perf_counter()
    train_embeddings = encoder.encode(training["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=int(config["seed"]))
    classifier.fit(train_embeddings, training["target"])
    validation_embeddings = encoder.encode(validation["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    scores = classifier.predict_proba(validation_embeddings)[:, 1]
    selected = choose_threshold(validation["target"].to_numpy(), scores)
    detector = FrozenEmbeddingDetector(encoder_name, classifier, float(selected["threshold"]))
    artifact = PROJECT_ROOT / "models" / "model_b_frozen.joblib"
    detector.save(artifact)
    metrics = binary_metrics(validation["target"], (scores >= detector.threshold).astype(int), scores)
    latency = benchmark(detector.score_text, validation["text"].head(min(50, len(validation))))
    return {
        "status": "trained",
        "mode": "frozen_embeddings_fallback",
        "model": encoder_name,
        "artifact": str(artifact.relative_to(PROJECT_ROOT)),
        "artifact_type": "frozen_embeddings",
        "threshold": detector.threshold,
        "threshold_selection": selected["selection_rule"],
        "metrics": metrics,
        "training_seconds": perf_counter() - started,
        "latency": latency,
        "validation_scores": scores,
    }


def train_transformer(training, validation, config: dict) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from src.models.multilingual_encoder import MultilingualDetector

    model_config = config["model_b"]
    model_name = model_config["pretrained_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.encoded = tokenizer(
                list(texts), truncation=True, padding=True,
                max_length=int(model_config["max_length"]),
            )
            self.labels = list(map(int, labels))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: torch.tensor(value[index]) for key, value in self.encoded.items()}
            item["labels"] = torch.tensor(self.labels[index])
            return item

    train_dataset = TextDataset(training["text"], training["target"])
    validation_dataset = TextDataset(validation["text"], validation["target"])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    artifact = PROJECT_ROOT / "models" / "model_b_transformer"
    arguments = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "model_b_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(model_config["learning_rate"]),
        per_device_train_batch_size=int(model_config["batch_size"]),
        per_device_eval_batch_size=int(model_config["batch_size"]),
        gradient_accumulation_steps=int(model_config["gradient_accumulation_steps"]),
        num_train_epochs=float(model_config["epochs"]),
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        report_to=[],
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    started = perf_counter()
    trainer.train()
    trainer.save_model(str(artifact))
    tokenizer.save_pretrained(str(artifact))
    output = trainer.predict(validation_dataset)
    logits = output.predictions
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    scores = exp[:, 1] / exp.sum(axis=1)
    selected = choose_threshold(validation["target"].to_numpy(), scores)
    detector = MultilingualDetector(artifact, float(selected["threshold"]), model_name, int(model_config["max_length"]))
    metrics = binary_metrics(validation["target"], (scores >= detector.threshold).astype(int), scores)
    latency = benchmark(detector.score_text, validation["text"].head(min(50, len(validation))))
    return {
        "status": "trained",
        "mode": "fine_tuned_transformer",
        "model": model_name,
        "artifact": str(artifact.relative_to(PROJECT_ROOT)),
        "artifact_type": "transformer",
        "threshold": detector.threshold,
        "threshold_selection": selected["selection_rule"],
        "metrics": metrics,
        "training_seconds": perf_counter() - started,
        "latency": latency,
        "validation_scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--mode", choices=["auto", "finetune", "frozen-embeddings"], default="auto")
    args = parser.parse_args()
    config = load_config()
    set_seed(int(config["seed"]))
    frame = load_training_frame(args.train_csv)
    training, validation = make_or_load_split(
        frame, PROJECT_ROOT / "outputs" / "internal_split.csv",
        float(config["validation_size"]), int(config["seed"]),
    )
    report_path = PROJECT_ROOT / "outputs" / "model_b_validation_metrics.json"
    try:
        if args.mode == "finetune":
            report = train_transformer(training, validation, config)
        elif args.mode == "frozen-embeddings":
            report = train_frozen_embeddings(training, validation, config)
        else:
            try:
                import torch
                import transformers  # noqa: F401
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA GPU not available for practical fine-tuning")
                report = train_transformer(training, validation, config)
            except (ImportError, RuntimeError) as primary_error:
                try:
                    report = train_frozen_embeddings(training, validation, config)
                    report["fallback_reason"] = str(primary_error)
                except Exception as fallback_error:
                    raise RuntimeError(f"fine-tuning unavailable: {primary_error}; frozen fallback unavailable: {fallback_error}") from fallback_error

        scores = report.pop("validation_scores")
        output = validation[["Id", "text", "target"]].copy()
        output["score"] = scores
        output["prediction"] = (scores >= float(report["threshold"])).astype(int)
        output.to_csv(PROJECT_ROOT / "outputs" / "model_b_validation_predictions.csv", index=False, encoding="utf-8")
    except Exception as exc:
        report = {
            "status": "unavailable",
            "reason": str(exc),
            "honest_fallback": "Model A remains available; no Model B result is claimed.",
            "traceback_tail": traceback.format_exc().splitlines()[-6:],
        }
    write_json(report_path, report)
    print(report)


if __name__ == "__main__":
    main()
