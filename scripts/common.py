from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: Path | None = None) -> dict:
    # config.yaml is deliberately JSON-compatible YAML, so bootstrap and
    # inference do not require PyYAML.
    return json.loads((path or PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def load_training_frame(train_csv: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install pandas before loading training data") from exc
    frame = pd.read_csv(train_csv, encoding="utf-8")
    if list(frame.columns) != ["Id", "text", "label"]:
        raise ValueError(f"train.csv schema must be Id,text,label; found {list(frame.columns)}")
    if frame[["Id", "text", "label"]].isnull().any().any():
        raise ValueError("train.csv contains null Id/text/label")
    mapping = {"benign": 0, "jailbreak": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    mapped = frame["label"].map(mapping)
    if mapped.isnull().any():
        unknown = sorted(frame.loc[mapped.isnull(), "label"].astype(str).unique())
        raise ValueError(f"unsupported labels: {unknown}")
    frame = frame.copy()
    frame["target"] = mapped.astype(int)
    frame["text"] = frame["text"].astype(str)
    return frame


def make_or_load_split(frame, split_path: Path, validation_size: float = 0.2, seed: int = 42):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if split_path.exists():
        split = pd.read_csv(split_path, dtype={"Id": str})
        if set(split.columns) != {"Id", "partition"}:
            raise ValueError("existing split file has unexpected schema")
        lookup = dict(zip(split["Id"], split["partition"]))
        partitions = frame["Id"].astype(str).map(lookup)
        if partitions.isnull().any() or set(partitions.unique()) != {"train", "validation"}:
            raise ValueError("existing split does not match current train.csv")
        return frame.loc[partitions == "train"].copy(), frame.loc[partitions == "validation"].copy()

    train_idx, validation_idx = train_test_split(
        frame.index,
        test_size=validation_size,
        stratify=frame["target"],
        random_state=seed,
    )
    split = pd.DataFrame({
        "Id": frame["Id"].astype(str),
        "partition": "train",
    })
    split.loc[validation_idx, "partition"] = "validation"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_path, index=False, encoding="utf-8")
    return frame.loc[train_idx].copy(), frame.loc[validation_idx].copy()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_label_column(frame):
    for candidate in ("label", "TARGET", "target", "ground_truth", "y"):
        if candidate in frame.columns:
            series = frame[candidate]
            mapping = {"benign": 0, "jailbreak": 1, "safe": 0, "attack": 1, 0: 0, 1: 1, "0": 0, "1": 1}
            mapped = series.map(mapping)
            if mapped.isnull().any():
                raise ValueError(f"unsupported values in ground-truth column {candidate}")
            return candidate, mapped.astype(int)
    raise ValueError("ground truth needs one of: label, TARGET, target, ground_truth, y")
