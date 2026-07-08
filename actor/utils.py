from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import yaml


def mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).expanduser().open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return dict(config)


def count_dataset_rows(data_config: Mapping[str, Any], name: str) -> int:
    path = Path(str(data_config["path"])).expanduser()
    max_samples = int(data_config.get("max_samples", -1))
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as data_file:
            row_count = sum(1 for line in data_file if line.strip())
    else:
        from datasets import load_dataset

        suffix = path.suffix.lower().removeprefix(".")
        dataset_type = "json" if suffix == "json" else suffix
        dataset = load_dataset(dataset_type, data_files={"data": str(path)}, split="data")
        row_count = len(dataset)

    if max_samples >= 0:
        row_count = min(row_count, max_samples)
    if row_count <= 0:
        raise ValueError(f"{name} is empty after max_samples filtering.")
    return row_count


def sft_steps_from_epochs(
    *,
    train_config: Mapping[str, Any],
    train_data: Mapping[str, Any],
    global_batch_size: int,
) -> tuple[int, int, int]:
    num_epochs = positive_int(
        train_config.get("num_train_epochs", 1),
        "train_actor.train.num_train_epochs",
    )
    train_rows = count_dataset_rows(train_data, "dataloader.train")
    drop_last = bool(train_data.get("drop_last", True))
    steps_per_epoch = (
        train_rows // global_batch_size
        if drop_last
        else math.ceil(train_rows / global_batch_size)
    )
    if steps_per_epoch <= 0:
        raise ValueError(
            "SFT dataset is smaller than train_actor.train.global_batch_size "
            "while dataloader.train.drop_last=true."
        )
    return num_epochs * steps_per_epoch, num_epochs, steps_per_epoch


class NoOpCallback:
    def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_train_end(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_epoch_begin(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_step_begin(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_step_end(self, *args: Any, **kwargs: Any) -> None:
        return None
