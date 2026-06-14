from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from monarch.actor import Actor, endpoint

from rl.types import DatasetSample


logger = logging.getLogger(__name__)


@dataclass
class DatasetActorStatus:
    initialized: bool
    dataset_size: int
    epoch: int
    cursor: int
    pad_token_id: int | None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


class DatasetActor(Actor):
    """Own the RL prompt dataset and provide deterministic shuffled batches."""

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self._dataset: Any | None = None
        self._order: list[int] = []
        self._cursor = 0
        self._epoch = 0
        self._seed = 0
        self._rng = random.Random()
        self._prompt_key = "question"
        self._response_key = "answer"
        self._system_prompt: str | None = None
        self._pad_token_id: int | None = None

    @endpoint
    def setup(self) -> DatasetActorStatus:
        if self._dataset is not None:
            raise RuntimeError("DatasetActor.setup() may only be called once.")

        with open(self.config_path, encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        veomni = _mapping(config.get("veomni"), "veomni")
        data = _mapping(veomni.get("data"), "veomni.data")
        adapter = _mapping(veomni.get("data_adapter"), "veomni.data_adapter")
        train = _mapping(veomni.get("train"), "veomni.train")
        model = _mapping(veomni.get("model"), "veomni.model")

        train_path = data.get("train_path")
        if not train_path:
            raise ValueError("veomni.data.train_path is required for DatasetActor.")
        train_path = Path(str(train_path)).expanduser().resolve()
        if not train_path.is_file():
            raise FileNotFoundError(f"RL training parquet does not exist: {train_path}")

        from datasets import load_dataset
        from transformers import AutoTokenizer

        self._dataset = load_dataset(
            "parquet", data_files={"train": str(train_path)}, split="train"
        )
        if len(self._dataset) == 0:
            raise ValueError(f"RL training dataset is empty: {train_path}")

        tokenizer_path = model.get("tokenizer_path") or model.get("model_path")
        if not tokenizer_path:
            raise ValueError("veomni.model.tokenizer_path or model_path is required.")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self._pad_token_id = tokenizer.pad_token_id
        if self._pad_token_id is None:
            self._pad_token_id = tokenizer.eos_token_id
        if self._pad_token_id is None:
            raise ValueError("The configured tokenizer has no pad or EOS token id.")

        self._prompt_key = str(adapter.get("prompt_key", "question"))
        self._response_key = str(adapter.get("response_key", "answer"))
        system_prompt = adapter.get("system_prompt")
        self._system_prompt = str(system_prompt) if system_prompt else None
        self._seed = int(train.get("seed", 0))
        self._rng.seed(self._seed)
        self._reset_order()
        logger.info("Loaded %d RL prompts from %s.", len(self._dataset), train_path)
        return self._status()

    def _reset_order(self) -> None:
        if self._dataset is None:
            raise RuntimeError("DatasetActor.setup() must complete first.")
        self._order = list(range(len(self._dataset)))
        random.Random(self._seed + self._epoch).shuffle(self._order)
        self._cursor = 0

    @staticmethod
    def _target_text(value: Any) -> str:
        target = str(value).strip()
        if "####" in target:
            target = target.rsplit("####", 1)[-1].strip()
        return target

    def _sample_at(self, index: int) -> DatasetSample:
        row = self._dataset[index]
        missing = [
            key for key in (self._prompt_key, self._response_key) if key not in row
        ]
        if missing:
            raise KeyError(f"Dataset row is missing required columns: {missing}")

        prompt = str(row[self._prompt_key])
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        return DatasetSample(
            sample_id=f"epoch-{self._epoch}-row-{index}",
            prompt=prompt,
            target=self._target_text(row[self._response_key]),
            messages=messages,
        )

    @endpoint
    def next_batch(self, batch_size: int = 1) -> list[DatasetSample]:
        if self._dataset is None:
            raise RuntimeError("DatasetActor.setup() must complete first.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        samples = []
        for _ in range(batch_size):
            if self._cursor >= len(self._order):
                self._epoch += 1
                self._reset_order()
            index = self._order[self._cursor]
            self._cursor += 1
            samples.append(self._sample_at(index))
        return samples

    @endpoint
    def get_pad_token_id(self) -> int:
        if self._pad_token_id is None:
            raise RuntimeError("DatasetActor.setup() must complete first.")
        return self._pad_token_id

    def _status(self) -> DatasetActorStatus:
        return DatasetActorStatus(
            initialized=self._dataset is not None,
            dataset_size=len(self._dataset) if self._dataset is not None else 0,
            epoch=self._epoch,
            cursor=self._cursor,
            pad_token_id=self._pad_token_id,
        )

    @endpoint
    def get_status(self) -> DatasetActorStatus:
        return self._status()
