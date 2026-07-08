from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from monarch.actor import Actor, endpoint

from actor.utils import load_yaml_config, mapping
from rl.types import DatasetSample


logger = logging.getLogger(__name__)


@dataclass
class DatasetActorStatus:
    initialized: bool
    dataset_name: str
    dataset_size: int
    epoch: int
    cursor: int
    pad_token_id: int | None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as data_file:
            for line_number, line in enumerate(data_file, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}.")
                rows.append(row)
        return rows

    from datasets import load_dataset

    suffix = path.suffix.lower().removeprefix(".")
    dataset_type = "json" if suffix == "json" else suffix
    if dataset_type not in {"json", "parquet", "csv", "arrow"}:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")
    dataset = load_dataset(dataset_type, data_files={"data": str(path)}, split="data")
    return [dict(row) for row in dataset]


def _rollout_messages(messages: Any, path: Path) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError(f"Dataset row messages must be a list: {path}")
    normalized = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError(f"Dataset row messages must contain mappings: {path}")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "assistant":
            continue
        if role not in {"system", "user"}:
            raise ValueError(f"Unsupported message role {role!r}: {path}")
        normalized.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in normalized):
        raise ValueError(f"Dataset row has no user message: {path}")
    return normalized


def _teacher_messages(messages: Any, path: Path) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError(f"Dataset row messages must be a list: {path}")
    normalized = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError(f"Dataset row messages must contain mappings: {path}")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role {role!r}: {path}")
        item = {"role": role, "content": content}
        if "loss_mask" in message:
            item["loss_mask"] = int(message["loss_mask"])
        normalized.append(item)
    if not any(message["role"] == "assistant" for message in normalized):
        raise ValueError(f"Dataset row has no assistant message: {path}")
    return normalized


def _prompt_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    raise ValueError("DatasetSample messages must contain a user message.")


def _tokenized_length(tokenized: Any) -> int:
    if isinstance(tokenized, Mapping):
        if "input_ids" not in tokenized:
            raise ValueError("Tokenized chat template output has no input_ids.")
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if (
        isinstance(tokenized, list)
        and tokenized
        and isinstance(tokenized[0], list)
    ):
        tokenized = tokenized[0]
    return len(tokenized)


class DatasetActor(Actor):
    """Load preprocessed RL data and provide prompt/target batches."""

    def __init__(self, config_path: str, dataset_name: str = "train"):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.dataset_name = str(dataset_name)
        self._rows: list[dict[str, Any]] = []
        self._order: list[int] = []
        self._cursor = 0
        self._epoch = 0
        self._seed = 0
        self._shuffle = False
        self._drop_last = False
        self._path: Path | None = None
        self._pad_token_id: int | None = None
        self._tokenizer: Any | None = None

    @endpoint
    def setup(self) -> DatasetActorStatus:
        if self._rows:
            raise RuntimeError("DatasetActor.setup() may only be called once.")

        config = load_yaml_config(self.config_path)

        dataloader = mapping(config.get("dataloader"), "dataloader")
        dataset_config = mapping(
            dataloader.get(self.dataset_name),
            f"dataloader.{self.dataset_name}",
        )
        path_value = dataset_config.get("path")
        if not path_value:
            raise ValueError(f"dataloader.{self.dataset_name}.path is required.")
        path = Path(str(path_value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")

        rows = _load_rows(path)
        max_samples = int(dataset_config.get("max_samples", -1))
        if max_samples >= 0:
            rows = rows[:max_samples]
        if not rows:
            raise ValueError(f"Dataset is empty: {path}")

        self._rows = [dict(row) for row in rows]
        self._seed = int(dataset_config.get("seed", 0))
        self._shuffle = bool(dataset_config.get("shuffle", self.dataset_name == "train"))
        self._drop_last = bool(dataset_config.get("drop_last", False))
        self._path = path
        self._tokenizer = self._load_tokenizer(config)
        self._pad_token_id = self._resolve_pad_token_id(self._tokenizer)
        self._reset_order()
        logger.info("Loaded %d %s rows from %s.", len(self._rows), self.dataset_name, path)
        return self._status()

    @staticmethod
    def _load_tokenizer(config: Mapping[str, Any]) -> Any:
        train_actor = mapping(config.get("train_actor"), "train_actor")
        model = mapping(train_actor.get("model"), "train_actor.model")
        tokenizer_path = model.get("tokenizer_path") or model.get("model_path")
        if not tokenizer_path:
            raise ValueError(
                "train_actor.model.tokenizer_path or model_path is required."
            )

        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(tokenizer_path))

    @staticmethod
    def _resolve_pad_token_id(tokenizer: Any) -> int:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The configured tokenizer has no pad or EOS token id.")
        return int(pad_token_id)

    def _reset_order(self) -> None:
        self._order = list(range(len(self._rows)))
        if self._shuffle:
            random.Random(self._seed + self._epoch).shuffle(self._order)
        self._cursor = 0

    def _sample_at(self, index: int, sample_id: str | None = None) -> DatasetSample:
        if self._path is None:
            raise RuntimeError("DatasetActor.setup() must complete first.")
        row = self._rows[index]
        if "messages" not in row or "label" not in row:
            raise KeyError("Preprocessed dataset rows must contain messages and label.")
        messages = _rollout_messages(row["messages"], self._path)
        return DatasetSample(
            sample_id=sample_id or f"{self.dataset_name}-epoch-{self._epoch}-row-{index}",
            prompt=_prompt_text(messages),
            target=str(row["label"]).strip(),
            messages=messages,
            teacher_messages=_teacher_messages(row["messages"], self._path),
        )

    @endpoint
    def next_batch(self, batch_size: int = 1) -> list[DatasetSample]:
        if not self._rows:
            raise RuntimeError("DatasetActor.setup() must complete first.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self._drop_last and len(self._order) - self._cursor < batch_size:
            self._epoch += 1
            self._reset_order()

        samples = []
        while len(samples) < batch_size:
            if self._cursor >= len(self._order):
                self._epoch += 1
                self._reset_order()
            index = self._order[self._cursor]
            self._cursor += 1
            samples.append(self._sample_at(index))
        return samples

    @endpoint
    def all_batches(self, batch_size: int = 1) -> list[list[DatasetSample]]:
        if not self._rows:
            raise RuntimeError("DatasetActor.setup() must complete first.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        batches = []
        for start in range(0, len(self._rows), batch_size):
            batch = []
            for index in range(start, min(start + batch_size, len(self._rows))):
                batch.append(
                    self._sample_at(index, f"{self.dataset_name}-row-{index}")
                )
            batches.append(batch)
        return batches

    @endpoint
    def get_pad_token_id(self) -> int:
        if self._pad_token_id is None:
            raise RuntimeError("DatasetActor.setup() must complete first.")
        return self._pad_token_id

    @endpoint
    def count_tokens(
        self, messages_batch: Sequence[Sequence[Mapping[str, str]]]
    ) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("DatasetActor.setup() must complete first.")

        counts = []
        for messages in messages_batch:
            token_ids = self._tokenizer.apply_chat_template(
                [dict(message) for message in messages],
                tokenize=True,
                add_generation_prompt=True,
            )
            counts.append(_tokenized_length(token_ids))
        return counts

    def _status(self) -> DatasetActorStatus:
        return DatasetActorStatus(
            initialized=bool(self._rows),
            dataset_name=self.dataset_name,
            dataset_size=len(self._rows),
            epoch=self._epoch,
            cursor=self._cursor,
            pad_token_id=self._pad_token_id,
        )

    @endpoint
    def get_status(self) -> DatasetActorStatus:
        return self._status()
