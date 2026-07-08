from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from monarch.actor import Actor, endpoint

from actor.utils import load_yaml_config
from rl.types import RLEpisode


logger = logging.getLogger(__name__)


@dataclass
class ReplayBufferStatus:
    size: int
    capacity: int
    batch_size_per_rank: int
    data_parallel_size: int
    oldest_policy_version: int | None
    newest_policy_version: int | None


def _load_settings(config_path: str) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    monarch_config = config.get("monarch", {})
    rl_config = config.get("rl", {})
    train_actor = config.get("train_actor", {})
    if not all(isinstance(item, dict) for item in (monarch_config, rl_config, train_actor)):
        raise ValueError("monarch, rl, and train_actor must be mappings.")
    replay = rl_config.get("replay_buffer", {})
    if not isinstance(replay, dict):
        raise ValueError("rl.replay_buffer must be a mapping.")
    sequence = monarch_config.get("sequence", {})
    if not isinstance(sequence, dict):
        raise ValueError("monarch.sequence must be a mapping.")
    train_config = train_actor.get("train", {})
    if not isinstance(train_config, dict):
        raise ValueError("train_actor.train must be a mapping.")
    train_num_gpus = int(train_actor.get("num_gpus", 0))
    if train_num_gpus <= 0:
        raise ValueError("train_actor.num_gpus must be positive.")
    global_batch_size = int(train_config.get("global_batch_size", 0))
    if global_batch_size <= 0:
        raise ValueError("train_actor.train.global_batch_size must be positive.")
    if global_batch_size % train_num_gpus:
        raise ValueError(
            "train_actor.train.global_batch_size must be divisible by "
            f"train_actor.num_gpus ({global_batch_size} vs {train_num_gpus})."
        )
    prompt_length = int(sequence.get("max_prompt_tokens", 1024))
    response_length = int(sequence.get("max_response_tokens", 1024))
    return {
        "capacity": int(replay.get("capacity", 256)),
        "batch_size_per_rank": global_batch_size // train_num_gpus,
        "max_policy_age": int(replay.get("max_policy_age", 0)),
        "consume_samples": bool(replay.get("consume_samples", True)),
        "seed": int(replay.get("seed", 0)),
        "data_parallel_size": train_num_gpus,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "mask_truncated": bool(replay.get("mask_truncated", False)),
    }


class ReplayBufferActor(Actor):
    """Store rollout episodes and collate one local batch per VeOmni rank."""

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self._buffer: deque[RLEpisode] = deque()
        self._capacity = 0
        self._batch_size_per_rank = 0
        self._data_parallel_size = 0
        self._max_policy_age = 0
        self._consume_samples = True
        self._prompt_length = 0
        self._response_length = 0
        self._mask_truncated = False
        self._pad_token_id: int | None = None
        self._rng = random.Random()

    @endpoint
    def setup(self, pad_token_id: int) -> ReplayBufferStatus:
        if self._capacity:
            raise RuntimeError("ReplayBufferActor.setup() may only be called once.")
        settings = _load_settings(self.config_path)
        for name in (
            "capacity",
            "batch_size_per_rank",
            "data_parallel_size",
            "prompt_length",
            "response_length",
        ):
            if settings[name] <= 0:
                raise ValueError(f"Replay buffer setting {name} must be positive.")

        self._capacity = settings["capacity"]
        self._batch_size_per_rank = settings["batch_size_per_rank"]
        self._data_parallel_size = settings["data_parallel_size"]
        self._max_policy_age = settings["max_policy_age"]
        self._consume_samples = settings["consume_samples"]
        self._prompt_length = settings["prompt_length"]
        self._response_length = settings["response_length"]
        self._mask_truncated = settings["mask_truncated"]
        self._pad_token_id = int(pad_token_id)
        self._buffer = deque(maxlen=self._capacity)
        self._rng.seed(settings["seed"])
        logger.info(
            "ReplayBufferActor initialized: capacity=%d, batch=%d x DP=%d.",
            self._capacity,
            self._batch_size_per_rank,
            self._data_parallel_size,
        )
        return self._status()

    @endpoint
    def add(self, episodes: Sequence[RLEpisode]) -> ReplayBufferStatus:
        if not self._capacity:
            raise RuntimeError("ReplayBufferActor.setup() must complete first.")
        for episode in episodes:
            episode.validate()
            self._buffer.append(episode)
        return self._status()

    def _evict_stale(self, current_policy_version: int) -> None:
        self._buffer = deque(
            (
                episode
                for episode in self._buffer
                if current_policy_version - episode.policy_version
                <= self._max_policy_age
            ),
            maxlen=self._capacity,
        )

    def _collate(self, episodes: Sequence[RLEpisode]) -> dict[str, torch.Tensor]:
        if self._pad_token_id is None:
            raise RuntimeError("ReplayBufferActor.setup() must complete first.")
        rows = [
            episode.to_training_tensors(
                prompt_length=self._prompt_length,
                response_length=self._response_length,
                pad_token_id=self._pad_token_id,
                mask_truncated=self._mask_truncated,
            )
            for episode in episodes
        ]
        return {
            key: torch.stack([row[key] for row in rows])
            for key in (
                "tokens",
                "attention_mask",
                "position_ids",
                "generator_logprobs",
                "loss_mask",
                "advantages",
            )
        }

    @endpoint
    def sample(self, current_policy_version: int) -> list[dict[str, torch.Tensor]] | None:
        if not self._capacity:
            raise RuntimeError("ReplayBufferActor.setup() must complete first.")
        self._evict_stale(int(current_policy_version))
        total_size = self._batch_size_per_rank * self._data_parallel_size
        if len(self._buffer) < total_size:
            return None

        indices = sorted(self._rng.sample(range(len(self._buffer)), total_size))
        episodes = [self._buffer[index] for index in indices]
        if self._consume_samples:
            selected = set(indices)
            self._buffer = deque(
                (
                    episode
                    for index, episode in enumerate(self._buffer)
                    if index not in selected
                ),
                maxlen=self._capacity,
            )

        return [
            self._collate(
                episodes[
                    rank * self._batch_size_per_rank :
                    (rank + 1) * self._batch_size_per_rank
                ]
            )
            for rank in range(self._data_parallel_size)
        ]

    def _status(self) -> ReplayBufferStatus:
        versions = [episode.policy_version for episode in self._buffer]
        return ReplayBufferStatus(
            size=len(self._buffer),
            capacity=self._capacity,
            batch_size_per_rank=self._batch_size_per_rank,
            data_parallel_size=self._data_parallel_size,
            oldest_policy_version=min(versions) if versions else None,
            newest_policy_version=max(versions) if versions else None,
        )

    @endpoint
    def get_status(self) -> ReplayBufferStatus:
        return self._status()

    @endpoint
    def clear(self) -> None:
        self._buffer.clear()
