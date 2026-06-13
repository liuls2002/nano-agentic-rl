from __future__ import annotations

import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from monarch.actor import Actor, endpoint
from veomni.arguments import VeOmniArguments, parse_args
from veomni.trainer.text_trainer import TextTrainer


logger = logging.getLogger(__name__)


def process_prompt_response_example(
    example: dict[str, Any],
    *,
    chat_template: Any,
    max_seq_len: int,
    prompt_key: str,
    response_key: str,
    system_prompt: str | None = None,
    **_: Any,
) -> list[dict[str, torch.Tensor]]:
    """Convert a prompt/response row to VeOmni's conversation SFT format."""
    missing_keys = [key for key in (prompt_key, response_key) if key not in example]
    if missing_keys:
        raise KeyError(f"Dataset row is missing required keys: {missing_keys}")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt, "loss_mask": 0})
    messages.extend(
        [
            {"role": "user", "content": str(example[prompt_key]), "loss_mask": 0},
            {"role": "assistant", "content": str(example[response_key]), "loss_mask": 1},
        ]
    )
    tokenized = chat_template.encode_messages(messages, max_seq_len=max_seq_len)
    return [{key: torch.tensor(value) for key, value in tokenized.items()}]


class PromptResponseTextTrainer(TextTrainer):
    """TextTrainer variant for datasets with separate prompt and response columns."""

    def __init__(self, args: VeOmniArguments, adapter_config: dict[str, Any]):
        self._adapter_config = adapter_config
        super().__init__(args)

    def _build_data_transform(self) -> None:
        args = self.base.args
        if args.data.data_type != "conversation":
            raise ValueError("The prompt_response adapter requires data.data_type='conversation'.")

        self.base.data_transform = partial(
            process_prompt_response_example,
            chat_template=self.base.chat_template,
            max_seq_len=args.data.max_seq_len,
            prompt_key=self._adapter_config.get("prompt_key", "question"),
            response_key=self._adapter_config.get("response_key", "answer"),
            system_prompt=self._adapter_config.get("system_prompt"),
        )


def load_veomni_args(config_path: str) -> VeOmniArguments:
    """Parse a VeOmni YAML file after Monarch has installed rank variables."""
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], config_path]
        return parse_args(VeOmniArguments)
    finally:
        sys.argv = original_argv


class TrainActor(Actor):
    """Monarch actor that owns one rank of a VeOmni text trainer."""

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.trainer: TextTrainer | None = None

    @endpoint
    def setup(self) -> None:
        if self.trainer is not None:
            raise RuntimeError("TrainActor.setup() may only be called once.")

        with open(self.config_path, encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}

        args = load_veomni_args(self.config_path)
        adapter_config = raw_config.get("data_adapter")
        if adapter_config is None:
            self.trainer = TextTrainer(args)
        elif adapter_config.get("type") == "prompt_response":
            self.trainer = PromptResponseTextTrainer(args, adapter_config)
        else:
            adapter_type = adapter_config.get("type")
            raise ValueError(f"Unsupported data_adapter.type: {adapter_type!r}")

        logger.info(
            "VeOmni trainer initialized on rank %s/%s (local rank %s).",
            os.environ.get("RANK", "0"),
            os.environ.get("WORLD_SIZE", "1"),
            os.environ.get("LOCAL_RANK", "0"),
        )

    @endpoint
    def train(self) -> None:
        if self.trainer is None:
            raise RuntimeError("TrainActor.setup() must complete before train().")

        try:
            self.trainer.train()
        finally:
            # TextTrainer destroys the process group on success. This also covers
            # failures in the middle of training so Monarch can stop the mesh cleanly.
            if dist.is_initialized():
                self.trainer.base.destroy_distributed()
