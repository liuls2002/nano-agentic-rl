from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class DatasetSample:
    """One prompt/target pair returned by DatasetActor."""

    sample_id: str
    prompt: str
    target: str
    messages: list[dict[str, str]]


@dataclass
class RLEpisode:
    """A rollout trajectory containing the tensors required by GRPO."""

    episode_id: str
    sample_id: str
    prompt: str
    target: str
    response: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    generator_logprobs: list[float]
    reward: float
    advantage: float
    policy_version: int
    finish_reason: str | None = None
    reward_breakdown: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("An RL episode must contain at least one prompt token.")
        if len(self.response_token_ids) != len(self.generator_logprobs):
            raise ValueError(
                "response_token_ids and generator_logprobs must have equal length."
            )

    def to_training_tensors(
        self,
        *,
        prompt_length: int,
        response_length: int,
        pad_token_id: int,
        mask_truncated: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Pad one episode and align old logprobs with next-token logits."""
        self.validate()
        if prompt_length <= 0 or response_length <= 0:
            raise ValueError("prompt_length and response_length must be positive.")

        prompt_ids = torch.tensor(self.prompt_token_ids[-prompt_length:], dtype=torch.long)
        prompt_token_count = prompt_ids.numel()
        prompt_ids = F.pad(
            prompt_ids,
            (prompt_length - prompt_token_count, 0),
            value=pad_token_id,
        )

        response_ids = torch.tensor(
            self.response_token_ids[:response_length], dtype=torch.long
        )
        response_token_count = response_ids.numel()
        response_ids = F.pad(
            response_ids,
            (0, response_length - response_token_count),
            value=pad_token_id,
        )

        sequence_length = prompt_length + response_length
        old_logprobs = torch.zeros(sequence_length, dtype=torch.float32)
        loss_mask = torch.zeros(sequence_length, dtype=torch.float32)
        attention_mask = torch.zeros(sequence_length, dtype=torch.bool)
        attention_mask[prompt_length - prompt_token_count : prompt_length] = True
        attention_mask[
            prompt_length : prompt_length + response_token_count
        ] = True
        position_ids = attention_mask.long().cumsum(dim=0) - 1
        position_ids.masked_fill_(~attention_mask, 0)
        if response_token_count:
            response_logprobs = torch.tensor(
                self.generator_logprobs[:response_token_count], dtype=torch.float32
            )
            start = prompt_length - 1
            end = start + response_token_count
            old_logprobs[start:end] = response_logprobs
            loss_mask[start:end] = 1.0

        if mask_truncated and self.finish_reason == "length":
            loss_mask.zero_()

        return {
            "tokens": torch.cat((prompt_ids, response_ids)),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "generator_logprobs": old_logprobs,
            "loss_mask": loss_mask,
            "advantages": torch.full(
                (sequence_length,), float(self.advantage), dtype=torch.float32
            ),
        }
