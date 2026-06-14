from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


@dataclass
class GRPOLossOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


def create_shifted_targets(
    tokens: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    targets = torch.roll(tokens, shifts=-1, dims=-1)
    targets[..., -1] = IGNORE_INDEX
    return torch.where(
        loss_mask.bool(), targets, torch.full_like(targets, IGNORE_INDEX)
    )


def selected_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    batch_size, sequence_length, vocab_size = logits.shape
    return -F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape(batch_size, sequence_length)


def grpo_loss(
    *,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    generator_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
) -> GRPOLossOutput:
    """Compute the clipped, token-level GRPO objective."""
    targets = create_shifted_targets(tokens, loss_mask)
    logprobs = selected_logprobs(logits, targets)
    log_ratio = logprobs - generator_logprobs.detach()
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    per_token_loss = torch.maximum(
        -ratio * advantages, -clipped_ratio * advantages
    )

    active_tokens = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / active_tokens.clamp_min(1.0)

    with torch.no_grad():
        active = loss_mask.bool()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "ratio_mean": (
                float(ratio[active].mean().detach().cpu()) if active.any() else 0.0
            ),
            "approx_kl": (
                float((-log_ratio[active]).mean().detach().cpu())
                if active.any()
                else 0.0
            ),
            "clip_fraction": (
                float(((ratio != clipped_ratio) & active).float().sum().cpu())
                / max(float(active_tokens.cpu()), 1.0)
            ),
            "active_tokens": float(active_tokens.cpu()),
        }
    return GRPOLossOutput(loss=loss, metrics=metrics)
