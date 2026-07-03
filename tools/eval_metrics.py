from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PassAtKMetrics:
    k: int
    pass_at_k: float


def compute_pass_at_k(
    correctness_groups: Sequence[Sequence[float | int | bool]], k: int
) -> PassAtKMetrics:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not correctness_groups:
        raise ValueError("At least one correctness group is required.")

    any_correct = 0.0
    for group in correctness_groups:
        if len(group) < k:
            raise ValueError("Each correctness group must contain at least k values.")
        all_values = [float(value) for value in group]
        num_samples = len(all_values)
        num_correct = sum(1 for value in all_values if value > 0.0)
        any_correct += _estimate_pass_at_k(num_samples, num_correct, k)

    count = float(len(correctness_groups))
    return PassAtKMetrics(
        k=k,
        pass_at_k=any_correct / count,
    )


def compute_pass_at_k_range(
    correctness_groups: Sequence[Sequence[float | int | bool]], max_k: int
) -> list[PassAtKMetrics]:
    return [
        compute_pass_at_k(correctness_groups, k)
        for k in slime_pass_at_k_values(max_k)
    ]


def slime_pass_at_k_values(max_k: int) -> list[int]:
    if max_k <= 0:
        raise ValueError("max_k must be positive.")
    values = []
    k = 1
    while k <= max_k:
        values.append(k)
        k *= 2
    return values


def _estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Slime/OpenAI-style unbiased pass@k estimator."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > num_samples:
        raise ValueError("k must not exceed num_samples.")
    if num_samples - num_correct < k:
        return 1.0

    product = 1.0
    for value in range(num_samples - num_correct + 1, num_samples + 1):
        product *= 1.0 - k / value
    return 1.0 - product
