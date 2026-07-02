from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PassAtKMetrics:
    k: int
    pass_at_k: float
    g_pass_at_k: float
    all_pass_at_k: float


def compute_pass_at_k(
    correctness_groups: Sequence[Sequence[float | int | bool]], k: int
) -> PassAtKMetrics:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not correctness_groups:
        raise ValueError("At least one correctness group is required.")

    any_correct = 0.0
    mean_correct = 0.0
    all_correct = 0.0
    for group in correctness_groups:
        if len(group) < k:
            raise ValueError("Each correctness group must contain at least k values.")
        values = [float(value) for value in group[:k]]
        any_correct += float(any(value > 0.0 for value in values))
        mean_correct += sum(values) / k
        all_correct += float(all(value > 0.0 for value in values))

    count = float(len(correctness_groups))
    return PassAtKMetrics(
        k=k,
        pass_at_k=any_correct / count,
        g_pass_at_k=mean_correct / count,
        all_pass_at_k=all_correct / count,
    )


def compute_pass_at_k_range(
    correctness_groups: Sequence[Sequence[float | int | bool]], max_k: int
) -> list[PassAtKMetrics]:
    return [compute_pass_at_k(correctness_groups, k) for k in range(1, max_k + 1)]
