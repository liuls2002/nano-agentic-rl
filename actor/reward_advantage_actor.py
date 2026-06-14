from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

import yaml
from monarch.actor import Actor, endpoint


logger = logging.getLogger(__name__)
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)


@dataclass
class RewardResult:
    reward: float
    breakdown: dict[str, float]
    predicted_answer: str | None


@dataclass
class AdvantageResult:
    advantages: list[float]
    reward_mean: float
    reward_std: float
    low_variance: bool


def _load_rl_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    monarch = config.get("monarch", {})
    if not isinstance(monarch, dict):
        raise ValueError("monarch must be a mapping.")
    rl_config = monarch.get("rl", {})
    if not isinstance(rl_config, dict):
        raise ValueError("monarch.rl must be a mapping.")
    return dict(rl_config)


def extract_numeric_answer(text: str) -> str | None:
    tagged = ANSWER_PATTERN.findall(text)
    search_text = tagged[-1] if tagged else text
    matches = NUMBER_PATTERN.findall(search_text)
    return matches[-1].replace(",", "") if matches else None


def _as_decimal(text: str | None) -> Decimal | None:
    if text is None:
        return None
    try:
        return Decimal(text.strip().replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


class RewardActor(Actor):
    """Evaluate GSM8K answer correctness and optional answer-tag formatting."""

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self._correctness_weight = 1.0
        self._format_weight = 0.0
        self._tolerance = Decimal("1e-6")
        self._initialized = False

    @endpoint
    def setup(self) -> None:
        reward_config = _load_rl_config(self.config_path).get("reward", {})
        if not isinstance(reward_config, dict):
            raise ValueError("monarch.rl.reward must be a mapping.")
        self._correctness_weight = float(
            reward_config.get("correctness_weight", 1.0)
        )
        self._format_weight = float(reward_config.get("format_weight", 0.0))
        self._tolerance = Decimal(str(reward_config.get("tolerance", 1e-6)))
        self._initialized = True
        logger.info(
            "RewardActor initialized (correctness=%s, format=%s).",
            self._correctness_weight,
            self._format_weight,
        )

    def _evaluate(self, response: str, target: str) -> RewardResult:
        predicted = extract_numeric_answer(response)
        target_answer = extract_numeric_answer(target)
        predicted_number = _as_decimal(predicted)
        target_number = _as_decimal(target_answer)
        correct = float(
            predicted_number is not None
            and target_number is not None
            and abs(predicted_number - target_number) <= self._tolerance
        )
        answer_format = float(bool(ANSWER_PATTERN.search(response)))
        breakdown = {
            "correctness": correct,
            "answer_format": answer_format,
        }
        reward = (
            self._correctness_weight * correct
            + self._format_weight * answer_format
        )
        return RewardResult(
            reward=reward,
            breakdown=breakdown,
            predicted_answer=predicted,
        )

    @endpoint
    def evaluate(self, response: str, target: str) -> RewardResult:
        if not self._initialized:
            raise RuntimeError("RewardActor.setup() must complete first.")
        return self._evaluate(response, target)

    @endpoint
    def evaluate_batch(
        self, responses: Sequence[str], targets: Sequence[str]
    ) -> list[RewardResult]:
        if not self._initialized:
            raise RuntimeError("RewardActor.setup() must complete first.")
        if len(responses) != len(targets):
            raise ValueError("responses and targets must have equal length.")
        return [
            self._evaluate(response, target)
            for response, target in zip(responses, targets)
        ]


class AdvantageActor(Actor):
    """Compute normalized, group-relative advantages for GRPO."""

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self._epsilon = 1e-4
        self._minimum_std = 1e-3
        self._initialized = False

    @endpoint
    def setup(self) -> None:
        advantage_config = _load_rl_config(self.config_path).get("advantage", {})
        if not isinstance(advantage_config, dict):
            raise ValueError("monarch.rl.advantage must be a mapping.")
        self._epsilon = float(advantage_config.get("epsilon", 1e-4))
        self._minimum_std = float(
            advantage_config.get("minimum_group_std", 1e-3)
        )
        self._initialized = True

    @endpoint
    def compute(self, rewards: Sequence[float]) -> AdvantageResult:
        if not self._initialized:
            raise RuntimeError("AdvantageActor.setup() must complete first.")
        if not rewards:
            raise ValueError("At least one reward is required.")
        values = [float(reward) for reward in rewards]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Rewards must be finite numbers.")

        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        advantages = [(value - mean) / (std + self._epsilon) for value in values]
        return AdvantageResult(
            advantages=advantages,
            reward_mean=mean,
            reward_std=std,
            low_variance=std < self._minimum_std,
        )
