from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

from monarch.actor import Actor, endpoint

from actor.utils import load_yaml_config


logger = logging.getLogger(__name__)
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
ANSWER_LINE_PATTERN = re.compile(
    r"(?im)(?:^|\n)\s*(?:final\s+)?answer\s*:\s*([^\n]+)"
)
LATEX_FRAC_PATTERN = re.compile(
    r"\\(?:d?frac)\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}"
)
PLAIN_FRAC_PATTERN = re.compile(r"^\s*([-+]?\d+)\s*/\s*([-+]?\d+)\s*$")
SIMPLE_EXPR_PATTERN = re.compile(r"^[0-9a-zA-Z_+\-*/(). ^]+$")


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
    config = load_yaml_config(config_path)
    rl_config = config.get("rl", {})
    if rl_config is None:
        return {}
    if not isinstance(rl_config, dict):
        raise ValueError("rl must be a mapping.")
    return dict(rl_config)


def _last_boxed_content(text: str) -> str | None:
    best_index = -1
    best_command = ""
    for command in ("\\boxed", "\\fbox"):
        index = text.rfind(command)
        if index > best_index:
            best_index = index
            best_command = command
    if best_index < 0:
        return None

    brace_start = text.find("{", best_index + len(best_command))
    if brace_start < 0:
        return None

    depth = 0
    for index in range(brace_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index].strip()
    return None


def extract_answer(text: str) -> str | None:
    tagged = ANSWER_PATTERN.findall(text)
    if tagged and tagged[-1].strip():
        return tagged[-1].strip()

    return None


def extract_target_answer(text: str) -> str | None:
    tagged = ANSWER_PATTERN.findall(text)
    if tagged and tagged[-1].strip():
        return tagged[-1].strip()

    boxed = _last_boxed_content(text)
    if boxed:
        return boxed

    answer_lines = ANSWER_LINE_PATTERN.findall(text)
    if answer_lines and answer_lines[-1].strip():
        return answer_lines[-1].strip()

    stripped = str(text).strip()
    if stripped and "\n" not in stripped and len(stripped) <= 128:
        return stripped

    matches = NUMBER_PATTERN.findall(text)
    return matches[-1].replace(",", "") if matches else None


def _as_decimal(text: str | None) -> Decimal | None:
    if text is None:
        return None
    text = _clean_answer_text(text)

    def safe_decimal(value: str) -> Decimal | None:
        try:
            number = Decimal(value)
            if not number.is_finite():
                return None
            return number
        except (InvalidOperation, ValueError, TypeError, AttributeError):
            return None

    latex_fraction = LATEX_FRAC_PATTERN.fullmatch(text)
    if latex_fraction:
        try:
            numerator = safe_decimal(latex_fraction.group(1))
            denominator = safe_decimal(latex_fraction.group(2))
            if numerator is None or denominator is None:
                return None
            number = numerator / denominator
            if not number.is_finite():
                return None
            return number
        except (InvalidOperation, DivisionByZero, ZeroDivisionError):
            return None

    plain_fraction = PLAIN_FRAC_PATTERN.fullmatch(text)
    if plain_fraction:
        try:
            numerator = safe_decimal(plain_fraction.group(1))
            denominator = safe_decimal(plain_fraction.group(2))
            if numerator is None or denominator is None:
                return None
            number = numerator / denominator
            if not number.is_finite():
                return None
            return number
        except (InvalidOperation, DivisionByZero, ZeroDivisionError):
            return None

    return safe_decimal(text)


def _clean_answer_text(text: str) -> str:
    text = str(text).strip()
    text = text.strip("$")
    text = text.rstrip(".。")
    text = text.replace(",", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    return text.strip()


def _normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    text = _clean_answer_text(text).lower()
    text = re.sub(r"\\text\s*\{\s*([^{}]*?)\s*\}", r"\1", text)
    text = LATEX_FRAC_PATTERN.sub(r"(\1)/(\2)", text)
    text = text.replace("^", "**")
    text = text.replace(" ", "")
    return text or None


def _sympy_equal(predicted: str | None, target: str | None) -> bool:
    predicted_normalized = _normalize_answer(predicted)
    target_normalized = _normalize_answer(target)
    if not predicted_normalized or not target_normalized:
        return False
    if len(predicted_normalized) > 128 or len(target_normalized) > 128:
        return False
    if not SIMPLE_EXPR_PATTERN.fullmatch(predicted_normalized):
        return False
    if not SIMPLE_EXPR_PATTERN.fullmatch(target_normalized):
        return False
    try:
        import sympy

        predicted_expr = sympy.sympify(predicted_normalized)
        target_expr = sympy.sympify(target_normalized)
        return bool(sympy.simplify(predicted_expr - target_expr) == 0)
    except Exception:
        return False


def answers_match(predicted: str | None, target: str | None, tolerance: Decimal) -> bool:
    predicted_number = _as_decimal(predicted)
    target_number = _as_decimal(target)
    if predicted_number is not None and target_number is not None:
        try:
            return abs(predicted_number - target_number) <= tolerance
        except (InvalidOperation, ValueError, TypeError, AttributeError):
            return False

    predicted_normalized = _normalize_answer(predicted)
    target_normalized = _normalize_answer(target)
    if predicted_normalized and target_normalized and predicted_normalized == target_normalized:
        return True

    return _sympy_equal(predicted, target)


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
            raise ValueError("rl.reward must be a mapping.")
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
        predicted = extract_answer(response)
        target_answer = extract_target_answer(target)
        correct = float(answers_match(predicted, target_answer, self._tolerance))
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
            raise ValueError("rl.advantage must be a mapping.")
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
