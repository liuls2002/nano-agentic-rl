from __future__ import annotations

import argparse
import re
from pathlib import Path

from dataset_utils import (
    RAW_ROOT,
    SYSTEM_PROMPT,
    TRAIN_ROOT,
    VAL_ROOT,
    extract_label,
    extract_problem,
    message,
    read_records,
    write_jsonl,
)


MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form <answer>$Answer</answer> where $Answer is the answer "
    "to the problem.\n\n{question}\n\nPut only the final answer inside the "
    "<answer></answer> tags on the last line."
)

OLD_BOXED_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: \\boxed{$Answer} where $Answer is the answer "
    "to the problem."
)
OLD_ANSWER_SUFFIX = re.compile(
    r'\n\nRemember to put your answer on its own line after "Answer:"\.\s*$'
)


def normalize_question(question: str) -> str:
    text = question.strip()
    if text.startswith(OLD_BOXED_PREFIX) and "\n\n" in text:
        text = text.split("\n\n", 1)[1].strip()
    text = OLD_ANSWER_SUFFIX.sub("", text).strip()
    return text


def build_record(question: str, label: str) -> dict[str, object]:
    user_content = MATH_PROMPT_TEMPLATE.format(
        question=normalize_question(question)
    )
    return {
        "messages": [
            message("system", SYSTEM_PROMPT, 0),
            message("user", user_content, 0),
        ],
        "label": label,
}


def convert_file(source: Path, destination: Path) -> int:
    records = [
        build_record(
            question=extract_problem(row, source),
            label=extract_label(row, source),
        )
        for row in read_records(source)
    ]
    write_jsonl(records, destination)
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DAPO-Math train and AIME validation JSONL files for RL."
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--val-root", type=Path, default=VAL_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conversions = [
        (
            args.raw_root / "train" / "dapo-math-17k.jsonl",
            args.train_root / "dapo-math-17k.jsonl",
        ),
        (
            args.raw_root / "val" / "aime-2024.jsonl",
            args.val_root / "aime-2024.jsonl",
        ),
        (
            args.raw_root / "val" / "aime-2025.jsonl",
            args.val_root / "aime-2025.jsonl",
        ),
    ]
    for source, destination in conversions:
        source = source.expanduser().resolve()
        destination = destination.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        count = convert_file(source, destination)
        print(f"{source} -> {destination} ({count} samples)")


if __name__ == "__main__":
    main()
