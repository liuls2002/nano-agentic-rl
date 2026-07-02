from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


GSM8K_ANSWER_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def extract_answer(row: dict[str, Any], path: Path) -> str:
    if "answer" not in row:
        raise KeyError(f"Could not find an answer field in {path}: {sorted(row)}")
    return str(row["answer"]).strip()


def build_record(question: str, answer: str, label: str) -> dict[str, Any]:
    return {
        "messages": [
            message("system", SYSTEM_PROMPT, 0),
            message("user", f"{question.strip()} {GSM8K_ANSWER_INSTRUCTION}", 0),
            message("assistant", answer.strip(), 1),
        ],
        "label": label,
    }


def convert_file(source: Path, destination: Path) -> int:
    records = [
        build_record(
            question=extract_problem(row, source),
            answer=extract_answer(row, source),
            label=extract_label(row, source),
        )
        for row in read_records(source)
    ]
    write_jsonl(records, destination)
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GSM8K SFT train/validation JSONL files."
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--val-root", type=Path, default=VAL_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conversions = [
        (
            args.raw_root / "train" / "gsm8k_train.parquet",
            args.train_root / "gsm8k_train.jsonl",
        ),
        (
            args.raw_root / "val" / "gsm8k_test.parquet",
            args.val_root / "gsm8k_val.jsonl",
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

