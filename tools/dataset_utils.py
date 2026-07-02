from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
TRAIN_ROOT = PROJECT_ROOT / "data" / "train"
VAL_ROOT = PROJECT_ROOT / "data" / "val"

SYSTEM_PROMPT = "You are a helpful assistant for answering user's math promblem."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    raise ValueError(f"Unsupported raw data format: {path}")


def write_jsonl(records: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prompt_text(prompt: Any) -> str | None:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    return None


def extract_problem(row: dict[str, Any], path: Path) -> str:
    if "question" in row:
        return str(row["question"]).strip()
    if "problem" in row:
        return str(row["problem"]).strip()
    if "prompt" in row:
        text = prompt_text(row["prompt"])
        if text is not None:
            return text.strip()
    if "messages" in row:
        text = prompt_text(row["messages"])
        if text is not None:
            return text.strip()
    raise KeyError(f"Could not find a problem field in {path}: {sorted(row)}")


def extract_label(row: dict[str, Any], path: Path) -> str:
    if "label" in row:
        return str(row["label"]).strip()
    if "answer" in row:
        answer = str(row["answer"]).strip()
        if "####" in answer:
            return answer.rsplit("####", 1)[-1].strip()
        return answer
    raise KeyError(f"Could not find a label field in {path}: {sorted(row)}")


def message(role: str, content: str, loss_mask: int) -> dict[str, Any]:
    return {"role": role, "content": content, "loss_mask": loss_mask}

