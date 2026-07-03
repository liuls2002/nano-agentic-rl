from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "train" / "dapo-math-17k.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "rollout_stats" / "train_rollout_n8.json"
DEFAULT_MODEL_PATH = Path("/ssd/liuls/data/hub/Qwen3-4B-Instruct-2507")

NCCL_IB_DISABLE = "1"
DATA_PARALLEL_SIZE = 6
TENSOR_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 2048
MAX_PROMPT_TOKENS = 1024
GPU_MEMORY_UTILIZATION = 0.8
DTYPE = "bfloat16"
SEED = 42

SAMPLE_N = 8
MAX_TOKENS = 1024
TEMPERATURE = 1.0
TOP_P = 0.95

CORRECTNESS_TOLERANCE = Decimal("1e-6")
LOW_VARIANCE_STD_THRESHOLD = 1.0e-3

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
logger = logging.getLogger("rollout_dataset_stats")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            rows.append(row)
    return rows


def rollout_messages(messages: Any, path: Path) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError(f"Dataset row messages must be a list: {path}")

    normalized = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError(f"Dataset row messages must contain mappings: {path}")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "assistant":
            continue
        if role not in {"system", "user"}:
            raise ValueError(f"Unsupported message role {role!r}: {path}")
        normalized.append({"role": role, "content": content})

    if not any(message["role"] == "user" for message in normalized):
        raise ValueError(f"Dataset row has no user message: {path}")
    return normalized


def prompt_text(messages: Sequence[Mapping[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return str(message["content"])
    raise ValueError("messages must contain a user message.")


def load_samples(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    samples = []
    for row_index, row in enumerate(rows):
        if "messages" not in row or "label" not in row:
            raise KeyError("Rows must contain preprocessed messages and label fields.")
        messages = rollout_messages(row["messages"], path)
        samples.append(
            {
                "sample_id": f"train-row-{row_index}",
                "row_index": row_index,
                "messages": messages,
                "prompt": prompt_text(messages),
                "target": str(row["label"]).strip(),
            }
        )
    return samples


def extract_numeric_answer(text: str) -> str | None:
    tagged = ANSWER_PATTERN.findall(text)
    search_text = tagged[-1] if tagged else text
    matches = NUMBER_PATTERN.findall(search_text)
    return matches[-1].replace(",", "") if matches else None


def as_decimal(text: str | None) -> Decimal | None:
    if text is None:
        return None
    try:
        return Decimal(text.strip().replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def is_correct(response: str, target: str) -> bool:
    predicted = as_decimal(extract_numeric_answer(response))
    expected = as_decimal(extract_numeric_answer(target))
    return (
        predicted is not None
        and expected is not None
        and abs(predicted - expected) <= CORRECTNESS_TOLERANCE
    )


def population_variance(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


async def generate_one(
    llm: Any, prompt: Any, sampling_params: Any, request_id: str
) -> Any:
    request_output = None
    async for output in llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        request_output = output
    if request_output is None:
        raise RuntimeError("vLLM returned no output.")
    return request_output


async def rollout_batch(
    *,
    llm: Any,
    samples: Sequence[Mapping[str, Any]],
    sampling_params: Any,
    requested_n: int,
) -> list[list[Any]]:
    from vllm.renderers.params import ChatParams, TokenizeParams

    _, prompts = await llm.renderer.render_chat_async(
        [sample["messages"] for sample in samples],
        ChatParams(
            chat_template_kwargs={
                "add_generation_prompt": True,
                "continue_final_message": False,
                "tools": None,
                "tokenize": False,
            }
        ),
        TokenizeParams(
            max_total_tokens=MAX_MODEL_LEN,
            max_output_tokens=MAX_TOKENS,
            truncate_prompt_tokens=MAX_PROMPT_TOKENS,
            truncation_side="left",
        ),
    )

    tasks = []
    task_keys = []
    for group_index, prompt in enumerate(prompts):
        sample_id = str(samples[group_index]["sample_id"])
        for sample_index in range(requested_n):
            request_id = (
                f"stats-{sample_id}-sample-{sample_index}-{uuid.uuid4().hex}"
            )
            tasks.append(
                generate_one(llm, prompt, sampling_params.clone(), request_id)
            )
            task_keys.append(group_index)

    outputs = await asyncio.gather(*tasks)
    grouped: list[list[Any]] = [[] for _ in samples]
    for group_index, request_output in zip(task_keys, outputs):
        if len(request_output.outputs) != 1:
            raise RuntimeError("Expanded n=1 request returned multiple outputs.")
        grouped[group_index].append(request_output.outputs[0])
    return grouped


def build_record(
    sample: Mapping[str, Any],
    outputs: Sequence[Any],
    include_responses: bool,
) -> dict[str, Any]:
    responses = [str(output.text) for output in outputs]
    correctness = [
        1.0 if is_correct(response, str(sample["target"])) else 0.0
        for response in responses
    ]
    variance = population_variance(correctness)
    std = math.sqrt(variance)
    record = {
        "sample_id": sample["sample_id"],
        "row_index": sample["row_index"],
        "prompt": sample["prompt"],
        "target": sample["target"],
        "num_samples": len(correctness),
        "correctness": correctness,
        "correct_count": int(sum(correctness)),
        "accuracy": sum(correctness) / len(correctness),
        "variance": variance,
        "std": std,
        "low_variance": std < LOW_VARIANCE_STD_THRESHOLD,
        "response_token_counts": [
            len(output.token_ids or []) for output in outputs
        ],
        "finish_reasons": [output.finish_reason for output in outputs],
    }
    if include_responses:
        record["responses"] = responses
    return record


def metric_at_k(
    records: Sequence[Mapping[str, Any]], k: int
) -> dict[str, float | int]:
    any_correct = 0.0
    mean_correct = 0.0
    all_correct = 0.0
    for record in records:
        values = [float(value) for value in record["correctness"][:k]]
        any_correct += float(any(value > 0.0 for value in values))
        mean_correct += sum(values) / k
        all_correct += float(all(value > 0.0 for value in values))

    count = float(len(records))
    return {
        "k": k,
        "pass_at_k": any_correct / count,
        "g_pass_at_k": mean_correct / count,
        "all_pass_at_k": all_correct / count,
    }


def build_summary(
    *,
    data_path: Path,
    model_path: Path,
    records: Sequence[Mapping[str, Any]],
    requested_n: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    histogram = {str(count): 0 for count in range(requested_n + 1)}
    for record in records:
        histogram[str(record["correct_count"])] += 1

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "model_path": str(model_path),
        "n": requested_n,
        "sampling": {
            "n": requested_n,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
        "num_questions": len(records),
        "num_completions": len(records) * requested_n,
        "elapsed_seconds": elapsed_seconds,
        "mean_accuracy": sum(float(record["accuracy"]) for record in records)
        / len(records),
        "low_variance_count": sum(1 for record in records if record["low_variance"]),
        "all_wrong_count": sum(
            1 for record in records if record["correct_count"] == 0
        ),
        "all_correct_count": sum(
            1 for record in records if record["correct_count"] == requested_n
        ),
        "correct_count_histogram": histogram,
        "pass_at_k": [metric_at_k(records, k) for k in range(1, requested_n + 1)],
    }


async def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("NCCL_IB_DISABLE", NCCL_IB_DISABLE)

    data_path = args.data_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_samples(data_path, args.max_samples)
    logger.info("Loaded %d samples from %s.", len(samples), data_path)

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=str(model_path),
        tokenizer=str(model_path),
        data_parallel_size=DATA_PARALLEL_SIZE,
        data_parallel_backend="mp",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=1,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        seed=SEED,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        n=1,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        logprobs=0,
    )
    sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

    logger.info(
        "Initializing vLLM: model=%s, DP=%d, TP=%d, n=%d.",
        model_path,
        DATA_PARALLEL_SIZE,
        TENSOR_PARALLEL_SIZE,
        args.n,
    )
    llm = AsyncLLM.from_engine_args(engine_args)

    records = []
    started_at = time.perf_counter()
    try:
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            grouped_outputs = await rollout_batch(
                llm=llm,
                samples=batch,
                sampling_params=sampling_params,
                requested_n=args.n,
            )
            for sample, outputs in zip(batch, grouped_outputs):
                records.append(
                    build_record(sample, outputs, args.include_responses)
                )

            if args.log_every and len(records) % args.log_every == 0:
                logger.info(
                    "Processed %d/%d questions in %.1fs.",
                    len(records),
                    len(samples),
                    time.perf_counter() - started_at,
                )

        elapsed_seconds = time.perf_counter() - started_at
        result = {
            "summary": build_summary(
                data_path=data_path,
                model_path=model_path,
                records=records,
                requested_n=args.n,
                elapsed_seconds=elapsed_seconds,
            ),
            "records": records,
        }
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                result,
                output_file,
                ensure_ascii=False,
                indent=2 if args.pretty else None,
            )
            output_file.write("\n")
        logger.info("Wrote rollout stats to %s.", output_path)
    finally:
        llm.shutdown(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out train data with n samples and save correctness stats."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n", type=int, default=SAMPLE_N)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--include-responses", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--log-every", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.n <= 0:
        raise ValueError("--n must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
