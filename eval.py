from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from actor.reward_advantage_actor import (  # noqa: E402
    ANSWER_PATTERN,
    answers_match,
    extract_answer,
    extract_target_answer,
)
from tools.eval_metrics import compute_pass_at_k_range  # noqa: E402


# Edit these constants for the common path. CLI flags can temporarily override
# model/data/output/batch/n without introducing a YAML dependency.
MODEL_PATH = Path("/ssd/liuls/data/hub/Qwen2.5-1.5B-Instruct")
DATA_PATH = REPO_ROOT / "data" / "val" / "aime-2024.jsonl"
OUTPUT_PATH = REPO_ROOT / "outputs" / "eval" / "eval_results.json"

DATA_PARALLEL_SIZE = 2
DATA_PARALLEL_BACKEND = "mp"
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
DTYPE = "bfloat16"
MAX_PROMPT_TOKENS = 1024
MAX_RESPONSE_TOKENS = 2048
MAX_MODEL_LEN = MAX_PROMPT_TOKENS + MAX_RESPONSE_TOKENS
GPU_MEMORY_UTILIZATION = 0.8
ENFORCE_EAGER = True
SEED = 42

SAMPLE_N = 8
BATCH_SIZE = 64
TEMPERATURE = 1.0
TOP_P = 0.95
LOGPROBS = 0

CORRECTNESS_WEIGHT = 1.0
FORMAT_WEIGHT = 0.0
CORRECTNESS_TOLERANCE = Decimal("1e-6")

NCCL_IB_DISABLE = "1"
LOG_EVERY = 64

logger = logging.getLogger("eval")


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
                "sample_id": row.get("sample_id") or f"eval-row-{row_index}",
                "row_index": row_index,
                "messages": messages,
                "prompt": prompt_text(messages),
                "target": str(row["label"]).strip(),
            }
        )
    return samples


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def evaluate_response(response: str, target: str) -> dict[str, Any]:
    predicted = extract_answer(response)
    target_answer = extract_target_answer(target)
    correctness = float(
        answers_match(predicted, target_answer, CORRECTNESS_TOLERANCE)
    )
    answer_format = float(bool(ANSWER_PATTERN.search(response)))
    reward = CORRECTNESS_WEIGHT * correctness + FORMAT_WEIGHT * answer_format
    return {
        "reward": reward,
        "correctness": correctness,
        "answer_format": answer_format,
        "predicted_answer": predicted,
        "target_answer": target_answer,
    }


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


async def generate_batch(
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
            max_output_tokens=MAX_RESPONSE_TOKENS,
            truncate_prompt_tokens=MAX_PROMPT_TOKENS,
            truncation_side="left",
        ),
    )

    tasks = []
    task_keys = []
    for group_index, prompt in enumerate(prompts):
        sample_id = str(samples[group_index]["sample_id"])
        for sample_index in range(requested_n):
            request_id = f"eval-{sample_id}-{sample_index}-{uuid.uuid4().hex}"
            tasks.append(generate_one(llm, prompt, sampling_params.clone(), request_id))
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
    samples = []
    correctness = []
    rewards = []
    response_token_counts = []
    finish_reasons = []
    for output in outputs:
        text = str(output.text)
        evaluation = evaluate_response(text, str(sample["target"]))
        correctness.append(float(evaluation["correctness"]))
        rewards.append(float(evaluation["reward"]))
        token_count = len(output.token_ids or [])
        response_token_counts.append(token_count)
        finish_reasons.append(output.finish_reason)

        sample_record = {
            "correctness": evaluation["correctness"],
            "reward": evaluation["reward"],
            "answer_format": evaluation["answer_format"],
            "predicted_answer": evaluation["predicted_answer"],
            "response_tokens": token_count,
            "finish_reason": output.finish_reason,
        }
        if include_responses:
            sample_record["response"] = text
        samples.append(sample_record)

    return {
        "sample_id": sample["sample_id"],
        "row_index": sample["row_index"],
        "prompt": sample["prompt"],
        "target": sample["target"],
        "correctness": correctness,
        "rewards": rewards,
        "accuracy": sum(correctness) / len(correctness),
        "reward_mean": sum(rewards) / len(rewards),
        "response_token_counts": response_token_counts,
        "finish_reasons": finish_reasons,
        "samples": samples,
    }


def build_summary(
    *,
    data_path: Path,
    model_path: Path,
    records: Sequence[Mapping[str, Any]],
    requested_n: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    correctness_groups = [record["correctness"] for record in records]
    pass_metrics = compute_pass_at_k_range(correctness_groups, requested_n)
    sample_count = sum(len(group) for group in correctness_groups)
    correct_count = sum(sum(float(value) for value in group) for group in correctness_groups)
    reward_values = [
        float(reward)
        for record in records
        for reward in record["rewards"]
    ]
    response_token_counts = [
        int(count)
        for record in records
        for count in record["response_token_counts"]
    ]
    finish_reasons = [
        reason
        for record in records
        for reason in record["finish_reasons"]
    ]
    truncated_count = sum(1 for reason in finish_reasons if reason == "length")

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "model_path": str(model_path),
        "engine": {
            "data_parallel_size": DATA_PARALLEL_SIZE,
            "data_parallel_backend": DATA_PARALLEL_BACKEND,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
            "dtype": DTYPE,
            "max_model_len": MAX_MODEL_LEN,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_response_tokens": MAX_RESPONSE_TOKENS,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "enforce_eager": ENFORCE_EAGER,
            "seed": SEED,
        },
        "sampling": {
            "n": requested_n,
            "max_tokens": MAX_RESPONSE_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "logprobs": LOGPROBS,
        },
        "reward": {
            "correctness_weight": CORRECTNESS_WEIGHT,
            "format_weight": FORMAT_WEIGHT,
            "tolerance": str(CORRECTNESS_TOLERANCE),
        },
        "num_groups": len(records),
        "num_samples": sample_count,
        "elapsed_seconds": elapsed_seconds,
        "accuracy": correct_count / sample_count if sample_count else 0.0,
        "reward_mean": mean(reward_values) if reward_values else 0.0,
        "truncated_samples": truncated_count,
        "truncated_sample_rate": truncated_count / sample_count if sample_count else 0.0,
        "response_tokens_mean": mean(response_token_counts)
        if response_token_counts
        else 0.0,
        "response_tokens_p95": percentile(response_token_counts, 0.95),
        "pass_at_k": [
            {"k": metric.k, "pass_at_k": metric.pass_at_k}
            for metric in pass_metrics
        ],
    }


async def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("NCCL_IB_DISABLE", NCCL_IB_DISABLE)

    data_path = args.data_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_samples(data_path, args.max_samples)
    logger.info("Loaded %d eval samples from %s.", len(samples), data_path)

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=str(model_path),
        tokenizer=str(model_path),
        data_parallel_size=DATA_PARALLEL_SIZE,
        data_parallel_backend=DATA_PARALLEL_BACKEND,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=ENFORCE_EAGER,
        seed=SEED,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        n=1,
        max_tokens=MAX_RESPONSE_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        logprobs=LOGPROBS,
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
            grouped_outputs = await generate_batch(
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
                    "Processed %d/%d groups in %.1fs.",
                    len(records),
                    len(samples),
                    time.perf_counter() - started_at,
                )

        elapsed_seconds = time.perf_counter() - started_at
        summary = build_summary(
            data_path=data_path,
            model_path=model_path,
            records=records,
            requested_n=args.n,
            elapsed_seconds=elapsed_seconds,
        )
        result = {"summary": summary, "records": records}
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                result,
                output_file,
                ensure_ascii=False,
                indent=2 if args.pretty else None,
            )
            output_file.write("\n")

        logger.info(
            "Eval complete: groups=%d, samples=%d, accuracy=%.4f, "
            "reward_mean=%.4f, truncated=%.4f, response_tokens_mean=%.1f, "
            "response_tokens_p95=%.1f, time=%.1fs.",
            summary["num_groups"],
            summary["num_samples"],
            summary["accuracy"],
            summary["reward_mean"],
            summary["truncated_sample_rate"],
            summary["response_tokens_mean"],
            summary["response_tokens_p95"],
            summary["elapsed_seconds"],
        )
        for metric in summary["pass_at_k"]:
            logger.info("pass@%d=%.4f", metric["k"], metric["pass_at_k"])
        logger.info("Wrote eval results to %s.", output_path)
    finally:
        llm.shutdown(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone vLLM evaluation for preprocessed math JSONL data."
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n", type=int, default=SAMPLE_N)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--include-responses", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
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
