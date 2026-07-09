from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Awaitable, Mapping, Sequence

import yaml
from monarch.actor import Actor, ProcMesh, endpoint, shutdown_context

from tools.config import mapping
from tools.eval_metrics import compute_pass_at_k_range


class EnvSetter(Actor):
    @endpoint
    def set_env(self, env_vars: Mapping[str, str]) -> None:
        os.environ.update({str(key): str(value) for key, value in env_vars.items()})

async def set_mesh_environment(
    proc_mesh: ProcMesh, name: str, env_vars: Mapping[str, str]
) -> None:
    setter = proc_mesh.spawn(f"{name}_env", EnvSetter)
    await setter.set_env.call(env_vars)

def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


def init_wandb_run(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    logger: logging.Logger,
) -> Any | None:
    wandb_config = mapping(settings.get("wandb"), "monarch.wandb")
    if not bool(wandb_config.get("enable", False)):
        return None

    import wandb

    kwargs: dict[str, Any] = {
        "project": wandb_config.get("project", "nano-agentic-rl"),
        "name": wandb_config.get("name", config_path.stem),
        "config": {
            "config_path": str(config_path),
            "config": dict(config),
            "train_gpus": list(settings["train_gpus"]),
            "rollout_gpus": list(settings["rollout_gpus"]),
        },
    }
    for key in ("entity", "group", "mode", "dir"):
        if wandb_config.get(key) is not None:
            kwargs[key] = wandb_config[key]
    if wandb_config.get("tags") is not None:
        tags = wandb_config["tags"]
        kwargs["tags"] = [tags] if isinstance(tags, str) else list(tags)

    run = wandb.init(**kwargs)
    logger.info(
        "Initialized W&B run: project=%s name=%s.",
        kwargs["project"],
        kwargs["name"],
    )
    return run


def log_wandb(run: Any | None, metrics: Mapping[str, Any], *, step: int) -> None:
    if run is None or not metrics:
        return
    run.log(dict(metrics), step=step)


def float_metric(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_lengths(lengths: Sequence[int]) -> dict[str, float]:
    if not lengths:
        return {
            "response_tokens_mean": 0.0,
            "response_tokens_p95": 0.0,
            "response_tokens_max": 0.0,
        }
    sorted_lengths = sorted(int(length) for length in lengths)
    p95_index = min(len(sorted_lengths) - 1, (95 * len(sorted_lengths) + 99) // 100 - 1)
    return {
        "response_tokens_mean": sum(sorted_lengths) / len(sorted_lengths),
        "response_tokens_p95": float(sorted_lengths[p95_index]),
        "response_tokens_max": float(sorted_lengths[-1]),
    }


class RolloutMetricsAccumulator:
    """Collect rollout producer metrics between controller train steps."""

    COUNT_KEYS = (
        "requested_groups",
        "processed_groups",
        "accepted_groups",
        "dynamic_sampling/dropped_low_variance_groups",
        "dynamic_sampling/dropped_overlong_prompt_groups",
        "dynamic_sampling/dropped_overlong_response_groups",
        "dynamic_sampling/dropped_overlong_responses",
        "dynamic_sampling/dropped_insufficient_sample_groups",
        "generated_samples",
        "truncated_samples",
        "truncated_groups",
        "accepted_samples",
        "interrupted_groups",
        "retried_groups",
    )

    def __init__(self) -> None:
        self._totals = {key: 0.0 for key in self.COUNT_KEYS}
        self._time_sec = 0.0
        self._iterations = 0
        self._reward_mean_weighted = 0.0
        self._reward_std_weighted = 0.0
        self._reward_weight = 0.0
        self._response_token_lengths: list[int] = []

    def add(self, metrics: Mapping[str, Any]) -> None:
        self._iterations += 1
        for key in self.COUNT_KEYS:
            self._totals[key] += float_metric(metrics, key)
        self._time_sec += float_metric(metrics, "time_sec")

        reward_weight = float_metric(metrics, "accepted_groups")
        if reward_weight > 0:
            self._reward_mean_weighted += (
                float_metric(metrics, "reward_mean") * reward_weight
            )
            self._reward_std_weighted += (
                float_metric(metrics, "reward_std_mean") * reward_weight
            )
            self._reward_weight += reward_weight

        lengths = metrics.get("response_token_lengths", ())
        if isinstance(lengths, Sequence) and not isinstance(lengths, (str, bytes)):
            self._response_token_lengths.extend(int(length) for length in lengths)

    def snapshot(self) -> dict[str, float]:
        result = dict(self._totals)
        generated_samples = result["generated_samples"]
        accepted_samples = result["accepted_samples"]
        processed_groups = result["processed_groups"]
        accepted_groups = result["accepted_groups"]
        result.update(
            {
                "rollout_iterations": float(self._iterations),
                "truncated_sample_rate": (
                    result["truncated_samples"] / generated_samples
                    if generated_samples
                    else 0.0
                ),
                "effective_sample_rate": (
                    accepted_samples / generated_samples if generated_samples else 0.0
                ),
                "effective_group_rate": (
                    accepted_groups / processed_groups if processed_groups else 0.0
                ),
                "reward_mean": (
                    self._reward_mean_weighted / self._reward_weight
                    if self._reward_weight
                    else 0.0
                ),
                "reward_std_mean": (
                    self._reward_std_weighted / self._reward_weight
                    if self._reward_weight
                    else 0.0
                ),
                "time_sec": self._time_sec,
                **summarize_lengths(self._response_token_lengths),
            }
        )
        return result

    def drain(self) -> dict[str, float]:
        result = self.snapshot()
        self.__init__()
        return result


def core_eval_wandb_metrics(eval_metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "core/eval_accuracy": float_metric(eval_metrics, "accuracy"),
        "core/eval_reward_mean": float_metric(eval_metrics, "reward_mean"),
        "core/eval_time_sec": float_metric(eval_metrics, "time_sec"),
        "core/eval_truncated_sample_rate": float_metric(
            eval_metrics, "truncated_sample_rate"
        ),
        "core/eval_response_tokens_mean": float_metric(
            eval_metrics, "response_tokens_mean"
        ),
        "core/eval_response_tokens_p95": float_metric(
            eval_metrics, "response_tokens_p95"
        ),
    }
    pass_keys = []
    for key in eval_metrics:
        if not str(key).startswith("pass@"):
            continue
        try:
            pass_keys.append((int(str(key).split("@", 1)[1]), str(key)))
        except ValueError:
            continue
    if pass_keys:
        pass_keys.sort()
        pass_dict = dict(pass_keys)
        if 1 in pass_dict:
            result["core/eval_pass_at_1"] = float_metric(eval_metrics, pass_dict[1])
        max_k, max_key = pass_keys[-1]
        result[f"core/eval_pass_at_{max_k}"] = float_metric(eval_metrics, max_key)
    return result


def summarize_grpo_train_results(train_results: Mapping[Any, Any]) -> dict[str, float]:
    results = list(train_results.values())
    if not results:
        raise RuntimeError("No train actor results returned for GRPO step.")

    count = float(len(results))

    def mean(field: str) -> float:
        return sum(float(getattr(result, field)) for result in results) / count

    return {
        "ranks": count,
        "loss_mean": mean("loss"),
        "lr_mean": mean("learning_rate"),
        "grad_norm_mean": mean("grad_norm"),
        "grad_norm_max": max(float(result.grad_norm) for result in results),
        "approx_kl_mean": mean("approx_kl"),
        "ratio_mean": mean("ratio_mean"),
        "clip_fraction_mean": mean("clip_fraction"),
        "active_tokens_total": sum(float(result.active_tokens) for result in results),
        "elapsed_seconds_max": max(float(result.elapsed_seconds) for result in results),
        "memory_allocated_mean_mb": mean("memory_allocated_mb"),
        "memory_allocated_max_mb": max(
            float(result.memory_allocated_mb) for result in results
        ),
        "memory_reserved_mean_mb": mean("memory_reserved_mb"),
        "memory_reserved_max_mb": max(
            float(result.memory_reserved_mb) for result in results
        ),
        "max_memory_allocated_mean_mb": mean("max_memory_allocated_mb"),
        "max_memory_allocated_max_mb": max(
            float(result.max_memory_allocated_mb) for result in results
        ),
        "max_memory_reserved_mean_mb": mean("max_memory_reserved_mb"),
        "max_memory_reserved_max_mb": max(
            float(result.max_memory_reserved_mb) for result in results
        ),
    }


def summarize_sft_train_results(
    train_results: Mapping[Any, Any]
) -> dict[str, float | bool]:
    results = list(train_results.values())
    if not results:
        raise RuntimeError("No train actor results returned for SFT step.")

    active_results = [result for result in results if not result.finished]
    if not active_results:
        return {
            "finished": True,
            "ranks": float(len(results)),
            "step": float(max(int(result.step) for result in results)),
            "epoch": float(max(int(result.epoch) for result in results)),
            "loss_mean": float("nan"),
            "lr_mean": float("nan"),
            "grad_norm_mean": float("nan"),
            "grad_norm_max": float("nan"),
            "elapsed_seconds_max": 0.0,
            "memory_allocated_mean_mb": 0.0,
            "memory_allocated_max_mb": 0.0,
            "memory_reserved_mean_mb": 0.0,
            "memory_reserved_max_mb": 0.0,
            "max_memory_allocated_mean_mb": 0.0,
            "max_memory_allocated_max_mb": 0.0,
            "max_memory_reserved_mean_mb": 0.0,
            "max_memory_reserved_max_mb": 0.0,
        }

    count = float(len(active_results))

    def mean(field: str) -> float:
        return sum(float(getattr(result, field)) for result in active_results) / count

    return {
        "finished": all(bool(result.finished) for result in results),
        "ranks": float(len(results)),
        "step": float(max(int(result.step) for result in active_results)),
        "epoch": float(max(int(result.epoch) for result in active_results)),
        "loss_mean": mean("loss"),
        "lr_mean": mean("learning_rate"),
        "grad_norm_mean": mean("grad_norm"),
        "grad_norm_max": max(float(result.grad_norm) for result in active_results),
        "elapsed_seconds_max": max(
            float(result.elapsed_seconds) for result in active_results
        ),
        "memory_allocated_mean_mb": mean("memory_allocated_mb"),
        "memory_allocated_max_mb": max(
            float(result.memory_allocated_mb) for result in active_results
        ),
        "memory_reserved_mean_mb": mean("memory_reserved_mb"),
        "memory_reserved_max_mb": max(
            float(result.memory_reserved_mb) for result in active_results
        ),
        "max_memory_allocated_mean_mb": mean("max_memory_allocated_mb"),
        "max_memory_allocated_max_mb": max(
            float(result.max_memory_allocated_mb) for result in active_results
        ),
        "max_memory_reserved_mean_mb": mean("max_memory_reserved_mb"),
        "max_memory_reserved_max_mb": max(
            float(result.max_memory_reserved_mb) for result in active_results
        ),
    }


def grpo_train_wandb_metrics(train_summary: Mapping[str, float]) -> dict[str, float]:
    return {
        "train/ranks": train_summary["ranks"],
        "train/loss_mean": train_summary["loss_mean"],
        "train/lr_mean": train_summary["lr_mean"],
        "train/grad_norm_mean": train_summary["grad_norm_mean"],
        "train/grad_norm_max": train_summary["grad_norm_max"],
        "train/approx_kl_mean": train_summary["approx_kl_mean"],
        "train/ratio_mean": train_summary["ratio_mean"],
        "train/clip_fraction_mean": train_summary["clip_fraction_mean"],
        "train/active_tokens_total": train_summary["active_tokens_total"],
        "train/time_max_sec": train_summary["elapsed_seconds_max"],
        "train/memory_allocated_mean_mb": train_summary["memory_allocated_mean_mb"],
        "train/memory_allocated_max_mb": train_summary["memory_allocated_max_mb"],
        "train/memory_reserved_mean_mb": train_summary["memory_reserved_mean_mb"],
        "train/memory_reserved_max_mb": train_summary["memory_reserved_max_mb"],
        "train/max_memory_allocated_mean_mb": train_summary[
            "max_memory_allocated_mean_mb"
        ],
        "train/max_memory_allocated_max_mb": train_summary[
            "max_memory_allocated_max_mb"
        ],
        "train/max_memory_reserved_mean_mb": train_summary[
            "max_memory_reserved_mean_mb"
        ],
        "train/max_memory_reserved_max_mb": train_summary[
            "max_memory_reserved_max_mb"
        ],
    }


def sft_train_wandb_metrics(train_summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        "train/ranks": float_metric(train_summary, "ranks"),
        "train/loss_mean": float_metric(train_summary, "loss_mean"),
        "train/lr_mean": float_metric(train_summary, "lr_mean"),
        "train/grad_norm_mean": float_metric(train_summary, "grad_norm_mean"),
        "train/grad_norm_max": float_metric(train_summary, "grad_norm_max"),
        "train/time_max_sec": float_metric(train_summary, "elapsed_seconds_max"),
        "train/memory_allocated_mean_mb": float_metric(
            train_summary, "memory_allocated_mean_mb"
        ),
        "train/memory_allocated_max_mb": float_metric(
            train_summary, "memory_allocated_max_mb"
        ),
        "train/memory_reserved_mean_mb": float_metric(
            train_summary, "memory_reserved_mean_mb"
        ),
        "train/memory_reserved_max_mb": float_metric(
            train_summary, "memory_reserved_max_mb"
        ),
        "train/max_memory_allocated_mean_mb": float_metric(
            train_summary, "max_memory_allocated_mean_mb"
        ),
        "train/max_memory_allocated_max_mb": float_metric(
            train_summary, "max_memory_allocated_max_mb"
        ),
        "train/max_memory_reserved_mean_mb": float_metric(
            train_summary, "max_memory_reserved_mean_mb"
        ),
        "train/max_memory_reserved_max_mb": float_metric(
            train_summary, "max_memory_reserved_max_mb"
        ),
    }


def grpo_core_step_wandb_metrics(
    *,
    policy_version: int,
    train_summary: Mapping[str, Any],
    rollout_metrics: Mapping[str, Any],
    sync_time_sec: float,
) -> dict[str, float]:
    return {
        "core/policy_version": float(policy_version),
        "core/train_loss": float_metric(train_summary, "loss_mean"),
        "core/train_lr": float_metric(train_summary, "lr_mean"),
        "core/train_grad_norm": float_metric(train_summary, "grad_norm_mean"),
        "core/train_approx_kl": float_metric(train_summary, "approx_kl_mean"),
        "core/train_active_tokens": float_metric(
            train_summary, "active_tokens_total"
        ),
        "core/train_time_sec": float_metric(train_summary, "elapsed_seconds_max"),
        "core/rollout_effective_group_rate": float_metric(
            rollout_metrics, "effective_group_rate"
        ),
        "core/rollout_reward_mean": float_metric(rollout_metrics, "reward_mean"),
        "core/rollout_truncated_sample_rate": float_metric(
            rollout_metrics, "truncated_sample_rate"
        ),
        "core/rollout_response_tokens_mean": float_metric(
            rollout_metrics, "response_tokens_mean"
        ),
        "core/rollout_response_tokens_p95": float_metric(
            rollout_metrics, "response_tokens_p95"
        ),
        "core/rollout_time_sec": float_metric(rollout_metrics, "time_sec"),
        "core/sync_time_sec": float(sync_time_sec),
    }


def sft_core_step_wandb_metrics(
    *,
    policy_version: int,
    train_summary: Mapping[str, Any],
    sync_time_sec: float,
) -> dict[str, float]:
    return {
        "core/policy_version": float(policy_version),
        "core/train_loss": float_metric(train_summary, "loss_mean"),
        "core/train_lr": float_metric(train_summary, "lr_mean"),
        "core/train_grad_norm": float_metric(train_summary, "grad_norm_mean"),
        "core/train_time_sec": float_metric(train_summary, "elapsed_seconds_max"),
        "core/sync_time_sec": float(sync_time_sec),
    }


def sft_core_train_wandb_metrics(
    *, policy_version: int, train_summary: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "core/policy_version": float(policy_version),
        "core/train_loss": float_metric(train_summary, "loss_mean"),
        "core/train_lr": float_metric(train_summary, "lr_mean"),
        "core/train_grad_norm": float_metric(train_summary, "grad_norm_mean"),
        "core/train_time_sec": float_metric(train_summary, "elapsed_seconds_max"),
    }


def reserve_local_port() -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", int(sock.getsockname()[1])


async def run_eval(
    *,
    eval_dataset: Any,
    reward_actor: Any,
    rollout_actor: Any,
    batch_size: int,
    epochs: int,
    sampling_params: Mapping[str, Any],
    step: int | None,
    logger: logging.Logger,
) -> dict[str, float]:
    started_at = time.perf_counter()
    correctness_groups: list[list[float]] = []
    reward_groups: list[list[float]] = []
    truncated_samples = 0
    response_token_lengths: list[int] = []
    for _ in range(epochs):
        eval_batches = await eval_dataset.all_batches.call_one(batch_size)
        for dataset_samples in eval_batches:
            rollout_outputs = await rollout_actor.chat.call_one(
                [sample.messages for sample in dataset_samples],
                sampling_params=dict(sampling_params),
            )
            if len(dataset_samples) != len(rollout_outputs):
                raise RuntimeError("Eval dataset and rollout batch sizes do not match.")

            for dataset_sample, rollout_output in zip(dataset_samples, rollout_outputs):
                response_token_lengths.extend(
                    len(sample.token_ids) for sample in rollout_output.samples
                )
                truncated_samples += sum(
                    1
                    for sample in rollout_output.samples
                    if sample.finish_reason == "length"
                )
                reward_results = await reward_actor.evaluate_batch.call_one(
                    [sample.text for sample in rollout_output.samples],
                    [dataset_sample.target] * len(rollout_output.samples),
                )
                correctness_groups.append(
                    [
                        float(result.breakdown["correctness"])
                        for result in reward_results
                    ]
                )
                reward_groups.append([float(result.reward) for result in reward_results])

    logger.info(
        "Eval %s processed %d sample group(s) across %d epoch(s).",
        "baseline" if step is None else f"step {step}",
        len(correctness_groups),
        epochs,
    )
    max_k = int(sampling_params.get("n", 1))
    metrics = compute_pass_at_k_range(correctness_groups, max_k)
    step_label = "baseline" if step is None else f"step {step}"
    num_samples = float(sum(len(group) for group in correctness_groups))
    accuracy = (
        sum(sum(group) for group in correctness_groups) / num_samples
        if num_samples
        else 0.0
    )
    reward_sample_count = float(sum(len(group) for group in reward_groups))
    reward_mean = (
        sum(sum(group) for group in reward_groups) / reward_sample_count
        if reward_sample_count
        else 0.0
    )
    truncated_sample_rate = truncated_samples / num_samples if num_samples else 0.0
    response_length_metrics = summarize_lengths(response_token_lengths)
    logger.info(
        "Eval %s: accuracy=%.4f, reward_mean=%.4f, truncated=%.4f, "
        "response_len_mean=%.1f, response_len_p95=%.0f.",
        step_label,
        accuracy,
        reward_mean,
        truncated_sample_rate,
        response_length_metrics["response_tokens_mean"],
        response_length_metrics["response_tokens_p95"],
    )
    for metric in metrics:
        logger.info("Eval %s: pass@%d=%.4f.", step_label, metric.k, metric.pass_at_k)

    result = {
        "time_sec": time.perf_counter() - started_at,
        "num_groups": float(len(correctness_groups)),
        "num_samples": num_samples,
        "accuracy": accuracy,
        "reward_mean": reward_mean,
        "truncated_samples": float(truncated_samples),
        "truncated_sample_rate": truncated_sample_rate,
        **response_length_metrics,
    }
    for metric in metrics:
        result[f"pass@{metric.k}"] = metric.pass_at_k
    return result


async def sync_policy_weights(
    *,
    rollout_actor: Any,
    train_actors: Any,
    weight_metadata: Mapping[str, Any],
    version: int,
    allow_same_version: bool = False,
) -> Any:
    update_result, _ = await asyncio.gather(
        rollout_actor.receive_weights.call_one(
            weight_metadata,
            version=version,
            allow_same_version=allow_same_version,
        ),
        train_actors.broadcast_weights.call(),
    )
    return update_result


async def close_resources(
    *,
    rollout_actor: Any | None,
    train_actors: Any | None,
    proc_meshes: Sequence[ProcMesh],
    timeout: float,
    logger: logging.Logger,
) -> None:
    phase_timeout = max(timeout / 3.0, 1.0)

    async def run_phase(name: str, awaitables: Sequence[Awaitable[Any]]) -> None:
        if not awaitables:
            return
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*awaitables, return_exceptions=True),
                timeout=phase_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("%s exceeded %.1fs.", name, phase_timeout)
            return
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("%s failed: %r", name, result)

    close_calls: list[Awaitable[Any]] = []
    if rollout_actor is not None:
        close_calls.append(rollout_actor.close.call_one())
    if train_actors is not None:
        close_calls.append(train_actors.close.call())
    await run_phase("Actor close", close_calls)

    stop_calls = [
        proc_mesh.stop("Controller shutdown")
        for proc_mesh in reversed(proc_meshes)
    ]
    await run_phase("Process mesh stop", stop_calls)

    try:
        await asyncio.wait_for(shutdown_context(), timeout=phase_timeout)
    except asyncio.TimeoutError:
        logger.warning("Monarch context shutdown exceeded %.1fs.", phase_timeout)
    except Exception:
        logger.exception("Failed to shut down Monarch context.")
