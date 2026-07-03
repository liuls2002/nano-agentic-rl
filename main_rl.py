from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Mapping, Sequence

import yaml
from monarch.actor import (
    Actor,
    ProcMesh,
    endpoint,
    shutdown_context,
    this_host,
)
from monarch.spmd import setup_torch_elastic_env_async

from actor.dataset_actor import DatasetActor
from actor.replay_buffer_actor import ReplayBufferActor
from actor.reward_advantage_actor import AdvantageActor, RewardActor
from actor.rollout_actor import RolloutActor, RolloutOutput
from actor.train_actor import TrainActor
from rl.types import DatasetSample, RLEpisode
from tools.eval_metrics import compute_pass_at_k_range
from tools.runtime_metrics import GpuMonitor, prefix_metrics


logger = logging.getLogger("main_rl")
DEFAULT_CONFIG = Path(__file__).parent / "config" / "qwen2_5_1_5b_gsm8k_grpo.yaml"
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 15.0


class EnvSetter(Actor):
    @endpoint
    def set_env(self, env_vars: Mapping[str, str]) -> None:
        os.environ.update({str(key): str(value) for key, value in env_vars.items()})


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _load_sequence_config(monarch: Mapping[str, Any]) -> dict[str, int]:
    sequence = _mapping(monarch.get("sequence"), "monarch.sequence")
    prompt_length = _positive_int(
        sequence.get("max_prompt_tokens", 1024),
        "monarch.sequence.max_prompt_tokens",
    )
    response_length = _positive_int(
        sequence.get("max_response_tokens", 1024),
        "monarch.sequence.max_response_tokens",
    )
    return {
        "max_prompt_tokens": prompt_length,
        "max_response_tokens": response_length,
        "max_seq_len": prompt_length + response_length,
    }


def split_worker_gpus(
    train_num_gpus: int, rollout_num_gpus: int
) -> tuple[list[str], list[str]]:
    required_gpus = train_num_gpus + rollout_num_gpus
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        gpu_pool = [
            device.strip()
            for device in visible_devices.split(",")
            if device.strip()
        ]
    else:
        gpu_pool = [str(gpu_id) for gpu_id in range(required_gpus)]

    if len(gpu_pool) < required_gpus:
        raise ValueError(
            "Not enough visible GPUs for train_actor.num_gpus + "
            f"rollout_actor.num_gpus ({required_gpus} required, "
            f"{len(gpu_pool)} visible from CUDA_VISIBLE_DEVICES="
            f"{visible_devices!r})."
        )
    return (
        gpu_pool[:train_num_gpus],
        gpu_pool[train_num_gpus:required_gpus],
    )


def summarize_train_results(train_results: Mapping[Any, Any]) -> dict[str, float]:
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


def validate_rl_config(config: Mapping[str, Any]) -> dict[str, Any]:
    monarch = _mapping(config.get("monarch"), "monarch")
    wandb_config = _mapping(monarch.get("wandb"), "monarch.wandb")
    sequence = _load_sequence_config(monarch)
    rl_config = _mapping(config.get("rl"), "rl")
    train_actor = _mapping(config.get("train_actor"), "train_actor")
    rollout_actor = _mapping(config.get("rollout_actor"), "rollout_actor")
    train_config = _mapping(train_actor.get("train"), "train_actor.train")
    dataloader = _mapping(config.get("dataloader"), "dataloader")
    train_data = _mapping(dataloader.get("train"), "dataloader.train")
    eval_data = _mapping(dataloader.get("eval"), "dataloader.eval")
    rollout_config = _mapping(rollout_actor.get("rollout"), "rollout_actor.rollout")
    eval_config = _mapping(rollout_actor.get("eval"), "rollout_actor.eval")
    rollout_sampling = _mapping(
        rollout_config.get("sampling"), "rollout_actor.rollout.sampling"
    )
    eval_sampling = _mapping(
        eval_config.get("sampling"), "rollout_actor.eval.sampling"
    )
    engine = _mapping(rollout_actor.get("engine"), "rollout_actor.engine")

    train_num_gpus = _positive_int(train_actor.get("num_gpus", 0), "train_actor.num_gpus")
    rollout_num_gpus = _positive_int(
        rollout_actor.get("num_gpus", 0), "rollout_actor.num_gpus"
    )
    train_gpus, rollout_gpus = split_worker_gpus(train_num_gpus, rollout_num_gpus)

    for name, data_config in (
        ("dataloader.train", train_data),
        ("dataloader.eval", eval_data),
    ):
        path = data_config.get("path")
        if not path:
            raise ValueError(f"{name}.path is required.")
        if not Path(str(path)).expanduser().is_file():
            raise FileNotFoundError(f"{name}.path does not exist: {path}")

    prompt_length = sequence["max_prompt_tokens"]
    response_length = sequence["max_response_tokens"]
    max_seq_len = sequence["max_seq_len"]
    train_max_seq_len = int(train_data.get("max_seq_len", max_seq_len))
    if train_max_seq_len < max_seq_len:
        raise ValueError(
            "dataloader.train.max_seq_len must be at least "
            "monarch.sequence.max_prompt_tokens + "
            f"max_response_tokens ({train_max_seq_len} vs {max_seq_len})."
        )
    rollout_max_tokens = int(rollout_sampling.get("max_tokens", response_length))
    eval_max_tokens = int(eval_sampling.get("max_tokens", response_length))
    if response_length < rollout_max_tokens:
        raise ValueError(
            "monarch.sequence.max_response_tokens must cover "
            "rollout_actor.rollout.sampling.max_tokens."
        )
    if response_length < eval_max_tokens:
        raise ValueError(
            "monarch.sequence.max_response_tokens must cover "
            "rollout_actor.eval.sampling.max_tokens."
        )
    engine_max_model_len = int(engine.get("max_model_len", max_seq_len))
    if max_seq_len > engine_max_model_len:
        raise ValueError(
            "rollout_actor.engine.max_model_len must be at least "
            "monarch.sequence.max_prompt_tokens + "
            f"max_response_tokens ({engine_max_model_len} vs {max_seq_len})."
        )
    if int(rollout_sampling.get("n", 1)) < 2:
        raise ValueError("GRPO requires rollout sampling.n >= 2.")
    if int(eval_sampling.get("n", 1)) <= 0:
        raise ValueError("eval sampling.n must be positive.")

    configured_env = _mapping(monarch.get("env"), "monarch.env")
    shutdown_timeout = float(
        monarch.get("shutdown_timeout_seconds", DEFAULT_SHUTDOWN_TIMEOUT_SECONDS)
    )
    if shutdown_timeout <= 0:
        raise ValueError("monarch.shutdown_timeout_seconds must be positive.")
    max_steps = int(rl_config.get("max_steps", 1))
    if max_steps <= 0:
        raise ValueError("rl.max_steps must be positive.")
    rollout_batch_size = int(rl_config.get("rollout_batch_size", 1))
    rollout_batch_size_multiplier = float(
        rl_config.get("rollout_batch_size_multiplier", 1.0)
    )
    max_rollout_groups = int(rl_config.get("max_rollout_groups_per_step", 16))
    if rollout_batch_size <= 0 or max_rollout_groups <= 0:
        raise ValueError("RL rollout batch size and group limit must be positive.")
    if rollout_batch_size_multiplier <= 0:
        raise ValueError("rl.rollout_batch_size_multiplier must be positive.")
    dynamic_sampling = _mapping(
        rl_config.get("dynamic_sampling"), "rl.dynamic_sampling"
    )
    dynamic_sampling_enabled = bool(dynamic_sampling.get("enabled", True))
    configured_global_batch = int(train_config.get("global_batch_size", 0))
    if configured_global_batch <= 0:
        raise ValueError("train_actor.train.global_batch_size must be positive.")
    if configured_global_batch % train_num_gpus:
        raise ValueError(
            "train_actor.train.global_batch_size must be divisible by "
            f"train_actor.num_gpus ({configured_global_batch} vs {train_num_gpus})."
        )
    expected_global_batch = configured_global_batch
    samples_per_prompt = int(rollout_sampling.get("n", 1))
    if expected_global_batch % samples_per_prompt:
        raise ValueError(
            "train_actor.train.global_batch_size must be divisible by rollout sampling.n "
            f"({expected_global_batch} vs {samples_per_prompt})."
        )
    weight_sync = _mapping(rl_config.get("weight_sync"), "rl.weight_sync")
    if weight_sync.get("backend", "nccl") != "nccl":
        raise ValueError(
            "This RL controller currently requires weight_sync.backend=nccl."
        )
    if int(weight_sync.get("packed_buffer_size_bytes", 0)) <= 0:
        raise ValueError("weight_sync.packed_buffer_size_bytes must be positive.")
    if int(weight_sync.get("packed_num_buffers", 0)) <= 0:
        raise ValueError("weight_sync.packed_num_buffers must be positive.")
    transfer_config = _mapping(
        engine.get("weight_transfer_config"),
        "rollout_actor.engine.weight_transfer_config",
    )
    if transfer_config.get("backend") != "nccl":
        raise ValueError(
            "rollout_actor.engine.weight_transfer_config.backend must be nccl."
        )
    eval_steps = _positive_int(
        eval_config.get("eval_steps", 1), "rollout_actor.eval.eval_steps"
    )
    eval_epochs = _positive_int(
        eval_config.get("eval_epochs", 1), "rollout_actor.eval.eval_epochs"
    )
    eval_batch_size = _positive_int(
        eval_config.get("batch_size", 1), "rollout_actor.eval.batch_size"
    )
    return {
        "train_gpus": train_gpus,
        "rollout_gpus": rollout_gpus,
        "worker_env": {str(key): str(value) for key, value in configured_env.items()},
        "sequence": sequence,
        "wandb": wandb_config,
        "gpu_sample_interval_seconds": float(
            wandb_config.get("gpu_sample_interval_seconds", 1.0)
        ),
        "shutdown_timeout": shutdown_timeout,
        "max_steps": max_steps,
        "rollout_batch_size": rollout_batch_size,
        "rollout_batch_size_multiplier": rollout_batch_size_multiplier,
        "max_rollout_groups_per_step": max_rollout_groups,
        "train_batch_episode_count": expected_global_batch,
        "samples_per_prompt": samples_per_prompt,
        "max_prompt_tokens": prompt_length,
        "max_response_tokens": response_length,
        "dynamic_sampling_enabled": dynamic_sampling_enabled,
        "dynamic_sampling_filter_low_variance": dynamic_sampling_enabled
        and bool(dynamic_sampling.get("filter_low_variance_groups", True)),
        "dynamic_sampling_filter_overlong_prompts": dynamic_sampling_enabled
        and bool(dynamic_sampling.get("filter_overlong_prompts", True)),
        "dynamic_sampling_filter_overlong_responses": dynamic_sampling_enabled
        and bool(dynamic_sampling.get("filter_overlong_responses", True)),
        "eval_steps": eval_steps,
        "eval_epochs": eval_epochs,
        "eval_batch_size": eval_batch_size,
        "eval_sampling": eval_sampling,
    }


def init_wandb_run(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> Any | None:
    wandb_config = _mapping(settings.get("wandb"), "monarch.wandb")
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
    logger.info("Initialized W&B run: project=%s name=%s.", kwargs["project"], kwargs["name"])
    return run


def log_wandb(run: Any | None, metrics: Mapping[str, Any], *, step: int) -> None:
    if run is None or not metrics:
        return
    run.log(dict(metrics), step=step)


def train_wandb_metrics(train_summary: Mapping[str, float]) -> dict[str, float]:
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


def _float_metric(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
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
    p95_index = min(
        len(sorted_lengths) - 1,
        max(0, math.ceil(0.95 * len(sorted_lengths)) - 1),
    )
    return {
        "response_tokens_mean": sum(sorted_lengths) / len(sorted_lengths),
        "response_tokens_p95": float(sorted_lengths[p95_index]),
        "response_tokens_max": float(sorted_lengths[-1]),
    }


def core_eval_wandb_metrics(eval_metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "core/eval_accuracy": _float_metric(eval_metrics, "accuracy"),
        "core/eval_reward_mean": _float_metric(eval_metrics, "reward_mean"),
        "core/eval_time_sec": _float_metric(eval_metrics, "time_sec"),
        "core/eval_truncated_sample_rate": _float_metric(
            eval_metrics, "truncated_sample_rate"
        ),
        "core/eval_response_tokens_mean": _float_metric(
            eval_metrics, "response_tokens_mean"
        ),
        "core/eval_response_tokens_p95": _float_metric(
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
            result["core/eval_pass_at_1"] = _float_metric(
                eval_metrics, pass_dict[1]
            )
        max_k, max_key = pass_keys[-1]
        result[f"core/eval_pass_at_{max_k}"] = _float_metric(eval_metrics, max_key)
    return result


def core_step_wandb_metrics(
    *,
    policy_version: int,
    train_summary: Mapping[str, Any],
    rollout_metrics: Mapping[str, Any],
    sync_time_sec: float,
) -> dict[str, float]:
    return {
        "core/policy_version": float(policy_version),
        "core/train_loss": _float_metric(train_summary, "loss_mean"),
        "core/train_lr": _float_metric(train_summary, "lr_mean"),
        "core/train_grad_norm": _float_metric(train_summary, "grad_norm_mean"),
        "core/train_approx_kl": _float_metric(train_summary, "approx_kl_mean"),
        "core/train_active_tokens": _float_metric(
            train_summary, "active_tokens_total"
        ),
        "core/train_time_sec": _float_metric(
            train_summary, "elapsed_seconds_max"
        ),
        "core/rollout_effective_group_rate": _float_metric(
            rollout_metrics, "effective_group_rate"
        ),
        "core/rollout_reward_mean": _float_metric(rollout_metrics, "reward_mean"),
        "core/rollout_truncated_sample_rate": _float_metric(
            rollout_metrics, "truncated_sample_rate"
        ),
        "core/rollout_response_tokens_mean": _float_metric(
            rollout_metrics, "response_tokens_mean"
        ),
        "core/rollout_response_tokens_p95": _float_metric(
            rollout_metrics, "response_tokens_p95"
        ),
        "core/rollout_time_sec": _float_metric(rollout_metrics, "time_sec"),
        "core/sync_time_sec": float(sync_time_sec),
    }


def reserve_local_port() -> tuple[str, int]:
    """Reserve a single-node rendezvous address for the NCCL transfer group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", int(sock.getsockname()[1])


async def _set_mesh_environment(
    proc_mesh: ProcMesh, name: str, env_vars: Mapping[str, str]
) -> None:
    setter = proc_mesh.spawn(f"{name}_env", EnvSetter)
    await setter.set_env.call(env_vars)


def build_episodes(
    dataset_sample: DatasetSample,
    rollout_output: RolloutOutput,
    rewards: Sequence[Any],
    advantages: Sequence[float],
) -> list[RLEpisode]:
    if len(rollout_output.samples) != len(rewards) or len(rewards) != len(advantages):
        raise ValueError("Rollout samples, rewards, and advantages must align.")
    episodes = []
    for sample, reward, advantage in zip(
        rollout_output.samples, rewards, advantages
    ):
        if sample.logprobs is None:
            raise ValueError(
                "Rollout sample has no token logprobs; set rollout sampling.logprobs."
            )
        episodes.append(
            RLEpisode(
                episode_id=str(uuid.uuid4()),
                sample_id=dataset_sample.sample_id,
                prompt=rollout_output.prompt,
                target=dataset_sample.target,
                response=sample.text,
                prompt_token_ids=rollout_output.prompt_token_ids,
                response_token_ids=sample.token_ids,
                generator_logprobs=sample.logprobs,
                reward=float(reward.reward),
                reward_breakdown=dict(reward.breakdown),
                advantage=float(advantage),
                policy_version=rollout_output.policy_version,
                finish_reason=sample.finish_reason,
            )
        )
    return episodes


async def fill_replay_buffer(
    *,
    dataset: Any,
    reward_actor: Any,
    advantage_actor: Any,
    replay_buffer: Any,
    rollout_actor: Any,
    current_policy_version: int,
    rollout_batch_size: int,
    rollout_batch_size_multiplier: float,
    max_groups: int,
    train_batch_episode_count: int,
    samples_per_prompt: int,
    max_prompt_tokens: int,
    max_response_tokens: int,
    dynamic_sampling_filter_low_variance: bool,
    dynamic_sampling_filter_overlong_prompts: bool,
    dynamic_sampling_filter_overlong_responses: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    started_at = time.perf_counter()
    requested_groups = 0
    groups_processed = 0
    accepted_groups = 0
    dynamic_sampling_dropped_low_variance_groups = 0
    dynamic_sampling_dropped_overlong_prompt_groups = 0
    dynamic_sampling_dropped_overlong_response_groups = 0
    dynamic_sampling_dropped_overlong_responses = 0
    dynamic_sampling_dropped_insufficient_sample_groups = 0
    generated_samples = 0
    truncated_samples = 0
    truncated_groups = 0
    accepted_samples = 0
    response_token_lengths: list[int] = []
    reward_means: list[float] = []
    reward_stds: list[float] = []

    def metrics() -> dict[str, float]:
        return {
            "requested_groups": float(requested_groups),
            "processed_groups": float(groups_processed),
            "accepted_groups": float(accepted_groups),
            "dynamic_sampling/dropped_low_variance_groups": float(
                dynamic_sampling_dropped_low_variance_groups
            ),
            "dynamic_sampling/dropped_overlong_prompt_groups": float(
                dynamic_sampling_dropped_overlong_prompt_groups
            ),
            "dynamic_sampling/dropped_overlong_response_groups": float(
                dynamic_sampling_dropped_overlong_response_groups
            ),
            "dynamic_sampling/dropped_overlong_responses": float(
                dynamic_sampling_dropped_overlong_responses
            ),
            "dynamic_sampling/dropped_insufficient_sample_groups": float(
                dynamic_sampling_dropped_insufficient_sample_groups
            ),
            "generated_samples": float(generated_samples),
            "truncated_samples": float(truncated_samples),
            "truncated_groups": float(truncated_groups),
            "truncated_sample_rate": (
                truncated_samples / generated_samples
                if generated_samples
                else 0.0
            ),
            "accepted_samples": float(accepted_samples),
            "effective_sample_rate": (
                accepted_samples / generated_samples if generated_samples else 0.0
            ),
            "effective_group_rate": (
                accepted_groups / groups_processed if groups_processed else 0.0
            ),
            "reward_mean": (
                sum(reward_means) / len(reward_means) if reward_means else 0.0
            ),
            "reward_std_mean": (
                sum(reward_stds) / len(reward_stds) if reward_stds else 0.0
            ),
            "time_sec": time.perf_counter() - started_at,
            **summarize_lengths(response_token_lengths),
        }

    while groups_processed < max_groups:
        batch = await replay_buffer.sample.call_one(current_policy_version)
        if batch is not None:
            return batch, metrics()

        status = await replay_buffer.get_status.call_one()
        groups_remaining = max_groups - groups_processed
        if groups_processed == 0:
            target_groups = rollout_batch_size
        else:
            missing_episodes = max(
                train_batch_episode_count - int(status.size), 0
            )
            target_groups = max(
                1, math.ceil(missing_episodes / samples_per_prompt)
            )
        planned_groups = min(
            groups_remaining,
            max(1, math.ceil(target_groups * rollout_batch_size_multiplier)),
        )
        logger.info(
            "Requesting %d rollout group(s): target=%d, multiplier=%.2f, "
            "buffer=%d/%d, processed=%d/%d.",
            planned_groups,
            target_groups,
            rollout_batch_size_multiplier,
            status.size,
            train_batch_episode_count,
            groups_processed,
            max_groups,
        )
        requested_groups += planned_groups

        dataset_samples = await dataset.next_batch.call_one(planned_groups)
        if dynamic_sampling_filter_overlong_prompts:
            prompt_lengths = await dataset.count_tokens.call_one(
                [sample.messages for sample in dataset_samples]
            )
            if len(prompt_lengths) != len(dataset_samples):
                raise RuntimeError("Dataset prompt token counts do not match batch size.")
            prompt_filtered_samples = []
            for dataset_sample, prompt_length in zip(
                dataset_samples, prompt_lengths
            ):
                if groups_processed >= max_groups:
                    break
                if int(prompt_length) > max_prompt_tokens:
                    groups_processed += 1
                    dynamic_sampling_dropped_overlong_prompt_groups += 1
                    logger.info(
                        "Dynamic sampling dropped group %s before rollout: "
                        "prompt length %d > %d.",
                        dataset_sample.sample_id,
                        int(prompt_length),
                        max_prompt_tokens,
                    )
                    continue
                prompt_filtered_samples.append(dataset_sample)
            dataset_samples = prompt_filtered_samples
            if not dataset_samples:
                continue

        rollout_outputs = await rollout_actor.chat.call_one(
            [sample.messages for sample in dataset_samples]
        )
        if len(dataset_samples) != len(rollout_outputs):
            raise RuntimeError("Dataset and rollout batch sizes do not match.")

        for dataset_sample, rollout_output in zip(dataset_samples, rollout_outputs):
            groups_processed += 1
            generated_samples += len(rollout_output.samples)
            response_token_lengths.extend(
                len(sample.token_ids) for sample in rollout_output.samples
            )
            truncated_in_group = sum(
                1
                for sample in rollout_output.samples
                if sample.finish_reason == "length"
            )
            truncated_samples += truncated_in_group
            if truncated_in_group:
                truncated_groups += 1
            if (
                dynamic_sampling_filter_overlong_prompts
                and len(rollout_output.prompt_token_ids) > max_prompt_tokens
            ):
                dynamic_sampling_dropped_overlong_prompt_groups += 1
                logger.info(
                    "Dynamic sampling dropped group %s: prompt length %d > %d.",
                    dataset_sample.sample_id,
                    len(rollout_output.prompt_token_ids),
                    max_prompt_tokens,
                )
                continue

            valid_samples = []
            for sample in rollout_output.samples:
                overlong_response = (
                    len(sample.token_ids) > max_response_tokens
                    or sample.finish_reason == "length"
                )
                if dynamic_sampling_filter_overlong_responses and overlong_response:
                    dynamic_sampling_dropped_overlong_responses += 1
                    continue
                valid_samples.append(sample)
            if len(valid_samples) != len(rollout_output.samples):
                dynamic_sampling_dropped_overlong_response_groups += 1
            if len(valid_samples) < 2:
                dynamic_sampling_dropped_insufficient_sample_groups += 1
                logger.info(
                    "Dynamic sampling dropped group %s: only %d valid response(s) after length filtering.",
                    dataset_sample.sample_id,
                    len(valid_samples),
                )
                continue

            reward_results = await reward_actor.evaluate_batch.call_one(
                [sample.text for sample in valid_samples],
                [dataset_sample.target] * len(valid_samples),
            )
            advantage_result = await advantage_actor.compute.call_one(
                [result.reward for result in reward_results]
            )
            reward_means.append(float(advantage_result.reward_mean))
            reward_stds.append(float(advantage_result.reward_std))
            if advantage_result.low_variance and dynamic_sampling_filter_low_variance:
                dynamic_sampling_dropped_low_variance_groups += 1
                logger.info(
                    "Dynamic sampling dropped low-variance group %s (std=%.6f).",
                    dataset_sample.sample_id,
                    advantage_result.reward_std,
                )
                continue

            filtered_rollout_output = RolloutOutput(
                prompt=rollout_output.prompt,
                prompt_token_ids=rollout_output.prompt_token_ids,
                samples=valid_samples,
                policy_version=rollout_output.policy_version,
                num_cached_tokens=rollout_output.num_cached_tokens,
            )
            episodes = build_episodes(
                dataset_sample,
                filtered_rollout_output,
                reward_results,
                advantage_result.advantages,
            )
            accepted_groups += 1
            accepted_samples += len(episodes)
            status = await replay_buffer.add.call_one(episodes)
            logger.info(
                "Rollout group %s: mean reward=%.3f, std=%.3f, buffer=%d.",
                dataset_sample.sample_id,
                advantage_result.reward_mean,
                advantage_result.reward_std,
                status.size,
            )
            if groups_processed >= max_groups:
                break

    batch = await replay_buffer.sample.call_one(current_policy_version)
    if batch is None:
        raise RuntimeError(
            "Replay buffer did not reach a full training batch within "
            f"{max_groups} rollout groups."
        )
    return batch, metrics()


async def run_eval(
    *,
    eval_dataset: Any,
    reward_actor: Any,
    rollout_actor: Any,
    batch_size: int,
    epochs: int,
    sampling_params: Mapping[str, Any],
    step: int | None,
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
                raise RuntimeError(
                    "Eval dataset and rollout batch sizes do not match."
                )

            for dataset_sample, rollout_output in zip(
                dataset_samples, rollout_outputs
            ):
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
    truncated_sample_rate = (
        truncated_samples / num_samples if num_samples else 0.0
    )
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
        logger.info(
            "Eval %s: pass@%d=%.4f.",
            step_label,
            metric.k,
            metric.pass_at_k,
        )
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
        proc_mesh.stop("RL controller shutdown")
        for proc_mesh in reversed(proc_meshes)
    ]
    await run_phase("Process mesh stop", stop_calls)

    try:
        await asyncio.wait_for(shutdown_context(), timeout=phase_timeout)
    except asyncio.TimeoutError:
        logger.warning("Monarch context shutdown exceeded %.1fs.", phase_timeout)
    except Exception:
        logger.exception("Failed to shut down Monarch context.")


async def run_rl(config_path: Path) -> None:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    settings = validate_rl_config(config)
    wandb_run = init_wandb_run(
        config_path=config_path,
        config=config,
        settings=settings,
    )
    proc_meshes: list[ProcMesh] = []
    train_actors = None
    rollout_actor = None

    try:
        host = this_host()
        support_actors = {}
        actor_specs = (
            ("train_dataset", DatasetActor, ("train",)),
            ("eval_dataset", DatasetActor, ("eval",)),
            ("reward", RewardActor, ()),
            ("advantage", AdvantageActor, ()),
            ("replay_buffer", ReplayBufferActor, ()),
        )
        for name, actor_type, actor_args in actor_specs:
            proc_mesh = host.spawn_procs(
                per_host={"procs": 1}, name=f"rl_{name}_procs"
            )
            proc_meshes.append(proc_mesh)
            support_actors[name] = proc_mesh.spawn(
                f"rl_{name}", actor_type, str(config_path), *actor_args
            )

        train_dataset = support_actors["train_dataset"]
        eval_dataset = support_actors["eval_dataset"]
        reward_actor = support_actors["reward"]
        advantage_actor = support_actors["advantage"]
        replay_buffer = support_actors["replay_buffer"]
        await asyncio.gather(
            train_dataset.setup.call_one(),
            eval_dataset.setup.call_one(),
        )
        pad_token_id = await train_dataset.get_pad_token_id.call_one()
        await asyncio.gather(
            reward_actor.setup.call_one(),
            advantage_actor.setup.call_one(),
            replay_buffer.setup.call_one(pad_token_id),
        )

        train_mesh = host.spawn_procs(
            per_host={"procs": len(settings["train_gpus"])},
            name="rl_train_procs",
        )
        proc_meshes.append(train_mesh)
        train_env = dict(settings["worker_env"])
        train_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id) for gpu_id in settings["train_gpus"]
        )
        await _set_mesh_environment(train_mesh, "rl_train", train_env)
        await setup_torch_elastic_env_async(train_mesh)
        train_actors = train_mesh.spawn("rl_train_actor", TrainActor, str(config_path))
        logger.info(
            "Initializing %d VeOmni training ranks.",
            len(settings["train_gpus"]),
        )
        await train_actors.setup.call()

        rollout_mesh = host.spawn_procs(
            per_host={"procs": 1}, name="rl_rollout_procs"
        )
        proc_meshes.append(rollout_mesh)
        rollout_env = dict(settings["worker_env"])
        rollout_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id) for gpu_id in settings["rollout_gpus"]
        )
        await _set_mesh_environment(rollout_mesh, "rl_rollout", rollout_env)
        rollout_actor = rollout_mesh.spawn(
            "rl_rollout_actor", RolloutActor, str(config_path)
        )
        rollout_status = await rollout_actor.setup.call_one()
        policy_version = rollout_status.policy_version
        metadata_results = await train_actors.get_weight_transfer_metadata.call()
        weight_metadata = next(iter(metadata_results.values()))
        transfer_address, transfer_port = reserve_local_port()
        transfer_world_size = 1 + len(settings["rollout_gpus"])
        await asyncio.gather(
            train_actors.init_weight_transfer.call(
                transfer_address, transfer_port, transfer_world_size
            ),
            rollout_actor.init_weight_transfer.call_one(
                transfer_address, transfer_port, transfer_world_size
            ),
        )
        logger.info(
            "NCCL policy transfer ready at %s:%d with %d vLLM worker(s).",
            transfer_address,
            transfer_port,
            transfer_world_size - 1,
        )
        async with GpuMonitor(
            settings["rollout_gpus"],
            interval_seconds=settings["gpu_sample_interval_seconds"],
        ) as sync_monitor:
            update_result = await sync_policy_weights(
                rollout_actor=rollout_actor,
                train_actors=train_actors,
                weight_metadata=weight_metadata,
                version=policy_version,
                allow_same_version=True,
            )
        policy_version = update_result.policy_version
        logger.info(
            "Initial policy sync complete: policy=v%d, NCCL sync=%.2fs.",
            policy_version,
            update_result.elapsed_seconds,
        )
        async with GpuMonitor(
            settings["rollout_gpus"],
            interval_seconds=settings["gpu_sample_interval_seconds"],
        ) as eval_monitor:
            eval_metrics = await run_eval(
                eval_dataset=eval_dataset,
                reward_actor=reward_actor,
                rollout_actor=rollout_actor,
                batch_size=settings["eval_batch_size"],
                epochs=settings["eval_epochs"],
                sampling_params=settings["eval_sampling"],
                step=None,
            )
        sync_gpu_metrics = sync_monitor.summary()
        eval_gpu_metrics = eval_monitor.summary()
        log_wandb(
            wandb_run,
            {
                "policy/version": float(policy_version),
                "core/policy_version": float(policy_version),
                "sync/initial_time_sec": update_result.elapsed_seconds,
                "core/sync_time_sec": update_result.elapsed_seconds,
                **core_eval_wandb_metrics(eval_metrics),
                **prefix_metrics("sync", sync_gpu_metrics),
                **prefix_metrics("eval", eval_metrics),
                **prefix_metrics("eval", eval_gpu_metrics),
            },
            step=0,
        )

        for step in range(1, settings["max_steps"] + 1):
            async with GpuMonitor(
                settings["rollout_gpus"],
                interval_seconds=settings["gpu_sample_interval_seconds"],
            ) as rollout_monitor:
                batches, rollout_metrics = await fill_replay_buffer(
                    dataset=train_dataset,
                    reward_actor=reward_actor,
                    advantage_actor=advantage_actor,
                    replay_buffer=replay_buffer,
                    rollout_actor=rollout_actor,
                    current_policy_version=policy_version,
                    rollout_batch_size=settings["rollout_batch_size"],
                    rollout_batch_size_multiplier=settings[
                        "rollout_batch_size_multiplier"
                    ],
                    max_groups=settings["max_rollout_groups_per_step"],
                    train_batch_episode_count=settings["train_batch_episode_count"],
                    samples_per_prompt=settings["samples_per_prompt"],
                    max_prompt_tokens=settings["max_prompt_tokens"],
                    max_response_tokens=settings["max_response_tokens"],
                    dynamic_sampling_filter_low_variance=settings[
                        "dynamic_sampling_filter_low_variance"
                    ],
                    dynamic_sampling_filter_overlong_prompts=settings[
                        "dynamic_sampling_filter_overlong_prompts"
                    ],
                    dynamic_sampling_filter_overlong_responses=settings[
                        "dynamic_sampling_filter_overlong_responses"
                    ],
                )
            async with GpuMonitor(
                settings["train_gpus"],
                interval_seconds=settings["gpu_sample_interval_seconds"],
            ) as train_monitor:
                train_results = await train_actors.train_grpo_step.call(batches)
            train_summary = summarize_train_results(train_results)

            async with GpuMonitor(
                settings["rollout_gpus"],
                interval_seconds=settings["gpu_sample_interval_seconds"],
            ) as sync_monitor:
                update_result = await sync_policy_weights(
                    rollout_actor=rollout_actor,
                    train_actors=train_actors,
                    weight_metadata=weight_metadata,
                    version=policy_version + 1,
                )
            policy_version = update_result.policy_version
            logger.info(
                "RL step %d/%d complete across %.0f rank(s): "
                "loss_mean=%.6f, lr_mean=%.3e, grad_norm_mean=%.4f, "
                "grad_norm_max=%.4f, KL_mean=%.6f, ratio_mean=%.4f, "
                "clip_mean=%.4f, tokens_total=%.0f, train_max=%.2fs, "
                "policy=v%d, NCCL sync=%.2fs.",
                step,
                settings["max_steps"],
                train_summary["ranks"],
                train_summary["loss_mean"],
                train_summary["lr_mean"],
                train_summary["grad_norm_mean"],
                train_summary["grad_norm_max"],
                train_summary["approx_kl_mean"],
                train_summary["ratio_mean"],
                train_summary["clip_fraction_mean"],
                train_summary["active_tokens_total"],
                train_summary["elapsed_seconds_max"],
                policy_version,
                update_result.elapsed_seconds,
            )
            rollout_gpu_metrics = rollout_monitor.summary()
            train_gpu_metrics = train_monitor.summary()
            sync_gpu_metrics = sync_monitor.summary()
            step_metrics = {
                "policy/version": float(policy_version),
                **prefix_metrics("rollout", rollout_metrics),
                **prefix_metrics("rollout", rollout_gpu_metrics),
                **train_wandb_metrics(train_summary),
                **prefix_metrics("train", train_gpu_metrics),
                "sync/time_sec": update_result.elapsed_seconds,
                **prefix_metrics("sync", sync_gpu_metrics),
                **core_step_wandb_metrics(
                    policy_version=policy_version,
                    train_summary=train_summary,
                    rollout_metrics=rollout_metrics,
                    sync_time_sec=update_result.elapsed_seconds,
                ),
            }
            log_wandb(wandb_run, step_metrics, step=step)
            if step % settings["eval_steps"] == 0:
                async with GpuMonitor(
                    settings["rollout_gpus"],
                    interval_seconds=settings["gpu_sample_interval_seconds"],
                ) as eval_monitor:
                    eval_metrics = await run_eval(
                        eval_dataset=eval_dataset,
                        reward_actor=reward_actor,
                        rollout_actor=rollout_actor,
                        batch_size=settings["eval_batch_size"],
                        epochs=settings["eval_epochs"],
                        sampling_params=settings["eval_sampling"],
                        step=step,
                    )
                eval_gpu_metrics = eval_monitor.summary()
                log_wandb(
                    wandb_run,
                    {
                        **core_eval_wandb_metrics(eval_metrics),
                        **prefix_metrics("eval", eval_metrics),
                        **prefix_metrics("eval", eval_gpu_metrics),
                    },
                    step=step,
                )

        logger.info("GRPO training completed successfully.")
    finally:
        await asyncio.shield(
            close_resources(
                rollout_actor=rollout_actor,
                train_actors=train_actors,
                proc_meshes=proc_meshes,
                timeout=settings["shutdown_timeout"],
            )
        )
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                logger.warning("Failed to finish W&B run cleanly.", exc_info=True)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal Monarch + VeOmni + vLLM GRPO pipeline."
    )
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--config", dest="config_option", type=Path)
    args = parser.parse_args()
    if args.config is not None and args.config_option is not None:
        parser.error("Specify the config positionally or with --config, not both.")
    args.config_path = args.config_option or args.config or DEFAULT_CONFIG
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_cli_args()
    try:
        asyncio.run(run_rl(args.config_path))
    except KeyboardInterrupt:
        logger.warning("Forced shutdown requested.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
