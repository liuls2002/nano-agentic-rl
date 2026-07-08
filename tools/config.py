from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from actor.utils import sft_steps_from_epochs


DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 15.0


def mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def load_sequence_config(monarch: Mapping[str, Any]) -> dict[str, int]:
    sequence = mapping(monarch.get("sequence"), "monarch.sequence")
    prompt_length = positive_int(
        sequence.get("max_prompt_tokens", 1024),
        "monarch.sequence.max_prompt_tokens",
    )
    response_length = positive_int(
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


def _validate_data_paths(train_data: Mapping[str, Any], eval_data: Mapping[str, Any]) -> None:
    for name, data_config in (
        ("dataloader.train", train_data),
        ("dataloader.eval", eval_data),
    ):
        path = data_config.get("path")
        if not path:
            raise ValueError(f"{name}.path is required.")
        if not Path(str(path)).expanduser().is_file():
            raise FileNotFoundError(f"{name}.path does not exist: {path}")


def validate_rl_config(config: Mapping[str, Any]) -> dict[str, Any]:
    monarch = mapping(config.get("monarch"), "monarch")
    wandb_config = mapping(monarch.get("wandb"), "monarch.wandb")
    sequence = load_sequence_config(monarch)
    rl_config = mapping(config.get("rl"), "rl")
    train_actor = mapping(config.get("train_actor"), "train_actor")
    rollout_actor = mapping(config.get("rollout_actor"), "rollout_actor")
    train_config = mapping(train_actor.get("train"), "train_actor.train")
    dataloader = mapping(config.get("dataloader"), "dataloader")
    train_data = mapping(dataloader.get("train"), "dataloader.train")
    eval_data = mapping(dataloader.get("eval"), "dataloader.eval")
    rollout_config = mapping(rollout_actor.get("rollout"), "rollout_actor.rollout")
    eval_config = mapping(rollout_actor.get("eval"), "rollout_actor.eval")
    rollout_sampling = mapping(
        rollout_config.get("sampling"), "rollout_actor.rollout.sampling"
    )
    eval_sampling = mapping(
        eval_config.get("sampling"), "rollout_actor.eval.sampling"
    )
    engine = mapping(rollout_actor.get("engine"), "rollout_actor.engine")

    train_num_gpus = positive_int(train_actor.get("num_gpus", 0), "train_actor.num_gpus")
    rollout_num_gpus = positive_int(
        rollout_actor.get("num_gpus", 0), "rollout_actor.num_gpus"
    )
    train_gpus, rollout_gpus = split_worker_gpus(train_num_gpus, rollout_num_gpus)
    _validate_data_paths(train_data, eval_data)

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

    configured_env = mapping(monarch.get("env"), "monarch.env")
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
    dynamic_sampling = mapping(
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
    weight_sync = mapping(rl_config.get("weight_sync"), "rl.weight_sync")
    if weight_sync.get("backend", "nccl") != "nccl":
        raise ValueError(
            "This RL controller currently requires weight_sync.backend=nccl."
        )
    if int(weight_sync.get("packed_buffer_size_bytes", 0)) <= 0:
        raise ValueError("weight_sync.packed_buffer_size_bytes must be positive.")
    if int(weight_sync.get("packed_num_buffers", 0)) <= 0:
        raise ValueError("weight_sync.packed_num_buffers must be positive.")
    transfer_config = mapping(
        engine.get("weight_transfer_config"),
        "rollout_actor.engine.weight_transfer_config",
    )
    if transfer_config.get("backend") != "nccl":
        raise ValueError(
            "rollout_actor.engine.weight_transfer_config.backend must be nccl."
        )
    eval_steps = positive_int(
        eval_config.get("eval_steps", 1), "rollout_actor.eval.eval_steps"
    )
    eval_epochs = positive_int(
        eval_config.get("eval_epochs", 1), "rollout_actor.eval.eval_epochs"
    )
    eval_batch_size = positive_int(
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


def validate_sft_config(config: Mapping[str, Any]) -> dict[str, Any]:
    monarch = mapping(config.get("monarch"), "monarch")
    wandb_config = mapping(monarch.get("wandb"), "monarch.wandb")
    sequence = load_sequence_config(monarch)
    train_actor = mapping(config.get("train_actor"), "train_actor")
    rollout_actor = mapping(config.get("rollout_actor"), "rollout_actor")
    train_config = mapping(train_actor.get("train"), "train_actor.train")
    dataloader = mapping(config.get("dataloader"), "dataloader")
    train_data = mapping(dataloader.get("train"), "dataloader.train")
    eval_data = mapping(dataloader.get("eval"), "dataloader.eval")
    engine = mapping(rollout_actor.get("engine"), "rollout_actor.engine")
    eval_config = mapping(rollout_actor.get("eval"), "rollout_actor.eval")
    eval_sampling = mapping(
        eval_config.get("sampling"), "rollout_actor.eval.sampling"
    )

    train_num_gpus = positive_int(train_actor.get("num_gpus", 0), "train_actor.num_gpus")
    rollout_num_gpus = positive_int(
        rollout_actor.get("num_gpus", 0), "rollout_actor.num_gpus"
    )
    train_gpus, rollout_gpus = split_worker_gpus(train_num_gpus, rollout_num_gpus)
    _validate_data_paths(train_data, eval_data)

    global_batch_size = positive_int(
        train_config.get("global_batch_size", 0),
        "train_actor.train.global_batch_size",
    )
    if global_batch_size % train_num_gpus:
        raise ValueError(
            "train_actor.train.global_batch_size must be divisible by "
            f"train_actor.num_gpus ({global_batch_size} vs {train_num_gpus})."
        )
    max_steps, num_train_epochs, steps_per_epoch = sft_steps_from_epochs(
        train_config=train_config,
        train_data=train_data,
        global_batch_size=global_batch_size,
    )
    eval_steps = positive_int(
        eval_config.get("eval_steps", 1), "rollout_actor.eval.eval_steps"
    )
    eval_epochs = positive_int(
        eval_config.get("eval_epochs", 1), "rollout_actor.eval.eval_epochs"
    )
    eval_batch_size = positive_int(
        eval_config.get("batch_size", 1), "rollout_actor.eval.batch_size"
    )
    eval_max_tokens = positive_int(
        eval_sampling.get("max_tokens", sequence["max_response_tokens"]),
        "rollout_actor.eval.sampling.max_tokens",
    )
    if int(eval_sampling.get("n", 1)) <= 0:
        raise ValueError("eval sampling.n must be positive.")
    max_model_len = int(engine.get("max_model_len", sequence["max_seq_len"]))
    if max_model_len < sequence["max_seq_len"]:
        raise ValueError(
            "rollout_actor.engine.max_model_len must be at least "
            "monarch.sequence.max_prompt_tokens + max_response_tokens."
        )
    if eval_max_tokens > max_model_len:
        raise ValueError("eval sampling.max_tokens must not exceed engine.max_model_len.")

    transfer_config = mapping(
        engine.get("weight_transfer_config"),
        "rollout_actor.engine.weight_transfer_config",
    )
    if transfer_config.get("backend") != "nccl":
        raise ValueError(
            "rollout_actor.engine.weight_transfer_config.backend must be nccl."
        )

    configured_env = mapping(monarch.get("env"), "monarch.env")
    shutdown_timeout = float(
        monarch.get("shutdown_timeout_seconds", DEFAULT_SHUTDOWN_TIMEOUT_SECONDS)
    )
    if shutdown_timeout <= 0:
        raise ValueError("monarch.shutdown_timeout_seconds must be positive.")

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
        "num_train_epochs": num_train_epochs,
        "steps_per_epoch": steps_per_epoch,
        "train_global_batch_size": global_batch_size,
        "train_batch_size_per_rank": global_batch_size // train_num_gpus,
        "eval_steps": eval_steps,
        "eval_epochs": eval_epochs,
        "eval_batch_size": eval_batch_size,
        "eval_sampling": eval_sampling,
    }
