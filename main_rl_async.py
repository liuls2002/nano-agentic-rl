from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

from monarch.actor import ProcMesh, this_host
from monarch.spmd import setup_torch_elastic_env_async

from actor.dataset_actor import DatasetActor
from actor.replay_buffer_actor import ReplayBufferActor
from actor.reward_advantage_actor import AdvantageActor, RewardActor
from actor.rollout_actor import RolloutActor, RolloutOutput
from actor.train_actor import TrainActor
from main_rl import build_episodes
from rl.types import DatasetSample
from tools.config import validate_rl_config
from tools.controller_utils import (
    RolloutMetricsAccumulator,
    close_resources,
    core_eval_wandb_metrics,
    grpo_core_step_wandb_metrics,
    grpo_train_wandb_metrics,
    init_wandb_run,
    load_config,
    log_wandb,
    reserve_local_port,
    run_eval,
    set_mesh_environment,
    summarize_grpo_train_results,
    summarize_lengths,
    sync_policy_weights,
)
from tools.runtime_metrics import GpuMonitor, prefix_metrics


logger = logging.getLogger("main_rl_async")
DEFAULT_CONFIG = Path(__file__).parent / "config" / "qwen3_1_7b_gsm8k_grpo_async.yaml"


def replay_status_metrics(status: Any) -> dict[str, float]:
    return {
        "async/buffer_size": float(status.size),
        "async/buffer_capacity": float(status.capacity),
        "async/buffer_oldest_policy_version": (
            float(status.oldest_policy_version)
            if status.oldest_policy_version is not None
            else -1.0
        ),
        "async/buffer_newest_policy_version": (
            float(status.newest_policy_version)
            if status.newest_policy_version is not None
            else -1.0
        ),
    }


def is_rollout_interruption(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "interrupted by policy update" in message
        or "pause_generation" in message
        or "aborted" in message
    )


async def next_rollout_samples(
    dataset: Any,
    retry_queue: deque[DatasetSample],
    batch_size: int,
) -> tuple[list[DatasetSample], int]:
    samples: list[DatasetSample] = []
    retried = 0
    while retry_queue and len(samples) < batch_size:
        samples.append(retry_queue.popleft())
        retried += 1
    if len(samples) < batch_size:
        samples.extend(await dataset.next_batch.call_one(batch_size - len(samples)))
    return samples, retried


def retry_rollout_samples(
    retry_queue: deque[DatasetSample],
    samples: list[DatasetSample],
) -> None:
    retry_queue.extendleft(reversed(samples))


def interrupted_rollout_metrics(
    *,
    requested_groups: int,
    retried_groups: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "requested_groups": float(requested_groups),
        "processed_groups": 0.0,
        "accepted_groups": 0.0,
        "dynamic_sampling/dropped_low_variance_groups": 0.0,
        "dynamic_sampling/dropped_overlong_prompt_groups": 0.0,
        "dynamic_sampling/dropped_overlong_response_groups": 0.0,
        "dynamic_sampling/dropped_overlong_responses": 0.0,
        "dynamic_sampling/dropped_insufficient_sample_groups": 0.0,
        "generated_samples": 0.0,
        "truncated_samples": 0.0,
        "truncated_groups": 0.0,
        "accepted_samples": 0.0,
        "interrupted_groups": float(requested_groups),
        "retried_groups": float(retried_groups),
        "effective_sample_rate": 0.0,
        "effective_group_rate": 0.0,
        "truncated_sample_rate": 0.0,
        "reward_mean": 0.0,
        "reward_std_mean": 0.0,
        "time_sec": float(elapsed_seconds),
        "response_token_lengths": [],
        **summarize_lengths([]),
    }


async def collect_rollout_iteration(
    *,
    dataset: Any,
    dataset_samples: list[DatasetSample],
    reward_actor: Any,
    advantage_actor: Any,
    replay_buffer: Any,
    rollout_actor: Any,
    max_prompt_tokens: int,
    max_response_tokens: int,
    dynamic_sampling_filter_low_variance: bool,
    dynamic_sampling_filter_overlong_prompts: bool,
    dynamic_sampling_filter_overlong_responses: bool,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    requested_groups = len(dataset_samples)
    groups_processed = 0
    accepted_groups = 0
    dropped_low_variance_groups = 0
    dropped_overlong_prompt_groups = 0
    dropped_overlong_response_groups = 0
    dropped_overlong_responses = 0
    dropped_insufficient_sample_groups = 0
    generated_samples = 0
    truncated_samples = 0
    truncated_groups = 0
    accepted_samples = 0
    response_token_lengths: list[int] = []
    reward_means: list[float] = []
    reward_stds: list[float] = []

    if dynamic_sampling_filter_overlong_prompts:
        prompt_lengths = await dataset.count_tokens.call_one(
            [sample.messages for sample in dataset_samples]
        )
        if len(prompt_lengths) != len(dataset_samples):
            raise RuntimeError("Dataset prompt token counts do not match batch size.")
        filtered_samples = []
        for dataset_sample, prompt_length in zip(dataset_samples, prompt_lengths):
            if int(prompt_length) > max_prompt_tokens:
                groups_processed += 1
                dropped_overlong_prompt_groups += 1
                logger.info(
                    "Async dynamic sampling dropped group %s before rollout: "
                    "prompt length %d > %d.",
                    dataset_sample.sample_id,
                    int(prompt_length),
                    max_prompt_tokens,
                )
                continue
            filtered_samples.append(dataset_sample)
        dataset_samples = filtered_samples

    if not dataset_samples:
        return {
            "requested_groups": float(requested_groups),
            "processed_groups": float(groups_processed),
            "accepted_groups": 0.0,
            "dynamic_sampling/dropped_low_variance_groups": 0.0,
            "dynamic_sampling/dropped_overlong_prompt_groups": float(
                dropped_overlong_prompt_groups
            ),
            "dynamic_sampling/dropped_overlong_response_groups": 0.0,
            "dynamic_sampling/dropped_overlong_responses": 0.0,
            "dynamic_sampling/dropped_insufficient_sample_groups": 0.0,
            "generated_samples": 0.0,
            "truncated_samples": 0.0,
            "truncated_groups": 0.0,
            "accepted_samples": 0.0,
            "interrupted_groups": 0.0,
            "retried_groups": 0.0,
            "effective_sample_rate": 0.0,
            "effective_group_rate": 0.0,
            "truncated_sample_rate": 0.0,
            "reward_mean": 0.0,
            "reward_std_mean": 0.0,
            "time_sec": time.perf_counter() - started_at,
            "response_token_lengths": [],
            **summarize_lengths([]),
        }

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
            1 for sample in rollout_output.samples if sample.finish_reason == "length"
        )
        truncated_samples += truncated_in_group
        if truncated_in_group:
            truncated_groups += 1

        if (
            dynamic_sampling_filter_overlong_prompts
            and len(rollout_output.prompt_token_ids) > max_prompt_tokens
        ):
            dropped_overlong_prompt_groups += 1
            logger.info(
                "Async dynamic sampling dropped group %s: prompt length %d > %d.",
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
                dropped_overlong_responses += 1
                continue
            valid_samples.append(sample)
        if len(valid_samples) != len(rollout_output.samples):
            dropped_overlong_response_groups += 1
        if len(valid_samples) < 2:
            dropped_insufficient_sample_groups += 1
            logger.info(
                "Async dynamic sampling dropped group %s: only %d valid response(s) "
                "after length filtering.",
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
            dropped_low_variance_groups += 1
            logger.info(
                "Async dynamic sampling dropped low-variance group %s (std=%.6f).",
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
            "Async rollout group %s: mean reward=%.3f, std=%.3f, "
            "policy=v%d, buffer=%d.",
            dataset_sample.sample_id,
            advantage_result.reward_mean,
            advantage_result.reward_std,
            rollout_output.policy_version,
            status.size,
        )

    return {
        "requested_groups": float(requested_groups),
        "processed_groups": float(groups_processed),
        "accepted_groups": float(accepted_groups),
        "dynamic_sampling/dropped_low_variance_groups": float(
            dropped_low_variance_groups
        ),
        "dynamic_sampling/dropped_overlong_prompt_groups": float(
            dropped_overlong_prompt_groups
        ),
        "dynamic_sampling/dropped_overlong_response_groups": float(
            dropped_overlong_response_groups
        ),
        "dynamic_sampling/dropped_overlong_responses": float(
            dropped_overlong_responses
        ),
        "dynamic_sampling/dropped_insufficient_sample_groups": float(
            dropped_insufficient_sample_groups
        ),
        "generated_samples": float(generated_samples),
        "truncated_samples": float(truncated_samples),
        "truncated_groups": float(truncated_groups),
        "interrupted_groups": 0.0,
        "retried_groups": 0.0,
        "truncated_sample_rate": (
            truncated_samples / generated_samples if generated_samples else 0.0
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
        "response_token_lengths": response_token_lengths,
        **summarize_lengths(response_token_lengths),
    }


async def rollout_producer_loop(
    *,
    dataset: Any,
    reward_actor: Any,
    advantage_actor: Any,
    replay_buffer: Any,
    rollout_actor: Any,
    settings: Mapping[str, Any],
    shutdown_event: asyncio.Event,
    rollout_gate: asyncio.Event,
    sync_in_progress: asyncio.Event,
    retry_queue: deque[DatasetSample],
    metrics_accumulator: RolloutMetricsAccumulator,
    producer_error: list[BaseException],
) -> None:
    planned_groups = min(
        settings["max_rollout_groups_per_step"],
        max(
            1,
            math.ceil(
                settings["rollout_batch_size"]
                * settings["rollout_batch_size_multiplier"]
            ),
        ),
    )
    logger.info(
        "Async rollout producer started: planned_groups=%d, max_groups=%d.",
        planned_groups,
        settings["max_rollout_groups_per_step"],
    )
    try:
        while not shutdown_event.is_set():
            await rollout_gate.wait()
            status = await replay_buffer.get_status.call_one()
            full_threshold = max(
                1,
                int(status.capacity) - int(settings["train_batch_episode_count"]),
            )
            if int(status.size) >= full_threshold:
                await asyncio.sleep(0.1)
                continue

            if shutdown_event.is_set() or not rollout_gate.is_set():
                continue
            status = await replay_buffer.get_status.call_one()
            if int(status.size) >= full_threshold:
                continue

            dataset_samples, retried_groups = await next_rollout_samples(
                dataset,
                retry_queue,
                planned_groups,
            )
            if not dataset_samples:
                await asyncio.sleep(0.1)
                continue
            started_at = time.perf_counter()
            try:
                metrics = await collect_rollout_iteration(
                    dataset=dataset,
                    dataset_samples=dataset_samples,
                    reward_actor=reward_actor,
                    advantage_actor=advantage_actor,
                    replay_buffer=replay_buffer,
                    rollout_actor=rollout_actor,
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
                metrics["retried_groups"] = float(retried_groups)
                metrics_accumulator.add(metrics)
            except BaseException as exc:
                if sync_in_progress.is_set() or is_rollout_interruption(exc):
                    retry_rollout_samples(retry_queue, dataset_samples)
                    metrics_accumulator.add(
                        interrupted_rollout_metrics(
                            requested_groups=len(dataset_samples),
                            retried_groups=len(dataset_samples),
                            elapsed_seconds=time.perf_counter() - started_at,
                        )
                    )
                    logger.info(
                        "Async rollout interrupted by policy update; queued %d "
                        "group(s) for retry (retry_queue=%d).",
                        len(dataset_samples),
                        len(retry_queue),
                    )
                    await asyncio.sleep(0)
                    continue
                raise
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        producer_error.append(exc)
        shutdown_event.set()
        logger.exception("Async rollout producer failed.")
        raise
    finally:
        logger.info("Async rollout producer stopped.")


def raise_producer_error(
    producer_task: asyncio.Task[Any],
    producer_error: list[BaseException],
) -> None:
    if producer_error:
        raise RuntimeError("Async rollout producer failed.") from producer_error[0]
    if producer_task.done() and not producer_task.cancelled():
        exc = producer_task.exception()
        if exc is not None:
            raise RuntimeError("Async rollout producer failed.") from exc


async def run_rl_async(config_path: Path) -> None:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    settings = validate_rl_config(config)
    replay_config = config.get("rl", {}).get("replay_buffer", {})
    if isinstance(replay_config, dict) and int(replay_config.get("max_policy_age", 0)) == 0:
        logger.warning(
            "rl.replay_buffer.max_policy_age is 0; async RL will only train on "
            "current-policy samples. Set it to 1 to allow one-step stale rollouts."
        )
    wandb_run = init_wandb_run(
        config_path=config_path,
        config=config,
        settings=settings,
        logger=logger,
    )
    proc_meshes: list[ProcMesh] = []
    train_actors = None
    rollout_actor = None
    producer_task: asyncio.Task[Any] | None = None
    shutdown_event = asyncio.Event()
    rollout_gate = asyncio.Event()
    sync_in_progress = asyncio.Event()
    retry_queue: deque[DatasetSample] = deque()
    rollout_metrics = RolloutMetricsAccumulator()
    producer_error: list[BaseException] = []

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
                per_host={"procs": 1}, name=f"rl_async_{name}_procs"
            )
            proc_meshes.append(proc_mesh)
            support_actors[name] = proc_mesh.spawn(
                f"rl_async_{name}", actor_type, str(config_path), *actor_args
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
            name="rl_async_train_procs",
        )
        proc_meshes.append(train_mesh)
        train_env = dict(settings["worker_env"])
        train_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id) for gpu_id in settings["train_gpus"]
        )
        await set_mesh_environment(train_mesh, "rl_async_train", train_env)
        await setup_torch_elastic_env_async(train_mesh)
        train_actors = train_mesh.spawn(
            "rl_async_train_actor", TrainActor, str(config_path)
        )
        logger.info(
            "Initializing %d VeOmni training ranks.",
            len(settings["train_gpus"]),
        )
        await train_actors.setup.call()

        rollout_mesh = host.spawn_procs(
            per_host={"procs": 1}, name="rl_async_rollout_procs"
        )
        proc_meshes.append(rollout_mesh)
        rollout_env = dict(settings["worker_env"])
        rollout_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id) for gpu_id in settings["rollout_gpus"]
        )
        await set_mesh_environment(rollout_mesh, "rl_async_rollout", rollout_env)
        rollout_actor = rollout_mesh.spawn(
            "rl_async_rollout_actor", RolloutActor, str(config_path)
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
                logger=logger,
            )
        log_wandb(
            wandb_run,
            {
                "policy/version": float(policy_version),
                "core/policy_version": float(policy_version),
                "sync/initial_time_sec": update_result.elapsed_seconds,
                "core/sync_time_sec": update_result.elapsed_seconds,
                **core_eval_wandb_metrics(eval_metrics),
                **prefix_metrics("sync", sync_monitor.summary()),
                **prefix_metrics("eval", eval_metrics),
                **prefix_metrics("eval", eval_monitor.summary()),
            },
            step=0,
        )

        rollout_gate.set()
        producer_task = asyncio.create_task(
            rollout_producer_loop(
                dataset=train_dataset,
                reward_actor=reward_actor,
                advantage_actor=advantage_actor,
                replay_buffer=replay_buffer,
                rollout_actor=rollout_actor,
                settings=settings,
                shutdown_event=shutdown_event,
                rollout_gate=rollout_gate,
                sync_in_progress=sync_in_progress,
                retry_queue=retry_queue,
                metrics_accumulator=rollout_metrics,
                producer_error=producer_error,
            )
        )

        last_eval_step = 0
        last_train_step = 0
        for step in range(1, settings["max_steps"] + 1):
            wait_started = time.perf_counter()
            batches = None
            while batches is None:
                raise_producer_error(producer_task, producer_error)
                batches = await replay_buffer.sample.call_one(policy_version)
                if batches is None:
                    await asyncio.sleep(0.1)
            train_wait_buffer_sec = time.perf_counter() - wait_started
            raise_producer_error(producer_task, producer_error)

            async with GpuMonitor(
                settings["train_gpus"],
                interval_seconds=settings["gpu_sample_interval_seconds"],
            ) as train_monitor:
                train_results = await train_actors.train_grpo_step.call(batches)
            train_summary = summarize_grpo_train_results(train_results)

            rollout_gate.clear()
            sync_in_progress.set()
            try:
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
            finally:
                sync_in_progress.clear()

            eval_metrics = None
            eval_gpu_metrics = {}
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
                        logger=logger,
                    )
                eval_gpu_metrics = eval_monitor.summary()
                last_eval_step = step
            rollout_gate.set()

            last_train_step = step
            buffer_status = await replay_buffer.get_status.call_one()
            rollout_window_metrics = rollout_metrics.drain()
            train_gpu_metrics = train_monitor.summary()
            sync_gpu_metrics = sync_monitor.summary()
            logger.info(
                "Async RL step %d/%d complete across %.0f rank(s): "
                "loss_mean=%.6f, lr_mean=%.3e, grad_norm_mean=%.4f, "
                "KL_mean=%.6f, tokens_total=%.0f, train_max=%.2fs, "
                "wait_buffer=%.2fs, buffer=%d/%d, policy_range=%s..%s, "
                "policy=v%d, NCCL sync=%.2fs.",
                step,
                settings["max_steps"],
                train_summary["ranks"],
                train_summary["loss_mean"],
                train_summary["lr_mean"],
                train_summary["grad_norm_mean"],
                train_summary["approx_kl_mean"],
                train_summary["active_tokens_total"],
                train_summary["elapsed_seconds_max"],
                train_wait_buffer_sec,
                buffer_status.size,
                buffer_status.capacity,
                buffer_status.oldest_policy_version,
                buffer_status.newest_policy_version,
                policy_version,
                update_result.elapsed_seconds,
            )
            step_metrics = {
                "policy/version": float(policy_version),
                **prefix_metrics("rollout", rollout_window_metrics),
                **grpo_train_wandb_metrics(train_summary),
                **prefix_metrics("train", train_gpu_metrics),
                "sync/time_sec": update_result.elapsed_seconds,
                **prefix_metrics("sync", sync_gpu_metrics),
                **grpo_core_step_wandb_metrics(
                    policy_version=policy_version,
                    train_summary=train_summary,
                    rollout_metrics=rollout_window_metrics,
                    sync_time_sec=update_result.elapsed_seconds,
                ),
                "async/train_wait_buffer_sec": train_wait_buffer_sec,
                "async/rollout_iterations": rollout_window_metrics[
                    "rollout_iterations"
                ],
                "async/retry_queue_size": float(len(retry_queue)),
                **replay_status_metrics(buffer_status),
            }
            log_wandb(wandb_run, step_metrics, step=step)
            if eval_metrics is not None:
                log_wandb(
                    wandb_run,
                    {
                        **core_eval_wandb_metrics(eval_metrics),
                        **prefix_metrics("eval", eval_metrics),
                        **prefix_metrics("eval", eval_gpu_metrics),
                    },
                    step=step,
                )

        rollout_gate.clear()
        shutdown_event.set()
        if producer_task is not None:
            producer_task.cancel()
            await asyncio.gather(producer_task, return_exceptions=True)

        if last_train_step and last_eval_step != last_train_step:
            logger.info("Running final async GRPO eval at step %d.", last_train_step)
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
                    step=last_train_step,
                    logger=logger,
                )
            log_wandb(
                wandb_run,
                {
                    **core_eval_wandb_metrics(eval_metrics),
                    **prefix_metrics("eval", eval_metrics),
                    **prefix_metrics("eval", eval_monitor.summary()),
                },
                step=last_train_step,
            )

        logger.info("Async GRPO training completed successfully.")
    finally:
        shutdown_event.set()
        rollout_gate.set()
        if producer_task is not None and not producer_task.done():
            producer_task.cancel()
            await asyncio.gather(producer_task, return_exceptions=True)
        await asyncio.shield(
            close_resources(
                rollout_actor=rollout_actor,
                train_actors=train_actors,
                proc_meshes=proc_meshes,
                timeout=settings["shutdown_timeout"],
                logger=logger,
            )
        )
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                logger.warning("Failed to finish W&B run cleanly.", exc_info=True)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the semi-async Monarch + VeOmni + vLLM GRPO pipeline."
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
        asyncio.run(run_rl_async(args.config_path))
    except KeyboardInterrupt:
        logger.warning("Forced shutdown requested.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
