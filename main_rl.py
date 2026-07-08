from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from monarch.actor import ProcMesh, this_host
from monarch.spmd import setup_torch_elastic_env_async

from actor.dataset_actor import DatasetActor
from actor.replay_buffer_actor import ReplayBufferActor
from actor.reward_advantage_actor import AdvantageActor, RewardActor
from actor.rollout_actor import RolloutActor, RolloutOutput
from actor.train_actor import TrainActor
from rl.types import DatasetSample, RLEpisode
from tools.config import validate_rl_config
from tools.controller_utils import (
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


logger = logging.getLogger("main_rl")
DEFAULT_CONFIG = Path(__file__).parent / "config" / "qwen3_1_7b_gsm8k_grpo.yaml"


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


async def run_rl(config_path: Path) -> None:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    settings = validate_rl_config(config)
    wandb_run = init_wandb_run(
        config_path=config_path,
        config=config,
        settings=settings,
        logger=logger,
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
        await set_mesh_environment(train_mesh, "rl_train", train_env)
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
        await set_mesh_environment(rollout_mesh, "rl_rollout", rollout_env)
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
                logger=logger,
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
        last_eval_step = 0
        last_train_step = 0

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
            train_summary = summarize_grpo_train_results(train_results)

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
                **grpo_train_wandb_metrics(train_summary),
                **prefix_metrics("train", train_gpu_metrics),
                "sync/time_sec": update_result.elapsed_seconds,
                **prefix_metrics("sync", sync_gpu_metrics),
                **grpo_core_step_wandb_metrics(
                    policy_version=policy_version,
                    train_summary=train_summary,
                    rollout_metrics=rollout_metrics,
                    sync_time_sec=update_result.elapsed_seconds,
                ),
            }
            log_wandb(wandb_run, step_metrics, step=step)
            last_train_step = step
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
                log_wandb(
                    wandb_run,
                    {
                        **core_eval_wandb_metrics(eval_metrics),
                        **prefix_metrics("eval", eval_metrics),
                        **prefix_metrics("eval", eval_gpu_metrics),
                    },
                    step=step,
                )
                last_eval_step = step

        if last_train_step and last_eval_step != last_train_step:
            logger.info("Running final GRPO eval at step %d.", last_train_step)
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

        logger.info("GRPO training completed successfully.")
    finally:
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
