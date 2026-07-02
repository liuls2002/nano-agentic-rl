from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
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


logger = logging.getLogger("main_rl")
DEFAULT_CONFIG = Path(__file__).parent / "config" / "qwen2_5_1_5b_gsm8k.yaml"
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
    }


def validate_rl_config(config: Mapping[str, Any]) -> dict[str, Any]:
    monarch = _mapping(config.get("monarch"), "monarch")
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
    train_gpus = list(range(train_num_gpus))
    rollout_gpus = list(range(train_num_gpus, train_num_gpus + rollout_num_gpus))

    for name, data_config in (
        ("dataloader.train", train_data),
        ("dataloader.eval", eval_data),
    ):
        path = data_config.get("path")
        if not path:
            raise ValueError(f"{name}.path is required.")
        if not Path(str(path)).expanduser().is_file():
            raise FileNotFoundError(f"{name}.path does not exist: {path}")

    prompt_length = int(rollout_config.get("max_prompt_tokens", 448))
    response_length = int(rollout_config.get("max_response_tokens", 64))
    if prompt_length <= 0 or response_length <= 0:
        raise ValueError("Rollout prompt and response token limits must be positive.")
    if prompt_length + response_length > int(train_data.get("max_seq_len", 0)):
        raise ValueError(
            "rollout prompt + response tokens must not exceed "
            "dataloader.train.max_seq_len."
        )
    if response_length < int(rollout_sampling.get("max_tokens", response_length)):
        raise ValueError(
            "rollout max_response_tokens must cover rollout sampling.max_tokens."
        )
    if prompt_length + response_length > int(engine.get("max_model_len", 0)):
        raise ValueError(
            "rollout prompt + response tokens must not exceed engine.max_model_len."
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
    max_rollout_groups = int(rl_config.get("max_rollout_groups_per_step", 16))
    if rollout_batch_size <= 0 or max_rollout_groups <= 0:
        raise ValueError("RL rollout batch size and group limit must be positive.")
    replay = _mapping(rl_config.get("replay_buffer"), "rl.replay_buffer")
    batch_size_per_rank = int(replay.get("batch_size_per_rank", 1))
    expected_global_batch = batch_size_per_rank * train_num_gpus
    configured_global_batch = int(train_config.get("global_batch_size", 0))
    if expected_global_batch != configured_global_batch:
        raise ValueError(
            "Replay buffer batch_size_per_rank * train DP size must equal "
            f"train_actor.train.global_batch_size ({expected_global_batch} vs "
            f"{configured_global_batch})."
        )
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
        "shutdown_timeout": shutdown_timeout,
        "max_steps": max_steps,
        "rollout_batch_size": rollout_batch_size,
        "max_rollout_groups_per_step": max_rollout_groups,
        "drop_low_variance_groups": bool(
            rl_config.get("drop_low_variance_groups", False)
        ),
        "eval_steps": eval_steps,
        "eval_epochs": eval_epochs,
        "eval_batch_size": eval_batch_size,
        "eval_sampling": eval_sampling,
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
    max_groups: int,
    drop_low_variance: bool,
) -> list[dict[str, Any]]:
    groups_processed = 0
    while groups_processed < max_groups:
        batch = await replay_buffer.sample.call_one(current_policy_version)
        if batch is not None:
            return batch

        dataset_samples = await dataset.next_batch.call_one(rollout_batch_size)
        rollout_outputs = await rollout_actor.chat.call_one(
            [sample.messages for sample in dataset_samples]
        )
        if len(dataset_samples) != len(rollout_outputs):
            raise RuntimeError("Dataset and rollout batch sizes do not match.")

        for dataset_sample, rollout_output in zip(dataset_samples, rollout_outputs):
            reward_results = await reward_actor.evaluate_batch.call_one(
                [sample.text for sample in rollout_output.samples],
                [dataset_sample.target] * len(rollout_output.samples),
            )
            advantage_result = await advantage_actor.compute.call_one(
                [result.reward for result in reward_results]
            )
            groups_processed += 1
            if advantage_result.low_variance and drop_low_variance:
                logger.info(
                    "Dropped low-variance rollout group %s (std=%.6f).",
                    dataset_sample.sample_id,
                    advantage_result.reward_std,
                )
                continue

            episodes = build_episodes(
                dataset_sample,
                rollout_output,
                reward_results,
                advantage_result.advantages,
            )
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
    return batch


async def run_eval(
    *,
    eval_dataset: Any,
    reward_actor: Any,
    rollout_actor: Any,
    batch_size: int,
    epochs: int,
    sampling_params: Mapping[str, Any],
    step: int | None,
) -> None:
    correctness_groups: list[list[float]] = []
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
    logger.info(
        "Eval %s processed %d sample group(s) across %d epoch(s).",
        "baseline" if step is None else f"step {step}",
        len(correctness_groups),
        epochs,
    )

    max_k = int(sampling_params.get("n", 1))
    metrics = compute_pass_at_k_range(correctness_groups, max_k)
    step_label = "baseline" if step is None else f"step {step}"
    for metric in metrics:
        logger.info(
            "Eval %s: pass@%d=%.4f, g-pass@%d=%.4f, all-pass@%d=%.4f.",
            step_label,
            metric.k,
            metric.pass_at_k,
            metric.k,
            metric.g_pass_at_k,
            metric.k,
            metric.all_pass_at_k,
        )


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
        await run_eval(
            eval_dataset=eval_dataset,
            reward_actor=reward_actor,
            rollout_actor=rollout_actor,
            batch_size=settings["eval_batch_size"],
            epochs=settings["eval_epochs"],
            sampling_params=settings["eval_sampling"],
            step=None,
        )

        for step in range(1, settings["max_steps"] + 1):
            batches = await fill_replay_buffer(
                dataset=train_dataset,
                reward_actor=reward_actor,
                advantage_actor=advantage_actor,
                replay_buffer=replay_buffer,
                rollout_actor=rollout_actor,
                current_policy_version=policy_version,
                rollout_batch_size=settings["rollout_batch_size"],
                max_groups=settings["max_rollout_groups_per_step"],
                drop_low_variance=settings["drop_low_variance_groups"],
            )
            train_results = await train_actors.train_grpo_step.call(batches)
            train_summary = summarize_train_results(train_results)

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
            if step % settings["eval_steps"] == 0:
                await run_eval(
                    eval_dataset=eval_dataset,
                    reward_actor=reward_actor,
                    rollout_actor=rollout_actor,
                    batch_size=settings["eval_batch_size"],
                    epochs=settings["eval_epochs"],
                    sampling_params=settings["eval_sampling"],
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
