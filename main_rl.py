from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
import uuid
from pathlib import Path
from typing import Any, Awaitable, Mapping, Sequence

import monarch.actor as monarch_actor
import yaml
from monarch.actor import (
    Actor,
    MeshFailure,
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


logger = logging.getLogger("main_rl")
DEFAULT_CONFIG = Path(__file__).parent / "config" / "qwen2_5_1_5b_gsm8k_sft.yaml"
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 15.0


class ControllerShutdown(Exception):
    pass


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


def _gpu_ids(resources: Mapping[str, Any], name: str) -> list[int]:
    gpu_ids = resources.get("gpu_ids")
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ValueError(f"{name}.gpu_ids must be a non-empty list.")
    if any(not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"{name}.gpu_ids must contain non-negative integers.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"{name}.gpu_ids must not contain duplicates.")
    return list(gpu_ids)


def validate_rl_config(config: Mapping[str, Any]) -> dict[str, Any]:
    monarch = _mapping(config.get("monarch"), "monarch")
    train_resources = _mapping(monarch.get("train"), "monarch.train")
    rollout_resources = _mapping(monarch.get("rollout"), "monarch.rollout")
    rl_config = _mapping(monarch.get("rl"), "monarch.rl")
    train_gpus = _gpu_ids(train_resources, "monarch.train")
    rollout_gpus = _gpu_ids(rollout_resources, "monarch.rollout")
    overlap = sorted(set(train_gpus).intersection(rollout_gpus))
    if overlap:
        raise ValueError(f"Training and rollout GPU sets overlap: {overlap}")

    veomni = _mapping(config.get("veomni"), "veomni")
    data = _mapping(veomni.get("data"), "veomni.data")
    train_config = _mapping(veomni.get("train"), "veomni.train")
    vllm = _mapping(config.get("vllm"), "vllm")
    sampling = _mapping(vllm.get("sampling"), "vllm.sampling")
    engine = _mapping(vllm.get("engine"), "vllm.engine")
    prompt_length = int(rl_config.get("max_prompt_tokens", 448))
    response_length = int(rl_config.get("max_response_tokens", 64))
    if prompt_length <= 0 or response_length <= 0:
        raise ValueError("RL prompt and response token limits must be positive.")
    if prompt_length + response_length > int(data.get("max_seq_len", 0)):
        raise ValueError(
            "monarch.rl max_prompt_tokens + max_response_tokens must not exceed "
            "veomni.data.max_seq_len."
        )
    if response_length < int(sampling.get("max_tokens", response_length)):
        raise ValueError(
            "monarch.rl.max_response_tokens must cover vllm.sampling.max_tokens."
        )
    if prompt_length + response_length > int(engine.get("max_model_len", 0)):
        raise ValueError(
            "monarch.rl max_prompt_tokens + max_response_tokens must not exceed "
            "vllm.engine.max_model_len."
        )
    if int(sampling.get("n", 1)) < 2:
        raise ValueError("GRPO requires vllm.sampling.n >= 2.")

    configured_env = _mapping(monarch.get("env"), "monarch.env")
    shutdown_timeout = float(
        monarch.get("shutdown_timeout_seconds", DEFAULT_SHUTDOWN_TIMEOUT_SECONDS)
    )
    if shutdown_timeout <= 0:
        raise ValueError("monarch.shutdown_timeout_seconds must be positive.")
    max_steps = int(rl_config.get("max_steps", 1))
    if max_steps <= 0:
        raise ValueError("monarch.rl.max_steps must be positive.")
    rollout_batch_size = int(rl_config.get("rollout_batch_size", 1))
    max_rollout_groups = int(rl_config.get("max_rollout_groups_per_step", 16))
    if rollout_batch_size <= 0 or max_rollout_groups <= 0:
        raise ValueError("RL rollout batch size and group limit must be positive.")
    replay = _mapping(rl_config.get("replay_buffer"), "monarch.rl.replay_buffer")
    batch_size_per_rank = int(replay.get("batch_size_per_rank", 1))
    expected_global_batch = batch_size_per_rank * len(train_gpus)
    configured_global_batch = int(train_config.get("global_batch_size", 0))
    if expected_global_batch != configured_global_batch:
        raise ValueError(
            "Replay buffer batch_size_per_rank * train DP size must equal "
            f"veomni.train.global_batch_size ({expected_global_batch} vs "
            f"{configured_global_batch})."
        )
    samples_per_prompt = int(sampling.get("n", 1))
    if expected_global_batch % samples_per_prompt:
        raise ValueError(
            "veomni.train.global_batch_size must be divisible by vllm.sampling.n "
            f"({expected_global_batch} vs {samples_per_prompt})."
        )
    weight_sync = _mapping(rl_config.get("weight_sync"), "monarch.rl.weight_sync")
    if weight_sync.get("backend", "nccl") != "nccl":
        raise ValueError(
            "This RL controller currently requires weight_sync.backend=nccl."
        )
    if int(weight_sync.get("packed_buffer_size_bytes", 0)) <= 0:
        raise ValueError("weight_sync.packed_buffer_size_bytes must be positive.")
    if int(weight_sync.get("packed_num_buffers", 0)) <= 0:
        raise ValueError("weight_sync.packed_num_buffers must be positive.")
    transfer_config = _mapping(
        engine.get("weight_transfer_config"), "vllm.engine.weight_transfer_config"
    )
    if transfer_config.get("backend") != "nccl":
        raise ValueError(
            "vllm.engine.weight_transfer_config.backend must be nccl."
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
    }


def reserve_local_port() -> tuple[str, int]:
    """Reserve a single-node rendezvous address for the NCCL transfer group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", int(sock.getsockname()[1])


async def await_or_shutdown(
    awaitable: Awaitable[Any], shutdown_event: asyncio.Event
) -> Any:
    call_task = asyncio.ensure_future(awaitable)
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    try:
        done, _ = await asyncio.wait(
            {call_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if call_task in done:
            return await call_task
        call_task.cancel()
        await asyncio.gather(call_task, return_exceptions=True)
        raise ControllerShutdown
    finally:
        shutdown_task.cancel()
        await asyncio.gather(shutdown_task, return_exceptions=True)


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
                "Rollout sample has no token logprobs; set vllm.sampling.logprobs."
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
    shutdown_event: asyncio.Event,
) -> list[dict[str, Any]]:
    groups_processed = 0
    while groups_processed < max_groups:
        batch = await await_or_shutdown(
            replay_buffer.sample.call_one(current_policy_version), shutdown_event
        )
        if batch is not None:
            return batch

        dataset_samples = await await_or_shutdown(
            dataset.next_batch.call_one(rollout_batch_size), shutdown_event
        )
        rollout_outputs = await await_or_shutdown(
            rollout_actor.chat.call_one(
                [sample.messages for sample in dataset_samples]
            ),
            shutdown_event,
        )
        if len(dataset_samples) != len(rollout_outputs):
            raise RuntimeError("Dataset and rollout batch sizes do not match.")

        for dataset_sample, rollout_output in zip(dataset_samples, rollout_outputs):
            reward_results = await await_or_shutdown(
                reward_actor.evaluate_batch.call_one(
                    [sample.text for sample in rollout_output.samples],
                    [dataset_sample.target] * len(rollout_output.samples),
                ),
                shutdown_event,
            )
            advantage_result = await await_or_shutdown(
                advantage_actor.compute.call_one(
                    [result.reward for result in reward_results]
                ),
                shutdown_event,
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
            status = await await_or_shutdown(
                replay_buffer.add.call_one(episodes), shutdown_event
            )
            logger.info(
                "Rollout group %s: mean reward=%.3f, std=%.3f, buffer=%d.",
                dataset_sample.sample_id,
                advantage_result.reward_mean,
                advantage_result.reward_std,
                status.size,
            )
            if groups_processed >= max_groups:
                break

    batch = await await_or_shutdown(
        replay_buffer.sample.call_one(current_policy_version), shutdown_event
    )
    if batch is None:
        raise RuntimeError(
            "Replay buffer did not reach a full training batch within "
            f"{max_groups} rollout groups."
        )
    return batch


async def close_resources(
    *,
    rollout_actor: Any | None,
    train_actors: Any | None,
    proc_meshes: Sequence[ProcMesh],
    timeout: float,
) -> None:
    phase_timeout = max(timeout / 3.0, 1.0)
    close_calls = []
    if rollout_actor is not None:
        close_calls.append(rollout_actor.close.call_one())
    if train_actors is not None:
        close_calls.append(train_actors.close.call())
    if close_calls:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*close_calls, return_exceptions=True),
                timeout=phase_timeout,
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning("Actor close failed: %r", result)
        except asyncio.TimeoutError:
            logger.warning("Actor close exceeded %.1fs; stopping meshes.", phase_timeout)

    stop_calls = [
        proc_mesh.stop("RL controller shutdown")
        for proc_mesh in reversed(proc_meshes)
    ]
    if stop_calls:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*stop_calls, return_exceptions=True),
                timeout=phase_timeout,
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning("Process mesh stop failed: %r", result)
        except asyncio.TimeoutError:
            logger.warning("Process mesh stop exceeded %.1fs.", phase_timeout)

    try:
        await asyncio.wait_for(shutdown_context(), timeout=phase_timeout)
    except asyncio.TimeoutError:
        logger.warning("Monarch context shutdown exceeded %.1fs.", phase_timeout)
    except Exception:
        logger.exception("Failed to shut down Monarch context.")


async def run_rl(config_path: Path) -> int | None:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    settings = validate_rl_config(config)
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    received_signal: int | None = None
    mesh_failure: MeshFailure | None = None
    cleaning_up = False
    proc_meshes: list[ProcMesh] = []
    train_actors = None
    rollout_actor = None

    previous_fault_hook = monarch_actor.unhandled_fault_hook
    previous_signal_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def request_shutdown(signum: int) -> None:
        nonlocal received_signal
        if shutdown_event.is_set():
            return
        received_signal = signum
        logger.info("Received %s; stopping RL.", signal.Signals(signum).name)
        shutdown_event.set()

    def signal_handler(signum: int, frame: Any) -> None:
        if shutdown_event.is_set() and signum == signal.SIGINT:
            signal.default_int_handler(signum, frame)
            return
        loop.call_soon_threadsafe(request_shutdown, signum)

    def record_mesh_failure(failure: MeshFailure) -> None:
        nonlocal mesh_failure
        if cleaning_up or received_signal is not None:
            return
        if mesh_failure is None:
            mesh_failure = failure
            logger.error("Monarch mesh failure:\n%s", failure.report())
        shutdown_event.set()

    def fault_hook(failure: MeshFailure) -> None:
        loop.call_soon_threadsafe(record_mesh_failure, failure)

    monarch_actor.unhandled_fault_hook = fault_hook
    for signum in previous_signal_handlers:
        signal.signal(signum, signal_handler)

    try:
        host = this_host()
        support_actors = {}
        for name, actor_type in (
            ("dataset", DatasetActor),
            ("reward", RewardActor),
            ("advantage", AdvantageActor),
            ("replay_buffer", ReplayBufferActor),
        ):
            proc_mesh = host.spawn_procs(
                per_host={"procs": 1}, name=f"rl_{name}_procs"
            )
            proc_meshes.append(proc_mesh)
            support_actors[name] = proc_mesh.spawn(
                f"rl_{name}", actor_type, str(config_path)
            )

        dataset = support_actors["dataset"]
        reward_actor = support_actors["reward"]
        advantage_actor = support_actors["advantage"]
        replay_buffer = support_actors["replay_buffer"]
        await await_or_shutdown(dataset.setup.call_one(), shutdown_event)
        pad_token_id = await await_or_shutdown(
            dataset.get_pad_token_id.call_one(), shutdown_event
        )
        await asyncio.gather(
            await_or_shutdown(reward_actor.setup.call_one(), shutdown_event),
            await_or_shutdown(advantage_actor.setup.call_one(), shutdown_event),
            await_or_shutdown(replay_buffer.setup.call_one(pad_token_id), shutdown_event),
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
        await await_or_shutdown(
            _set_mesh_environment(train_mesh, "rl_train", train_env), shutdown_event
        )
        await await_or_shutdown(
            setup_torch_elastic_env_async(train_mesh), shutdown_event
        )
        train_actors = train_mesh.spawn("rl_train_actor", TrainActor, str(config_path))
        logger.info("Initializing %d VeOmni training ranks.", len(settings["train_gpus"]))
        await await_or_shutdown(train_actors.setup.call(), shutdown_event)

        rollout_mesh = host.spawn_procs(
            per_host={"procs": 1}, name="rl_rollout_procs"
        )
        proc_meshes.append(rollout_mesh)
        rollout_env = dict(settings["worker_env"])
        rollout_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id) for gpu_id in settings["rollout_gpus"]
        )
        await await_or_shutdown(
            _set_mesh_environment(rollout_mesh, "rl_rollout", rollout_env),
            shutdown_event,
        )
        rollout_actor = rollout_mesh.spawn(
            "rl_rollout_actor", RolloutActor, str(config_path)
        )
        rollout_status = await await_or_shutdown(
            rollout_actor.setup.call_one(), shutdown_event
        )
        policy_version = rollout_status.policy_version
        metadata_results = await await_or_shutdown(
            train_actors.get_weight_transfer_metadata.call(), shutdown_event
        )
        weight_metadata = next(iter(metadata_results.values()))
        transfer_address, transfer_port = reserve_local_port()
        transfer_world_size = 1 + len(settings["rollout_gpus"])
        await await_or_shutdown(
            asyncio.gather(
                train_actors.init_weight_transfer.call(
                    transfer_address, transfer_port, transfer_world_size
                ),
                rollout_actor.init_weight_transfer.call_one(
                    transfer_address, transfer_port, transfer_world_size
                ),
            ),
            shutdown_event,
        )
        logger.info(
            "NCCL policy transfer ready at %s:%d with %d vLLM worker(s).",
            transfer_address,
            transfer_port,
            transfer_world_size - 1,
        )

        for step in range(1, settings["max_steps"] + 1):
            batches = await fill_replay_buffer(
                dataset=dataset,
                reward_actor=reward_actor,
                advantage_actor=advantage_actor,
                replay_buffer=replay_buffer,
                rollout_actor=rollout_actor,
                current_policy_version=policy_version,
                rollout_batch_size=settings["rollout_batch_size"],
                max_groups=settings["max_rollout_groups_per_step"],
                drop_low_variance=settings["drop_low_variance_groups"],
                shutdown_event=shutdown_event,
            )
            train_results = await await_or_shutdown(
                train_actors.train_grpo_step.call(batches), shutdown_event
            )
            rank_zero_result = next(iter(train_results.values()))

            policy_version += 1
            update_result, _ = await await_or_shutdown(
                asyncio.gather(
                    rollout_actor.receive_weights.call_one(
                        weight_metadata,
                        version=policy_version,
                    ),
                    train_actors.broadcast_weights.call(),
                ),
                shutdown_event,
            )
            logger.info(
                "RL step %d/%d complete: loss=%.6f, policy=v%d, NCCL sync=%.2fs.",
                step,
                settings["max_steps"],
                rank_zero_result.loss,
                policy_version,
                update_result.elapsed_seconds,
            )

        logger.info("GRPO training completed successfully.")
    except ControllerShutdown:
        logger.info("RL controller shutdown requested.")
    finally:
        cleaning_up = True
        try:
            await asyncio.shield(
                close_resources(
                    rollout_actor=rollout_actor,
                    train_actors=train_actors,
                    proc_meshes=proc_meshes,
                    timeout=settings["shutdown_timeout"],
                )
            )
        finally:
            for signum, previous_handler in previous_signal_handlers.items():
                signal.signal(signum, previous_handler)
            monarch_actor.unhandled_fault_hook = previous_fault_hook

    if mesh_failure is not None:
        raise RuntimeError(f"Monarch mesh failure:\n{mesh_failure.report()}")
    return received_signal


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
        received_signal = asyncio.run(run_rl(args.config_path))
    except KeyboardInterrupt:
        logger.warning("Forced shutdown requested.")
        raise SystemExit(130) from None
    if received_signal is not None:
        raise SystemExit(128 + received_signal)


if __name__ == "__main__":
    main()
