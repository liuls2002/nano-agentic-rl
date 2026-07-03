from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
from pathlib import Path
from typing import Any, Awaitable, Mapping, Sequence

import yaml
from monarch.actor import Actor, ProcMesh, endpoint, shutdown_context, this_host
from monarch.spmd import setup_torch_elastic_env_async

from actor.dataset_actor import DatasetActor
from actor.reward_advantage_actor import RewardActor
from actor.rollout_actor import RolloutActor
from actor.train_actor import TrainActor
from tools.eval_metrics import compute_pass_at_k_range


logger = logging.getLogger("main_sft")
DEFAULT_CONFIG = Path(__file__).parent / "config" / "qwen2_5_1_5b_gsm8k_sft.yaml"
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


def validate_sft_config(config: Mapping[str, Any]) -> dict[str, Any]:
    monarch = _mapping(config.get("monarch"), "monarch")
    sequence = _load_sequence_config(monarch)
    train_actor = _mapping(config.get("train_actor"), "train_actor")
    rollout_actor = _mapping(config.get("rollout_actor"), "rollout_actor")
    train_config = _mapping(train_actor.get("train"), "train_actor.train")
    dataloader = _mapping(config.get("dataloader"), "dataloader")
    train_data = _mapping(dataloader.get("train"), "dataloader.train")
    eval_data = _mapping(dataloader.get("eval"), "dataloader.eval")
    engine = _mapping(rollout_actor.get("engine"), "rollout_actor.engine")
    eval_config = _mapping(rollout_actor.get("eval"), "rollout_actor.eval")
    eval_sampling = _mapping(
        eval_config.get("sampling"), "rollout_actor.eval.sampling"
    )

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

    max_steps = _positive_int(train_config.get("max_steps", 1), "train_actor.train.max_steps")
    eval_steps = _positive_int(
        eval_config.get("eval_steps", 1), "rollout_actor.eval.eval_steps"
    )
    eval_epochs = _positive_int(
        eval_config.get("eval_epochs", 1), "rollout_actor.eval.eval_epochs"
    )
    eval_batch_size = _positive_int(
        eval_config.get("batch_size", 1), "rollout_actor.eval.batch_size"
    )
    eval_max_tokens = _positive_int(
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

    transfer_config = _mapping(
        engine.get("weight_transfer_config"),
        "rollout_actor.engine.weight_transfer_config",
    )
    if transfer_config.get("backend") != "nccl":
        raise ValueError(
            "rollout_actor.engine.weight_transfer_config.backend must be nccl."
        )

    configured_env = _mapping(monarch.get("env"), "monarch.env")
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
        "shutdown_timeout": shutdown_timeout,
        "max_steps": max_steps,
        "eval_steps": eval_steps,
        "eval_epochs": eval_epochs,
        "eval_batch_size": eval_batch_size,
        "eval_sampling": eval_sampling,
    }


def summarize_sft_results(train_results: Mapping[Any, Any]) -> dict[str, float | bool]:
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
    }


def reserve_local_port() -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", int(sock.getsockname()[1])


async def _set_mesh_environment(
    proc_mesh: ProcMesh, name: str, env_vars: Mapping[str, str]
) -> None:
    setter = proc_mesh.spawn(f"{name}_env", EnvSetter)
    await setter.set_env.call(env_vars)


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
    reward_groups: list[list[float]] = []
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
    logger.info(
        "Eval %s: accuracy=%.4f, reward_mean=%.4f.",
        step_label,
        accuracy,
        reward_mean,
    )
    for metric in metrics:
        logger.info(
            "Eval %s: pass@%d=%.4f.",
            step_label,
            metric.k,
            metric.pass_at_k,
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
        proc_mesh.stop("SFT controller shutdown")
        for proc_mesh in reversed(proc_meshes)
    ]
    await run_phase("Process mesh stop", stop_calls)

    try:
        await asyncio.wait_for(shutdown_context(), timeout=phase_timeout)
    except asyncio.TimeoutError:
        logger.warning("Monarch context shutdown exceeded %.1fs.", phase_timeout)
    except Exception:
        logger.exception("Failed to shut down Monarch context.")


async def run_sft(config_path: Path) -> None:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    settings = validate_sft_config(config)
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
        )
        for name, actor_type, actor_args in actor_specs:
            proc_mesh = host.spawn_procs(
                per_host={"procs": 1}, name=f"sft_{name}_procs"
            )
            proc_meshes.append(proc_mesh)
            support_actors[name] = proc_mesh.spawn(
                f"sft_{name}", actor_type, str(config_path), *actor_args
            )

        train_dataset = support_actors["train_dataset"]
        eval_dataset = support_actors["eval_dataset"]
        reward_actor = support_actors["reward"]
        await asyncio.gather(
            train_dataset.setup.call_one(),
            eval_dataset.setup.call_one(),
            reward_actor.setup.call_one(),
        )

        train_mesh = host.spawn_procs(
            per_host={"procs": len(settings["train_gpus"])},
            name="sft_train_procs",
        )
        proc_meshes.append(train_mesh)
        train_env = dict(settings["worker_env"])
        train_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id) for gpu_id in settings["train_gpus"]
        )
        await _set_mesh_environment(train_mesh, "sft_train", train_env)
        await setup_torch_elastic_env_async(train_mesh)
        train_actors = train_mesh.spawn("sft_train_actor", TrainActor, str(config_path))
        logger.info(
            "Initializing %d VeOmni SFT training ranks.",
            len(settings["train_gpus"]),
        )
        await train_actors.setup.call()

        rollout_mesh = host.spawn_procs(
            per_host={"procs": 1}, name="sft_rollout_procs"
        )
        proc_meshes.append(rollout_mesh)
        rollout_env = dict(settings["worker_env"])
        rollout_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id) for gpu_id in settings["rollout_gpus"]
        )
        await _set_mesh_environment(rollout_mesh, "sft_rollout", rollout_env)
        rollout_actor = rollout_mesh.spawn(
            "sft_rollout_actor", RolloutActor, str(config_path)
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

        for _ in range(settings["max_steps"]):
            train_results = await train_actors.train_sft_step.call()
            train_summary = summarize_sft_results(train_results)
            if train_summary["finished"]:
                logger.info("SFT dataloader/epoch schedule finished.")
                break

            step = int(train_summary["step"])
            logger.info(
                "SFT step %d/%d complete across %.0f rank(s): "
                "epoch=%.0f, loss_mean=%.6f, lr_mean=%.3e, "
                "grad_norm_mean=%.4f, grad_norm_max=%.4f, train_max=%.2fs, "
                "policy=v%d.",
                step,
                settings["max_steps"],
                train_summary["ranks"],
                train_summary["epoch"],
                train_summary["loss_mean"],
                train_summary["lr_mean"],
                train_summary["grad_norm_mean"],
                train_summary["grad_norm_max"],
                train_summary["elapsed_seconds_max"],
                policy_version,
            )
            if step % settings["eval_steps"] == 0:
                update_result = await sync_policy_weights(
                    rollout_actor=rollout_actor,
                    train_actors=train_actors,
                    weight_metadata=weight_metadata,
                    version=policy_version + 1,
                )
                policy_version = update_result.policy_version
                logger.info(
                    "SFT eval sync complete at step %d: policy=v%d, NCCL sync=%.2fs.",
                    step,
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
                    step=step,
                )

        logger.info("SFT training completed successfully.")
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
        description="Run Monarch + VeOmni SFT with vLLM eval."
    )
    parser.add_argument("config", nargs="?", type=Path, help="Path to the YAML config.")
    parser.add_argument(
        "--config", dest="config_option", type=Path, help="Path to the YAML config."
    )
    args = parser.parse_args()
    if args.config is not None and args.config_option is not None:
        parser.error("Specify the config either positionally or with --config, not both.")
    args.config_path = args.config_option or args.config or DEFAULT_CONFIG
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_cli_args()
    try:
        asyncio.run(run_sft(args.config_path))
    except KeyboardInterrupt:
        logger.warning("Forced shutdown requested.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
