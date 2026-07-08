from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Mapping

from monarch.actor import ProcMesh, this_host
from monarch.spmd import setup_torch_elastic_env_async

from actor.dataset_actor import DatasetActor
from actor.reward_advantage_actor import RewardActor
from actor.rollout_actor import RolloutActor
from actor.train_actor import TrainActor
from tools.config import validate_sft_config
from tools.controller_utils import (
    close_resources,
    core_eval_wandb_metrics,
    init_wandb_run,
    load_config,
    log_wandb,
    reserve_local_port,
    run_eval,
    set_mesh_environment,
    sft_core_step_wandb_metrics,
    sft_core_train_wandb_metrics,
    sft_train_wandb_metrics,
    summarize_sft_train_results,
    sync_policy_weights,
)
from tools.runtime_metrics import GpuMonitor, prefix_metrics


logger = logging.getLogger("main_sft")
DEFAULT_CONFIG = Path(__file__).parent / "config" / "qwen3_1_7b_gsm8k_sft.yaml"


async def run_sft(config_path: Path) -> None:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    settings = validate_sft_config(config)
    logger.info(
        "SFT schedule: num_train_epochs=%d, steps_per_epoch=%d, total_steps=%d.",
        settings["num_train_epochs"],
        settings["steps_per_epoch"],
        settings["max_steps"],
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
        await set_mesh_environment(train_mesh, "sft_train", train_env)
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
        await set_mesh_environment(rollout_mesh, "sft_rollout", rollout_env)
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
        last_eval_step = 0
        last_train_step = 0
        last_train_summary: dict[str, float | bool] | None = None

        for _ in range(settings["max_steps"]):
            dataset_samples = await train_dataset.next_batch.call_one(
                settings["train_global_batch_size"]
            )
            per_rank = settings["train_batch_size_per_rank"]
            sample_batches = [
                dataset_samples[rank * per_rank : (rank + 1) * per_rank]
                for rank in range(len(settings["train_gpus"]))
            ]
            async with GpuMonitor(
                settings["train_gpus"],
                interval_seconds=settings["gpu_sample_interval_seconds"],
            ) as train_monitor:
                train_results = await train_actors.train_sft_step.call(sample_batches)
            train_summary = summarize_sft_train_results(train_results)
            if train_summary["finished"]:
                logger.info("SFT step schedule finished.")
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
            last_train_step = step
            last_train_summary = train_summary
            step_metrics: dict[str, float] = {
                "policy/version": float(policy_version),
                **sft_train_wandb_metrics(train_summary),
                **prefix_metrics("train", train_monitor.summary()),
                **sft_core_train_wandb_metrics(
                    policy_version=policy_version,
                    train_summary=train_summary,
                ),
            }
            if step % settings["eval_steps"] == 0:
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
                    "SFT eval sync complete at step %d: policy=v%d, NCCL sync=%.2fs.",
                    step,
                    policy_version,
                    update_result.elapsed_seconds,
                )
                step_metrics.update(
                    {
                        "policy/version": float(policy_version),
                        "sync/time_sec": update_result.elapsed_seconds,
                        **prefix_metrics("sync", sync_monitor.summary()),
                        **sft_core_step_wandb_metrics(
                            policy_version=policy_version,
                            train_summary=train_summary,
                            sync_time_sec=update_result.elapsed_seconds,
                        ),
                    }
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
                        step=step,
                        logger=logger,
                    )
                step_metrics.update(
                    {
                        **core_eval_wandb_metrics(eval_metrics),
                        **prefix_metrics("eval", eval_metrics),
                        **prefix_metrics("eval", eval_monitor.summary()),
                    }
                )
                last_eval_step = step
            log_wandb(wandb_run, step_metrics, step=step)

        if (
            last_train_step
            and last_eval_step != last_train_step
            and last_train_summary is not None
        ):
            logger.info("Running final SFT eval at step %d.", last_train_step)
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
                "SFT final eval sync complete at step %d: policy=v%d, NCCL sync=%.2fs.",
                last_train_step,
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
                    step=last_train_step,
                    logger=logger,
                )
            log_wandb(
                wandb_run,
                {
                    "policy/version": float(policy_version),
                    "sync/time_sec": update_result.elapsed_seconds,
                    **prefix_metrics("sync", sync_monitor.summary()),
                    **sft_core_step_wandb_metrics(
                        policy_version=policy_version,
                        train_summary=last_train_summary,
                        sync_time_sec=update_result.elapsed_seconds,
                    ),
                    **core_eval_wandb_metrics(eval_metrics),
                    **prefix_metrics("eval", eval_metrics),
                    **prefix_metrics("eval", eval_monitor.summary()),
                },
                step=last_train_step,
            )

        logger.info("SFT training completed successfully.")
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
