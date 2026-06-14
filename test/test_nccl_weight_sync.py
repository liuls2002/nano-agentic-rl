from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from monarch.actor import Actor, ProcMesh, endpoint, shutdown_context, this_host
from monarch.spmd import setup_torch_elastic_env_async

from actor.rollout_actor import RolloutActor
from actor.train_actor import TrainActor
from main_rl import load_config, reserve_local_port, validate_rl_config


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "qwen2_5_1_5b_gsm8k_sft.yaml"


class EnvSetter(Actor):
    @endpoint
    def set_env(self, env_vars: Mapping[str, str]) -> None:
        os.environ.update({str(key): str(value) for key, value in env_vars.items()})


async def set_mesh_environment(
    proc_mesh: ProcMesh, name: str, env_vars: Mapping[str, str]
) -> None:
    setter = proc_mesh.spawn(f"{name}_env", EnvSetter)
    await setter.set_env.call(env_vars)


async def run_smoke_test(config_path: Path) -> None:
    config_path = config_path.expanduser().resolve()
    settings = validate_rl_config(load_config(config_path))
    host = this_host()
    meshes = []
    train_actors = None
    rollout_actor = None
    try:
        train_mesh = host.spawn_procs(
            per_host={"procs": len(settings["train_gpus"])},
            name="test_nccl_train_procs",
        )
        meshes.append(train_mesh)
        train_env = dict(settings["worker_env"])
        train_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id) for gpu_id in settings["train_gpus"]
        )
        await set_mesh_environment(train_mesh, "test_nccl_train", train_env)
        await setup_torch_elastic_env_async(train_mesh)
        train_actors = train_mesh.spawn(
            "test_nccl_train_actor", TrainActor, str(config_path)
        )
        await train_actors.setup.call()

        rollout_mesh = host.spawn_procs(
            per_host={"procs": 1}, name="test_nccl_rollout_procs"
        )
        meshes.append(rollout_mesh)
        rollout_env = dict(settings["worker_env"])
        rollout_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id) for gpu_id in settings["rollout_gpus"]
        )
        await set_mesh_environment(rollout_mesh, "test_nccl_rollout", rollout_env)
        rollout_actor = rollout_mesh.spawn(
            "test_nccl_rollout_actor", RolloutActor, str(config_path)
        )
        status = await rollout_actor.setup.call_one()

        metadata_results = await train_actors.get_weight_transfer_metadata.call()
        metadata = next(iter(metadata_results.values()))
        address, port = reserve_local_port()
        world_size = 1 + len(settings["rollout_gpus"])
        await asyncio.gather(
            train_actors.init_weight_transfer.call(address, port, world_size),
            rollout_actor.init_weight_transfer.call_one(address, port, world_size),
        )
        update_result, _ = await asyncio.gather(
            rollout_actor.receive_weights.call_one(
                metadata, version=status.policy_version + 1
            ),
            train_actors.broadcast_weights.call(),
        )
        final_status = await rollout_actor.get_status.call_one()
        assert final_status.policy_version == status.policy_version + 1
        assert update_result.num_tensors == len(metadata["names"])
        print(
            "NCCL weight sync passed:",
            f"tensors={update_result.num_tensors}",
            f"elapsed={update_result.elapsed_seconds:.2f}s",
            f"policy=v{final_status.policy_version}",
        )
    finally:
        close_calls = []
        if rollout_actor is not None:
            close_calls.append(rollout_actor.close.call_one())
        if train_actors is not None:
            close_calls.append(train_actors.close.call())
        if close_calls:
            await asyncio.gather(*close_calls, return_exceptions=True)
        for mesh in reversed(meshes):
            await mesh.stop("NCCL weight sync smoke test complete")
        await shutdown_context()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one VeOmni FSDP2 to vLLM NCCL weight synchronization."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    asyncio.run(run_smoke_test(parse_args().config))


if __name__ == "__main__":
    main()
