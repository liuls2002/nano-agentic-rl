from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Any

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

from actor.train_actor import TrainActor


logger = logging.getLogger("main_sft")
DEFAULT_CONFIG = Path(__file__).parent / "config" / "qwen2_5_1_5b_gsm8k_sft.yaml"
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 15.0


class EnvSetter(Actor):
    """Set process-wide environment variables before importing GPU runtimes."""

    @endpoint
    def set_env(self, env_vars: dict[str, str]) -> None:
        os.environ.update(env_vars)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


def get_worker_config(config: dict[str, Any]) -> tuple[int, dict[str, str], float]:
    monarch_config = config.get("monarch", {})
    if not isinstance(monarch_config, dict):
        raise ValueError("monarch must be a mapping.")
    train_config = monarch_config.get("train", {})
    if not isinstance(train_config, dict):
        raise ValueError("monarch.train must be a mapping.")

    gpu_ids = train_config.get("gpu_ids", [0])
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ValueError("monarch.train.gpu_ids must be a non-empty list.")
    if any(not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(
            "monarch.train.gpu_ids entries must be non-negative integers."
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("monarch.train.gpu_ids must not contain duplicates.")
    num_gpus = len(gpu_ids)

    configured_env = monarch_config.get("env", {})
    if not isinstance(configured_env, dict):
        raise ValueError("monarch.env must be a mapping of environment variables.")
    worker_env = {str(key): str(value) for key, value in configured_env.items()}
    worker_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)

    shutdown_timeout = monarch_config.get(
        "shutdown_timeout_seconds", DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    )
    if not isinstance(shutdown_timeout, (int, float)) or shutdown_timeout <= 0:
        raise ValueError("monarch.shutdown_timeout_seconds must be positive.")
    return num_gpus, worker_env, float(shutdown_timeout)


async def wait_for_call_or_shutdown(call: Any, shutdown_event: asyncio.Event) -> bool:
    """Wait for a Monarch call, returning False when controller shutdown wins."""
    call_task = asyncio.ensure_future(call)
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    try:
        done, _ = await asyncio.wait(
            {call_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if call_task in done:
            await call_task
            return True

        call_task.cancel()
        await asyncio.gather(call_task, return_exceptions=True)
        return False
    finally:
        shutdown_task.cancel()
        await asyncio.gather(shutdown_task, return_exceptions=True)


async def shutdown_monarch(proc_mesh: ProcMesh | None, timeout: float) -> None:
    """Stop the worker mesh and global Monarch context with bounded waits."""
    async def shutdown_sequence() -> None:
        if proc_mesh is not None:
            try:
                await proc_mesh.stop("SFT controller shutdown")
            except Exception:
                logger.exception("Failed while stopping the training mesh.")

        try:
            await shutdown_context()
        except Exception:
            logger.exception("Failed while shutting down Monarch context.")

    try:
        await asyncio.wait_for(shutdown_sequence(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Monarch shutdown exceeded the %.1fs timeout.", timeout)


async def run_sft(config_path: Path) -> int | None:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    num_gpus, worker_env, shutdown_timeout = get_worker_config(config)

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    received_signal: int | None = None
    mesh_failure: MeshFailure | None = None
    proc_mesh: ProcMesh | None = None

    previous_fault_hook = monarch_actor.unhandled_fault_hook
    previous_signal_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def request_shutdown(signum: int) -> None:
        nonlocal received_signal
        if shutdown_event.is_set():
            return
        received_signal = signum
        logger.info("Received %s; stopping training.", signal.Signals(signum).name)
        shutdown_event.set()

    def signal_handler(signum: int, frame: Any) -> None:
        if shutdown_event.is_set() and signum == signal.SIGINT:
            signal.default_int_handler(signum, frame)
            return
        loop.call_soon_threadsafe(request_shutdown, signum)

    def record_mesh_failure(failure: MeshFailure) -> None:
        nonlocal mesh_failure
        if received_signal is not None:
            logger.debug(
                "Ignoring supervision event raised during %s shutdown:\n%s",
                signal.Signals(received_signal).name,
                failure.report(),
            )
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

    proceed = True
    try:
        logger.info(
            "Starting VeOmni SFT with %d Monarch processes on CUDA devices %s.",
            num_gpus,
            worker_env["CUDA_VISIBLE_DEVICES"],
        )
        proc_mesh = this_host().spawn_procs(
            per_host={"procs": num_gpus},
            name="sft_train_procs",
        )

        env_setter = proc_mesh.spawn("sft_env_setter", EnvSetter)
        if not await wait_for_call_or_shutdown(
            env_setter.set_env.call(worker_env), shutdown_event
        ):
            proceed = False

        if proceed:
            elastic_setup = asyncio.create_task(setup_torch_elastic_env_async(proc_mesh))
            if not await wait_for_call_or_shutdown(elastic_setup, shutdown_event):
                proceed = False

        if proceed:
            trainers = proc_mesh.spawn("train_actor", TrainActor, str(config_path))

            logger.info("Initializing VeOmni trainers on all ranks.")
            if not await wait_for_call_or_shutdown(trainers.setup.call(), shutdown_event):
                proceed = False

        if proceed:
            logger.info("Running SFT on all ranks.")
            if await wait_for_call_or_shutdown(trainers.train.call(), shutdown_event):
                logger.info("SFT completed successfully.")
    finally:
        try:
            if proc_mesh is not None or shutdown_event.is_set():
                logger.info("Shutting down Monarch resources.")
            cleanup_task = asyncio.create_task(shutdown_monarch(proc_mesh, shutdown_timeout))
            await asyncio.shield(cleanup_task)
        finally:
            for signum, previous_handler in previous_signal_handlers.items():
                signal.signal(signum, previous_handler)
            monarch_actor.unhandled_fault_hook = previous_fault_hook

    if mesh_failure is not None:
        raise RuntimeError(f"Monarch mesh failure:\n{mesh_failure.report()}")
    return received_signal


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VeOmni text SFT with Monarch.")
    parser.add_argument("config", nargs="?", type=Path, help="Path to the YAML config.")
    parser.add_argument("--config", dest="config_option", type=Path, help="Path to the YAML config.")
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
        received_signal = asyncio.run(run_sft(args.config_path))
    except KeyboardInterrupt:
        logger.warning("Forced shutdown requested.")
        raise SystemExit(130) from None
    if received_signal is not None:
        raise SystemExit(128 + received_signal)


if __name__ == "__main__":
    main()
