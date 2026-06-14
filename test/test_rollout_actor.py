from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from monarch.actor import shutdown_context, this_host

from actor.rollout_actor import RolloutActor


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "qwen2_5_1_5b_gsm8k_sft.yaml"
DEFAULT_CHECKPOINT = Path("/ssd/liuls/data/hub/Qwen2.5-1.5B-Instruct")


async def run_test(config_path: Path, checkpoint_path: Path, version: int) -> None:
    proc_mesh = this_host().spawn_procs(
        per_host={"procs": 1},
        name="rollout_actor_manual_test",
    )
    rollout = proc_mesh.spawn(
        "rollout_actor",
        RolloutActor,
        str(config_path.expanduser().resolve()),
    )

    setup_complete = False
    try:
        status = await rollout.setup.call_one()
        setup_complete = True
        print("setup:", status)

        outputs = await rollout.generate.call_one(
            [
                "What is 12 + 30? Answer with only the number.",
                "What is 7 + 8? Answer with only the number.",
            ],
            {"max_tokens": 16, "temperature": 0.8, "n": 2},
        )
        print("generate before update:", outputs)

        update_result = await rollout.update_weights.call_one(
            checkpoint_path=str(checkpoint_path.expanduser().resolve()),
            version=version,
        )
        print("weight update:", update_result)

        outputs = await rollout.chat.call_one(
            [{"role": "user", "content": "What is 2 + 2?"}],
            {"max_tokens": 16, "temperature": 0.0, "n": 1},
        )
        print("chat after update:", outputs)
        print("final status:", await rollout.get_status.call_one())
    finally:
        if setup_complete:
            try:
                await rollout.close.call()
            except Exception:
                logging.exception("Failed to close RolloutActor cleanly.")
        try:
            await proc_mesh.stop("rollout actor manual test complete")
        finally:
            await shutdown_context()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually test RolloutActor generation and checkpoint reload."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--version", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    asyncio.run(run_test(args.config, args.checkpoint, args.version))


if __name__ == "__main__":
    main()
