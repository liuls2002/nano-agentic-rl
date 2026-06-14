from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from monarch.actor import shutdown_context, this_host

from actor.dataset_actor import DatasetActor
from actor.replay_buffer_actor import ReplayBufferActor
from actor.reward_advantage_actor import AdvantageActor, RewardActor
from actor.rollout_actor import RolloutActor
from rl.loss import grpo_loss
from rl.types import RLEpisode


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "qwen2_5_1_5b_gsm8k_sft.yaml"


def invoke_local(actor: object, endpoint_name: str, *args: object) -> object:
    """Invoke endpoint logic directly, without creating Monarch sockets."""
    endpoint_property = type(actor).__dict__[endpoint_name]
    return endpoint_property._method(actor, *args)


def test_grpo_loss() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11, requires_grad=True)
    tokens = torch.tensor([[1, 2, 3, 4, 0], [1, 5, 6, 7, 8]])
    loss_mask = torch.tensor(
        [[0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 0.0]]
    )
    old_logprobs = torch.zeros_like(loss_mask)
    advantages = torch.tensor(
        [[1.0] * 5, [-1.0] * 5], dtype=torch.float32
    )
    output = grpo_loss(
        logits=logits,
        tokens=tokens,
        generator_logprobs=old_logprobs,
        loss_mask=loss_mask,
        advantages=advantages,
    )
    assert torch.isfinite(output.loss)
    assert output.metrics["active_tokens"] == 5.0
    output.loss.backward()
    assert logits.grad is not None


async def test_nccl_receive_protocol(config_path: Path) -> None:
    class FakeAsyncLLM:
        def __init__(self) -> None:
            self.calls = []

        async def pause_generation(self, mode: str) -> None:
            self.calls.append(("pause", mode))

        async def start_weight_update(self, is_checkpoint_format: bool) -> None:
            self.calls.append(("start", is_checkpoint_format))

        async def update_weights(self, request: object) -> None:
            self.calls.append(("update", request.update_info))

        async def finish_weight_update(self) -> None:
            self.calls.append(("finish", None))

        async def reset_prefix_cache(self) -> bool:
            self.calls.append(("reset", None))
            return True

        async def resume_generation(self) -> None:
            self.calls.append(("resume", None))

    rollout = RolloutActor(str(config_path.resolve()))
    fake_llm = FakeAsyncLLM()
    rollout.llm = fake_llm
    rollout._weight_transfer_initialized = True
    rollout._max_prompt_tokens = 512
    rollout._max_model_len = 1024
    tokenize_params = rollout._make_tokenize_params(SimpleNamespace(max_tokens=256))
    assert tokenize_params.truncate_prompt_tokens == 512
    assert tokenize_params.max_output_tokens == 256

    result = await type(rollout).__dict__["receive_weights"]._method(
        rollout,
        {
            "names": ["model.embed_tokens.weight"],
            "dtype_names": ["bfloat16"],
            "shapes": [[8, 4]],
        },
        version=1,
    )
    assert result.policy_version == 1
    assert result.source == "nccl"
    assert [name for name, _ in fake_llm.calls] == [
        "pause",
        "start",
        "update",
        "finish",
        "reset",
        "resume",
    ]
    update_info = fake_llm.calls[2][1]
    assert update_info["packed"] is True
    assert update_info["packed_buffer_size_bytes"] == 256 * 1024 * 1024
    assert update_info["packed_num_buffers"] == 2


def make_episode(index: int, reward: float, advantage: float) -> RLEpisode:
    return RLEpisode(
        episode_id=f"episode-{index}",
        sample_id="sample-0",
        prompt="test prompt",
        target="42",
        response=str(42 - index),
        prompt_token_ids=[10, 11, 12],
        response_token_ids=[20 + index, 30 + index],
        generator_logprobs=[-0.2, -0.3],
        reward=reward,
        advantage=advantage,
        policy_version=0,
        finish_reason="stop",
    )


async def run_test(config_path: Path) -> None:
    test_grpo_loss()
    await test_nccl_receive_protocol(config_path)
    host = this_host()
    meshes = []
    try:
        actors = {}
        for name, actor_type in (
            ("dataset", DatasetActor),
            ("reward", RewardActor),
            ("advantage", AdvantageActor),
            ("replay", ReplayBufferActor),
        ):
            mesh = host.spawn_procs(
                per_host={"procs": 1}, name=f"test_rl_{name}_procs"
            )
            meshes.append(mesh)
            actors[name] = mesh.spawn(
                f"test_rl_{name}", actor_type, str(config_path.resolve())
            )

        dataset_status = await actors["dataset"].setup.call_one()
        sample = (await actors["dataset"].next_batch.call_one(1))[0]
        pad_token_id = await actors["dataset"].get_pad_token_id.call_one()
        assert dataset_status.dataset_size > 0
        assert sample.prompt and sample.target and sample.messages

        await actors["reward"].setup.call_one()
        await actors["advantage"].setup.call_one()
        rewards = await actors["reward"].evaluate_batch.call_one(
            ["The answer is <answer>42</answer>.", "The answer is 41."],
            ["42", "42"],
        )
        assert [result.reward for result in rewards] == [1.0, 0.0]
        advantage = await actors["advantage"].compute.call_one(
            [result.reward for result in rewards]
        )
        assert not advantage.low_variance
        assert advantage.advantages[0] > 0 > advantage.advantages[1]

        replay_status = await actors["replay"].setup.call_one(pad_token_id)
        assert replay_status.data_parallel_size == 2
        episodes = [
            make_episode(
                index,
                rewards[index % 2].reward,
                advantage.advantages[index % 2],
            )
            for index in range(16)
        ]
        await actors["replay"].add.call_one(episodes)
        batches = await actors["replay"].sample.call_one(0)
        assert batches is not None and len(batches) == 2
        for batch in batches:
            assert batch["tokens"].shape == (8, 768)
            assert batch["loss_mask"].sum().item() == 16.0
            assert batch["loss_mask"][:, 511].sum().item() == 8.0
            assert batch["attention_mask"].sum().item() == 40
            assert batch["position_ids"][:, 511].tolist() == [2] * 8
            assert batch["position_ids"][:, 512].tolist() == [3] * 8

        await actors["replay"].add.call_one(episodes)
        assert await actors["replay"].sample.call_one(1) is None
        assert (await actors["replay"].get_status.call_one()).size == 0
        print("RL actor CPU test passed:", sample.sample_id, sample.target)
    finally:
        for mesh in reversed(meshes):
            try:
                await mesh.stop("RL actor CPU test complete")
            except Exception:
                logging.exception("Failed to stop test process mesh.")
        await shutdown_context()


def run_local_test(config_path: Path) -> None:
    """Run the same component checks in-process for restricted environments."""
    test_grpo_loss()
    asyncio.run(test_nccl_receive_protocol(config_path))
    dataset = DatasetActor(str(config_path.resolve()))
    reward_actor = RewardActor(str(config_path.resolve()))
    advantage_actor = AdvantageActor(str(config_path.resolve()))
    replay = ReplayBufferActor(str(config_path.resolve()))

    dataset_status = invoke_local(dataset, "setup")
    sample = invoke_local(dataset, "next_batch", 1)[0]
    pad_token_id = invoke_local(dataset, "get_pad_token_id")
    assert dataset_status.dataset_size > 0
    assert sample.prompt and sample.target and sample.messages

    invoke_local(reward_actor, "setup")
    invoke_local(advantage_actor, "setup")
    rewards = invoke_local(
        reward_actor,
        "evaluate_batch",
        ["The answer is <answer>42</answer>.", "The answer is 41."],
        ["42", "42"],
    )
    advantage = invoke_local(
        advantage_actor, "compute", [result.reward for result in rewards]
    )
    assert [result.reward for result in rewards] == [1.0, 0.0]
    assert advantage.advantages[0] > 0 > advantage.advantages[1]

    replay_status = invoke_local(replay, "setup", pad_token_id)
    assert replay_status.data_parallel_size == 2
    episodes = [
        make_episode(
            index,
            rewards[index % 2].reward,
            advantage.advantages[index % 2],
        )
        for index in range(16)
    ]
    invoke_local(replay, "add", episodes)
    batches = invoke_local(replay, "sample", 0)
    assert batches is not None and len(batches) == 2
    for batch in batches:
        assert batch["tokens"].shape == (8, 768)
        assert batch["loss_mask"].sum().item() == 16.0
        assert batch["loss_mask"][:, 511].sum().item() == 8.0
        assert batch["attention_mask"].sum().item() == 40
        assert batch["position_ids"][:, 511].tolist() == [2] * 8
        assert batch["position_ids"][:, 512].tolist() == [3] * 8
    invoke_local(replay, "add", episodes)
    assert invoke_local(replay, "sample", 1) is None
    assert invoke_local(replay, "get_status").size == 0
    print("RL actor local test passed:", sample.sample_id, sample.target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Dataset/Reward/Advantage/ReplayBuffer actors without GPUs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run endpoint logic in-process without Monarch sockets.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    if args.local:
        run_local_test(args.config)
    else:
        asyncio.run(run_test(args.config))


if __name__ == "__main__":
    main()
