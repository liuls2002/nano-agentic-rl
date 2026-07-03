from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from monarch.actor import shutdown_context, this_host

from actor.dataset_actor import DatasetActor
from actor.replay_buffer_actor import ReplayBufferActor
from actor.reward_advantage_actor import (
    AdvantageActor,
    RewardActor,
    answers_match,
    extract_answer,
    extract_target_answer,
)
from actor.rollout_actor import RolloutActor
from rl.loss import grpo_loss
from rl.types import RLEpisode
from tools.eval_metrics import compute_pass_at_k, compute_pass_at_k_range


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "qwen2_5_1_5b_gsm8k.yaml"


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


def test_eval_metrics() -> None:
    groups = [
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ]
    metrics = compute_pass_at_k(groups, 2)
    assert abs(metrics.pass_at_k - 11 / 18) < 1e-12
    assert [metric.k for metric in compute_pass_at_k_range(groups, 4)] == [1, 2, 4]


def test_reward_answer_matching() -> None:
    tolerance = Decimal("1e-6")
    assert extract_answer("<answer>0.5</answer>") == "0.5"
    assert extract_answer("final: \\boxed{\\frac{1}{2}}") is None
    assert extract_answer("Answer: 1/2") is None
    assert extract_target_answer("final: \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_target_answer("Answer: 1/2") == "1/2"
    assert answers_match("\\frac{1}{2}", "0.5", tolerance)
    assert answers_match("2^3", "8", tolerance)
    assert answers_match("1,024", "1024", tolerance)

    reward_actor = RewardActor("config")
    reward_actor._initialized = True
    assert reward_actor._evaluate("<answer>0.5</answer>", "1/2").reward == 1.0
    assert reward_actor._evaluate("\\boxed{0.5}", "1/2").reward == 0.0
    assert reward_actor._evaluate("Answer: 0.5", "1/2").reward == 0.0


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


async def test_expanded_sampling(config_path: Path) -> None:
    class FakeSamplingParams:
        def __init__(self, *, n: int, max_tokens: int = 8, **kwargs: object) -> None:
            self.n = n
            self.max_tokens = max_tokens
            self.output_kind = kwargs.get("output_kind")

        def clone(self) -> "FakeSamplingParams":
            cloned = FakeSamplingParams(n=self.n, max_tokens=self.max_tokens)
            cloned.output_kind = self.output_kind
            return cloned

    class FakeAsyncLLM:
        def __init__(self) -> None:
            self.calls = []

        async def generate(self, *, prompt: object, sampling_params: object, request_id: str):
            self.calls.append((prompt, sampling_params.n))
            index = len(self.calls)
            yield SimpleNamespace(
                prompt="prompt text",
                prompt_token_ids=[1, 2],
                outputs=[
                    SimpleNamespace(
                        text=f"sample-{index}",
                        token_ids=[10 + index],
                        logprobs=None,
                        cumulative_logprob=None,
                        finish_reason="stop",
                        stop_reason=None,
                    )
                ],
                num_cached_tokens=1,
            )

    rollout = RolloutActor(str(config_path.resolve()))
    rollout.llm = FakeAsyncLLM()
    rollout._sampling_params_type = FakeSamplingParams
    rollout._request_output_kind = SimpleNamespace(FINAL_ONLY="final_only")
    rollout.policy_version = 7
    params = rollout._make_sampling_params({"n": 3, "max_tokens": 8})

    output = await rollout._generate_expanded("prompt", params, rollout.policy_version)
    assert [call[1] for call in rollout.llm.calls] == [1, 1, 1]
    assert output.policy_version == 7
    assert output.prompt_token_ids == [1, 2]
    assert [sample.text for sample in output.samples] == [
        "sample-1",
        "sample-2",
        "sample-3",
    ]
    assert output.num_cached_tokens == 3


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


def config_has_replay(config_path: Path) -> bool:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    rl_config = config.get("rl") or {}
    rollout_actor = config.get("rollout_actor") or {}
    if not isinstance(rl_config, dict) or not isinstance(rollout_actor, dict):
        return False
    return isinstance(rl_config.get("replay_buffer"), dict) and isinstance(
        rollout_actor.get("rollout"), dict
    )


def replay_expectations(config_path: Path) -> tuple[int, int, int, int]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    rl_config = config["rl"]
    train_actor = config["train_actor"]
    train_config = train_actor["train"]
    sequence = config.get("monarch", {}).get("sequence", {})
    _ = rl_config["replay_buffer"]
    data_parallel_size = int(train_actor["num_gpus"])
    global_batch_size = int(train_config["global_batch_size"])
    if global_batch_size % data_parallel_size:
        raise AssertionError(
            "global_batch_size must be divisible by train_actor.num_gpus"
        )
    batch_size_per_rank = global_batch_size // data_parallel_size
    prompt_length = int(sequence.get("max_prompt_tokens", 1024))
    response_length = int(sequence.get("max_response_tokens", 1024))
    return batch_size_per_rank, data_parallel_size, prompt_length, response_length


async def run_test(config_path: Path) -> None:
    test_grpo_loss()
    test_eval_metrics()
    test_reward_answer_matching()
    await test_nccl_receive_protocol(config_path)
    await test_expanded_sampling(config_path)
    host = this_host()
    meshes = []
    try:
        actors = {}
        for name, actor_type, actor_args in (
            ("train_dataset", DatasetActor, ("train",)),
            ("eval_dataset", DatasetActor, ("eval",)),
            ("reward", RewardActor, ()),
            ("advantage", AdvantageActor, ()),
            ("replay", ReplayBufferActor, ()),
        ):
            mesh = host.spawn_procs(
                per_host={"procs": 1}, name=f"test_rl_{name}_procs"
            )
            meshes.append(mesh)
            actors[name] = mesh.spawn(
                f"test_rl_{name}", actor_type, str(config_path.resolve()), *actor_args
            )

        train_status, eval_status = await asyncio.gather(
            actors["train_dataset"].setup.call_one(),
            actors["eval_dataset"].setup.call_one(),
        )
        sample = (await actors["train_dataset"].next_batch.call_one(1))[0]
        eval_sample = (await actors["eval_dataset"].next_batch.call_one(1))[0]
        pad_token_id = await actors["train_dataset"].get_pad_token_id.call_one()
        assert train_status.dataset_size > 0 and eval_status.dataset_size > 0
        for checked_sample in (sample, eval_sample):
            assert checked_sample.prompt and checked_sample.target
            assert checked_sample.messages
            assert checked_sample.target
            assert all(
                message["role"] != "assistant"
                for message in checked_sample.messages
            )

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

        if config_has_replay(config_path):
            replay_status = await actors["replay"].setup.call_one(pad_token_id)
            (
                batch_size_per_rank,
                data_parallel_size,
                prompt_length,
                response_length,
            ) = replay_expectations(config_path)
            sequence_length = prompt_length + response_length
            assert replay_status.data_parallel_size == data_parallel_size
            assert replay_status.batch_size_per_rank == batch_size_per_rank
            episodes = [
                make_episode(
                    index,
                    rewards[index % 2].reward,
                    advantage.advantages[index % 2],
                )
                for index in range(batch_size_per_rank * data_parallel_size)
            ]
            await actors["replay"].add.call_one(episodes)
            batches = await actors["replay"].sample.call_one(0)
            assert batches is not None and len(batches) == data_parallel_size
            for batch in batches:
                assert batch["tokens"].shape == (batch_size_per_rank, sequence_length)
                assert batch["loss_mask"].sum().item() == 2.0 * batch_size_per_rank
                assert batch["loss_mask"][:, prompt_length - 1].sum().item() == float(
                    batch_size_per_rank
                )
                assert batch["attention_mask"].sum().item() == 5 * batch_size_per_rank
                assert batch["position_ids"][:, prompt_length - 1].tolist() == [2] * batch_size_per_rank
                assert batch["position_ids"][:, prompt_length].tolist() == [3] * batch_size_per_rank

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
    test_eval_metrics()
    test_reward_answer_matching()
    asyncio.run(test_nccl_receive_protocol(config_path))
    asyncio.run(test_expanded_sampling(config_path))
    dataset = DatasetActor(str(config_path.resolve()), "train")
    eval_dataset = DatasetActor(str(config_path.resolve()), "eval")
    reward_actor = RewardActor(str(config_path.resolve()))
    advantage_actor = AdvantageActor(str(config_path.resolve()))
    replay = ReplayBufferActor(str(config_path.resolve()))

    dataset_status = invoke_local(dataset, "setup")
    eval_status = invoke_local(eval_dataset, "setup")
    sample = invoke_local(dataset, "next_batch", 1)[0]
    eval_sample = invoke_local(eval_dataset, "next_batch", 1)[0]
    pad_token_id = invoke_local(dataset, "get_pad_token_id")
    assert dataset_status.dataset_size > 0 and eval_status.dataset_size > 0
    for checked_sample in (sample, eval_sample):
        assert checked_sample.prompt and checked_sample.target
        assert checked_sample.messages
        assert all(
            message["role"] != "assistant"
            for message in checked_sample.messages
        )

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

    if config_has_replay(config_path):
        replay_status = invoke_local(replay, "setup", pad_token_id)
        (
            batch_size_per_rank,
            data_parallel_size,
            prompt_length,
            response_length,
        ) = replay_expectations(config_path)
        sequence_length = prompt_length + response_length
        assert replay_status.data_parallel_size == data_parallel_size
        assert replay_status.batch_size_per_rank == batch_size_per_rank
        episodes = [
            make_episode(
                index,
                rewards[index % 2].reward,
                advantage.advantages[index % 2],
            )
            for index in range(batch_size_per_rank * data_parallel_size)
        ]
        invoke_local(replay, "add", episodes)
        batches = invoke_local(replay, "sample", 0)
        assert batches is not None and len(batches) == data_parallel_size
        for batch in batches:
            assert batch["tokens"].shape == (batch_size_per_rank, sequence_length)
            assert batch["loss_mask"].sum().item() == 2.0 * batch_size_per_rank
            assert batch["loss_mask"][:, prompt_length - 1].sum().item() == float(
                batch_size_per_rank
            )
            assert batch["attention_mask"].sum().item() == 5 * batch_size_per_rank
            assert batch["position_ids"][:, prompt_length - 1].tolist() == [2] * batch_size_per_rank
            assert batch["position_ids"][:, prompt_length].tolist() == [3] * batch_size_per_rank
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
