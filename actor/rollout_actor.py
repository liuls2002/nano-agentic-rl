from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monarch.actor import Actor, endpoint

from actor.utils import load_yaml_config, mapping


logger = logging.getLogger(__name__)


@dataclass
class RolloutSample:
    """One sampled response returned across the Monarch RPC boundary."""

    text: str
    token_ids: list[int]
    logprobs: list[float] | None
    cumulative_logprob: float | None
    finish_reason: str | None
    stop_reason: str | int | None


@dataclass
class RolloutOutput:
    """All samples generated for one prompt by one policy version."""

    prompt: str
    prompt_token_ids: list[int]
    samples: list[RolloutSample]
    policy_version: int
    num_cached_tokens: int


@dataclass
class WeightUpdateResult:
    """Metadata for a completed in-place policy update."""

    policy_version: int
    source: str
    num_tensors: int | None
    elapsed_seconds: float


@dataclass
class RolloutActorStatus:
    """Small status payload suitable for controller health checks."""

    initialized: bool
    policy_version: int
    model_path: str


def load_rollout_config(config_path: str) -> dict[str, Any]:
    """Load the shared Train/Rollout YAML configuration."""
    return load_yaml_config(Path(config_path).expanduser().resolve())


def _apply_environment(config: Mapping[str, Any]) -> None:
    monarch_config = mapping(config.get("monarch"), "monarch")
    rollout_actor = mapping(config.get("rollout_actor"), "rollout_actor")

    environment = mapping(monarch_config.get("env"), "monarch.env")
    environment.update(mapping(rollout_actor.get("env"), "rollout_actor.env"))
    for name, value in environment.items():
        os.environ[str(name)] = str(value)


def _sequence_config(config: Mapping[str, Any]) -> dict[str, int]:
    monarch_config = mapping(config.get("monarch"), "monarch")
    sequence = mapping(monarch_config.get("sequence"), "monarch.sequence")
    prompt_length = int(sequence.get("max_prompt_tokens", 1024))
    response_length = int(sequence.get("max_response_tokens", 1024))
    if prompt_length <= 0 or response_length <= 0:
        raise ValueError("monarch.sequence token limits must be positive.")
    return {
        "max_prompt_tokens": prompt_length,
        "max_response_tokens": response_length,
        "max_seq_len": prompt_length + response_length,
    }


def _build_engine_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    rollout_actor = mapping(config.get("rollout_actor"), "rollout_actor")
    engine_config = mapping(
        rollout_actor.get("engine"), "rollout_actor.engine"
    )
    sequence = _sequence_config(config)

    model_path = engine_config.get("model")
    if not model_path:
        raise ValueError(
            "rollout_actor.engine.model is required to initialize RolloutActor."
        )

    engine_config.setdefault("tokenizer", str(model_path))
    engine_config.setdefault("data_parallel_size", 1)
    engine_config.setdefault("data_parallel_backend", "mp")
    engine_config.setdefault("tensor_parallel_size", 1)
    engine_config.setdefault("pipeline_parallel_size", 1)
    engine_config.setdefault("dtype", "bfloat16")
    engine_config.setdefault("gpu_memory_utilization", 0.85)
    engine_config.setdefault("enforce_eager", True)
    engine_config.setdefault("max_model_len", sequence["max_seq_len"])
    engine_config.setdefault("seed", 0)
    engine_config.setdefault("disable_log_stats", True)

    num_gpus = int(rollout_actor.get("num_gpus", 0))
    if num_gpus <= 0:
        raise ValueError("rollout_actor.num_gpus must be positive.")
    required_gpus = (
        int(engine_config["data_parallel_size"])
        * int(engine_config["tensor_parallel_size"])
        * int(engine_config["pipeline_parallel_size"])
    )
    if num_gpus != required_gpus:
        raise ValueError(
            "rollout_actor.num_gpus must equal "
            "engine.data_parallel_size * tensor_parallel_size * "
            "pipeline_parallel_size entries "
            f"({required_gpus} required, got {num_gpus})."
        )
    return engine_config


def _build_default_sampling_config(config: Mapping[str, Any]) -> dict[str, Any]:
    rollout_actor = mapping(config.get("rollout_actor"), "rollout_actor")
    sequence = _sequence_config(config)
    rollout_config = mapping(
        rollout_actor.get("rollout"), "rollout_actor.rollout"
    )
    if rollout_config:
        sampling_config = mapping(
            rollout_config.get("sampling"), "rollout_actor.rollout.sampling"
        )
    else:
        eval_config = mapping(
            rollout_actor.get("eval"), "rollout_actor.eval"
        )
        sampling_config = mapping(
            eval_config.get("sampling"), "rollout_actor.eval.sampling"
        )

    sampling_config.setdefault("n", 1)
    sampling_config.setdefault("max_tokens", sequence["max_response_tokens"])
    sampling_config.setdefault("temperature", 1.0)
    sampling_config.setdefault("top_p", 1.0)
    sampling_config.setdefault("logprobs", 0 if not rollout_config else 1)
    return sampling_config


def _normalize_prompts(prompts: str | Sequence[str]) -> list[str]:
    if isinstance(prompts, str):
        prompts = [prompts]
    elif not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a string or a sequence of strings.")

    normalized = list(prompts)
    if any(not isinstance(prompt, str) for prompt in normalized):
        raise TypeError("Every prompt must be a string.")
    return normalized


def _normalize_conversations(
    conversations: Sequence[Mapping[str, Any]]
    | Sequence[Sequence[Mapping[str, Any]]],
) -> list[list[dict[str, Any]]]:
    if not isinstance(conversations, Sequence) or isinstance(
        conversations, (str, bytes)
    ):
        raise TypeError("conversations must be a conversation or a batch of them.")
    if not conversations:
        return []

    first_item = conversations[0]
    if isinstance(first_item, Mapping):
        batch: list[Sequence[Mapping[str, Any]]] = [
            conversations  # type: ignore[list-item]
        ]
    else:
        batch = list(conversations)  # type: ignore[arg-type]

    normalized = []
    for conversation in batch:
        if not isinstance(conversation, Sequence) or isinstance(
            conversation, (str, bytes)
        ):
            raise TypeError("Each conversation must be a sequence of messages.")
        messages = []
        for message in conversation:
            if not isinstance(message, Mapping):
                raise TypeError("Each chat message must be a mapping.")
            if "role" not in message or "content" not in message:
                raise ValueError("Each chat message requires role and content fields.")
            messages.append(dict(message))
        normalized.append(messages)
    return normalized


def _extract_sample_logprobs(output: Any) -> list[float] | None:
    if output.logprobs is None:
        return None

    selected_logprobs = []
    for token_id, top_logprobs in zip(output.token_ids, output.logprobs):
        selected = top_logprobs.get(token_id)
        if selected is None:
            raise RuntimeError(f"vLLM did not return a logprob for token {token_id}.")
        selected_logprobs.append(float(selected.logprob))
    return selected_logprobs


def _convert_request_output(request_output: Any, policy_version: int) -> RolloutOutput:
    samples = []
    for output in request_output.outputs:
        cumulative_logprob = getattr(output, "cumulative_logprob", None)
        samples.append(
            RolloutSample(
                text=output.text,
                token_ids=list(output.token_ids or []),
                logprobs=_extract_sample_logprobs(output),
                cumulative_logprob=(
                    float(cumulative_logprob)
                    if cumulative_logprob is not None
                    else None
                ),
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
            )
        )

    return RolloutOutput(
        prompt=request_output.prompt or "",
        prompt_token_ids=list(request_output.prompt_token_ids or []),
        samples=samples,
        policy_version=policy_version,
        num_cached_tokens=int(getattr(request_output, "num_cached_tokens", 0) or 0),
    )


class RolloutActor(Actor):
    """Monarch actor that owns a vLLM offline inference engine.

    The actor serializes generation and weight updates. Therefore, once an
    update endpoint returns, all later rollouts are guaranteed to use the new
    policy version.
    """

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.model_path = ""
        self.policy_version = 0
        self.llm: Any | None = None
        self._sampling_params_type: Any | None = None
        self._request_output_kind: Any | None = None
        self._default_sampling_config: dict[str, Any] = {}
        self._max_prompt_tokens = 512
        self._max_response_tokens = 256
        self._max_model_len = 768
        self._weight_transfer_initialized = False
        self._weight_transfer_packed = True
        self._weight_transfer_buffer_size = 256 * 1024 * 1024
        self._weight_transfer_num_buffers = 2
        self._lock = asyncio.Lock()

    def _require_llm(self) -> Any:
        if self.llm is None:
            raise RuntimeError("RolloutActor.setup() must complete first.")
        return self.llm

    def _make_sampling_params(
        self, overrides: Mapping[str, Any] | None
    ) -> Any:
        if self._sampling_params_type is None:
            raise RuntimeError("RolloutActor.setup() must complete first.")
        sampling_config = dict(self._default_sampling_config)
        sampling_config.update(mapping(overrides, "sampling_params"))
        params = self._sampling_params_type(**sampling_config)
        if int(params.max_tokens) > self._max_response_tokens:
            raise ValueError(
                f"sampling max_tokens cannot exceed {self._max_response_tokens}."
            )
        params.output_kind = self._request_output_kind.FINAL_ONLY
        return params

    def _make_single_sample_params(self, sampling_params: Any) -> tuple[Any, int]:
        requested_n = int(getattr(sampling_params, "n", 1))
        if requested_n <= 0:
            raise ValueError("sampling n must be positive.")
        if requested_n == 1:
            return sampling_params, 1

        single_params = sampling_params.clone()
        single_params.n = 1
        single_params.output_kind = self._request_output_kind.FINAL_ONLY
        return single_params, requested_n

    def _next_policy_version(
        self, version: int | None, *, allow_same_version: bool = False
    ) -> int:
        next_version = self.policy_version + 1 if version is None else int(version)
        if allow_same_version and next_version == self.policy_version:
            return next_version
        if next_version <= self.policy_version:
            raise ValueError(
                f"Policy version must increase beyond {self.policy_version}, "
                f"got {next_version}."
            )
        return next_version

    async def _generate_one(
        self, prompt: Any, sampling_params: Any, request_id: str | None = None
    ) -> Any:
        request_output = None
        async for output in self._require_llm().generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id or str(uuid.uuid4()),
        ):
            request_output = output
        if request_output is None:
            raise RuntimeError("vLLM returned no output for a rollout request.")
        return request_output

    async def _generate_expanded(
        self,
        prompt: Any,
        sampling_params: Any,
        policy_version: int,
        group_id: str,
    ) -> RolloutOutput:
        single_params, requested_n = self._make_single_sample_params(sampling_params)
        request_outputs = await asyncio.gather(
            *(
                self._generate_one(
                    prompt,
                    single_params.clone() if requested_n > 1 else single_params,
                    request_id=f"{group_id}-sample-{sample_index}-{uuid.uuid4().hex}",
                )
                for sample_index in range(requested_n)
            )
        )

        converted_outputs = [
            _convert_request_output(output, policy_version)
            for output in request_outputs
        ]
        if not converted_outputs:
            raise RuntimeError("vLLM returned no outputs for a rollout request.")

        merged = converted_outputs[0]
        for output in converted_outputs[1:]:
            merged.samples.extend(output.samples)
            merged.num_cached_tokens += output.num_cached_tokens
        return merged

    @endpoint
    async def setup(self) -> RolloutActorStatus:
        async with self._lock:
            if self.llm is not None:
                raise RuntimeError("RolloutActor.setup() may only be called once.")

            config = load_rollout_config(self.config_path)
            _apply_environment(config)
            engine_kwargs = _build_engine_kwargs(config)
            self._default_sampling_config = _build_default_sampling_config(config)
            rl_config = mapping(config.get("rl"), "rl")
            rollout_actor = mapping(
                config.get("rollout_actor"), "rollout_actor"
            )
            rollout_config = mapping(
                rollout_actor.get("rollout"), "rollout_actor.rollout"
            )
            sequence = _sequence_config(config)
            self._max_prompt_tokens = sequence["max_prompt_tokens"]
            self._max_response_tokens = sequence["max_response_tokens"]
            weight_sync_config = mapping(
                rl_config.get("weight_sync"), "rl.weight_sync"
            )
            self._weight_transfer_packed = bool(
                weight_sync_config.get("packed", True)
            )
            self._weight_transfer_buffer_size = int(
                weight_sync_config.get(
                    "packed_buffer_size_bytes", 256 * 1024 * 1024
                )
            )
            self._weight_transfer_num_buffers = int(
                weight_sync_config.get("packed_num_buffers", 2)
            )
            self._max_model_len = int(engine_kwargs.get("max_model_len", 0))
            if self._max_model_len < sequence["max_seq_len"]:
                raise ValueError(
                    "rollout_actor.engine.max_model_len must be at least "
                    "monarch.sequence.max_prompt_tokens + max_response_tokens."
                )

            # Import after applying environment variables because vLLM reads many
            # runtime settings during module import and engine construction.
            from vllm import SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.sampling_params import RequestOutputKind
            from vllm.v1.engine.async_llm import AsyncLLM

            logger.info(
                "Initializing vLLM rollout engine: model=%s, DP=%s, TP=%s.",
                engine_kwargs["model"],
                engine_kwargs["data_parallel_size"],
                engine_kwargs["tensor_parallel_size"],
            )
            self.llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
            self._sampling_params_type = SamplingParams
            self._request_output_kind = RequestOutputKind
            self.model_path = str(engine_kwargs["model"])

            self.policy_version = int(
                rollout_actor.get("initial_policy_version", 0)
            )
            logger.info(
                "vLLM rollout engine initialized at policy version %s.",
                self.policy_version,
            )
            return self._status()

    def _make_tokenize_params(self, sampling_params: Any) -> Any:
        from vllm.renderers.params import TokenizeParams

        return TokenizeParams(
            max_total_tokens=self._max_model_len,
            max_output_tokens=int(sampling_params.max_tokens),
            truncate_prompt_tokens=self._max_prompt_tokens,
            truncation_side="left",
        )

    @endpoint
    async def generate(
        self,
        prompts: str | Sequence[str],
        sampling_params: Mapping[str, Any] | None = None,
    ) -> list[RolloutOutput]:
        """Sample one or more text prompts with token-level log probabilities."""
        normalized_prompts = _normalize_prompts(prompts)
        if not normalized_prompts:
            return []

        async with self._lock:
            llm = self._require_llm()
            policy_version = self.policy_version
            started_at = time.perf_counter()
            resolved_sampling_params = self._make_sampling_params(sampling_params)
            engine_prompts = await llm.renderer.render_cmpl_async(
                [{"prompt": prompt} for prompt in normalized_prompts],
                self._make_tokenize_params(resolved_sampling_params),
            )
            outputs = await asyncio.gather(
                *(
                    self._generate_expanded(
                        prompt,
                        resolved_sampling_params,
                        policy_version,
                        group_id=f"completion-{prompt_index}",
                    )
                    for prompt_index, prompt in enumerate(engine_prompts)
                )
            )
            logger.info(
                "Generated %s sequence(s) for %s prompt(s) with policy v%s in %.2fs.",
                sum(len(output.samples) for output in outputs),
                len(outputs),
                policy_version,
                time.perf_counter() - started_at,
            )
            return outputs

    @endpoint
    async def chat(
        self,
        conversations: Sequence[Mapping[str, Any]]
        | Sequence[Sequence[Mapping[str, Any]]],
        sampling_params: Mapping[str, Any] | None = None,
    ) -> list[RolloutOutput]:
        """Apply the model chat template and sample one or more conversations."""
        normalized_conversations = _normalize_conversations(conversations)
        if not normalized_conversations:
            return []

        async with self._lock:
            llm = self._require_llm()
            policy_version = self.policy_version
            from vllm.renderers.params import ChatParams

            resolved_sampling_params = self._make_sampling_params(sampling_params)

            _, engine_prompts = await llm.renderer.render_chat_async(
                normalized_conversations,
                ChatParams(
                    chat_template_kwargs={
                        "add_generation_prompt": True,
                        "continue_final_message": False,
                        "tools": None,
                        "tokenize": False,
                    }
                    ),
                    self._make_tokenize_params(resolved_sampling_params),
                )
            return await asyncio.gather(
                *(
                    self._generate_expanded(
                        prompt,
                        resolved_sampling_params,
                        policy_version,
                        group_id=f"chat-{prompt_index}",
                    )
                    for prompt_index, prompt in enumerate(engine_prompts)
                )
            )

    @endpoint
    async def update_weights(
        self,
        state_dict: Mapping[str, Any] | None = None,
        *,
        checkpoint_path: str | None = None,
        version: int | None = None,
        is_checkpoint_format: bool = True,
    ) -> WeightUpdateResult:
        """Reload a HF state dict or checkpoint into all vLLM workers.

        Exactly one of ``state_dict`` and ``checkpoint_path`` must be supplied.
        State dict tensors are normalized to contiguous CPU tensors before vLLM
        shards and loads them on each tensor-parallel worker.
        """
        if (state_dict is None) == (checkpoint_path is None):
            raise ValueError(
                "Exactly one of state_dict and checkpoint_path must be provided."
            )

        async with self._lock:
            llm = self._require_llm()
            next_version = self._next_policy_version(version)
            started_at = time.perf_counter()

            if checkpoint_path is not None:
                expanded_path = str(Path(checkpoint_path).expanduser())
                reload_kwargs = {
                    "weights_path": expanded_path,
                    "is_checkpoint_format": bool(is_checkpoint_format),
                }
                source = expanded_path
                num_tensors = None
            else:
                import torch

                if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
                    raise RuntimeError(
                        "In-memory vLLM weight updates require "
                        "VLLM_ALLOW_INSECURE_SERIALIZATION=1 in the trusted "
                        "rollout process environment."
                    )
                weights = []
                for name, tensor in state_dict.items():
                    if not isinstance(name, str):
                        raise TypeError("Every state_dict key must be a string.")
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(
                            f"state_dict[{name!r}] must be a torch.Tensor."
                        )
                    weights.append((name, tensor.detach().cpu().contiguous()))
                if not weights:
                    raise ValueError("state_dict must contain at least one tensor.")
                reload_kwargs = {
                    # AsyncLLM sends utility RPC arguments through an untyped
                    # msgpack boundary. A plain list would encode each tensor as
                    # [dtype, shape, buffer_index] without restoring its type.
                    # list_iterator uses vLLM's trusted pickle fallback and keeps
                    # the contained torch.Tensor objects intact.
                    "weights_iterator": iter(weights),
                    "is_checkpoint_format": bool(is_checkpoint_format),
                }
                source = "state_dict"
                num_tensors = len(weights)

            # The actor lock ensures no rollout request is in flight.
            # reload_weights is vLLM's documented in-place checkpoint API.
            await llm.collective_rpc("reload_weights", kwargs=reload_kwargs)
            if not await llm.reset_prefix_cache():
                logger.warning("vLLM reported that the prefix cache was not reset.")
            self.policy_version = next_version

            result = WeightUpdateResult(
                policy_version=next_version,
                source=source,
                num_tensors=num_tensors,
                elapsed_seconds=time.perf_counter() - started_at,
            )
            logger.info(
                "Updated rollout weights to policy v%s from %s in %.2fs.",
                result.policy_version,
                result.source,
                result.elapsed_seconds,
            )
            return result

    @endpoint
    async def init_weight_transfer(
        self, master_address: str, master_port: int, world_size: int
    ) -> None:
        """Join vLLM workers to the trainer-owned NCCL transfer group."""
        async with self._lock:
            if self._weight_transfer_initialized:
                raise RuntimeError(
                    "RolloutActor weight transfer is already initialized."
                )
            if world_size <= 1:
                raise ValueError(
                    "NCCL weight-transfer world_size must be greater than one."
                )

            from vllm.distributed.weight_transfer.base import (
                WeightTransferInitRequest,
            )

            await self._require_llm().init_weight_transfer_engine(
                WeightTransferInitRequest(
                    init_info={
                        "master_address": str(master_address),
                        "master_port": int(master_port),
                        "rank_offset": 1,
                        "world_size": int(world_size),
                    }
                )
            )
            self._weight_transfer_initialized = True
            logger.info("vLLM workers joined the NCCL weight-transfer group.")

    @endpoint
    async def receive_weights(
        self,
        metadata: Mapping[str, Any],
        *,
        version: int | None = None,
        allow_same_version: bool = False,
    ) -> WeightUpdateResult:
        """Receive one HF-format policy update directly from the trainer GPUs."""
        async with self._lock:
            if not self._weight_transfer_initialized:
                raise RuntimeError(
                    "RolloutActor.init_weight_transfer() must complete first."
                )
            names = list(metadata.get("names", []))
            dtype_names = list(metadata.get("dtype_names", []))
            shapes = [list(shape) for shape in metadata.get("shapes", [])]
            if (
                not names
                or len(names) != len(dtype_names)
                or len(names) != len(shapes)
            ):
                raise ValueError(
                    "Weight metadata names, dtype_names, and shapes must be non-empty "
                    "lists of equal length."
                )

            from vllm.distributed.weight_transfer.base import (
                WeightTransferUpdateRequest,
            )

            llm = self._require_llm()
            next_version = self._next_policy_version(
                version,
                allow_same_version=allow_same_version,
            )
            started_at = time.perf_counter()
            update_started = False
            paused = False
            try:
                await llm.pause_generation(mode="abort")
                paused = True
                await llm.start_weight_update(is_checkpoint_format=True)
                update_started = True
                await llm.update_weights(
                    WeightTransferUpdateRequest(
                        update_info={
                            "names": names,
                            "dtype_names": dtype_names,
                            "shapes": shapes,
                            "packed": self._weight_transfer_packed,
                            "packed_buffer_size_bytes": (
                                self._weight_transfer_buffer_size
                            ),
                            "packed_num_buffers": self._weight_transfer_num_buffers,
                        }
                    )
                )
                await llm.finish_weight_update()
                update_started = False
                if not await llm.reset_prefix_cache():
                    logger.warning("vLLM reported that the prefix cache was not reset.")
                self.policy_version = next_version
            finally:
                if update_started:
                    try:
                        await llm.finish_weight_update()
                    except Exception:
                        logger.exception(
                            "Failed to finish an interrupted weight update."
                        )
                if paused:
                    await llm.resume_generation()

            result = WeightUpdateResult(
                policy_version=next_version,
                source="nccl",
                num_tensors=len(names),
                elapsed_seconds=time.perf_counter() - started_at,
            )
            logger.info(
                "Received %d policy tensors over NCCL for policy v%d in %.2fs.",
                len(names),
                next_version,
                result.elapsed_seconds,
            )
            return result

    @endpoint
    async def reset_prefix_cache(self) -> bool:
        async with self._lock:
            return bool(await self._require_llm().reset_prefix_cache())

    def _status(self) -> RolloutActorStatus:
        return RolloutActorStatus(
            initialized=self.llm is not None,
            policy_version=self.policy_version,
            model_path=self.model_path,
        )

    @endpoint
    async def get_status(self) -> RolloutActorStatus:
        async with self._lock:
            return self._status()

    @endpoint
    async def close(self) -> None:
        """Stop vLLM subprocesses before the controller tears down the mesh."""
        async with self._lock:
            if self.llm is None:
                return

            llm = self.llm
            self.llm = None
            self._sampling_params_type = None
            self._request_output_kind = None
            self._weight_transfer_initialized = False
            try:
                async_tokenizer = getattr(llm.renderer, "_async_tokenizer", None)
                tokenizer_tasks = list(
                    getattr(async_tokenizer, "_batcher_tasks", [])
                )
                for task in tokenizer_tasks:
                    task.cancel()
                if tokenizer_tasks:
                    await asyncio.gather(*tokenizer_tasks, return_exceptions=True)
                llm.shutdown()
            finally:
                del llm
                gc.collect()

                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    logger.debug("CUDA cache cleanup failed.", exc_info=True)
            logger.info("vLLM rollout engine closed.")
