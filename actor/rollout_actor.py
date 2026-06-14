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

import yaml
from monarch.actor import Actor, endpoint


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


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def load_rollout_config(config_path: str) -> dict[str, Any]:
    """Load the shared Train/Rollout YAML configuration."""
    resolved_path = Path(config_path).expanduser().resolve()
    with resolved_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, Mapping):
        raise TypeError("The top-level YAML configuration must be a mapping.")
    return dict(config)


def _apply_environment(config: Mapping[str, Any]) -> None:
    monarch_config = _require_mapping(config.get("monarch"), "monarch")
    rollout_config = _require_mapping(
        monarch_config.get("rollout"), "monarch.rollout"
    )

    environment = _require_mapping(monarch_config.get("env"), "monarch.env")
    environment.update(
        _require_mapping(rollout_config.get("env"), "monarch.rollout.env")
    )
    for name, value in environment.items():
        os.environ[str(name)] = str(value)

    gpu_ids = rollout_config.get("gpu_ids")
    if gpu_ids is not None:
        if not isinstance(gpu_ids, Sequence) or isinstance(gpu_ids, (str, bytes)):
            raise TypeError(
                "monarch.rollout.gpu_ids must be a sequence of GPU identifiers."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)


def _build_engine_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    vllm_config = _require_mapping(config.get("vllm"), "vllm")
    engine_config = _require_mapping(vllm_config.get("engine"), "vllm.engine")

    model_path = engine_config.get("model")
    if not model_path:
        raise ValueError("vllm.engine.model is required to initialize RolloutActor.")

    engine_config.setdefault("tokenizer", str(model_path))
    engine_config.setdefault("data_parallel_size", 1)
    engine_config.setdefault("data_parallel_backend", "mp")
    engine_config.setdefault("tensor_parallel_size", 1)
    engine_config.setdefault("pipeline_parallel_size", 1)
    engine_config.setdefault("dtype", "bfloat16")
    engine_config.setdefault("gpu_memory_utilization", 0.85)
    engine_config.setdefault("enforce_eager", True)
    engine_config.setdefault("seed", 0)
    engine_config.setdefault("disable_log_stats", True)

    monarch_config = _require_mapping(config.get("monarch"), "monarch")
    rollout_resources = _require_mapping(
        monarch_config.get("rollout"), "monarch.rollout"
    )
    gpu_ids = rollout_resources.get("gpu_ids")
    if not isinstance(gpu_ids, Sequence) or isinstance(gpu_ids, (str, bytes)):
        raise TypeError(
            "monarch.rollout.gpu_ids must be a sequence of GPU identifiers."
        )
    required_gpus = (
        int(engine_config["data_parallel_size"])
        * int(engine_config["tensor_parallel_size"])
        * int(engine_config["pipeline_parallel_size"])
    )
    if len(gpu_ids) != required_gpus:
        raise ValueError(
            "monarch.rollout.gpu_ids must contain exactly "
            "vllm.engine.data_parallel_size * tensor_parallel_size * "
            "pipeline_parallel_size entries "
            f"({required_gpus} required, got {len(gpu_ids)})."
        )
    return engine_config


def _build_default_sampling_config(config: Mapping[str, Any]) -> dict[str, Any]:
    rollout_config = _require_mapping(config.get("vllm"), "vllm")
    sampling_config = _require_mapping(
        rollout_config.get("sampling"), "vllm.sampling"
    )

    sampling_config.setdefault("n", 1)
    sampling_config.setdefault("max_tokens", 512)
    sampling_config.setdefault("temperature", 1.0)
    sampling_config.setdefault("top_p", 1.0)
    sampling_config.setdefault("logprobs", 1)
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
        sampling_config.update(_require_mapping(overrides, "sampling_params"))
        params = self._sampling_params_type(**sampling_config)
        params.output_kind = self._request_output_kind.FINAL_ONLY
        return params

    def _next_policy_version(self, version: int | None) -> int:
        next_version = self.policy_version + 1 if version is None else int(version)
        if next_version <= self.policy_version:
            raise ValueError(
                f"Policy version must increase beyond {self.policy_version}, "
                f"got {next_version}."
            )
        return next_version

    async def _generate_one(self, prompt: Any, sampling_params: Any) -> Any:
        request_output = None
        async for output in self._require_llm().generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=str(uuid.uuid4()),
        ):
            request_output = output
        if request_output is None:
            raise RuntimeError("vLLM returned no output for a rollout request.")
        return request_output

    @endpoint
    async def setup(self) -> RolloutActorStatus:
        async with self._lock:
            if self.llm is not None:
                raise RuntimeError("RolloutActor.setup() may only be called once.")

            config = load_rollout_config(self.config_path)
            _apply_environment(config)
            engine_kwargs = _build_engine_kwargs(config)
            self._default_sampling_config = _build_default_sampling_config(config)

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

            rollout_config = _require_mapping(config.get("vllm"), "vllm")
            self.policy_version = int(rollout_config.get("initial_policy_version", 0))
            logger.info(
                "vLLM rollout engine initialized at policy version %s.",
                self.policy_version,
            )
            return self._status()

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
            engine_prompts = await llm.renderer.render_cmpl_async(
                [{"prompt": prompt} for prompt in normalized_prompts]
            )
            request_outputs = await asyncio.gather(
                *(
                    self._generate_one(
                        prompt, self._make_sampling_params(sampling_params)
                    )
                    for prompt in engine_prompts
                )
            )
            outputs = [
                _convert_request_output(output, policy_version)
                for output in request_outputs
            ]
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
            )
            request_outputs = await asyncio.gather(
                *(
                    self._generate_one(
                        prompt, self._make_sampling_params(sampling_params)
                    )
                    for prompt in engine_prompts
                )
            )
            return [
                _convert_request_output(output, policy_version)
                for output in request_outputs
            ]

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
                    "weights_iterator": weights,
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
