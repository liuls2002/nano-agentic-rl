from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import yaml
from monarch.actor import Actor, endpoint
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from veomni.arguments import VeOmniArguments, parse_args
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.models.module_utils import save_model_weights
from veomni.trainer.text_trainer import TextTrainer
from veomni.utils.device import synchronize

from rl.loss import grpo_loss


logger = logging.getLogger(__name__)


@dataclass
class GRPOTrainStepResult:
    step: int
    loss: float
    ratio_mean: float
    approx_kl: float
    clip_fraction: float
    active_tokens: float
    grad_norm: float
    learning_rate: float
    elapsed_seconds: float


def process_prompt_response_example(
    example: dict[str, Any],
    *,
    chat_template: Any,
    max_seq_len: int,
    prompt_key: str,
    response_key: str,
    system_prompt: str | None = None,
    **_: Any,
) -> list[dict[str, torch.Tensor]]:
    """Convert a prompt/response row to VeOmni's conversation SFT format."""
    missing_keys = [key for key in (prompt_key, response_key) if key not in example]
    if missing_keys:
        raise KeyError(f"Dataset row is missing required keys: {missing_keys}")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt, "loss_mask": 0})
    messages.extend(
        [
            {"role": "user", "content": str(example[prompt_key]), "loss_mask": 0},
            {"role": "assistant", "content": str(example[response_key]), "loss_mask": 1},
        ]
    )
    tokenized = chat_template.encode_messages(messages, max_seq_len=max_seq_len)
    return [{key: torch.tensor(value) for key, value in tokenized.items()}]


class PromptResponseTextTrainer(TextTrainer):
    """TextTrainer variant for datasets with separate prompt and response columns."""

    def __init__(self, args: VeOmniArguments, adapter_config: dict[str, Any]):
        self._adapter_config = adapter_config
        super().__init__(args)

    def _build_data_transform(self) -> None:
        args = self.base.args
        if args.data.data_type != "conversation":
            raise ValueError("The prompt_response adapter requires data.data_type='conversation'.")

        self.base.data_transform = partial(
            process_prompt_response_example,
            chat_template=self.base.chat_template,
            max_seq_len=args.data.max_seq_len,
            prompt_key=self._adapter_config.get("prompt_key", "question"),
            response_key=self._adapter_config.get("response_key", "answer"),
            system_prompt=self._adapter_config.get("system_prompt"),
        )


def load_veomni_args(
    config_path: str, veomni_config: dict[str, Any] | None = None
) -> VeOmniArguments:
    """Extract and parse VeOmni config after Monarch installs rank variables."""
    if veomni_config is None:
        with open(config_path, encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}
        veomni_config = raw_config.get("veomni")
    if not isinstance(veomni_config, dict):
        raise ValueError("veomni must be a mapping.")

    parser_config = dict(veomni_config)
    parser_config.pop("data_adapter", None)
    original_argv = sys.argv
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", encoding="utf-8"
    ) as config_file:
        yaml.safe_dump(parser_config, config_file, sort_keys=False)
        config_file.flush()
        try:
            sys.argv = [original_argv[0], config_file.name]
            return parse_args(VeOmniArguments)
        finally:
            sys.argv = original_argv


class TrainActor(Actor):
    """Monarch actor that owns one rank of a VeOmni text trainer."""

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.trainer: TextTrainer | None = None
        self._grpo_step = 0
        self._clip_low = 0.2
        self._clip_high = 0.28
        self._weight_transfer_group: Any | None = None
        self._weight_transfer_initialized = False
        self._weight_transfer_dtype = torch.bfloat16
        self._weight_transfer_packed = True
        self._weight_transfer_buffer_size = 256 * 1024 * 1024
        self._weight_transfer_num_buffers = 2

    def _require_trainer(self) -> TextTrainer:
        if self.trainer is None:
            raise RuntimeError("TrainActor.setup() must complete first.")
        return self.trainer

    @endpoint
    def setup(self) -> None:
        if self.trainer is not None:
            raise RuntimeError("TrainActor.setup() may only be called once.")

        with open(self.config_path, encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}

        veomni_config = raw_config.get("veomni")
        if not isinstance(veomni_config, dict):
            raise ValueError("veomni must be a mapping.")

        monarch_config = raw_config.get("monarch", {})
        if not isinstance(monarch_config, dict):
            raise ValueError("monarch must be a mapping.")
        rl_config = monarch_config.get("rl", {})
        if not isinstance(rl_config, dict):
            raise ValueError("monarch.rl must be a mapping.")
        self._clip_low = float(rl_config.get("clip_low", 0.2))
        self._clip_high = float(rl_config.get("clip_high", 0.28))
        weight_sync_config = rl_config.get("weight_sync", {})
        if not isinstance(weight_sync_config, dict):
            raise ValueError("monarch.rl.weight_sync must be a mapping.")
        self._weight_transfer_packed = bool(weight_sync_config.get("packed", True))
        self._weight_transfer_buffer_size = int(
            weight_sync_config.get("packed_buffer_size_bytes", 256 * 1024 * 1024)
        )
        self._weight_transfer_num_buffers = int(
            weight_sync_config.get("packed_num_buffers", 2)
        )

        vllm_config = raw_config.get("vllm", {})
        if not isinstance(vllm_config, dict):
            raise ValueError("vllm must be a mapping.")
        engine_config = vllm_config.get("engine", {})
        if not isinstance(engine_config, dict):
            raise ValueError("vllm.engine must be a mapping.")
        transfer_dtype_name = str(engine_config.get("dtype", "bfloat16"))
        transfer_dtype_name = transfer_dtype_name.removeprefix("torch.")
        transfer_dtype = getattr(torch, transfer_dtype_name, None)
        if not isinstance(transfer_dtype, torch.dtype):
            raise ValueError(
                f"vllm.engine.dtype must name a torch dtype, got {transfer_dtype_name!r}."
            )
        self._weight_transfer_dtype = transfer_dtype

        args = load_veomni_args(self.config_path, veomni_config)
        adapter_config = veomni_config.get("data_adapter")
        if adapter_config is None:
            self.trainer = TextTrainer(args)
        elif not isinstance(adapter_config, dict):
            raise ValueError("veomni.data_adapter must be a mapping.")
        elif adapter_config.get("type") == "prompt_response":
            self.trainer = PromptResponseTextTrainer(args, adapter_config)
        else:
            adapter_type = adapter_config.get("type")
            raise ValueError(f"Unsupported data_adapter.type: {adapter_type!r}")

        logger.info(
            "VeOmni trainer initialized on rank %s/%s (local rank %s).",
            os.environ.get("RANK", "0"),
            os.environ.get("WORLD_SIZE", "1"),
            os.environ.get("LOCAL_RANK", "0"),
        )

    @endpoint
    def train(self) -> None:
        trainer = self._require_trainer()

        try:
            trainer.train()
        finally:
            # TextTrainer destroys the process group on success. This also covers
            # failures in the middle of training so Monarch can stop the mesh cleanly.
            if dist.is_initialized():
                trainer.base.destroy_distributed()

    @staticmethod
    def _validate_grpo_batch(batch: Mapping[str, Any]) -> None:
        required = {
            "tokens",
            "attention_mask",
            "position_ids",
            "generator_logprobs",
            "loss_mask",
            "advantages",
        }
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"GRPO batch is missing keys: {sorted(missing)}")
        shapes = []
        for name in sorted(required):
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"GRPO batch {name!r} must be a torch.Tensor.")
            if value.ndim != 2:
                raise ValueError(f"GRPO batch {name!r} must have shape [B, S].")
            shapes.append(tuple(value.shape))
        if len(set(shapes)) != 1:
            raise ValueError(f"All GRPO batch tensors must share a shape: {shapes}")

    @endpoint
    def train_grpo_step(
        self, batches: Sequence[Mapping[str, torch.Tensor]]
    ) -> GRPOTrainStepResult:
        """Run one distributed GRPO optimizer step on a rank-local batch."""
        trainer = self._require_trainer()
        base = trainer.base
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if len(batches) != world_size:
            raise ValueError(
                f"Expected one GRPO batch per rank ({world_size}), got {len(batches)}."
            )

        batch = batches[rank]
        self._validate_grpo_batch(batch)
        local_batch_size = int(batch["tokens"].shape[0])
        micro_batch_size = int(base.args.train.micro_batch_size)
        if local_batch_size % micro_batch_size:
            raise ValueError(
                "Rank-local GRPO batch size must be divisible by "
                f"veomni.train.micro_batch_size ({local_batch_size} vs {micro_batch_size})."
            )

        started_at = time.perf_counter()
        base.model.train()
        base.optimizer.zero_grad()
        base.state.global_step += 1
        synchronize()

        micro_batches = []
        for start in range(0, local_batch_size, micro_batch_size):
            micro_batches.append(
                {
                    key: value[start : start + micro_batch_size].to(
                        base.device, non_blocking=True
                    )
                    for key, value in batch.items()
                }
            )

        metric_totals = {
            "loss": 0.0,
            "ratio_mean": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "active_tokens": 0.0,
        }
        num_micro_batches = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches):
            base.model_reshard(micro_step, num_micro_batches)
            base._configure_hsdp_allreduce(micro_step, num_micro_batches)
            tokens = micro_batch["tokens"]
            with base.model_fwd_context:
                outputs = base.model(
                    input_ids=tokens,
                    attention_mask=micro_batch["attention_mask"],
                    position_ids=micro_batch["position_ids"],
                    use_cache=False,
                )
            loss_output = grpo_loss(
                logits=outputs.logits,
                tokens=tokens,
                generator_logprobs=micro_batch["generator_logprobs"],
                loss_mask=micro_batch["loss_mask"],
                advantages=micro_batch["advantages"],
                clip_low=self._clip_low,
                clip_high=self._clip_high,
            )
            with base.model_bwd_context:
                (loss_output.loss / num_micro_batches).backward()
            for name, value in loss_output.metrics.items():
                metric_totals[name] += value

        grad_norm = veomni_clip_grad_norm(
            base.model, base.args.train.optimizer.max_grad_norm
        )
        base.optimizer.step()
        base.lr_scheduler.step()
        base.optimizer.zero_grad()
        self._grpo_step += 1

        for name in ("loss", "ratio_mean", "approx_kl", "clip_fraction"):
            metric_totals[name] /= num_micro_batches
        learning_rate = float(base.optimizer.param_groups[0]["lr"])
        grad_norm_value = (
            float(grad_norm.detach().cpu())
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm)
        )
        result = GRPOTrainStepResult(
            step=self._grpo_step,
            loss=metric_totals["loss"],
            ratio_mean=metric_totals["ratio_mean"],
            approx_kl=metric_totals["approx_kl"],
            clip_fraction=metric_totals["clip_fraction"],
            active_tokens=metric_totals["active_tokens"],
            grad_norm=grad_norm_value,
            learning_rate=learning_rate,
            elapsed_seconds=time.perf_counter() - started_at,
        )
        logger.info(
            "GRPO step %d on rank %d: loss=%.6f, KL=%.6f, tokens=%.0f.",
            result.step,
            rank,
            result.loss,
            result.approx_kl,
            result.active_tokens,
        )
        return result

    @endpoint
    def get_weight_transfer_metadata(self) -> dict[str, Any]:
        """Describe the HF-format parameters sent to vLLM in iteration order."""
        trainer = self._require_trainer()
        names = []
        dtype_names = []
        shapes = []
        for name, parameter in trainer.base.model.named_parameters():
            names.append(name)
            dtype = (
                self._weight_transfer_dtype
                if parameter.is_floating_point()
                else parameter.dtype
            )
            dtype_names.append(str(dtype).removeprefix("torch."))
            shapes.append(list(parameter.shape))
        if not names:
            raise RuntimeError("VeOmni model has no parameters to transfer.")
        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
        }

    @endpoint
    def init_weight_transfer(
        self, master_address: str, master_port: int, world_size: int
    ) -> bool:
        """Join the vLLM weight-transfer NCCL group; trainer rank 0 is sender."""
        self._require_trainer()
        if self._weight_transfer_initialized:
            raise RuntimeError("TrainActor weight transfer is already initialized.")
        if world_size <= 1:
            raise ValueError("NCCL weight-transfer world_size must be greater than one.")

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLWeightTransferEngine,
            )

            self._weight_transfer_group = NCCLWeightTransferEngine.trainer_init(
                {
                    "master_address": str(master_address),
                    "master_port": int(master_port),
                    "world_size": int(world_size),
                }
            )
        self._weight_transfer_initialized = True
        logger.info("NCCL weight transfer initialized on training rank %d.", rank)
        return rank == 0

    @endpoint
    def broadcast_weights(self) -> float:
        """All-gather FSDP2 parameters and broadcast them from rank 0 to vLLM."""
        trainer = self._require_trainer()
        if not self._weight_transfer_initialized:
            raise RuntimeError("TrainActor.init_weight_transfer() must complete first.")

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and self._weight_transfer_group is None:
            raise RuntimeError("Training rank 0 has no NCCL weight-transfer group.")
        started_at = time.perf_counter()

        def materialize(parameter: torch.Tensor) -> torch.Tensor:
            full_tensor = getattr(parameter, "full_tensor", None)
            tensor = full_tensor() if callable(full_tensor) else parameter
            if tensor.is_floating_point() and tensor.dtype != self._weight_transfer_dtype:
                tensor = tensor.to(dtype=self._weight_transfer_dtype)
            return tensor.contiguous()

        with torch.no_grad():
            if rank == 0:
                from vllm.distributed.weight_transfer.nccl_engine import (
                    NCCLTrainerSendWeightsArgs,
                    NCCLWeightTransferEngine,
                )

                def full_parameter_iterator():
                    for name, parameter in trainer.base.model.named_parameters():
                        yield name, materialize(parameter)

                NCCLWeightTransferEngine.trainer_send_weights(
                    iterator=full_parameter_iterator(),
                    trainer_args=NCCLTrainerSendWeightsArgs(
                        group=self._weight_transfer_group,
                        packed=self._weight_transfer_packed,
                        packed_buffer_size_bytes=self._weight_transfer_buffer_size,
                        packed_num_buffers=self._weight_transfer_num_buffers,
                    ),
                )
            else:
                # DTensor.full_tensor() is collective, so every FSDP rank must
                # materialize parameters in exactly the same order as rank 0.
                for _, parameter in trainer.base.model.named_parameters():
                    full_tensor = getattr(parameter, "full_tensor", None)
                    tensor = full_tensor() if callable(full_tensor) else parameter
                    del tensor

        elapsed = time.perf_counter() - started_at
        logger.info(
            "Broadcast policy weights from training rank %d in %.2fs.", rank, elapsed
        )
        return elapsed

    @endpoint
    def export_state_dict(self) -> dict[str, torch.Tensor]:
        """Gather a complete CPU state dict; only rank 0 returns its tensors."""
        trainer = self._require_trainer()
        rank = dist.get_rank() if dist.is_initialized() else 0
        state_dict = get_model_state_dict(
            trainer.base.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if rank != 0:
            return {}
        return {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in state_dict.items()
        }

    @endpoint
    def save_policy_checkpoint(self, output_dir: str) -> str:
        """Collectively gather and save one HF checkpoint for vLLM reload."""
        trainer = self._require_trainer()
        rank = dist.get_rank() if dist.is_initialized() else 0
        resolved_output_dir = str(Path(output_dir).expanduser().resolve())
        state_dict = get_model_state_dict(
            trainer.base.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if rank == 0:
            save_model_weights(
                resolved_output_dir,
                state_dict,
                save_dtype="bfloat16",
                safe_serialization=True,
                model_assets=trainer.base.model_assets,
            )
        del state_dict
        if dist.is_initialized():
            dist.barrier()
        return resolved_output_dir

    @endpoint
    def close(self) -> None:
        """Release the VeOmni process group before Monarch stops the mesh."""
        if self._weight_transfer_group is not None:
            self._weight_transfer_group.destroy()
            self._weight_transfer_group = None
        self._weight_transfer_initialized = False
        if self.trainer is not None and dist.is_initialized():
            self.trainer.base.destroy_distributed()
