from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
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
from veomni.trainer.base import VeOmniIter
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


@dataclass
class SFTTrainStepResult:
    step: int
    epoch: int
    loss: float
    grad_norm: float
    learning_rate: float
    elapsed_seconds: float
    finished: bool


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def load_veomni_args(config_path: str) -> VeOmniArguments:
    """Extract and parse VeOmni config after Monarch installs rank variables."""
    with open(config_path, encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    train_actor = _mapping(raw_config.get("train_actor"), "train_actor")
    dataloader = _mapping(raw_config.get("dataloader"), "dataloader")
    train_loader = _mapping(dataloader.get("train"), "dataloader.train")
    train_path = train_loader.get("path")
    if not train_path:
        raise ValueError("dataloader.train.path is required.")

    data_config = {
        "train_path": str(train_path),
        "datasets_type": train_loader.get("datasets_type", "mapping"),
        "data_type": train_loader.get("data_type", "conversation"),
        "text_keys": train_loader.get("text_keys", "messages"),
        "max_seq_len": int(train_loader.get("max_seq_len", 2048)),
        "dataloader": {
            "type": train_loader.get("type", "native"),
            "num_workers": int(train_loader.get("num_workers", 0)),
            "drop_last": bool(train_loader.get("drop_last", True)),
            "pin_memory": bool(train_loader.get("pin_memory", True)),
        },
    }
    for key in ("worker_num_threads", "prefetch_factor", "use_background_prefetcher"):
        if key in train_loader:
            data_config["dataloader"][key] = train_loader[key]

    parser_config = {
        "model": _mapping(train_actor.get("model"), "train_actor.model"),
        "data": data_config,
        "train": _mapping(train_actor.get("train"), "train_actor.train"),
    }
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
        self._sft_started = False
        self._sft_finished = False
        self._sft_epoch = 0
        self._sft_step_in_epoch = 0

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

        rl_config = _mapping(raw_config.get("rl"), "rl")
        self._clip_low = float(rl_config.get("clip_low", 0.2))
        self._clip_high = float(rl_config.get("clip_high", 0.28))
        weight_sync_config = rl_config.get("weight_sync", {})
        if not isinstance(weight_sync_config, dict):
            raise ValueError("rl.weight_sync must be a mapping.")
        self._weight_transfer_packed = bool(weight_sync_config.get("packed", True))
        self._weight_transfer_buffer_size = int(
            weight_sync_config.get("packed_buffer_size_bytes", 256 * 1024 * 1024)
        )
        self._weight_transfer_num_buffers = int(
            weight_sync_config.get("packed_num_buffers", 2)
        )

        rollout_actor = _mapping(raw_config.get("rollout_actor"), "rollout_actor")
        engine_config = _mapping(rollout_actor.get("engine"), "rollout_actor.engine")
        transfer_dtype_name = str(engine_config.get("dtype", "bfloat16"))
        transfer_dtype_name = transfer_dtype_name.removeprefix("torch.")
        transfer_dtype = getattr(torch, transfer_dtype_name, None)
        if not isinstance(transfer_dtype, torch.dtype):
            raise ValueError(
                "rollout_actor.engine.dtype must name a torch dtype, "
                f"got {transfer_dtype_name!r}."
            )
        self._weight_transfer_dtype = transfer_dtype

        self.trainer = TextTrainer(load_veomni_args(self.config_path))
        self._sft_epoch = int(getattr(self.trainer.base, "start_epoch", 0))
        self._sft_step_in_epoch = int(getattr(self.trainer.base, "start_step", 0))

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

    def _start_sft_epoch(self) -> None:
        trainer = self._require_trainer()
        base = trainer.base
        args = base.args
        if self._sft_epoch >= args.train.num_train_epochs:
            self._sft_finished = True
            return
        if hasattr(base.train_dataloader, "set_epoch"):
            base.train_dataloader.set_epoch(self._sft_epoch)
        base.state.epoch = self._sft_epoch
        trainer.on_epoch_begin()
        base.data_iterator = VeOmniIter(
            base.train_dataloader,
            use_background_prefetcher=args.data.dataloader.use_background_prefetcher,
        )

    def _finish_sft_epoch(self) -> None:
        trainer = self._require_trainer()
        base = trainer.base
        args = base.args
        trainer.on_epoch_end()
        if args.data.dataloader.use_background_prefetcher:
            data_iterator = getattr(base, "data_iterator", None)
            if data_iterator is not None:
                data_iterator.stop()
        base.start_step = 0
        self._sft_epoch += 1
        self._sft_step_in_epoch = 0

    def _finish_sft_training(self) -> None:
        trainer = self._require_trainer()
        if self._sft_finished:
            return
        trainer.on_train_end()
        data_iterator = getattr(trainer.base, "data_iterator", None)
        if (
            data_iterator is not None
            and trainer.base.args.data.dataloader.use_background_prefetcher
        ):
            data_iterator.stop()
        synchronize()
        self._sft_finished = True

    @endpoint
    def train_sft_step(self) -> SFTTrainStepResult:
        """Run one VeOmni SFT optimizer step and keep trainer state alive."""
        trainer = self._require_trainer()
        base = trainer.base
        args = base.args
        if self._sft_finished:
            return SFTTrainStepResult(
                step=int(base.state.global_step),
                epoch=int(base.state.epoch),
                loss=float("nan"),
                grad_norm=float("nan"),
                learning_rate=float(base.optimizer.param_groups[0]["lr"]),
                elapsed_seconds=0.0,
                finished=True,
            )

        if not self._sft_started:
            trainer.on_train_begin()
            logger.info(
                "Rank%s Start stepwise SFT. Start step: %s. Train steps: %s. "
                "Start epoch: %s. Train epochs: %s.",
                args.train.local_rank,
                base.start_step,
                args.train_steps,
                base.start_epoch,
                args.train.num_train_epochs,
            )
            self._sft_started = True
            self._start_sft_epoch()

        started_at = time.perf_counter()
        while not self._sft_finished:
            if self._sft_epoch >= args.train.num_train_epochs:
                self._finish_sft_training()
                break
            if self._sft_step_in_epoch >= args.train_steps:
                self._finish_sft_epoch()
                if self._sft_epoch >= args.train.num_train_epochs:
                    self._finish_sft_training()
                    break
                self._start_sft_epoch()
                continue

            try:
                trainer.train_step(base.data_iterator)
            except StopIteration:
                logger.info(
                    "epoch:%d Dataloader finished with drop_last %s",
                    self._sft_epoch,
                    args.data.dataloader.drop_last,
                )
                self._finish_sft_epoch()
                if self._sft_epoch < args.train.num_train_epochs:
                    self._start_sft_epoch()
                continue

            self._sft_step_in_epoch += 1
            step_metrics = dict(getattr(base, "step_train_metrics", {}) or {})
            loss = float(
                step_metrics.get(
                    "training/total_loss",
                    step_metrics.get("total_loss", float("nan")),
                )
            )
            grad_norm = float(
                step_metrics.get(
                    "training/grad_norm",
                    step_metrics.get("grad_norm", float("nan")),
                )
            )
            learning_rate = float(
                step_metrics.get("training/lr", base.optimizer.param_groups[0]["lr"])
            )
            result = SFTTrainStepResult(
                step=int(base.state.global_step),
                epoch=int(base.state.epoch),
                loss=loss,
                grad_norm=grad_norm,
                learning_rate=learning_rate,
                elapsed_seconds=time.perf_counter() - started_at,
                finished=False,
            )
            logger.info(
                "SFT step %d on rank %d: epoch=%d, loss=%.6f, lr=%.3e, "
                "grad_norm=%.4f.",
                result.step,
                dist.get_rank() if dist.is_initialized() else 0,
                result.epoch,
                result.loss,
                result.learning_rate,
                result.grad_norm,
            )
            return result

        return SFTTrainStepResult(
            step=int(base.state.global_step),
            epoch=int(base.state.epoch),
            loss=float("nan"),
            grad_norm=float("nan"),
            learning_rate=float(base.optimizer.param_groups[0]["lr"]),
            elapsed_seconds=time.perf_counter() - started_at,
            finished=True,
        )

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
            "GRPO step %d on rank %d: loss=%.6f, lr=%.3e, "
            "grad_norm=%.4f, KL=%.6f, ratio=%.4f, clip=%.4f, tokens=%.0f.",
            result.step,
            rank,
            result.loss,
            result.learning_rate,
            result.grad_norm,
            result.approx_kl,
            result.ratio_mean,
            result.clip_fraction,
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
        if self.trainer is not None and self._sft_started and not self._sft_finished:
            try:
                self._finish_sft_training()
            except Exception:
                logger.exception("Failed to finish stepwise SFT before close.")
        if self._weight_transfer_group is not None:
            self._weight_transfer_group.destroy()
            self._weight_transfer_group = None
        self._weight_transfer_initialized = False
        if self.trainer is not None and dist.is_initialized():
            self.trainer.base.destroy_distributed()
