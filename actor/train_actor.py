from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
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
from veomni.trainer.base import BaseTrainer
from veomni.trainer.text_trainer import TextTrainer
from veomni.utils.device import synchronize
from veomni.utils.loss_utils import count_loss_token

from actor.utils import (
    NoOpCallback,
    load_yaml_config,
    mapping,
    positive_int,
    sft_steps_from_epochs,
)
from rl.loss import create_shifted_targets, grpo_loss
from rl.types import DatasetSample


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
    memory_allocated_mb: float
    memory_reserved_mb: float
    max_memory_allocated_mb: float
    max_memory_reserved_mb: float


@dataclass
class SFTTrainStepResult:
    step: int
    epoch: int
    loss: float
    grad_norm: float
    learning_rate: float
    elapsed_seconds: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    max_memory_allocated_mb: float
    max_memory_reserved_mb: float
    finished: bool


@dataclass
class TrainActorStatus:
    scheduler_horizon: int
    steps_per_epoch: int
    num_train_epochs: int
    global_batch_size: int
    micro_batch_size: int
    data_parallel_size: int


def load_veomni_args(config_path: str) -> VeOmniArguments:
    """Extract and parse VeOmni config after Monarch installs rank variables."""
    raw_config = load_yaml_config(config_path)

    monarch = mapping(raw_config.get("monarch"), "monarch")
    sequence = mapping(monarch.get("sequence"), "monarch.sequence")
    default_max_seq_len = int(sequence.get("max_prompt_tokens", 1024)) + int(
        sequence.get("max_response_tokens", 1024)
    )
    train_actor = mapping(raw_config.get("train_actor"), "train_actor")
    dataloader = mapping(raw_config.get("dataloader"), "dataloader")
    train_loader = mapping(dataloader.get("train"), "dataloader.train")
    train_path = train_loader.get("path")
    if not train_path:
        raise ValueError("dataloader.train.path is required.")

    data_config = {
        "train_path": str(train_path),
        "datasets_type": train_loader.get("datasets_type", "mapping"),
        "data_type": train_loader.get("data_type", "conversation"),
        "chat_template": train_loader.get("chat_template", "default"),
        "text_keys": train_loader.get("text_keys", "messages"),
        "max_seq_len": int(train_loader.get("max_seq_len", default_max_seq_len)),
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
        "model": mapping(train_actor.get("model"), "train_actor.model"),
        "data": data_config,
        "train": mapping(train_actor.get("train"), "train_actor.train"),
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


def _training_schedule(raw_config: Mapping[str, Any]) -> tuple[int, int, int]:
    rl_config = mapping(raw_config.get("rl"), "rl")
    if rl_config:
        horizon = positive_int(rl_config.get("max_steps", 0), "rl.max_steps")
        return horizon, horizon, 1

    train_actor = mapping(raw_config.get("train_actor"), "train_actor")
    train_config = mapping(train_actor.get("train"), "train_actor.train")
    dataloader = mapping(raw_config.get("dataloader"), "dataloader")
    train_data = mapping(dataloader.get("train"), "dataloader.train")
    global_batch_size = positive_int(
        train_config.get("global_batch_size", 0),
        "train_actor.train.global_batch_size",
    )
    horizon, num_epochs, steps_per_epoch = sft_steps_from_epochs(
        train_config=train_config,
        train_data=train_data,
        global_batch_size=global_batch_size,
    )
    return horizon, steps_per_epoch, num_epochs


def _training_horizon(raw_config: Mapping[str, Any]) -> int:
    horizon, _, _ = _training_schedule(raw_config)
    return horizon


def _rank_and_world() -> tuple[int, int]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return rank, world_size


class VeOmniPolicyTrainer(TextTrainer):
    """VeOmni runtime without VeOmni dataloader/epoch ownership."""

    def __init__(
        self,
        args: VeOmniArguments,
        scheduler_horizon: int,
        *,
        steps_per_epoch: int | None = None,
        num_train_epochs: int | None = None,
    ):
        self.scheduler_horizon = int(scheduler_horizon)
        if self.scheduler_horizon <= 0:
            raise ValueError("scheduler_horizon must be positive.")
        self.steps_per_epoch = int(steps_per_epoch or self.scheduler_horizon)
        self.num_train_epochs = int(num_train_epochs or 1)
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive.")
        if self.steps_per_epoch * self.num_train_epochs != self.scheduler_horizon:
            raise ValueError(
                "steps_per_epoch * num_train_epochs must equal scheduler_horizon "
                f"({self.steps_per_epoch} * {self.num_train_epochs} vs "
                f"{self.scheduler_horizon})."
            )

        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args
        # Keep VeOmni scheduler/checkpoint math in controller-step units.
        self.base.args._train_steps = self.steps_per_epoch
        self.base.args.train.max_steps = None
        self.base.args.train.num_train_epochs = self.num_train_epochs

        self.base._setup()
        self.base._build_model()
        self.base._freeze_model_module()
        self._build_model_assets()
        self._build_data_transform()
        self.base._build_collate_fn()
        self.base._build_parallelized_model()
        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base.train_dataset = None
        self.base.train_dataloader = None
        self.base.LOG_SAMPLE = False
        self.base._init_callbacks()
        self.base.tqdm_callback = NoOpCallback()
        self.base.state.epoch = 0
        self.base.start_epoch = 0
        self.base.start_step = 0
        self._train_started = False
        self._train_finished = False
        self._epoch_started = False
        self.ensure_train_started()

    def ensure_train_started(self) -> None:
        if not self._train_started:
            self.on_train_begin()
            self._train_started = True
        if not self._epoch_started:
            self.on_epoch_begin()
            self._epoch_started = True
            logger.info(
                "Rank%s started VeOmni policy trainer with scheduler horizon %d "
                "(%d epoch(s) x %d step(s)).",
                self.base.args.train.local_rank,
                self.scheduler_horizon,
                self.num_train_epochs,
                self.steps_per_epoch,
            )

    def finish_training(self) -> None:
        if self._train_started and not self._train_finished:
            if self._epoch_started:
                self.on_epoch_end()
                self._epoch_started = False
            self.on_train_end()
            synchronize()
            self._train_finished = True

    def advance_epoch_if_needed(self) -> None:
        if not self._epoch_started:
            return
        if self.global_step <= 0 or self.global_step % self.steps_per_epoch != 0:
            return
        self.on_epoch_end()
        self._epoch_started = False
        if self.global_step < self.scheduler_horizon:
            self.base.state.epoch += 1
            self.base.start_step = 0
            self.on_epoch_begin()
            self._epoch_started = True

    def _memory_metrics(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {
                "memory_allocated_mb": 0.0,
                "memory_reserved_mb": 0.0,
                "max_memory_allocated_mb": 0.0,
                "max_memory_reserved_mb": 0.0,
            }
        return {
            "memory_allocated_mb": torch.cuda.memory_allocated(self.base.device)
            / (1024**2),
            "memory_reserved_mb": torch.cuda.memory_reserved(self.base.device)
            / (1024**2),
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated(
                self.base.device
            )
            / (1024**2),
            "max_memory_reserved_mb": torch.cuda.max_memory_reserved(
                self.base.device
            )
            / (1024**2),
        }

    @property
    def global_step(self) -> int:
        return int(self.base.state.global_step)

    @property
    def finished(self) -> bool:
        return self.global_step >= self.scheduler_horizon


class VeOmniGRPOTrainer(VeOmniPolicyTrainer):
    def __init__(
        self,
        args: VeOmniArguments,
        scheduler_horizon: int,
        *,
        clip_low: float,
        clip_high: float,
    ):
        super().__init__(args, scheduler_horizon)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)

    @staticmethod
    def validate_batch(batch: Mapping[str, Any]) -> None:
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

    def train_step(self, batch: Mapping[str, torch.Tensor]) -> GRPOTrainStepResult:
        self.validate_batch(batch)
        base = self.base
        rank, _ = _rank_and_world()
        local_batch_size = int(batch["tokens"].shape[0])
        micro_batch_size = int(base.args.train.micro_batch_size)
        if local_batch_size % micro_batch_size:
            raise ValueError(
                "Rank-local GRPO batch size must be divisible by "
                f"veomni.train.micro_batch_size ({local_batch_size} vs {micro_batch_size})."
            )

        self.ensure_train_started()
        started_at = time.perf_counter()
        base.model.train()
        base.optimizer.zero_grad()
        base.state.global_step += 1
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(base.device)

        micro_batches = [
            {
                key: value[start : start + micro_batch_size].to(
                    base.device, non_blocking=True
                )
                for key, value in batch.items()
            }
            for start in range(0, local_batch_size, micro_batch_size)
        ]
        callback_batches = [
            {
                "input_ids": micro_batch["tokens"],
                "attention_mask": micro_batch["attention_mask"],
                "labels": create_shifted_targets(
                    micro_batch["tokens"], micro_batch["loss_mask"]
                ),
            }
            for micro_batch in micro_batches
        ]
        self.on_step_begin(micro_batches=callback_batches)
        synchronize()

        metric_sums = {
            "loss": 0.0,
            "ratio_mean_weighted": 0.0,
            "approx_kl_weighted": 0.0,
            "clip_fraction_weighted": 0.0,
            "active_tokens": 0.0,
        }
        active_tokens_total = torch.stack(
            [micro_batch["loss_mask"].sum() for micro_batch in micro_batches]
        ).sum().clamp_min(1.0)
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
                clip_low=self.clip_low,
                clip_high=self.clip_high,
                normalizer=active_tokens_total,
            )
            with base.model_bwd_context:
                loss_output.loss.backward()
            active_tokens = float(loss_output.metrics["active_tokens"])
            metric_sums["loss"] += float(loss_output.metrics["loss"])
            metric_sums["ratio_mean_weighted"] += (
                float(loss_output.metrics["ratio_mean"]) * active_tokens
            )
            metric_sums["approx_kl_weighted"] += (
                float(loss_output.metrics["approx_kl"]) * active_tokens
            )
            metric_sums["clip_fraction_weighted"] += (
                float(loss_output.metrics["clip_fraction"]) * active_tokens
            )
            metric_sums["active_tokens"] += active_tokens

        grad_norm = veomni_clip_grad_norm(
            base.model, base.args.train.optimizer.max_grad_norm
        )
        base.optimizer.step()
        base.lr_scheduler.step()
        base.optimizer.zero_grad()
        synchronize()

        grad_norm_value = (
            float(grad_norm.detach().cpu())
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm)
        )
        active_tokens_float = max(metric_sums["active_tokens"], 1.0)
        loss_dict = {
            "grpo_loss": metric_sums["loss"],
            "approx_kl": metric_sums["approx_kl_weighted"] / active_tokens_float,
            "ratio_mean": metric_sums["ratio_mean_weighted"] / active_tokens_float,
            "clip_fraction": metric_sums["clip_fraction_weighted"]
            / active_tokens_float,
            "active_tokens": metric_sums["active_tokens"],
        }
        self.on_step_end(
            loss=metric_sums["loss"], loss_dict=loss_dict, grad_norm=grad_norm
        )
        memory = self._memory_metrics()
        result = GRPOTrainStepResult(
            step=int(base.state.global_step),
            loss=metric_sums["loss"],
            ratio_mean=loss_dict["ratio_mean"],
            approx_kl=loss_dict["approx_kl"],
            clip_fraction=loss_dict["clip_fraction"],
            active_tokens=metric_sums["active_tokens"],
            grad_norm=grad_norm_value,
            learning_rate=float(base.optimizer.param_groups[0]["lr"]),
            elapsed_seconds=time.perf_counter() - started_at,
            **memory,
        )
        self.advance_epoch_if_needed()
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


class VeOmniSFTTrainer(VeOmniPolicyTrainer):
    def _features_from_samples(
        self, samples: Sequence[DatasetSample]
    ) -> list[dict[str, torch.Tensor]]:
        features: list[dict[str, torch.Tensor]] = []
        for sample in samples:
            messages = sample.teacher_messages or [
                *sample.messages,
                {"role": "assistant", "content": sample.target, "loss_mask": 1},
            ]
            transformed = self.base.data_transform({"messages": messages})
            features.extend(transformed)
        return features

    def train_step(self, samples: Sequence[DatasetSample]) -> SFTTrainStepResult:
        base = self.base
        if self.finished:
            return SFTTrainStepResult(
                step=int(base.state.global_step),
                epoch=min(int(base.state.epoch) + 1, self.num_train_epochs),
                loss=float("nan"),
                grad_norm=float("nan"),
                learning_rate=float(base.optimizer.param_groups[0]["lr"]),
                elapsed_seconds=0.0,
                **self._memory_metrics(),
                finished=True,
            )
        if not samples:
            raise ValueError("SFT train_step requires at least one sample.")

        self.ensure_train_started()
        started_at = time.perf_counter()
        features = self._features_from_samples(samples)
        micro_batch_size = int(base.args.train.micro_batch_size)
        if len(features) % micro_batch_size:
            raise ValueError(
                "Rank-local SFT feature count must be divisible by "
                f"veomni.train.micro_batch_size ({len(features)} vs {micro_batch_size})."
            )
        micro_batches = [
            base.collate_fn(features[start : start + micro_batch_size])
            for start in range(0, len(features), micro_batch_size)
        ]

        base.model.train()
        base.optimizer.zero_grad()
        base.state.global_step += 1
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(base.device)
        self.on_step_begin(micro_batches=micro_batches)
        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(float)
        base.micro_batches_token_len = count_loss_token(micro_batches)
        num_micro_batches = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches):
            base.model_reshard(micro_step, num_micro_batches)
            base._configure_hsdp_allreduce(micro_step, num_micro_batches)
            base.micro_batch_token_len = count_loss_token(micro_batch)
            loss, loss_dict = base.forward_backward_step(micro_batch)
            total_loss += float(loss.detach().cpu())
            for key, value in loss_dict.items():
                total_loss_dict[key] += float(value.detach().cpu())

        grad_norm = veomni_clip_grad_norm(
            base.model, base.args.train.optimizer.max_grad_norm
        )
        base.optimizer.step()
        base.lr_scheduler.step()
        base.optimizer.zero_grad()
        self.on_step_end(
            loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm
        )
        synchronize()

        result_epoch = min(int(base.state.epoch) + 1, self.num_train_epochs)
        grad_norm_value = (
            float(grad_norm.detach().cpu())
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm)
        )
        step_metrics = dict(getattr(base, "step_train_metrics", {}) or {})
        loss_value = float(
            step_metrics.get(
                "training/total_loss",
                step_metrics.get("total_loss", total_loss),
            )
        )
        lr = float(step_metrics.get("training/lr", base.optimizer.param_groups[0]["lr"]))
        memory = self._memory_metrics()
        result = SFTTrainStepResult(
            step=int(base.state.global_step),
            epoch=result_epoch,
            loss=loss_value,
            grad_norm=grad_norm_value,
            learning_rate=lr,
            elapsed_seconds=time.perf_counter() - started_at,
            **memory,
            finished=False,
        )
        self.advance_epoch_if_needed()
        rank, _ = _rank_and_world()
        logger.info(
            "SFT step %d on rank %d: loss=%.6f, lr=%.3e, grad_norm=%.4f.",
            result.step,
            rank,
            result.loss,
            result.learning_rate,
            result.grad_norm,
        )
        return result


class TrainActor(Actor):
    """Monarch actor that owns one rank of a VeOmni text trainer."""

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.trainer: VeOmniPolicyTrainer | None = None
        self._clip_low = 0.2
        self._clip_high = 0.28
        self._weight_transfer_group: Any | None = None
        self._weight_transfer_initialized = False
        self._weight_transfer_dtype = torch.bfloat16
        self._weight_transfer_packed = True
        self._weight_transfer_buffer_size = 256 * 1024 * 1024
        self._weight_transfer_num_buffers = 2

    def _require_trainer(self) -> VeOmniPolicyTrainer:
        if self.trainer is None:
            raise RuntimeError("TrainActor.setup() must complete first.")
        return self.trainer

    @endpoint
    def setup(self) -> TrainActorStatus:
        if self.trainer is not None:
            raise RuntimeError("TrainActor.setup() may only be called once.")

        raw_config = load_yaml_config(self.config_path)

        rl_config = mapping(raw_config.get("rl"), "rl")
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

        rollout_actor = mapping(raw_config.get("rollout_actor"), "rollout_actor")
        engine_config = mapping(rollout_actor.get("engine"), "rollout_actor.engine")
        transfer_dtype_name = str(engine_config.get("dtype", "bfloat16"))
        transfer_dtype_name = transfer_dtype_name.removeprefix("torch.")
        transfer_dtype = getattr(torch, transfer_dtype_name, None)
        if not isinstance(transfer_dtype, torch.dtype):
            raise ValueError(
                "rollout_actor.engine.dtype must name a torch dtype, "
                f"got {transfer_dtype_name!r}."
        )
        self._weight_transfer_dtype = transfer_dtype

        args = load_veomni_args(self.config_path)
        scheduler_horizon, steps_per_epoch, num_train_epochs = _training_schedule(
            raw_config
        )
        if rl_config:
            self.trainer = VeOmniGRPOTrainer(
                args,
                scheduler_horizon,
                clip_low=self._clip_low,
                clip_high=self._clip_high,
            )
        else:
            self.trainer = VeOmniSFTTrainer(
                args,
                scheduler_horizon,
                steps_per_epoch=steps_per_epoch,
                num_train_epochs=num_train_epochs,
            )

        logger.info(
            "VeOmni policy trainer initialized on rank %s/%s (local rank %s, "
            "scheduler horizon %d, %d epoch(s) x %d step(s)).",
            os.environ.get("RANK", "0"),
            os.environ.get("WORLD_SIZE", "1"),
            os.environ.get("LOCAL_RANK", "0"),
            scheduler_horizon,
            num_train_epochs,
            steps_per_epoch,
        )
        return TrainActorStatus(
            scheduler_horizon=scheduler_horizon,
            steps_per_epoch=steps_per_epoch,
            num_train_epochs=num_train_epochs,
            global_batch_size=int(args.train.global_batch_size),
            micro_batch_size=int(args.train.micro_batch_size),
            data_parallel_size=int(args.train.accelerator.dp_size),
        )

    @endpoint
    def train(self) -> None:
        raise RuntimeError("TrainActor.train() is disabled for stepwise policy training.")

    @endpoint
    def train_sft_step(
        self, sample_batches: Sequence[Sequence[DatasetSample]]
    ) -> SFTTrainStepResult:
        """Run one stepwise SFT optimizer step on a rank-local dataset batch."""
        trainer = self._require_trainer()
        if not isinstance(trainer, VeOmniSFTTrainer):
            raise RuntimeError("This TrainActor was not initialized for SFT.")
        rank, world_size = _rank_and_world()
        if len(sample_batches) != world_size:
            raise ValueError(
                f"Expected one SFT sample batch per rank ({world_size}), "
                f"got {len(sample_batches)}."
            )
        return trainer.train_step(sample_batches[rank])

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
        if not isinstance(trainer, VeOmniGRPOTrainer):
            raise RuntimeError("This TrainActor was not initialized for GRPO.")
        rank, world_size = _rank_and_world()
        if len(batches) != world_size:
            raise ValueError(
                f"Expected one GRPO batch per rank ({world_size}), got {len(batches)}."
            )
        return trainer.train_step(batches[rank])

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
        if self.trainer is not None:
            try:
                self.trainer.finish_training()
            except Exception:
                logger.exception("Failed to finish policy trainer before close.")
        if self._weight_transfer_group is not None:
            self._weight_transfer_group.destroy()
            self._weight_transfer_group = None
        self._weight_transfer_initialized = False
        if self.trainer is not None and dist.is_initialized():
            self.trainer.base.destroy_distributed()
