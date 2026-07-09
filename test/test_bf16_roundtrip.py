from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "qwen3_1_7b_gsm8k_grpo_async.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.expanduser().open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


def model_path_from_config(config_path: Path) -> Path:
    config = load_config(config_path)
    train_actor = config.get("train_actor") or {}
    model = train_actor.get("model") or {}
    model_path = model.get("model_path")
    if not model_path:
        raise ValueError("train_actor.model.model_path is missing from config.")
    return Path(str(model_path)).expanduser()


def checkpoint_files(model_path: Path) -> list[Path]:
    if model_path.is_file():
        return [model_path]
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if safetensors_files:
        return safetensors_files

    bin_files = sorted(model_path.glob("pytorch_model*.bin"))
    if bin_files:
        return bin_files

    raise FileNotFoundError(
        f"No *.safetensors or pytorch_model*.bin files found in {model_path}"
    )


def iter_safetensors(path: Path) -> Iterable[tuple[str, torch.Tensor]]:
    try:
        from safetensors.torch import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to inspect safetensors checkpoint shards."
        ) from exc

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            yield key, handle.get_tensor(key)


def iter_torch_bin(path: Path) -> Iterable[tuple[str, torch.Tensor]]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(
        state["state_dict"], dict
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}")
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            yield key, value


def iter_checkpoint_tensors(path: Path) -> Iterable[tuple[str, torch.Tensor]]:
    if path.suffix == ".safetensors":
        yield from iter_safetensors(path)
    else:
        yield from iter_torch_bin(path)


def max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if lhs.numel() == 0:
        return 0.0
    return float((lhs.float() - rhs.float()).abs().max().item())


def check_roundtrip(
    files: list[Path],
    *,
    max_tensors: int | None,
    verbose: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "files": len(files),
        "tensors": 0,
        "floating_tensors": 0,
        "bf16_tensors": 0,
        "bf16_numel": 0,
        "roundtrip_failures": [],
        "non_bf16_float_dtypes": {},
    }

    visited = 0
    for shard in files:
        if verbose:
            print(f"[scan] {shard}")
        for name, tensor in iter_checkpoint_tensors(shard):
            summary["tensors"] += 1
            if not tensor.is_floating_point():
                continue

            summary["floating_tensors"] += 1
            if tensor.dtype != torch.bfloat16:
                dtype_name = str(tensor.dtype).removeprefix("torch.")
                non_bf16 = summary["non_bf16_float_dtypes"]
                non_bf16[dtype_name] = int(non_bf16.get(dtype_name, 0)) + 1
                continue

            summary["bf16_tensors"] += 1
            summary["bf16_numel"] += int(tensor.numel())
            roundtrip = tensor.float().bfloat16()
            if not torch.equal(tensor, roundtrip):
                summary["roundtrip_failures"].append(
                    {
                        "file": str(shard),
                        "name": name,
                        "shape": list(tensor.shape),
                        "max_abs_diff": max_abs_diff(tensor, roundtrip),
                    }
                )

            visited += 1
            if max_tensors is not None and visited >= max_tensors:
                return summary

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether checkpoint bf16 tensors survive "
            "bf16 -> fp32 -> bf16 exactly."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override train_actor.model.model_path from the config.",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=None,
        help="Inspect only the first N bf16 tensors for a faster smoke test.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = (
        args.model_path.expanduser()
        if args.model_path is not None
        else model_path_from_config(args.config)
    )
    files = checkpoint_files(model_path)
    summary = check_roundtrip(
        files,
        max_tensors=args.max_tensors,
        verbose=bool(args.verbose),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["bf16_tensors"] == 0:
        print("No bf16 tensors were found in the checkpoint.", file=sys.stderr)
        raise SystemExit(2)
    if summary["roundtrip_failures"]:
        print("bf16 -> fp32 -> bf16 round-trip mismatch found.", file=sys.stderr)
        raise SystemExit(1)

    print(
        "OK: all inspected bf16 tensors are exactly stable under "
        "bf16 -> fp32 -> bf16."
    )


if __name__ == "__main__":
    main()
