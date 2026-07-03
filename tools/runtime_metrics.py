from __future__ import annotations

import asyncio
import logging
from statistics import mean
from typing import Sequence


logger = logging.getLogger(__name__)
_NVIDIA_SMI_DISABLED = False
_NVIDIA_SMI_WARNING_LOGGED = False


def _disable_gpu_metrics(message: str, *, exc_info: bool = False) -> None:
    global _NVIDIA_SMI_DISABLED, _NVIDIA_SMI_WARNING_LOGGED
    _NVIDIA_SMI_DISABLED = True
    if not _NVIDIA_SMI_WARNING_LOGGED:
        logger.warning(message, exc_info=exc_info)
        _NVIDIA_SMI_WARNING_LOGGED = True


class GpuMonitor:
    """Sample nvidia-smi stats during an async phase.

    The monitor is best-effort: if nvidia-smi is unavailable or cannot see the
    driver, it returns an empty metrics dict and lets training continue.
    """

    def __init__(
        self,
        gpu_ids: Sequence[str | int],
        *,
        interval_seconds: float = 1.0,
    ):
        self.gpu_ids = {str(gpu_id) for gpu_id in gpu_ids}
        self.interval_seconds = max(float(interval_seconds), 0.1)
        self._samples: list[dict[str, float]] = []
        self._task: asyncio.Task[None] | None = None
        self._disabled = False

    async def __aenter__(self) -> "GpuMonitor":
        if _NVIDIA_SMI_DISABLED:
            self._disabled = True
            return self
        await self._sample_once()
        if not self._disabled:
            self._task = asyncio.create_task(self._sample_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if not self._disabled:
            await self._sample_once()

    async def _sample_loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            await self._sample_once()

    async def _sample_once(self) -> None:
        if self._disabled:
            return
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
        except FileNotFoundError:
            _disable_gpu_metrics("nvidia-smi is not available; GPU metrics disabled.")
            self._disabled = True
            return
        except Exception:
            _disable_gpu_metrics(
                "Failed to run nvidia-smi; GPU metrics disabled.",
                exc_info=True,
            )
            self._disabled = True
            return

        if process.returncode != 0:
            message = (
                stderr.decode("utf-8", errors="replace").strip()
                or stdout.decode("utf-8", errors="replace").strip()
                or f"exit code {process.returncode}"
            )
            _disable_gpu_metrics(f"nvidia-smi failed; GPU metrics disabled: {message}")
            self._disabled = True
            return

        values = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                continue
            index, uuid, util, memory_used, memory_total = parts
            if self.gpu_ids and index not in self.gpu_ids and uuid not in self.gpu_ids:
                continue
            try:
                values.append(
                    {
                        "util_pct": float(util),
                        "memory_used_mb": float(memory_used),
                        "memory_total_mb": float(memory_total),
                    }
                )
            except ValueError:
                continue

        if values:
            self._samples.append(
                {
                    "gpu_util_mean_pct": mean(value["util_pct"] for value in values),
                    "gpu_mem_used_mean_mb": mean(
                        value["memory_used_mb"] for value in values
                    ),
                    "gpu_mem_used_peak_mb": max(
                        value["memory_used_mb"] for value in values
                    ),
                    "gpu_mem_total_mean_mb": mean(
                        value["memory_total_mb"] for value in values
                    ),
                }
            )

    def summary(self) -> dict[str, float]:
        if not self._samples:
            return {}
        return {
            "gpu_util_mean_pct": mean(
                sample["gpu_util_mean_pct"] for sample in self._samples
            ),
            "gpu_mem_used_mean_mb": mean(
                sample["gpu_mem_used_mean_mb"] for sample in self._samples
            ),
            "gpu_mem_used_peak_mb": max(
                sample["gpu_mem_used_peak_mb"] for sample in self._samples
            ),
            "gpu_mem_total_mean_mb": mean(
                sample["gpu_mem_total_mean_mb"] for sample in self._samples
            ),
        }


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}
