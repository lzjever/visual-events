from __future__ import annotations

import importlib
import json
import math
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from visual_events_server.protocol import FrameMessage, SCHEMA_VERSION

_AUTO_TORCH = object()


class MetricsSink(Protocol):
    def write_frame_metrics(
        self,
        frame: FrameMessage,
        phase_latencies_ms: Mapping[str, float],
    ) -> None:
        ...


class JsonlMetricsSink:
    def __init__(
        self,
        path: str | Path,
        *,
        resource_sampler: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._resource_sampler = resource_sampler or collect_resource_snapshot
        self.write_error_count = 0

    def write_frame_metrics(
        self,
        frame: FrameMessage,
        phase_latencies_ms: Mapping[str, float],
    ) -> None:
        try:
            resources = self._resource_sampler()
        except Exception:
            resources = {
                "rss": {"available": False, "reason": "resource_sampler_error"},
                "vram": {"available": False, "reason": "resource_sampler_error"},
            }

        payload = {
            "type": "frame_metrics",
            "schema_version": SCHEMA_VERSION,
            "camera": frame.camera,
            "frame_id": frame.frame_id,
            "frame_timestamp_ms": frame.timestamp_ms,
            "phase_latencies_ms": _normalize_phase_latencies(phase_latencies_ms),
            "resources": resources,
        }
        try:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
        except OSError as exc:
            self.write_error_count += 1
            print(
                "metrics write failure "
                f"path={self.path} "
                f"error={type(exc).__name__}: {exc} "
                f"count={self.write_error_count}",
                file=sys.stderr,
            )
            return


def collect_resource_snapshot(
    *,
    status_path: str | Path = Path("/proc/self/status"),
    torch_module: Any = _AUTO_TORCH,
) -> dict[str, Any]:
    return {
        "rss": _rss_snapshot(Path(status_path)),
        "vram": _vram_snapshot(torch_module),
    }


def _normalize_phase_latencies(
    phase_latencies_ms: Mapping[str, float],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name, value in phase_latencies_ms.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        normalized[str(name)] = round(max(0.0, numeric), 3)
    return normalized


def _rss_snapshot(status_path: Path) -> dict[str, Any]:
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"available": False, "reason": "rss_unavailable"}

    for line in lines:
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            break
        try:
            return {"available": True, "bytes": int(parts[1]) * 1024}
        except ValueError:
            break
    return {"available": False, "reason": "rss_unavailable"}


def _vram_snapshot(torch_module: Any) -> dict[str, Any]:
    if torch_module is _AUTO_TORCH:
        try:
            torch_module = importlib.import_module("torch")
        except Exception:
            return {"available": False, "reason": "torch_unavailable"}
    if torch_module is None:
        return {"available": False, "reason": "torch_unavailable"}

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return {"available": False, "reason": "cuda_unavailable"}
    try:
        if not bool(cuda.is_available()):
            return {"available": False, "reason": "cuda_unavailable"}
        device = int(cuda.current_device())
        return {
            "available": True,
            "device": device,
            "allocated_bytes": int(cuda.memory_allocated(device)),
            "reserved_bytes": int(cuda.memory_reserved(device)),
        }
    except Exception:
        return {"available": False, "reason": "vram_unavailable"}
