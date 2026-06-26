from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from visual_events_server.metrics import (
    JsonlMetricsSink,
    collect_resource_snapshot,
)
from visual_events_server.protocol import FrameMessage


def frame_message() -> FrameMessage:
    return FrameMessage(
        camera="front",
        frame_id=42,
        timestamp_ms=1710000000123,
        width=640,
        height=480,
        jpeg_bytes=b"\xff\xd8fake\xff\xd9",
        head_motion_state="stationary",
    )


def test_jsonl_metrics_sink_writes_frame_identity_phases_and_resources(tmp_path):
    metrics_path = tmp_path / "nested" / "metrics.jsonl"
    sink = JsonlMetricsSink(
        metrics_path,
        resource_sampler=lambda: {
            "rss": {"available": True, "bytes": 1234},
            "vram": {"available": False, "reason": "cuda_unavailable"},
        },
    )

    sink.write_frame_metrics(
        frame_message(),
        {
            "infer": 1.23456,
            "tracking": 0.5,
            "total": 2,
        },
    )

    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["type"] == "frame_metrics"
    assert payload["schema_version"] == 1
    assert payload["camera"] == "front"
    assert payload["frame_id"] == 42
    assert payload["frame_timestamp_ms"] == 1710000000123
    assert payload["phase_latencies_ms"] == {
        "infer": 1.235,
        "tracking": 0.5,
        "total": 2.0,
    }
    assert payload["resources"] == {
        "rss": {"available": True, "bytes": 1234},
        "vram": {"available": False, "reason": "cuda_unavailable"},
    }


def test_jsonl_metrics_sink_ignores_write_errors_after_startup(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.jsonl"
    sink = JsonlMetricsSink(metrics_path, resource_sampler=lambda: {})
    original_open = Path.open

    def fail_metrics_open(self, *args, **kwargs):
        if self == metrics_path:
            raise OSError("disk went away")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_metrics_open)

    sink.write_frame_metrics(frame_message(), {"total": 1.0})


def test_resource_snapshot_reports_unavailable_without_proc_or_torch(tmp_path):
    resources = collect_resource_snapshot(
        status_path=tmp_path / "missing-status",
        torch_module=None,
    )

    assert resources["rss"]["available"] is False
    assert resources["rss"]["reason"] == "rss_unavailable"
    assert resources["vram"]["available"] is False
    assert resources["vram"]["reason"] == "torch_unavailable"


def test_resource_snapshot_reads_rss_and_cuda_when_available(tmp_path):
    status_path = tmp_path / "status"
    status_path.write_text("Name:\tpython\nVmRSS:\t      37 kB\n", encoding="utf-8")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 2,
            memory_allocated=lambda device: 2048 if device == 2 else 0,
            memory_reserved=lambda device: 4096 if device == 2 else 0,
        )
    )

    resources = collect_resource_snapshot(
        status_path=status_path,
        torch_module=fake_torch,
    )

    assert resources["rss"] == {"available": True, "bytes": 37 * 1024}
    assert resources["vram"] == {
        "available": True,
        "device": 2,
        "allocated_bytes": 2048,
        "reserved_bytes": 4096,
    }
