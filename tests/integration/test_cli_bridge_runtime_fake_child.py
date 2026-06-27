from __future__ import annotations

import asyncio
import base64
import json
import os
import stat
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_cli.config import default_config
from visual_events_cli.dds.bridge_process import DdsBridgeProcess
from visual_events_cli.runtime import run_runtime
from visual_events_cli.runtime_factories import bridge_runtime_factories


class AdvancingClock:
    def __init__(
        self,
        *,
        step_ms: int = 1,
        sleep_seconds: float = 0.001,
    ) -> None:
        self.now_ms = time.time_ns() // 1_000_000
        self._step_ms = int(step_ms)
        self._sleep_seconds = float(sleep_seconds)

    def __call__(self) -> int:
        if self._sleep_seconds > 0:
            time.sleep(self._sleep_seconds)
        self.now_ms += self._step_ms
        return self.now_ms


class RecordingTrackingService:
    def __init__(self) -> None:
        self.requests: list[tuple[dict[str, Any], bytes]] = []

    async def request_frame(self, header: dict[str, Any], jpeg: bytes) -> Any:
        self.requests.append((header, jpeg))
        return SimpleNamespace(
            visual_state={
                "type": "visual_state",
                "schema_version": 1,
                "camera": header["camera"],
                "frame_id": header["frame_id"],
                "frame_timestamp_ms": header["timestamp_ms"],
                "image_size": [header["width"], header["height"]],
                "attention": {
                    "target_track_id": 17,
                    "target_uv": [320.0, 180.0],
                    "confidence": 0.87,
                    "reason": "integration-fake",
                },
                "tracks": [{"track_id": 17}],
                "semantic_events": [],
            },
            error=None,
        )


class HangingService:
    def __init__(self, *, delay_seconds: float = 1.0) -> None:
        self.requests: list[tuple[dict[str, Any], bytes]] = []
        self._delay_seconds = float(delay_seconds)

    async def request_frame(self, header: dict[str, Any], jpeg: bytes) -> Any:
        self.requests.append((header, jpeg))
        await asyncio.sleep(self._delay_seconds)
        return SimpleNamespace(visual_state=None, error=None)


def test_fake_child_camera_head_to_service_to_gaze_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_bin, stdin_path = _write_fake_bridge_child(tmp_path, monkeypatch, mode="tracking")
    recorded_processes: list[DdsBridgeProcess] = []
    service = RecordingTrackingService()
    factories = _runtime_factories(
        bridge_bin=bridge_bin,
        recorded_processes=recorded_processes,
        service=service,
    )
    config = _bridge_config(gaze_stale_ms=2_000, head_stale_ms=1_000)

    result = run_runtime(
        config,
        factories=factories,
        max_ticks=80,
        sleep_seconds=0,
        clock_ms=AdvancingClock(step_ms=1),
    )

    assert result == 0
    assert len(recorded_processes) == 1
    assert len(service.requests) == 1
    header, jpeg = service.requests[0]
    assert header["camera"] == "logical-front"
    assert header["camera"] != "dds-front"
    assert header["width"] == 1280
    assert header["height"] == 720
    assert jpeg == JPEG_1280X720
    assert header["head_motion"]["state"] == "moving"
    assert header["head_motion"]["yaw_vel_rad_s"] == pytest.approx(0.08)

    payloads = _read_jsonl(stdin_path)
    tracking = _only_payload(payloads, state="tracking")
    assert tracking["protocol_version"] == 1
    assert tracking["type"] == "gaze_target"
    assert tracking["camera"] == "logical-front"
    assert tracking["valid"] is True
    assert tracking["state"] == "tracking"
    assert tracking["target_track_id"] == 17
    assert tracking["target_u"] == pytest.approx(320.0)
    assert tracking["target_v"] == pytest.approx(180.0)
    assert tracking["image_width"] == 1280
    assert tracking["image_height"] == 720
    assert tracking["stale_after_ms"] == 2_000

    _assert_process_closed(recorded_processes)


def test_fake_child_stale_publish_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_bin, stdin_path = _write_fake_bridge_child(tmp_path, monkeypatch, mode="stale")
    recorded_processes: list[DdsBridgeProcess] = []
    service = HangingService(delay_seconds=1.0)
    factories = _runtime_factories(
        bridge_bin=bridge_bin,
        recorded_processes=recorded_processes,
        service=service,
    )
    config = _bridge_config(gaze_stale_ms=5, head_stale_ms=1_000)

    result = run_runtime(
        config,
        factories=factories,
        max_ticks=90,
        sleep_seconds=0,
        clock_ms=AdvancingClock(step_ms=2),
    )

    assert result == 0
    assert len(service.requests) == 1
    payloads = _read_jsonl(stdin_path)
    stale_payloads = [payload for payload in payloads if payload.get("state") == "stale"]
    assert len(stale_payloads) == 1
    stale = stale_payloads[0]
    assert stale["protocol_version"] == 1
    assert stale["type"] == "gaze_target"
    assert stale["camera"] == "logical-front"
    assert stale["valid"] is False
    assert stale["state"] == "stale"
    assert stale["frame_id"] == 1
    assert stale["stale_after_ms"] == 5

    _assert_process_closed(recorded_processes)


def test_fake_child_fatal_error_frame_fails_runtime_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_bin, _stdin_path = _write_fake_bridge_child(tmp_path, monkeypatch, mode="fatal")
    recorded_processes: list[DdsBridgeProcess] = []
    factories = _runtime_factories(
        bridge_bin=bridge_bin,
        recorded_processes=recorded_processes,
        service=RecordingTrackingService(),
    )

    with pytest.raises(RuntimeError, match="dds_init_failed"):
        run_runtime(
            _bridge_config(),
            factories=factories,
            max_ticks=80,
            sleep_seconds=0,
            clock_ms=AdvancingClock(step_ms=1),
        )

    _assert_process_closed(recorded_processes)


def test_fake_child_nonzero_exit_fails_runtime_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_bin, _stdin_path = _write_fake_bridge_child(tmp_path, monkeypatch, mode="exit42")
    recorded_processes: list[DdsBridgeProcess] = []
    factories = _runtime_factories(
        bridge_bin=bridge_bin,
        recorded_processes=recorded_processes,
        service=RecordingTrackingService(),
    )

    with pytest.raises(RuntimeError, match="code 42"):
        run_runtime(
            _bridge_config(),
            factories=factories,
            max_ticks=80,
            sleep_seconds=0,
            clock_ms=AdvancingClock(step_ms=1),
        )

    _assert_process_closed(recorded_processes)


def _runtime_factories(
    *,
    bridge_bin: str,
    recorded_processes: list[DdsBridgeProcess],
    service: Any,
) -> Any:
    def recording_process_factory(process_config: Any) -> DdsBridgeProcess:
        process = DdsBridgeProcess(process_config)
        recorded_processes.append(process)
        return process

    factories = bridge_runtime_factories(
        bridge_bin=bridge_bin,
        process_factory=recording_process_factory,
    )
    return replace(
        factories,
        service_client=lambda _config: service,
        botified_writer=lambda _config: None,
    )


def _bridge_config(
    *,
    gaze_stale_ms: int = 250,
    head_stale_ms: int = 250,
) -> Any:
    base = default_config()
    return replace(
        base,
        camera=replace(
            base.camera,
            name="logical-front",
            image_topic="/camera/front/jpeg",
        ),
        head_state=replace(base.head_state, stale_ms=head_stale_ms),
        gaze_target=replace(base.gaze_target, stale_ms=gaze_stale_ms),
        botified=replace(base.botified, enabled=False, stdout=False),
    )


def _write_fake_bridge_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
) -> tuple[str, Path]:
    stdin_path = tmp_path / f"{mode}-stdin.jsonl"
    bridge_bin = tmp_path / "fake_jsonl_bridge_child.py"
    jpeg_base64 = base64.b64encode(JPEG_1280X720).decode("ascii")
    bridge_bin.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            from __future__ import annotations

            import json
            import os
            import sys
            import threading
            import time

            JPEG_BASE64 = {jpeg_base64!r}
            MODE = os.environ["VISUAL_EVENTS_FAKE_BRIDGE_MODE"]
            STDIN_PATH = os.environ["VISUAL_EVENTS_FAKE_BRIDGE_STDIN_PATH"]


            def capture_stdin() -> None:
                with open(STDIN_PATH, "ab", buffering=0) as output:
                    while True:
                        line = sys.stdin.buffer.readline()
                        if not line:
                            return
                        output.write(line)


            def emit(payload: dict[str, object]) -> None:
                sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\\n")
                sys.stdout.flush()


            def head_state() -> dict[str, object]:
                return {{
                    "protocol_version": 1,
                    "type": "head_state",
                    "dds_timestamp_ns": time.time_ns(),
                    "received_monotonic_ns": time.monotonic_ns(),
                    "valid": True,
                    "state": "moving",
                    "yaw_rad": 0.1,
                    "pitch_rad": -0.2,
                    "yaw_vel_rad_s": 0.08,
                    "pitch_vel_rad_s": -0.01,
                }}


            def camera_jpeg() -> dict[str, object]:
                return {{
                    "protocol_version": 1,
                    "type": "camera_jpeg",
                    "dds_timestamp_ns": time.time_ns(),
                    "received_monotonic_ns": time.monotonic_ns(),
                    "camera_name": "dds-front",
                    "width": 1280,
                    "height": 720,
                    "encoding": "JPEG",
                    "step": {len(JPEG_1280X720)},
                    "data_size_bytes": {len(JPEG_1280X720)},
                    "data_base64": JPEG_BASE64,
                }}


            threading.Thread(target=capture_stdin, daemon=True).start()

            if MODE in {{"tracking", "stale"}}:
                emit(head_state())
                time.sleep(0.03)
                emit(camera_jpeg())
                while True:
                    time.sleep(0.1)
            elif MODE == "fatal":
                emit({{
                    "protocol_version": 1,
                    "type": "error",
                    "code": "dds_init_failed",
                    "message": "fake child fatal",
                    "fatal": True,
                }})
                while True:
                    time.sleep(0.1)
            elif MODE == "exit42":
                raise SystemExit(42)
            else:
                raise SystemExit(f"unknown fake bridge mode: {{MODE}}")
            """
        ),
        encoding="utf-8",
    )
    bridge_bin.chmod(bridge_bin.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("VISUAL_EVENTS_FAKE_BRIDGE_MODE", mode)
    monkeypatch.setenv("VISUAL_EVENTS_FAKE_BRIDGE_STDIN_PATH", os.fspath(stdin_path))
    return os.fspath(bridge_bin), stdin_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.exists(), f"expected fake child stdin capture at {path}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _only_payload(payloads: list[dict[str, Any]], *, state: str) -> dict[str, Any]:
    matches = [payload for payload in payloads if payload.get("state") == state]
    assert len(matches) == 1
    return matches[0]


def _assert_process_closed(processes: list[DdsBridgeProcess]) -> None:
    assert len(processes) == 1
    process = processes[0]
    with pytest.raises(RuntimeError, match="closed"):
        process.poll_latest_camera()

    child = getattr(process, "_process", None)
    assert child is not None
    deadline = time.monotonic() + 1.0
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child.poll() is not None
