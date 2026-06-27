from __future__ import annotations

import base64
import contextlib
import json
import os
import socket
import stat
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_cli.config import default_config
from visual_events_cli.dds.bridge_process import DdsBridgeProcess
from visual_events_cli.runtime import run_runtime
from visual_events_cli.runtime_factories import bridge_runtime_factories


REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_SERVER_TOOL = REPO_ROOT / "tools" / "mock_visual_state_server.py"


class AdvancingClock:
    def __init__(
        self,
        *,
        step_ms: int = 1,
        sleep_seconds: float = 0.0005,
    ) -> None:
        self.now_ms = time.time_ns() // 1_000_000
        self._step_ms = int(step_ms)
        self._sleep_seconds = float(sleep_seconds)

    def __call__(self) -> int:
        if self._sleep_seconds > 0:
            time.sleep(self._sleep_seconds)
        self.now_ms += self._step_ms
        return self.now_ms


def test_event_profile_writes_only_botified_stdout_and_tracking_gaze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    bridge_bin, stdin_path = _write_fake_bridge_child(tmp_path, monkeypatch)
    recorded_processes: list[DdsBridgeProcess] = []

    with _mock_visual_state_server(profile="event") as service_url:
        result = run_runtime(
            _bridge_config(service_url=service_url, gaze_stale_ms=10_000),
            factories=_runtime_factories(
                bridge_bin=bridge_bin,
                recorded_processes=recorded_processes,
            ),
            stop_requested=_StopAfterGazeState(
                stdin_path,
                state="tracking",
                extra_ticks=40,
            ),
            max_ticks=400,
            sleep_seconds=0,
            clock_ms=AdvancingClock(step_ms=1),
        )

    assert result == 0
    stdout = capfd.readouterr().out
    stdout_lines = stdout.splitlines()
    assert len(stdout_lines) == 1
    line = stdout_lines[0]
    assert line.startswith("<botified>")
    assert line.endswith("</botified>")

    payload = json.loads(line.removeprefix("<botified>").removesuffix("</botified>"))
    assert payload["id"] == "visual:front:mock_evt_000001"
    assert "event=person_waving" in payload["request"]
    assert not any(output_line.startswith("{") for output_line in stdout_lines)
    for forbidden in (
        "attention_target_changed",
        "visual_state",
        "gaze_target",
        "status",
    ):
        assert forbidden not in stdout

    gaze = _only_gaze_target(_read_jsonl(stdin_path))
    assert gaze["state"] == "tracking"
    assert gaze["valid"] is True
    assert gaze["camera"] == "front"
    assert gaze["target_track_id"] == 7

    _assert_process_closed(recorded_processes)


def test_lost_profile_writes_no_botified_stdout_and_lost_gaze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    bridge_bin, stdin_path = _write_fake_bridge_child(tmp_path, monkeypatch)
    recorded_processes: list[DdsBridgeProcess] = []

    with _mock_visual_state_server(profile="lost") as service_url:
        result = run_runtime(
            _bridge_config(service_url=service_url, gaze_stale_ms=10_000),
            factories=_runtime_factories(
                bridge_bin=bridge_bin,
                recorded_processes=recorded_processes,
            ),
            stop_requested=_StopAfterGazeState(stdin_path, state="lost"),
            max_ticks=400,
            sleep_seconds=0,
            clock_ms=AdvancingClock(step_ms=1),
        )

    assert result == 0
    assert capfd.readouterr().out == ""

    gaze = _only_gaze_target(_read_jsonl(stdin_path))
    assert gaze["state"] == "lost"
    assert gaze["valid"] is False
    assert gaze["camera"] == "front"

    _assert_process_closed(recorded_processes)


def test_slow_tracking_profile_publishes_one_stale_gaze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    bridge_bin, stdin_path = _write_fake_bridge_child(tmp_path, monkeypatch)
    recorded_processes: list[DdsBridgeProcess] = []

    with _mock_visual_state_server(profile="tracking", delay_ms=100) as service_url:
        result = run_runtime(
            _bridge_config(
                service_url=service_url,
                gaze_stale_ms=5,
                response_timeout_ms=20,
            ),
            factories=_runtime_factories(
                bridge_bin=bridge_bin,
                recorded_processes=recorded_processes,
            ),
            stop_requested=_StopAfterGazeState(
                stdin_path,
                state="stale",
                extra_ticks=5,
            ),
            max_ticks=400,
            sleep_seconds=0,
            clock_ms=AdvancingClock(step_ms=2),
        )

    assert result == 0
    assert capfd.readouterr().out == ""

    gaze_payloads = [
        payload
        for payload in _read_jsonl(stdin_path)
        if payload.get("type") == "gaze_target"
    ]
    assert len(gaze_payloads) == 1
    stale = gaze_payloads[0]
    assert stale["valid"] is False
    assert stale["state"] == "stale"
    assert stale["camera"] == "front"
    assert stale["frame_id"] == 1
    assert stale["stale_after_ms"] == 5

    _assert_process_closed(recorded_processes)


@contextlib.contextmanager
def _mock_visual_state_server(
    *,
    profile: str,
    delay_ms: int = 0,
) -> Iterator[str]:
    host = "127.0.0.1"
    port = _free_loopback_port()
    process = subprocess.Popen(
        [
            sys.executable,
            os.fspath(MOCK_SERVER_TOOL),
            "--host",
            host,
            "--port",
            str(port),
            "--profile",
            profile,
            "--delay-ms",
            str(delay_ms),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(f"http://{host}:{port}/healthz")
        yield f"ws://{host}:{port}/v1/stream"
    finally:
        _terminate_server_process(process)


def _runtime_factories(
    *,
    bridge_bin: str,
    recorded_processes: list[DdsBridgeProcess],
) -> Any:
    def recording_process_factory(process_config: Any) -> DdsBridgeProcess:
        process = DdsBridgeProcess(process_config)
        recorded_processes.append(process)
        return process

    return bridge_runtime_factories(
        bridge_bin=bridge_bin,
        process_factory=recording_process_factory,
    )


def _bridge_config(
    *,
    service_url: str,
    gaze_stale_ms: int,
    response_timeout_ms: int = 1_000,
) -> Any:
    base = default_config()
    return replace(
        base,
        camera=replace(
            base.camera,
            name="front",
            image_topic="/camera/front/jpeg",
        ),
        head_state=replace(base.head_state, stale_ms=1_000),
        service=replace(
            base.service,
            url=service_url,
            response_timeout_ms=response_timeout_ms,
        ),
        gaze_target=replace(base.gaze_target, stale_ms=gaze_stale_ms),
        botified=replace(
            base.botified,
            enabled=True,
            stdout=True,
            stdout_queue_max=8,
        ),
    )


def _write_fake_bridge_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, Path]:
    stdin_path = tmp_path / "fake-bridge-stdin.jsonl"
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
            emit(head_state())
            emit(camera_jpeg())

            while True:
                time.sleep(0.1)
            """
        ),
        encoding="utf-8",
    )
    bridge_bin.chmod(bridge_bin.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("VISUAL_EVENTS_FAKE_BRIDGE_STDIN_PATH", os.fspath(stdin_path))
    return os.fspath(bridge_bin), stdin_path


class _StopAfterGazeState:
    def __init__(
        self,
        path: Path,
        *,
        state: str,
        extra_ticks: int = 0,
    ) -> None:
        self._path = path
        self._state = state
        self._remaining_extra_ticks = int(extra_ticks)
        self._seen = False

    def __call__(self) -> bool:
        if not self._seen:
            self._seen = any(
                payload.get("type") == "gaze_target"
                and payload.get("state") == self._state
                for payload in _read_jsonl_if_exists(self._path)
            )
        if not self._seen:
            return False
        if self._remaining_extra_ticks > 0:
            self._remaining_extra_ticks -= 1
            return False
        return True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.exists(), f"expected fake child stdin capture at {path}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return payloads


def _only_gaze_target(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    gaze_payloads = [
        payload for payload in payloads if payload.get("type") == "gaze_target"
    ]
    assert len(gaze_payloads) == 1
    return gaze_payloads[0]


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(url: str) -> None:
    deadline = time.monotonic() + 2.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.01)
    raise AssertionError(f"mock visual_state server did not become healthy: {last_error}")


def _terminate_server_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        process.wait(timeout=0.1)
        return

    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


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
