from __future__ import annotations

import base64
import json
import subprocess
import threading
import time
from typing import Any

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_cli.target_mapper import GazeTargetPayload


def import_bridge_process() -> Any:
    try:
        import visual_events_cli.dds.bridge_process as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.dds.bridge_process module: {exc}")
    return module


class QueueReadablePipe:
    def __init__(self) -> None:
        self._items: list[bytes | None] = []
        self._condition = threading.Condition()
        self.read_count = 0
        self.closed = False

    def push_json(self, payload: dict[str, Any]) -> None:
        self.push_line(json.dumps(payload))

    def push_line(self, line: str) -> None:
        with self._condition:
            if self.closed:
                raise RuntimeError("pipe closed")
            self._items.append(line.encode("utf-8") + b"\n")
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if not self.closed:
                self.closed = True
                self._items.append(None)
                self._condition.notify_all()

    def readline(self) -> bytes:
        with self._condition:
            while not self._items:
                self._condition.wait(timeout=0.1)
            item = self._items.pop(0)
        if item is None:
            return b""
        self.read_count += 1
        return item


class WritablePipe:
    def __init__(self, *, block_first_write: bool = False) -> None:
        self.lines: list[bytes] = []
        self.flush_count = 0
        self.closed = False
        self.write_started = threading.Event()
        self.release_write = threading.Event()
        self._block_first_write = block_first_write
        self._write_count = 0
        self._lock = threading.Lock()

    def write(self, data: bytes) -> int:
        self.write_started.set()
        if self._block_first_write and self._write_count == 0:
            self.release_write.wait(timeout=1.0)
        with self._lock:
            self._write_count += 1
            self.lines.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True
        self.release_write.set()


class FakeProcess:
    def __init__(
        self,
        *,
        stdin: WritablePipe | None = None,
        timeout_on_wait: bool = False,
    ) -> None:
        self.stdin = stdin or WritablePipe()
        self.stdout = QueueReadablePipe()
        self.stderr = QueueReadablePipe()
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self._timeout_on_wait = timeout_on_wait

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.stdout.close()
        self.stderr.close()
        self.stdin.close()

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self._timeout_on_wait and self.kill_calls == 0:
            raise subprocess.TimeoutExpired("fake-bridge", timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()
        self.stdin.close()


def make_process(
    module: Any,
    fake: FakeProcess | None = None,
    calls: list[dict[str, Any]] | None = None,
    wall_clock_ms: Any | None = None,
    monotonic_ns: Any | None = None,
) -> Any:
    fake = fake or FakeProcess()
    calls = calls if calls is not None else []

    def popen(args: list[str], **kwargs: Any) -> FakeProcess:
        calls.append({"args": args, "kwargs": kwargs})
        return fake

    config = module.DdsBridgeProcessConfig(
        bridge_bin="/tmp/visual-events-dds-bridge",
        dds_domain=57,
        dds_network="lo",
        camera_topic="/camera/image/jpeg",
        head_state_topic="/robot/head_state",
        gaze_topic="/visual_events/gaze_target",
        logical_camera_name="logical-front",
        base_env={"PATH": "/bin", "LD_LIBRARY_PATH": "/opt/dds/lib"},
    )
    kwargs: dict[str, Any] = {"popen": popen}
    if wall_clock_ms is not None:
        kwargs["wall_clock_ms"] = wall_clock_ms
    if monotonic_ns is not None:
        kwargs["monotonic_ns"] = monotonic_ns
    return module.DdsBridgeProcess(config, **kwargs)


def camera_payload(
    timestamp_ns: int,
    *,
    data: bytes = JPEG_1280X720,
    received_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "type": "camera_jpeg",
        "dds_timestamp_ns": timestamp_ns,
        "received_monotonic_ns": (
            timestamp_ns if received_monotonic_ns is None else received_monotonic_ns
        ),
        "camera_name": "dds-front",
        "width": 1280,
        "height": 720,
        "encoding": "JPEG",
        "step": len(data),
        "data_size_bytes": len(data),
        "data_base64": base64.b64encode(data).decode("ascii"),
    }


def head_payload(
    dds_timestamp_ns: int,
    *,
    received_monotonic_ns: int | None = None,
    yaw_vel: float = 0.01,
) -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "type": "head_state",
        "dds_timestamp_ns": dds_timestamp_ns,
        "received_monotonic_ns": (
            dds_timestamp_ns if received_monotonic_ns is None else received_monotonic_ns
        ),
        "valid": True,
        "state": "moving" if abs(yaw_vel) > 0.03 else "stationary",
        "yaw_rad": 0.1,
        "pitch_rad": -0.2,
        "yaw_vel_rad_s": yaw_vel,
        "pitch_vel_rad_s": 0.02,
    }


def gaze_payload(frame_id: int = 1) -> GazeTargetPayload:
    return GazeTargetPayload(
        schema_version=1,
        camera="front",
        frame_id=frame_id,
        frame_timestamp_ms=1_710_000_000_000 + frame_id,
        publish_timestamp_ms=1_710_000_000_100 + frame_id,
        valid=True,
        state="tracking",
        target_track_id=7,
        target_u=640.0,
        target_v=360.0,
        target_norm_x=0.0,
        target_norm_y=0.0,
        image_width=1280,
        image_height=720,
        confidence=0.9,
        reason="nearest",
        stale_after_ms=250,
    )


def wait_until(predicate: Any, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


def test_start_idempotent_and_close_idempotent_and_closed_operations_raise():
    module = import_bridge_process()
    fake = FakeProcess()
    calls: list[dict[str, Any]] = []
    process = make_process(module, fake, calls)

    process.start()
    process.start()

    assert len(calls) == 1
    assert calls[0]["args"] == ["/tmp/visual-events-dds-bridge"]
    child_env = calls[0]["kwargs"]["env"]
    assert child_env["LD_LIBRARY_PATH"] == "/opt/dds/lib"
    assert child_env["VISUAL_EVENTS_DDS_DOMAIN"] == "57"
    assert child_env["VISUAL_EVENTS_DDS_NETWORK"] == "lo"
    assert child_env["VISUAL_EVENTS_CAMERA_TOPIC"] == "/camera/image/jpeg"
    assert child_env["VISUAL_EVENTS_HEAD_STATE_TOPIC"] == "/robot/head_state"
    assert child_env["VISUAL_EVENTS_GAZE_TOPIC"] == "/visual_events/gaze_target"
    assert child_env["VISUAL_EVENTS_LOGICAL_CAMERA_NAME"] == "logical-front"

    process.close()
    process.close()

    assert fake.terminate_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        process.start()
    with pytest.raises(RuntimeError, match="closed"):
        process.poll_latest_camera()
    with pytest.raises(RuntimeError, match="closed"):
        process.send_gaze_target(gaze_payload())


def test_stdout_reader_keeps_latest_valid_camera_and_head_only_and_drops_bad_jpeg():
    module = import_bridge_process()
    fake = FakeProcess()
    process = make_process(
        module,
        fake,
        wall_clock_ms=lambda: 0,
        monotonic_ns=lambda: 0,
    )
    process.start()

    fake.stdout.push_json(camera_payload(1_000_000_000))
    fake.stdout.push_json(head_payload(1_100_000_000, yaw_vel=0.01))
    fake.stdout.push_json(camera_payload(2_000_000_000))
    fake.stdout.push_json(camera_payload(3_000_000_000, data=b"\xff\xd8bad\xff\xd9"))
    fake.stdout.push_json(head_payload(1_200_000_000, yaw_vel=0.05))
    wait_until(lambda: fake.stdout.read_count >= 5)

    frame = process.poll_latest_camera()
    assert frame is not None
    assert frame.camera == "logical-front"
    assert frame.timestamp_ms == 2_000
    assert process.poll_latest_camera() is None

    sample = process.latest_head_state()
    assert sample is not None
    assert sample.timestamp_ms == 1_200
    assert sample.yaw_vel_rad_s == pytest.approx(0.05)

    process.close()


def test_camera_source_monotonic_timestamp_is_converted_to_wall_time():
    module = import_bridge_process()
    fake = FakeProcess()
    process = make_process(
        module,
        fake,
        wall_clock_ms=lambda: 1_700_000_000_000,
        monotonic_ns=lambda: 5_000_000_000,
    )
    process.start()

    fake.stdout.push_json(
        camera_payload(
            5_050_000_000,
            received_monotonic_ns=5_060_000_000,
        )
    )
    wait_until(lambda: fake.stdout.read_count >= 1)

    frame = process.poll_latest_camera()
    assert frame is not None
    assert frame.timestamp_ms == 1_700_000_000_050
    process.close()


@pytest.mark.parametrize("dds_timestamp_ns", [0, 1_710_000_000_050_000_000])
def test_camera_zero_or_cross_domain_source_timestamp_falls_back_to_received_monotonic(
    dds_timestamp_ns: int,
) -> None:
    module = import_bridge_process()
    fake = FakeProcess()
    process = make_process(
        module,
        fake,
        wall_clock_ms=lambda: 1_700_000_000_000,
        monotonic_ns=lambda: 5_000_000_000,
    )
    process.start()

    fake.stdout.push_json(
        camera_payload(
            dds_timestamp_ns,
            received_monotonic_ns=5_050_000_000,
        )
    )
    wait_until(lambda: fake.stdout.read_count >= 1)

    frame = process.poll_latest_camera()
    assert frame is not None
    assert frame.timestamp_ms == 1_700_000_000_050
    process.close()


def test_head_state_freshness_uses_received_monotonic_not_dds_timestamp():
    module = import_bridge_process()
    adapters = __import__(
        "visual_events_cli.dds.bridge_adapters",
        fromlist=["BridgeDdsHeadStateSubscriber"],
    )
    fake = FakeProcess()
    process = make_process(
        module,
        fake,
        wall_clock_ms=lambda: 10_000,
        monotonic_ns=lambda: 5_000_000_000,
    )
    subscriber = adapters.BridgeDdsHeadStateSubscriber(
        process,
        stale_ms=250,
        stationary_yaw_vel_rad_s=0.03,
        stationary_pitch_vel_rad_s=0.03,
    )
    subscriber.start()

    fake.stdout.push_json(
        head_payload(
            123_000_000,
            received_monotonic_ns=5_050_000_000,
            yaw_vel=0.05,
        )
    )
    wait_until(lambda: process.latest_head_state() is not None)

    fresh_sample = process.latest_head_state()
    assert fresh_sample is not None
    assert fresh_sample.timestamp_ms == 10_050
    moving = subscriber.current_motion(now_ms=10_100)
    assert moving.state == "moving"
    assert moving.yaw_vel_rad_s == pytest.approx(0.05)

    fake.stdout.push_json(
        head_payload(
            123_000_000,
            received_monotonic_ns=4_000_000_000,
            yaw_vel=0.01,
        )
    )
    wait_until(
        lambda: (
            (sample := process.latest_head_state()) is not None
            and sample.timestamp_ms == 9_000
        )
    )

    stale = subscriber.current_motion(now_ms=10_100)
    assert stale.state == "unknown"
    process.close()


def test_stderr_json_is_drained_but_never_parsed_as_protocol():
    module = import_bridge_process()
    fake = FakeProcess()
    process = make_process(module, fake)
    process.start()

    fake.stderr.push_json(
        {
            "protocol_version": 1,
            "type": "error",
            "code": "fatal-on-stderr",
            "message": "must be ignored",
            "fatal": True,
        }
    )
    wait_until(lambda: fake.stderr.read_count >= 1)

    process.send_gaze_target(gaze_payload())
    wait_until(lambda: len(fake.stdin.lines) == 1)
    assert json.loads(fake.stdin.lines[0])["type"] == "gaze_target"

    process.close()


def test_writer_emits_json_line_flushes_and_uses_latest_only_bounded_queue():
    module = import_bridge_process()
    stdin = WritablePipe(block_first_write=True)
    fake = FakeProcess(stdin=stdin)
    process = make_process(module, fake)
    process.start()

    process.send_gaze_target(gaze_payload(frame_id=1))
    assert stdin.write_started.wait(timeout=1.0)
    process.send_gaze_target(gaze_payload(frame_id=2))
    process.send_gaze_target(gaze_payload(frame_id=3))
    assert process.writer_queue_maxsize == 1

    stdin.release_write.set()
    wait_until(lambda: len(stdin.lines) >= 2)

    decoded = [json.loads(line) for line in stdin.lines]
    assert [payload["frame_id"] for payload in decoded] == [1, 3]
    assert all(line.endswith(b"\n") for line in stdin.lines)
    assert stdin.flush_count >= 2

    process.close()


def test_close_after_timeout_kills_child():
    module = import_bridge_process()
    fake = FakeProcess(timeout_on_wait=True)
    process = make_process(module, fake)

    process.start()
    process.close()

    assert fake.terminate_calls == 1
    assert fake.kill_calls == 1


def test_fatal_error_frame_marks_process_failed():
    module = import_bridge_process()
    fake = FakeProcess()
    process = make_process(module, fake)
    process.start()

    fake.stdout.push_json(
        {
            "protocol_version": 1,
            "type": "error",
            "code": "dds_init_failed",
            "message": "native bridge failed",
            "fatal": True,
        }
    )

    def failed() -> bool:
        try:
            process.poll_latest_camera()
        except RuntimeError as exc:
            return "dds_init_failed" in str(exc)
        return False

    wait_until(failed)
    process.close()


def test_nonzero_child_exit_marks_process_failed():
    module = import_bridge_process()
    fake = FakeProcess()
    process = make_process(module, fake)
    process.start()
    fake.returncode = 42

    with pytest.raises(RuntimeError, match="exited with code 42"):
        process.send_gaze_target(gaze_payload())

    process.close()


def test_clean_child_exit_during_runtime_marks_process_failed():
    module = import_bridge_process()
    fake = FakeProcess()
    process = make_process(module, fake)
    process.start()

    fake.returncode = 0
    fake.stdout.close()

    def failed() -> bool:
        try:
            process.poll_latest_camera()
        except RuntimeError as exc:
            return "exited unexpectedly with code 0" in str(exc)
        return False

    wait_until(failed)
    process.close()
