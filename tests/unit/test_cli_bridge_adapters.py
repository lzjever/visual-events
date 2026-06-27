from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_cli.dds.types import CameraJpegMessage, HeadStateSample
from visual_events_cli.frame_pump import HeadMotion, InputFrame
from visual_events_cli.target_mapper import GazeTargetPayload


def import_bridge_adapters() -> Any:
    try:
        import visual_events_cli.dds.bridge_adapters as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.dds.bridge_adapters module: {exc}")
    return module


class FakeBridgeProcess:
    def __init__(self) -> None:
        self.start_calls = 0
        self.close_calls = 0
        self.frames: list[InputFrame] = []
        self.head_sample: HeadStateSample | None = None
        self.payloads: list[Any] = []

    def start(self) -> None:
        self.start_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def push_camera(self, message: CameraJpegMessage) -> None:
        frame = message.to_input_frame()
        if frame is not None:
            self.frames = [frame]

    def poll_latest_camera(self) -> InputFrame | None:
        if not self.frames:
            return None
        return self.frames.pop(0)

    def latest_head_state(self) -> HeadStateSample | None:
        return self.head_sample

    def send_gaze_target(self, payload: Any) -> None:
        self.payloads.append(payload)


def camera_message(**overrides: Any) -> CameraJpegMessage:
    values = {
        "camera": "front",
        "timestamp_ms": 1_710_000_000_000,
        "width": 1280,
        "height": 720,
        "encoding": "JPEG",
        "data": JPEG_1280X720,
    }
    values.update(overrides)
    return CameraJpegMessage(**values)


def head_sample(**overrides: Any) -> HeadStateSample:
    values = {
        "timestamp_ms": 1_710_000_000_000,
        "valid": True,
        "yaw_rad": 0.1,
        "pitch_rad": -0.2,
        "yaw_vel_rad_s": 0.01,
        "pitch_vel_rad_s": 0.02,
    }
    values.update(overrides)
    return HeadStateSample(**values)


def gaze_payload() -> GazeTargetPayload:
    return GazeTargetPayload(
        schema_version=1,
        camera="front",
        frame_id=42,
        frame_timestamp_ms=1_710_000_000_000,
        publish_timestamp_ms=1_710_000_000_082,
        valid=True,
        state="tracking",
        target_track_id=7,
        target_u=640.0,
        target_v=360.0,
        target_norm_x=0.0,
        target_norm_y=0.0,
        image_width=1280,
        image_height=720,
        confidence=0.91,
        reason="nearest",
        stale_after_ms=250,
    )


def assert_unknown_motion(motion: HeadMotion) -> None:
    assert isinstance(motion, HeadMotion)
    assert motion.state == "unknown"
    assert motion.yaw_vel_rad_s is None
    assert motion.pitch_vel_rad_s is None


def test_image_subscriber_wraps_process_and_returns_latest_valid_frame_only():
    module = import_bridge_adapters()
    process = FakeBridgeProcess()
    subscriber = module.BridgeDdsImageSubscriber(process)

    subscriber.start()
    subscriber.start()
    process.push_camera(camera_message(timestamp_ms=1_000))
    process.push_camera(camera_message(timestamp_ms=2_000, data=b"\xff\xd8bad\xff\xd9"))

    frame = subscriber.poll_latest()
    assert isinstance(frame, InputFrame)
    assert frame.timestamp_ms == 1_000
    assert subscriber.poll_latest() is None

    subscriber.close()
    subscriber.close()
    assert process.start_calls == 2
    assert process.close_calls == 2


def test_head_state_subscriber_returns_unknown_without_sample_and_maps_motion():
    module = import_bridge_adapters()
    process = FakeBridgeProcess()
    subscriber = module.BridgeDdsHeadStateSubscriber(
        process,
        stale_ms=250,
        stationary_yaw_vel_rad_s=0.03,
        stationary_pitch_vel_rad_s=0.03,
    )

    subscriber.start()
    assert_unknown_motion(subscriber.current_motion(now_ms=1_710_000_000_010))

    process.head_sample = head_sample(yaw_vel_rad_s=0.04)
    motion = subscriber.current_motion(now_ms=1_710_000_000_010)

    assert motion.state == "moving"
    assert motion.yaw_vel_rad_s == pytest.approx(0.04)
    assert motion.pitch_vel_rad_s == pytest.approx(0.02)


def test_gaze_target_publisher_sends_payload_to_shared_process():
    module = import_bridge_adapters()
    process = FakeBridgeProcess()
    publisher = module.BridgeDdsGazeTargetPublisher(process)
    payload = gaze_payload()

    publisher.start()
    publisher.publish(payload)
    publisher.close()

    assert process.payloads == [payload]


def test_importing_bridge_modules_does_not_load_denied_dds_ml_or_motion_modules():
    denied_roots = {
        "cyclonedds",
        "fastdds",
        "rclpy",
        "rtidds",
        "torch",
        "ultralytics",
        "unitree",
        "unitree_sdk2py",
    }
    script = f"""
import importlib
import sys

before = set(sys.modules)
for name in (
    "visual_events_cli.dds.bridge_protocol",
    "visual_events_cli.dds.bridge_process",
    "visual_events_cli.dds.bridge_adapters",
    "visual_events_cli.runtime_factories",
):
    importlib.import_module(name)
loaded = set(sys.modules) - before
violations = sorted(
    root for root in {sorted(denied_roots)!r}
    if any(name == root or name.startswith(root + ".") for name in loaded)
)
if violations:
    print("\\n".join(violations))
    raise SystemExit(1)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
