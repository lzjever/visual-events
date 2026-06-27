from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import is_dataclass
from typing import Any

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_cli.frame_pump import HeadMotion, InputFrame


def import_dds_module(name: str) -> Any:
    try:
        return __import__(f"visual_events_cli.dds.{name}", fromlist=[name])
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.dds.{name} module: {exc}")


def make_camera_message(module: Any, **overrides: Any) -> Any:
    values = {
        "camera": "front",
        "timestamp_ms": 1710000000000,
        "width": 1280,
        "height": 720,
        "encoding": "JPEG",
        "data": JPEG_1280X720,
    }
    values.update(overrides)
    return module.CameraJpegMessage(**values)


def make_head_sample(module: Any, **overrides: Any) -> Any:
    values = {
        "timestamp_ms": 1710000000000,
        "valid": True,
        "yaw_rad": 0.1,
        "pitch_rad": -0.2,
        "yaw_vel_rad_s": 0.01,
        "pitch_vel_rad_s": 0.02,
    }
    values.update(overrides)
    return module.HeadStateSample(**values)


def assert_qos_profile(profile: Any, *, deadline_ms: int, lifespan_ms: int, lease_ms: int) -> None:
    assert profile.reliability == "best_effort"
    assert profile.durability == "volatile"
    assert profile.history == "keep_last"
    assert profile.depth == 1
    assert profile.deadline_ms == deadline_ms
    assert profile.lifespan_ms == lifespan_ms
    assert profile.liveliness_lease_ms == lease_ms


def assert_unknown_motion(motion: HeadMotion) -> None:
    assert isinstance(motion, HeadMotion)
    assert motion.state == "unknown"
    assert motion.yaw_vel_rad_s is None
    assert motion.pitch_vel_rad_s is None


def test_qos_profiles_pin_camera_head_and_gaze_contract_values():
    module = import_dds_module("qos")

    assert is_dataclass(module.DdsQosProfile)
    assert_qos_profile(
        module.CAMERA_JPEG_QOS,
        deadline_ms=150,
        lifespan_ms=300,
        lease_ms=1000,
    )
    assert_qos_profile(
        module.HEAD_STATE_QOS,
        deadline_ms=150,
        lifespan_ms=250,
        lease_ms=500,
    )
    assert_qos_profile(
        module.GAZE_TARGET_QOS,
        deadline_ms=150,
        lifespan_ms=250,
        lease_ms=500,
    )


def test_camera_jpeg_message_converts_valid_jpeg_to_input_frame():
    module = import_dds_module("types")

    frame = make_camera_message(module).to_input_frame()

    assert isinstance(frame, InputFrame)
    assert frame.camera == "front"
    assert frame.timestamp_ms == 1710000000000
    assert frame.width == 1280
    assert frame.height == 720
    assert frame.jpeg == JPEG_1280X720


@pytest.mark.parametrize(
    "overrides",
    [
        {"encoding": "jpeg"},
        {"encoding": "RGB8"},
        {"width": 0},
        {"height": 0},
        {"width": -1},
        {"height": -1},
        {"data": b""},
        {"data": bytearray(JPEG_1280X720)},
        {"data": "not-bytes"},
        {"data": b"\xff\xd8garbage\xff\xd9"},
        {"width": 640},
        {"height": 360},
        {"data": b"\x00\x01not-jpeg\xff\xd9"},
        {"data": b"\xff\xd8not-complete"},
        {"data": b"not-started\xff\xd9"},
    ],
)
def test_camera_jpeg_message_rejects_invalid_input_with_none(overrides: dict[str, Any]):
    module = import_dds_module("types")

    assert make_camera_message(module, **overrides).to_input_frame() is None


def test_fake_image_subscriber_keeps_latest_valid_frame_only_and_drops_invalid():
    types = import_dds_module("types")
    fake = import_dds_module("fake")
    subscriber = fake.FakeDdsImageSubscriber()

    subscriber.push(make_camera_message(types, timestamp_ms=1000))
    subscriber.push(make_camera_message(types, timestamp_ms=2000))
    subscriber.push(make_camera_message(types, timestamp_ms=3000))

    frame = subscriber.poll_latest()
    assert isinstance(frame, InputFrame)
    assert frame.timestamp_ms == 3000
    assert subscriber.poll_latest() is None

    subscriber.push(make_camera_message(types, timestamp_ms=4000))
    subscriber.push(make_camera_message(types, timestamp_ms=5000, data=b"bad-jpeg"))

    frame = subscriber.poll_latest()
    assert isinstance(frame, InputFrame)
    assert frame.timestamp_ms == 4000
    assert subscriber.poll_latest() is None


def test_fake_image_subscriber_poll_latest_raises_after_close():
    types = import_dds_module("types")
    fake = import_dds_module("fake")
    subscriber = fake.FakeDdsImageSubscriber()

    subscriber.push(make_camera_message(types, timestamp_ms=1000))
    subscriber.close()

    with pytest.raises(RuntimeError, match="closed"):
        subscriber.poll_latest()


@pytest.mark.parametrize(
    ("velocities", "expected_state"),
    [
        ({"yaw_vel_rad_s": 0.029, "pitch_vel_rad_s": 0.02}, "stationary"),
        ({"yaw_vel_rad_s": 0.03, "pitch_vel_rad_s": 0.02}, "stationary"),
        ({"yaw_vel_rad_s": 0.031, "pitch_vel_rad_s": 0.02}, "moving"),
        ({"yaw_vel_rad_s": 0.01, "pitch_vel_rad_s": -0.031}, "moving"),
    ],
)
def test_head_state_sample_maps_fresh_valid_motion_by_velocity_threshold(
    velocities: dict[str, float],
    expected_state: str,
):
    module = import_dds_module("types")
    sample = make_head_sample(module, **velocities)

    motion = sample.to_head_motion(
        now_ms=1710000000100,
        stale_ms=250,
        stationary_yaw_vel_rad_s=0.03,
        stationary_pitch_vel_rad_s=0.03,
    )

    assert isinstance(motion, HeadMotion)
    assert motion.state == expected_state
    assert motion.yaw_vel_rad_s == pytest.approx(velocities["yaw_vel_rad_s"])
    assert motion.pitch_vel_rad_s == pytest.approx(velocities["pitch_vel_rad_s"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"valid": False},
        {"yaw_rad": math.nan},
        {"pitch_rad": math.inf},
        {"yaw_vel_rad_s": math.nan},
        {"pitch_vel_rad_s": -math.inf},
        {"timestamp_ms": None},
        {"timestamp_ms": "1710000000000"},
        {"valid": None},
        {"yaw_rad": None},
        {"pitch_rad": None},
        {"yaw_vel_rad_s": None},
        {"pitch_vel_rad_s": None},
    ],
)
def test_head_state_sample_maps_invalid_or_non_finite_values_to_unknown(overrides: dict[str, Any]):
    module = import_dds_module("types")

    motion = make_head_sample(module, **overrides).to_head_motion(
        now_ms=1710000000100,
        stale_ms=250,
        stationary_yaw_vel_rad_s=0.03,
        stationary_pitch_vel_rad_s=0.03,
    )

    assert_unknown_motion(motion)


def test_head_state_sample_stale_boundary_is_inclusive_at_exact_stale_ms():
    module = import_dds_module("types")

    fresh_at_boundary = make_head_sample(module, timestamp_ms=1000).to_head_motion(
        now_ms=1250,
        stale_ms=250,
        stationary_yaw_vel_rad_s=0.03,
        stationary_pitch_vel_rad_s=0.03,
    )
    stale_after_boundary = make_head_sample(module, timestamp_ms=1000).to_head_motion(
        now_ms=1251,
        stale_ms=250,
        stationary_yaw_vel_rad_s=0.03,
        stationary_pitch_vel_rad_s=0.03,
    )

    assert isinstance(fresh_at_boundary, HeadMotion)
    assert fresh_at_boundary.state == "stationary"
    assert_unknown_motion(stale_after_boundary)


def test_head_state_sample_future_timestamp_maps_to_unknown():
    module = import_dds_module("types")

    motion = make_head_sample(module, timestamp_ms=1251).to_head_motion(
        now_ms=1250,
        stale_ms=250,
        stationary_yaw_vel_rad_s=0.03,
        stationary_pitch_vel_rad_s=0.03,
    )

    assert_unknown_motion(motion)


def test_fake_head_state_subscriber_returns_current_motion_and_unknown_without_sample():
    types = import_dds_module("types")
    fake = import_dds_module("fake")
    subscriber = fake.FakeDdsHeadStateSubscriber(
        stale_ms=250,
        stationary_yaw_vel_rad_s=0.03,
        stationary_pitch_vel_rad_s=0.03,
    )

    assert_unknown_motion(subscriber.current_motion(now_ms=1710000000000))

    subscriber.push(make_head_sample(types, timestamp_ms=1710000000000, yaw_vel_rad_s=0.04))
    moving = subscriber.current_motion(now_ms=1710000000100)
    assert isinstance(moving, HeadMotion)
    assert moving.state == "moving"
    assert moving.yaw_vel_rad_s == pytest.approx(0.04)
    assert moving.pitch_vel_rad_s == pytest.approx(0.02)

    assert_unknown_motion(subscriber.current_motion(now_ms=1710000000300))


def test_fake_head_state_subscriber_current_motion_raises_after_close():
    types = import_dds_module("types")
    fake = import_dds_module("fake")
    subscriber = fake.FakeDdsHeadStateSubscriber()

    subscriber.push(make_head_sample(types, timestamp_ms=1000))
    subscriber.close()

    with pytest.raises(RuntimeError, match="closed"):
        subscriber.current_motion(now_ms=1000)


def test_fake_gaze_target_publisher_lifecycle_and_payload_history():
    fake = import_dds_module("fake")
    publisher = fake.FakeDdsGazeTargetPublisher()
    payload1 = {"schema_version": 1, "state": "tracking", "frame_id": 1}
    payload2 = {"schema_version": 1, "state": "lost", "frame_id": 2}

    publisher.start()
    publisher.start()
    publisher.publish(payload1)
    publisher.publish(payload2)

    assert publisher.latest() == payload2
    assert publisher.all() == [payload1, payload2]

    publisher.close()
    publisher.close()
    with pytest.raises(RuntimeError, match="closed"):
        publisher.publish({"schema_version": 1, "state": "tracking", "frame_id": 3})


def test_protocol_module_exports_dds_protocol_names():
    module = import_dds_module("protocols")

    assert module.DdsImageSubscriber is not None
    assert module.DdsHeadStateSubscriber is not None
    assert module.DdsGazeTargetPublisher is not None


def test_importing_fake_dds_adapter_does_not_load_robot_dds_or_ml_modules():
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
importlib.import_module("visual_events_cli.dds.fake")
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
