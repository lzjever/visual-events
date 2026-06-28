from __future__ import annotations

import base64
import json
import math
from typing import Any

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_cli.target_mapper import GazeTargetPayload


def import_bridge_protocol() -> Any:
    try:
        import visual_events_cli.dds.bridge_protocol as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.dds.bridge_protocol module: {exc}")
    return module


def camera_line(**overrides: Any) -> str:
    payload = {
        "protocol_version": 1,
        "type": "camera_jpeg",
        "dds_timestamp_ns": 1_710_000_000_123_456_789,
        "received_monotonic_ns": 99_000_000,
        "camera_name": "dds-front",
        "width": 1280,
        "height": 720,
        "encoding": "JPEG",
        "step": len(JPEG_1280X720),
        "data_size_bytes": len(JPEG_1280X720),
        "data_base64": base64.b64encode(JPEG_1280X720).decode("ascii"),
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_camera_base64_decodes_to_bridge_frame_with_logical_camera():
    module = import_bridge_protocol()

    frame = module.decode_bridge_line(
        camera_line(camera_name="dds-source"),
        logical_camera_name="logical-front",
    )

    assert isinstance(frame, module.BridgeCameraJpegFrame)
    assert frame.dds_timestamp_ns == 1_710_000_000_123_456_789
    assert frame.received_monotonic_ns == 99_000_000
    assert frame.camera == "logical-front"
    assert frame.width == 1280
    assert frame.height == 720
    assert frame.encoding == "JPEG"
    assert frame.data == JPEG_1280X720


@pytest.mark.parametrize(
    "overrides",
    [
        {"data_size_bytes": len(JPEG_1280X720) + 1},
        {"data_base64": "not valid base64"},
        {"data_base64": [255, 216, 255, 217]},
        {"data": list(JPEG_1280X720), "data_base64": None},
    ],
)
def test_camera_decode_rejects_size_mismatch_raw_bytes_and_bad_base64(
    overrides: dict[str, Any],
):
    module = import_bridge_protocol()

    with pytest.raises(module.ProtocolError):
        module.decode_bridge_line(camera_line(**overrides), logical_camera_name="front")


def test_camera_decode_allows_protocol_frame_with_bad_jpeg_bytes():
    module = import_bridge_protocol()
    bad_jpeg = b"\xff\xd8bad-jpeg\xff\xd9"

    frame = module.decode_bridge_line(
        camera_line(
            width=1280,
            height=720,
            data_size_bytes=len(bad_jpeg),
            data_base64=base64.b64encode(bad_jpeg).decode("ascii"),
        ),
        logical_camera_name="front",
    )

    assert isinstance(frame, module.BridgeCameraJpegFrame)
    assert frame.camera == "front"
    assert frame.width == 1280
    assert frame.height == 720
    assert frame.data == bad_jpeg


def test_head_state_decode_preserves_bridge_timestamps_and_motion_fields():
    module = import_bridge_protocol()
    line = json.dumps(
        {
            "protocol_version": 1,
            "type": "head_state",
            "dds_timestamp_ns": 123_000_000,
            "received_monotonic_ns": 9_999_000_000,
            "valid": True,
            "state": "moving",
            "yaw_rad": 0.1,
            "pitch_rad": -0.2,
            "yaw_vel_rad_s": 0.05,
            "pitch_vel_rad_s": 0.01,
        }
    )

    frame = module.decode_bridge_line(line, logical_camera_name="front")

    assert isinstance(frame, module.BridgeHeadStateFrame)
    assert frame.dds_timestamp_ns == 123_000_000
    assert frame.received_monotonic_ns == 9_999_000_000
    assert frame.valid is True
    assert frame.state == "moving"
    assert frame.yaw_rad == pytest.approx(0.1)
    assert frame.pitch_rad == pytest.approx(-0.2)
    assert frame.yaw_vel_rad_s == pytest.approx(0.05)
    assert frame.pitch_vel_rad_s == pytest.approx(0.01)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dds_timestamp_ns": None},
        {"received_monotonic_ns": None},
        {"valid": "true"},
        {"state": None},
        {"state": "drifted"},
        {"yaw_rad": math.nan},
        {"pitch_rad": math.inf},
        {"yaw_vel_rad_s": None},
        {"pitch_vel_rad_s": -math.inf},
    ],
)
def test_head_state_decode_rejects_missing_or_non_finite_fields(
    overrides: dict[str, Any],
):
    module = import_bridge_protocol()
    payload = {
        "protocol_version": 1,
        "type": "head_state",
        "dds_timestamp_ns": 1_710_000_000_000_000_000,
        "received_monotonic_ns": 99_000_000,
        "valid": True,
        "state": "stationary",
        "yaw_rad": 0.1,
        "pitch_rad": -0.2,
        "yaw_vel_rad_s": 0.01,
        "pitch_vel_rad_s": 0.02,
    }
    payload.update(overrides)

    with pytest.raises(module.ProtocolError):
        module.decode_bridge_line(json.dumps(payload), logical_camera_name="front")


def test_head_state_decode_rejects_timestamp_ms_as_inbound_abi_substitute():
    module = import_bridge_protocol()
    payload = {
        "protocol_version": 1,
        "type": "head_state",
        "timestamp_ms": 1_710_000_000_000,
        "received_monotonic_ns": 99_000_000,
        "valid": True,
        "state": "stationary",
        "yaw_rad": 0.1,
        "pitch_rad": -0.2,
        "yaw_vel_rad_s": 0.01,
        "pitch_vel_rad_s": 0.02,
    }

    with pytest.raises(module.ProtocolError):
        module.decode_bridge_line(json.dumps(payload), logical_camera_name="front")


def test_gaze_encode_includes_only_canonical_payload_fields():
    module = import_bridge_protocol()
    payload = GazeTargetPayload(
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

    line = module.encode_gaze_target_line(payload)
    encoded = json.loads(line)

    assert line.endswith("\n")
    assert encoded == {
        "protocol_version": 1,
        "type": "gaze_target",
        "schema_version": 1,
        "camera": "front",
        "frame_id": 42,
        "frame_timestamp_ms": 1_710_000_000_000,
        "publish_timestamp_ms": 1_710_000_000_082,
        "valid": True,
        "state": "tracking",
        "target_track_id": 7,
        "target_u": 640.0,
        "target_v": 360.0,
        "target_norm_x": 0.0,
        "target_norm_y": 0.0,
        "image_width": 1280,
        "image_height": 720,
        "confidence": 0.91,
        "reason": "nearest",
        "stale_after_ms": 250,
    }
    assert "received_monotonic_ns" not in encoded
    assert "box_norm_width" not in encoded
    assert "box_norm_height" not in encoded
    assert "track_id" not in encoded


def test_status_frame_requires_code_and_message_fields():
    module = import_bridge_protocol()

    frame = module.decode_bridge_line(
        json.dumps(
            {
                "protocol_version": 1,
                "type": "status",
                "code": "ready",
                "message": "bridge started",
            }
        ),
        logical_camera_name="front",
    )

    assert isinstance(frame, module.BridgeStatusFrame)
    assert frame.code == "ready"
    assert frame.message == "bridge started"

    for missing_field in ("code", "message"):
        payload = {
            "protocol_version": 1,
            "type": "status",
            "code": "ready",
            "message": "bridge started",
        }
        payload.pop(missing_field)
        with pytest.raises(module.ProtocolError):
            module.decode_bridge_line(json.dumps(payload), logical_camera_name="front")


def test_fatal_error_frame_parses_for_process_layer():
    module = import_bridge_protocol()
    frame = module.decode_bridge_line(
        json.dumps(
            {
                "protocol_version": 1,
                "type": "error",
                "code": "dds_init_failed",
                "message": "native bridge failed",
                "fatal": True,
            }
        ),
        logical_camera_name="front",
    )

    assert isinstance(frame, module.BridgeErrorFrame)
    assert frame.code == "dds_init_failed"
    assert frame.message == "native bridge failed"
    assert frame.fatal is True


@pytest.mark.parametrize("missing_field", ["code", "message", "fatal"])
def test_error_frame_requires_code_message_and_fatal(missing_field: str):
    module = import_bridge_protocol()
    payload = {
        "protocol_version": 1,
        "type": "error",
        "code": "dds_init_failed",
        "message": "native bridge failed",
        "fatal": True,
    }
    payload.pop(missing_field)

    with pytest.raises(module.ProtocolError):
        module.decode_bridge_line(json.dumps(payload), logical_camera_name="front")
