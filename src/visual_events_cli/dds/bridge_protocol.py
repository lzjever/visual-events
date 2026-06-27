from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from visual_events_cli.dds.types import CameraJpegMessage
from visual_events_cli.target_mapper import GazeTargetPayload


PROTOCOL_VERSION = 1
GAZE_TARGET_FIELDS = (
    "schema_version",
    "camera",
    "frame_id",
    "frame_timestamp_ms",
    "publish_timestamp_ms",
    "valid",
    "state",
    "target_track_id",
    "target_u",
    "target_v",
    "target_norm_x",
    "target_norm_y",
    "image_width",
    "image_height",
    "confidence",
    "reason",
    "stale_after_ms",
)
HEAD_STATE_STATES = frozenset({"stationary", "moving", "unknown"})


class ProtocolError(ValueError):
    """Raised when a bridge JSONL frame violates the Python-side contract."""


@dataclass(frozen=True)
class BridgeStatusFrame:
    code: str
    message: str


@dataclass(frozen=True)
class BridgeErrorFrame:
    code: str
    message: str
    fatal: bool


@dataclass(frozen=True)
class BridgeHeadStateFrame:
    dds_timestamp_ns: int
    received_monotonic_ns: int
    valid: bool
    state: str
    yaw_rad: float
    pitch_rad: float
    yaw_vel_rad_s: float
    pitch_vel_rad_s: float


BridgeInboundFrame = (
    CameraJpegMessage | BridgeHeadStateFrame | BridgeStatusFrame | BridgeErrorFrame
)


def decode_bridge_line(
    line: str | bytes,
    *,
    logical_camera_name: str,
) -> BridgeInboundFrame:
    payload = _decode_json_object(line)
    _require_protocol_version(payload)

    frame_type = payload.get("type")
    if frame_type == "camera_jpeg":
        return _decode_camera_jpeg(payload, logical_camera_name=logical_camera_name)
    if frame_type == "head_state":
        return _decode_head_state(payload)
    if frame_type == "status":
        return _decode_status(payload)
    if frame_type == "error":
        return _decode_error(payload)
    raise ProtocolError("unsupported bridge frame type")


def encode_gaze_target_line(payload: GazeTargetPayload | Mapping[str, Any]) -> str:
    if isinstance(payload, GazeTargetPayload):
        source = payload.to_dict()
    elif isinstance(payload, Mapping):
        source = payload
    else:
        raise ProtocolError("gaze_target payload must be a mapping")

    missing = [field for field in GAZE_TARGET_FIELDS if field not in source]
    if missing:
        raise ProtocolError(f"gaze_target payload missing fields: {', '.join(missing)}")

    frame = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "gaze_target",
    }
    for field in GAZE_TARGET_FIELDS:
        frame[field] = source[field]

    try:
        return json.dumps(
            frame,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"could not encode gaze_target payload: {exc}") from exc


def _decode_json_object(line: str | bytes) -> dict[str, Any]:
    if isinstance(line, bytes):
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("bridge frame must be UTF-8") from exc
    elif isinstance(line, str):
        text = line
    else:
        raise ProtocolError("bridge frame must be str or bytes")

    text = text.rstrip("\r\n")
    if "\n" in text or "\r" in text:
        raise ProtocolError("bridge frame must be a single line")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON bridge frame: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("bridge frame must be a JSON object")
    return payload


def _require_protocol_version(payload: Mapping[str, Any]) -> None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported bridge protocol_version")


def _decode_camera_jpeg(
    payload: Mapping[str, Any],
    *,
    logical_camera_name: str,
) -> CameraJpegMessage:
    if "data" in payload:
        raise ProtocolError("camera_jpeg raw data field is not accepted")

    dds_timestamp_ns = _required_int(payload, "dds_timestamp_ns")
    _required_int(payload, "received_monotonic_ns")
    _required_str(payload, "camera_name")
    width = _required_positive_int(payload, "width")
    height = _required_positive_int(payload, "height")
    encoding = _required_str(payload, "encoding")
    _required_int(payload, "step")
    data_size_bytes = _required_nonnegative_int(payload, "data_size_bytes")
    data_base64 = _required_str(payload, "data_base64")

    try:
        data = base64.b64decode(data_base64.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ProtocolError("camera_jpeg data_base64 is invalid") from exc

    if len(data) != data_size_bytes:
        raise ProtocolError("camera_jpeg data_size_bytes does not match payload")

    return CameraJpegMessage(
        camera=str(logical_camera_name),
        timestamp_ms=dds_timestamp_ns // 1_000_000,
        width=width,
        height=height,
        encoding=encoding,
        data=data,
    )


def _decode_head_state(payload: Mapping[str, Any]) -> BridgeHeadStateFrame:
    dds_timestamp_ns = _required_int(payload, "dds_timestamp_ns")
    received_monotonic_ns = _required_int(payload, "received_monotonic_ns")
    valid = _required_bool(payload, "valid")
    state = _required_str(payload, "state")
    if state not in HEAD_STATE_STATES:
        raise ProtocolError("head_state state must be stationary, moving, or unknown")

    return BridgeHeadStateFrame(
        dds_timestamp_ns=dds_timestamp_ns,
        received_monotonic_ns=received_monotonic_ns,
        valid=valid,
        state=state,
        yaw_rad=_required_finite_number(payload, "yaw_rad"),
        pitch_rad=_required_finite_number(payload, "pitch_rad"),
        yaw_vel_rad_s=_required_finite_number(payload, "yaw_vel_rad_s"),
        pitch_vel_rad_s=_required_finite_number(payload, "pitch_vel_rad_s"),
    )


def _decode_status(payload: Mapping[str, Any]) -> BridgeStatusFrame:
    return BridgeStatusFrame(
        code=_required_str(payload, "code"),
        message=_required_str(payload, "message"),
    )


def _decode_error(payload: Mapping[str, Any]) -> BridgeErrorFrame:
    return BridgeErrorFrame(
        code=_required_str(payload, "code"),
        message=_required_str(payload, "message"),
        fatal=_required_bool(payload, "fatal"),
    )


def _required_str(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ProtocolError(f"{field} must be a string")
    return value


def _required_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if type(value) is not int:
        raise ProtocolError(f"{field} must be an integer")
    return value


def _required_bool(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    if type(value) is not bool:
        raise ProtocolError(f"{field} must be a boolean")
    return value


def _required_nonnegative_int(payload: Mapping[str, Any], field: str) -> int:
    value = _required_int(payload, field)
    if value < 0:
        raise ProtocolError(f"{field} must be non-negative")
    return value


def _required_positive_int(payload: Mapping[str, Any], field: str) -> int:
    value = _required_int(payload, field)
    if value <= 0:
        raise ProtocolError(f"{field} must be positive")
    return value


def _required_finite_number(payload: Mapping[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{field} must be finite")
    return result
