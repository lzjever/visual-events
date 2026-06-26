from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = 1
MAX_HEADER_BYTES = 16 * 1024
MAX_JPEG_BYTES = 2 * 1024 * 1024
_HEADER_PREFIX_BYTES = 4
_SUPPORTED_HEAD_MOTION_STATES = {"stationary", "moving", "unknown"}


@dataclass(frozen=True)
class FrameMessage:
    camera: str
    frame_id: int
    timestamp_ms: int
    width: int
    height: int
    jpeg_bytes: bytes
    head_motion_state: str = "unknown"
    header: dict[str, Any] | None = None


class ProtocolError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        frame_id: int | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.frame_id = frame_id
        self.retryable = retryable


def encode_frame_message(
    header: dict[str, Any],
    jpeg_bytes: bytes,
    *,
    validate: bool = True,
) -> bytes:
    header_json = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(header_json) > MAX_HEADER_BYTES:
        raise ProtocolError("invalid_header", "header exceeds 16 KiB")
    if validate:
        _validate_jpeg(jpeg_bytes, frame_id=_frame_id_from_header(header))
    return struct.pack(">I", len(header_json)) + header_json + jpeg_bytes


def decode_frame_message(message: bytes) -> FrameMessage:
    if len(message) < _HEADER_PREFIX_BYTES:
        raise ProtocolError("invalid_header", "message is missing header length")

    header_len = struct.unpack(">I", message[:_HEADER_PREFIX_BYTES])[0]
    if header_len > MAX_HEADER_BYTES:
        raise ProtocolError("invalid_header", "header exceeds 16 KiB")

    header_end = _HEADER_PREFIX_BYTES + header_len
    if len(message) < header_end:
        raise ProtocolError("invalid_header", "message ended before header")

    header_bytes = message[_HEADER_PREFIX_BYTES:header_end]
    header = _decode_header(header_bytes)
    frame_id = _frame_id_from_header(header)
    jpeg_bytes = message[header_end:]

    _validate_frame_header(header)
    _validate_jpeg_size(jpeg_bytes, frame_id=frame_id)
    _validate_jpeg(jpeg_bytes, frame_id=frame_id)

    head_motion = header.get("head_motion")
    head_motion_state = "unknown"
    if isinstance(head_motion, dict):
        head_motion_state = head_motion.get("state", "unknown")

    return FrameMessage(
        camera=header["camera"],
        frame_id=header["frame_id"],
        timestamp_ms=header["timestamp_ms"],
        width=header["width"],
        height=header["height"],
        jpeg_bytes=jpeg_bytes,
        head_motion_state=head_motion_state,
        header=header,
    )


def serialize_json_message(message: dict[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def serialize_error(
    *,
    code: str,
    message: str,
    frame_id: int | None = None,
    retryable: bool = True,
) -> str:
    payload: dict[str, Any] = {
        "type": "error",
        "schema_version": SCHEMA_VERSION,
        "frame_id": frame_id,
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    return serialize_json_message(payload)


def serialize_protocol_error(error: ProtocolError) -> str:
    return serialize_error(
        code=error.code,
        message=error.message,
        frame_id=error.frame_id,
        retryable=error.retryable,
    )


def _decode_header(header_bytes: bytes) -> dict[str, Any]:
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid_header", "header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise ProtocolError("invalid_header", "header JSON must be an object")
    return header


def _validate_frame_header(header: dict[str, Any]) -> None:
    frame_id = _frame_id_from_header(header)
    if header.get("type") != "frame":
        raise ProtocolError("invalid_header", "header type must be frame", frame_id=frame_id)
    if header.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError(
            "invalid_header",
            "unsupported schema_version",
            frame_id=frame_id,
        )

    for key in ("camera", "frame_id", "timestamp_ms", "encoding", "width", "height"):
        if key not in header:
            raise ProtocolError(
                "invalid_header",
                f"header is missing required field {key}",
                frame_id=frame_id,
            )

    if not isinstance(header["camera"], str) or not header["camera"]:
        raise ProtocolError("invalid_header", "camera must be a non-empty string", frame_id=frame_id)
    if not isinstance(header["frame_id"], int):
        raise ProtocolError("invalid_header", "frame_id must be an integer", frame_id=frame_id)
    if not isinstance(header["timestamp_ms"], int):
        raise ProtocolError("invalid_header", "timestamp_ms must be an integer", frame_id=frame_id)
    if header["encoding"] != "jpeg":
        raise ProtocolError(
            "unsupported_encoding",
            "only jpeg encoding is supported",
            frame_id=frame_id,
        )
    if not _positive_int(header["width"]) or not _positive_int(header["height"]):
        raise ProtocolError(
            "invalid_header",
            "width and height must be positive integers",
            frame_id=frame_id,
        )

    head_motion = header.get("head_motion")
    if head_motion is not None:
        if not isinstance(head_motion, dict):
            raise ProtocolError("invalid_header", "head_motion must be an object", frame_id=frame_id)
        state = head_motion.get("state", "unknown")
        if state not in _SUPPORTED_HEAD_MOTION_STATES:
            raise ProtocolError(
                "invalid_header",
                "head_motion.state is unsupported",
                frame_id=frame_id,
            )


def _validate_jpeg_size(jpeg_bytes: bytes, *, frame_id: int | None) -> None:
    if len(jpeg_bytes) > MAX_JPEG_BYTES:
        raise ProtocolError(
            "frame_too_large",
            "jpeg payload exceeds 2 MiB",
            frame_id=frame_id,
        )


def _validate_jpeg(jpeg_bytes: bytes, *, frame_id: int | None) -> None:
    _validate_jpeg_size(jpeg_bytes, frame_id=frame_id)
    if not jpeg_bytes.startswith(b"\xff\xd8") or not jpeg_bytes.endswith(b"\xff\xd9"):
        raise ProtocolError(
            "invalid_frame",
            "jpeg payload is invalid",
            frame_id=frame_id,
        )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _frame_id_from_header(header: dict[str, Any]) -> int | None:
    frame_id = header.get("frame_id")
    return frame_id if isinstance(frame_id, int) else None
