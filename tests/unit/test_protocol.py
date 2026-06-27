import json
import struct

import pytest

from tests.jpeg_fixtures import jpeg_with_sof0_dimensions
from visual_events_server.protocol import (
    MAX_HEADER_BYTES,
    MAX_JPEG_BYTES,
    ProtocolError,
    decode_frame_message,
    encode_frame_message,
    serialize_error,
    serialize_json_message,
)


JPEG_BYTES = jpeg_with_sof0_dimensions(width=1280, height=720)


def frame_header(**overrides):
    header = {
        "type": "frame",
        "schema_version": 1,
        "camera": "front",
        "frame_id": 42,
        "timestamp_ms": 1710000000000,
        "encoding": "jpeg",
        "width": 1280,
        "height": 720,
        "head_motion": {"state": "stationary"},
    }
    header.update(overrides)
    return header


def test_decode_accepts_jpeg_when_header_dimensions_match():
    payload = encode_frame_message(frame_header(), JPEG_BYTES)

    decoded = decode_frame_message(payload)

    assert decoded.camera == "front"
    assert decoded.frame_id == 42
    assert decoded.timestamp_ms == 1710000000000
    assert decoded.width == 1280
    assert decoded.height == 720
    assert decoded.head_motion_state == "stationary"
    assert decoded.jpeg_bytes == JPEG_BYTES


def test_decode_rejects_jpeg_dimension_mismatch():
    payload = encode_frame_message(
        frame_header(width=1280, height=720),
        jpeg_with_sof0_dimensions(width=640, height=480),
        validate=False,
    )

    with pytest.raises(ProtocolError) as exc:
        decode_frame_message(payload)

    assert exc.value.code == "invalid_frame"
    assert exc.value.frame_id == 42


def test_decode_rejects_soi_eoi_bytes_without_decodable_dimensions():
    payload = encode_frame_message(
        frame_header(width=1280, height=720),
        b"\xff\xd8minimal-envelope-only\xff\xd9",
        validate=False,
    )

    with pytest.raises(ProtocolError) as exc:
        decode_frame_message(payload)

    assert exc.value.code == "invalid_frame"
    assert exc.value.frame_id == 42


def test_decode_defaults_missing_head_motion_to_unknown():
    header = frame_header()
    header.pop("head_motion")

    decoded = decode_frame_message(encode_frame_message(header, JPEG_BYTES))

    assert decoded.head_motion_state == "unknown"


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"\x00\x00\x00", "invalid_header"),
        (struct.pack(">I", 2) + b"{]" + JPEG_BYTES, "invalid_header"),
        (
            struct.pack(">I", MAX_HEADER_BYTES + 1)
            + (b" " * (MAX_HEADER_BYTES + 1)),
            "invalid_header",
        ),
    ],
)
def test_decode_rejects_invalid_headers(payload, expected_code):
    with pytest.raises(ProtocolError) as exc:
        decode_frame_message(payload)

    assert exc.value.code == expected_code
    assert exc.value.retryable is True


def test_decode_rejects_unsupported_encoding():
    payload = encode_frame_message(frame_header(encoding="png"), JPEG_BYTES, validate=False)

    with pytest.raises(ProtocolError) as exc:
        decode_frame_message(payload)

    assert exc.value.code == "unsupported_encoding"
    assert exc.value.frame_id == 42


def test_decode_rejects_invalid_jpeg_payload():
    payload = encode_frame_message(frame_header(), b"not-jpeg", validate=False)

    with pytest.raises(ProtocolError) as exc:
        decode_frame_message(payload)

    assert exc.value.code == "invalid_frame"
    assert exc.value.frame_id == 42


def test_decode_rejects_oversized_jpeg_payload():
    payload = encode_frame_message(
        frame_header(),
        b"\xff\xd8" + (b"0" * (MAX_JPEG_BYTES + 1)) + b"\xff\xd9",
        validate=False,
    )

    with pytest.raises(ProtocolError) as exc:
        decode_frame_message(payload)

    assert exc.value.code == "frame_too_large"
    assert exc.value.frame_id == 42


def test_json_serializers_emit_protocol_json_messages():
    visual_state = {
        "type": "visual_state",
        "schema_version": 1,
        "camera": "front",
        "frame_id": 42,
    }

    encoded_state = serialize_json_message(visual_state)
    encoded_error = serialize_error(
        code="invalid_frame",
        message="jpeg payload is invalid",
        frame_id=42,
        retryable=True,
    )

    assert json.loads(encoded_state)["type"] == "visual_state"
    assert json.loads(encoded_error) == {
        "type": "error",
        "schema_version": 1,
        "frame_id": 42,
        "code": "invalid_frame",
        "message": "jpeg payload is invalid",
        "retryable": True,
    }
