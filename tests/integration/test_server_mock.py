import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from visual_events_server.app import create_app
from visual_events_server.protocol import encode_frame_message


JPEG_BYTES = b"\xff\xd8\xff\xe0minimal-jpeg\xff\xd9"


def frame_header(**overrides):
    header = {
        "type": "frame",
        "schema_version": 1,
        "camera": "front",
        "frame_id": 7,
        "timestamp_ms": 1710000000000,
        "encoding": "jpeg",
        "width": 1280,
        "height": 720,
        "head_motion": {"state": "stationary"},
    }
    header.update(overrides)
    return header


def test_healthz_returns_ok():
    client = TestClient(create_app())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_websocket_stream_returns_mock_visual_state_for_valid_frame():
    client = TestClient(create_app())

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
        message = json.loads(websocket.receive_text())

    assert message["type"] == "visual_state"
    assert message["schema_version"] == 1
    assert message["camera"] == "front"
    assert message["frame_id"] == 7
    assert message["frame_timestamp_ms"] == 1710000000000
    assert message["image_size"] == [1280, 720]
    assert message["tracks"] == []
    assert message["attention"] is None
    assert message["scene_flags"] == {
        "has_person": False,
        "person_count": 0,
        "largest_person_stable": False,
        "someone_near_center": False,
    }
    assert message["semantic_events"] == []


def test_websocket_stream_returns_protocol_error_for_invalid_jpeg_and_continues():
    client = TestClient(create_app())

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), b"not-jpeg", validate=False))
        error = json.loads(websocket.receive_text())

        websocket.send_bytes(encode_frame_message(frame_header(frame_id=8), JPEG_BYTES))
        ok = json.loads(websocket.receive_text())

    assert error["type"] == "error"
    assert error["frame_id"] == 7
    assert error["code"] == "invalid_frame"
    assert error["retryable"] is True
    assert ok["type"] == "visual_state"
    assert ok["frame_id"] == 8


def test_websocket_stream_returns_error_for_text_messages():
    client = TestClient(create_app())

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_text("not a binary frame")
        message = json.loads(websocket.receive_text())

    assert message["type"] == "error"
    assert message["code"] == "invalid_frame"
    assert message["retryable"] is True


def test_websocket_stream_rejects_camera_switch_and_closes_connection():
    client = TestClient(create_app())

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(camera="front"), JPEG_BYTES))
        first = json.loads(websocket.receive_text())

        websocket.send_bytes(
            encode_frame_message(
                frame_header(camera="rear", frame_id=8),
                JPEG_BYTES,
            )
        )
        error = json.loads(websocket.receive_text())

        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()

    assert first["type"] == "visual_state"
    assert first["camera"] == "front"
    assert error["type"] == "error"
    assert error["frame_id"] == 8
    assert error["code"] == "invalid_header"
    assert error["retryable"] is False
