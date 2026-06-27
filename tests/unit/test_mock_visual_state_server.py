from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_server.protocol import encode_frame_message


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "mock_visual_state_server.py"


def import_tool():
    try:
        import tools.mock_visual_state_server as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected tools.mock_visual_state_server module: {exc}")
    return module


def frame_header(**overrides: Any) -> dict[str, Any]:
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


def frame_message(**header_overrides: Any) -> bytes:
    return encode_frame_message(frame_header(**header_overrides), JPEG_1280X720)


def websocket_client(*args: str) -> TestClient:
    module = import_tool()
    return TestClient(module.create_app(module.parse_args(list(args))))


def receive_json(websocket) -> dict[str, Any]:
    return json.loads(websocket.receive_text())


def assert_error(
    message: dict[str, Any],
    *,
    code: str,
    frame_id: int | None,
    retryable: bool,
) -> None:
    assert message["type"] == "error"
    assert message["schema_version"] == 1
    assert message["frame_id"] == frame_id
    assert message["code"] == code
    assert message["retryable"] is retryable


def test_parse_args_defaults_to_local_tracking_server():
    args = import_tool().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8767
    assert args.profile == "tracking"
    assert args.delay_ms == 0
    assert args.disconnect_after is None


def test_parse_args_rejects_negative_delay():
    with pytest.raises(SystemExit):
        import_tool().parse_args(["--delay-ms", "-1"])


def test_parse_args_rejects_zero_disconnect_after():
    with pytest.raises(SystemExit):
        import_tool().parse_args(["--disconnect-after", "0"])


def test_healthz_returns_profile_and_process_identity():
    client = websocket_client("--profile", "lost")

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "profile": "lost", "pid": os.getpid()}


def test_tracking_response_maps_to_valid_gaze_target():
    module = import_tool()
    from visual_events_cli.target_mapper import map_visual_state_to_gaze_target

    client = TestClient(module.create_app(module.parse_args(["--delay-ms", "12"])))

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(frame_message())
        state = receive_json(websocket)

    assert state["type"] == "visual_state"
    assert state["camera"] == "front"
    assert state["frame_id"] == 7
    assert state["frame_timestamp_ms"] == 1710000000000
    assert state["server_timestamp_ms"] == 1710000000012
    assert state["image_size"] == [1280, 720]
    assert len(state["tracks"]) == 1
    assert state["attention"]["target_track_id"] == state["tracks"][0]["track_id"]
    assert state["semantic_events"] == []

    gaze = map_visual_state_to_gaze_target(
        state,
        publish_timestamp_ms=state["server_timestamp_ms"],
    )

    assert gaze.valid is True
    assert gaze.state == "tracking"
    assert gaze.camera == "front"
    assert gaze.frame_id == 7


def test_lost_response_maps_to_invalid_lost_gaze_target():
    from visual_events_cli.target_mapper import map_visual_state_to_gaze_target

    client = websocket_client("--profile", "lost")

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(frame_message())
        state = receive_json(websocket)

    assert state["type"] == "visual_state"
    assert state["tracks"] == []
    assert state["attention"] is None
    assert state["scene_flags"] == {
        "has_person": False,
        "person_count": 0,
        "largest_person_stable": False,
        "someone_near_center": False,
    }
    assert state["semantic_events"] == []

    gaze = map_visual_state_to_gaze_target(
        state,
        publish_timestamp_ms=state["server_timestamp_ms"],
    )

    assert gaze.valid is False
    assert gaze.state == "lost"
    assert gaze.reason == "lost"


def test_event_profile_emits_allowlist_event_once_per_connection():
    client = websocket_client("--profile", "event")

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(frame_message(frame_id=7))
        first = receive_json(websocket)
        websocket.send_bytes(frame_message(frame_id=8, timestamp_ms=1710000000033))
        second = receive_json(websocket)

    events = first["semantic_events"]
    assert [event["event"] for event in events] == ["person_waving"]
    assert all(event["event"] != "attention_target_changed" for event in events)
    assert second["semantic_events"] == []

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(frame_message(frame_id=9, timestamp_ms=1710000000066))
        new_connection_first = receive_json(websocket)

    assert [event["event"] for event in new_connection_first["semantic_events"]] == [
        "person_waving"
    ]


def test_text_message_returns_retryable_error_and_connection_continues():
    client = websocket_client()

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_text("not a binary frame")
        error = receive_json(websocket)

        websocket.send_bytes(frame_message(frame_id=8))
        state = receive_json(websocket)

    assert_error(error, code="invalid_frame", frame_id=None, retryable=True)
    assert state["type"] == "visual_state"
    assert state["frame_id"] == 8


def test_invalid_jpeg_returns_protocol_error_and_connection_continues():
    client = websocket_client()

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(
            encode_frame_message(frame_header(frame_id=7), b"not-jpeg", validate=False)
        )
        error = receive_json(websocket)

        websocket.send_bytes(frame_message(frame_id=8))
        state = receive_json(websocket)

    assert_error(error, code="invalid_frame", frame_id=7, retryable=True)
    assert state["type"] == "visual_state"
    assert state["frame_id"] == 8


def test_camera_switch_returns_non_retryable_error_and_closes_policy_violation():
    client = websocket_client()

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(frame_message(camera="front", frame_id=7))
        first = receive_json(websocket)

        websocket.send_bytes(frame_message(camera="rear", frame_id=8))
        error = receive_json(websocket)

        with pytest.raises(WebSocketDisconnect) as exc:
            websocket.receive_text()

    assert first["type"] == "visual_state"
    assert_error(error, code="invalid_header", frame_id=8, retryable=False)
    assert exc.value.code == 1008


def test_disconnect_after_closes_after_legal_decode_without_response():
    client = websocket_client("--disconnect-after", "1")

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(frame_message(frame_id=7))

        with pytest.raises(WebSocketDisconnect):
            websocket.receive_text()


def test_source_audit_avoids_heavy_runtime_and_cli_bridge_imports():
    tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    forbidden_roots = {"torch", "ultralytics", "subprocess", "visual_events_cli"}
    assert {
        name.split(".", 1)[0]
        for name in imported_modules
        if name.split(".", 1)[0] in forbidden_roots
    } == set()
    assert all(".dds" not in name.lower() for name in imported_modules)
    assert all("dds_bridge" not in name.lower() for name in imported_modules)
