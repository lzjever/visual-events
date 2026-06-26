from __future__ import annotations

import json

from fastapi.testclient import TestClient

from visual_events_server.app import create_app
from visual_events_server.inference.base import PersonPoseDetection, PoseDetections
from visual_events_server.processor import BackendVisualFrameProcessor
from visual_events_server.protocol import encode_frame_message


JPEG_BYTES = b"\xff\xd8\xff\xe0minimal-jpeg\xff\xd9"


def frame_header(**overrides):
    header = {
        "type": "frame",
        "schema_version": 1,
        "camera": "front",
        "frame_id": 1,
        "timestamp_ms": 1710000000000,
        "encoding": "jpeg",
        "width": 1280,
        "height": 720,
        "head_motion": {"state": "stationary"},
    }
    header.update(overrides)
    return header


def person(
    bbox=(100.0, 100.0, 220.0, 460.0),
    *,
    confidence: float = 0.9,
) -> PersonPoseDetection:
    x1, y1, x2, y2 = bbox
    return PersonPoseDetection(
        bbox_xyxy=bbox,
        bbox_area=(x2 - x1) * (y2 - y1),
        confidence=confidence,
        keypoints=[],
    )


class SequenceBackend:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    async def infer(self, frame):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def visual_state_from(websocket, *, frame_id: int, timestamp_ms: int) -> dict:
    websocket.send_bytes(
        encode_frame_message(
            frame_header(frame_id=frame_id, timestamp_ms=timestamp_ms),
            JPEG_BYTES,
        )
    )
    return json.loads(websocket.receive_text())


def test_fake_backend_multi_frame_websocket_outputs_stable_track_id():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person((100.0, 100.0, 220.0, 460.0))]),
                PoseDetections(persons=[person((106.0, 100.0, 226.0, 460.0))]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        first = visual_state_from(websocket, frame_id=1, timestamp_ms=1000)
        second = visual_state_from(websocket, frame_id=2, timestamp_ms=1200)

    assert first["type"] == "visual_state"
    assert second["type"] == "visual_state"
    assert len(first["tracks"]) == 1
    assert len(second["tracks"]) == 1
    assert second["tracks"][0]["track_id"] == first["tracks"][0]["track_id"]
    assert second["tracks"][0]["age_ms"] == 200
    assert second["tracks"][0]["lost_ms"] == 0
    assert second["tracks"][0]["velocity_uv_s"][0] > 0
    assert second["scene_flags"]["has_person"] is True
    assert second["scene_flags"]["person_count"] == 1
    assert second["attention"] is None
    assert second["semantic_events"] == []


def test_backend_error_does_not_advance_or_damage_tracker_state():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person((100.0, 100.0, 220.0, 460.0))]),
                RuntimeError("backend exploded"),
                PoseDetections(persons=[person((106.0, 100.0, 226.0, 460.0))]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        first = visual_state_from(websocket, frame_id=1, timestamp_ms=1000)
        websocket.send_bytes(
            encode_frame_message(
                frame_header(frame_id=2, timestamp_ms=1100),
                JPEG_BYTES,
            )
        )
        error = json.loads(websocket.receive_text())
        recovered = visual_state_from(websocket, frame_id=3, timestamp_ms=1200)

    assert error["type"] == "error"
    assert error["code"] == "backend_unavailable"
    assert recovered["type"] == "visual_state"
    assert recovered["tracks"][0]["track_id"] == first["tracks"][0]["track_id"]
    assert recovered["tracks"][0]["lost_ms"] == 0
    assert recovered["scene_flags"]["person_count"] == 1


def test_websocket_connections_have_isolated_tracking_state():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person((100.0, 100.0, 220.0, 460.0))]),
                PoseDetections(persons=[person((106.0, 100.0, 226.0, 460.0))]),
                PoseDetections(persons=[person((800.0, 100.0, 920.0, 460.0))]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as first_socket:
        first = visual_state_from(first_socket, frame_id=1, timestamp_ms=1000)
        second = visual_state_from(first_socket, frame_id=2, timestamp_ms=1200)

    with client.websocket_connect("/v1/stream") as second_socket:
        new_connection = visual_state_from(second_socket, frame_id=1, timestamp_ms=5000)

    assert first["tracks"][0]["track_id"] == 1
    assert second["tracks"][0]["track_id"] == 1
    assert [track["track_id"] for track in new_connection["tracks"]] == [1]
    assert new_connection["tracks"][0]["age_ms"] == 0
    assert new_connection["attention"] is None
    assert new_connection["semantic_events"] == []
