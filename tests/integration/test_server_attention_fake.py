from __future__ import annotations

import json

from fastapi.testclient import TestClient

from visual_events_server.app import create_app
from visual_events_server.inference.base import (
    PersonPoseDetection,
    PoseDetections,
    PoseKeypoint,
)
from visual_events_server.processor import BackendVisualFrameProcessor
from visual_events_server.protocol import encode_frame_message


JPEG_BYTES = b"\xff\xd8\xff\xe0minimal-jpeg\xff\xd9"


def frame_header(**overrides):
    header = {
        "type": "frame",
        "schema_version": 1,
        "camera": "front",
        "frame_id": 1,
        "timestamp_ms": 1000,
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
    nose: tuple[float, float] | None = None,
) -> PersonPoseDetection:
    x1, y1, x2, y2 = bbox
    keypoints = []
    if nose is not None:
        keypoints.append(
            PoseKeypoint(name="nose", x=nose[0], y=nose[1], confidence=0.9)
        )
    return PersonPoseDetection(
        bbox_xyxy=bbox,
        bbox_area=(x2 - x1) * (y2 - y1),
        confidence=confidence,
        keypoints=keypoints,
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


def test_fake_backend_multi_frame_stream_outputs_attention_after_stable_track():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person(nose=(160.0, 130.0))]),
                PoseDetections(persons=[person((104.0, 100.0, 224.0, 460.0))]),
                PoseDetections(persons=[person((108.0, 100.0, 228.0, 460.0))]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        first = visual_state_from(websocket, frame_id=1, timestamp_ms=1000)
        second = visual_state_from(websocket, frame_id=2, timestamp_ms=1200)
        third = visual_state_from(websocket, frame_id=3, timestamp_ms=1400)

    assert first["attention"] is None
    assert second["attention"] is None
    assert third["attention"] == {
        "target_track_id": third["tracks"][0]["track_id"],
        "target_uv": third["tracks"][0]["head_uv"],
        "reason": "largest_stable_person",
        "confidence": third["tracks"][0]["confidence"],
    }
    assert third["scene_flags"]["largest_person_stable"] is True
    assert third["semantic_events"] == []


def test_websocket_connections_have_isolated_attention_state():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person((100.0, 100.0, 220.0, 460.0))]),
                PoseDetections(persons=[person((104.0, 100.0, 224.0, 460.0))]),
                PoseDetections(persons=[person((108.0, 100.0, 228.0, 460.0))]),
                PoseDetections(persons=[person((800.0, 100.0, 920.0, 460.0))]),
                PoseDetections(persons=[person((804.0, 100.0, 924.0, 460.0))]),
                PoseDetections(persons=[person((808.0, 100.0, 928.0, 460.0))]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as first_socket:
        visual_state_from(first_socket, frame_id=1, timestamp_ms=1000)
        visual_state_from(first_socket, frame_id=2, timestamp_ms=1200)
        first_stable = visual_state_from(first_socket, frame_id=3, timestamp_ms=1400)

    with client.websocket_connect("/v1/stream") as second_socket:
        second_first = visual_state_from(second_socket, frame_id=1, timestamp_ms=5000)
        visual_state_from(second_socket, frame_id=2, timestamp_ms=5200)
        second_stable = visual_state_from(second_socket, frame_id=3, timestamp_ms=5400)

    assert first_stable["attention"]["target_track_id"] == 1
    assert second_first["tracks"][0]["track_id"] == 1
    assert second_first["attention"] is None
    assert second_stable["attention"]["target_track_id"] == 1


def test_backend_error_does_not_advance_or_pollute_attention_state():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person((100.0, 100.0, 220.0, 460.0))]),
                PoseDetections(persons=[person((104.0, 100.0, 224.0, 460.0))]),
                PoseDetections(persons=[person((108.0, 100.0, 228.0, 460.0))]),
                RuntimeError("backend exploded"),
                PoseDetections(persons=[person((112.0, 100.0, 232.0, 460.0))]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        visual_state_from(websocket, frame_id=1, timestamp_ms=1000)
        visual_state_from(websocket, frame_id=2, timestamp_ms=1200)
        selected = visual_state_from(websocket, frame_id=3, timestamp_ms=1400)
        websocket.send_bytes(
            encode_frame_message(
                frame_header(frame_id=4, timestamp_ms=1500),
                JPEG_BYTES,
            )
        )
        error = json.loads(websocket.receive_text())
        recovered = visual_state_from(websocket, frame_id=5, timestamp_ms=1600)

    assert selected["attention"]["target_track_id"] == 1
    assert error["type"] == "error"
    assert error["code"] == "backend_unavailable"
    assert recovered["tracks"][0]["track_id"] == 1
    assert recovered["attention"]["target_track_id"] == 1
    assert recovered["semantic_events"] == []


def test_lost_detection_sequence_outputs_short_attention_hold():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person((100.0, 100.0, 220.0, 460.0))]),
                PoseDetections(persons=[person((104.0, 100.0, 224.0, 460.0))]),
                PoseDetections(persons=[person((108.0, 100.0, 228.0, 460.0))]),
                PoseDetections(persons=[]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        visual_state_from(websocket, frame_id=1, timestamp_ms=1000)
        visual_state_from(websocket, frame_id=2, timestamp_ms=1200)
        selected = visual_state_from(websocket, frame_id=3, timestamp_ms=1400)
        held = visual_state_from(websocket, frame_id=4, timestamp_ms=1600)

    assert selected["attention"]["target_track_id"] == 1
    assert held["attention"]["target_track_id"] == 1
    assert held["attention"]["reason"] == "held_lost_target"
    assert held["tracks"][0]["lost_ms"] == 200
    assert held["scene_flags"]["person_count"] == 0
    assert held["scene_flags"]["largest_person_stable"] is False
    assert held["semantic_events"] == []
