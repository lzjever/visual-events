from __future__ import annotations

import json

import pytest
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
    bbox=(460.0, 100.0, 820.0, 620.0),
    *,
    confidence: float = 0.9,
    keypoints: list[PoseKeypoint] | None = None,
) -> PersonPoseDetection:
    x1, y1, x2, y2 = bbox
    return PersonPoseDetection(
        bbox_xyxy=bbox,
        bbox_area=(x2 - x1) * (y2 - y1),
        confidence=confidence,
        keypoints=keypoints or [],
    )


def wave_keypoints(wrist_x: float) -> list[PoseKeypoint]:
    return [
        PoseKeypoint(name="left_shoulder", x=640.0, y=260.0, confidence=0.9),
        PoseKeypoint(name="left_wrist", x=wrist_x, y=240.0, confidence=0.9),
    ]


class SequenceBackend:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    async def infer(self, frame):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def visual_state_from(
    websocket,
    *,
    frame_id: int,
    timestamp_ms: int,
    head_motion: str = "stationary",
) -> dict:
    websocket.send_bytes(
        encode_frame_message(
            frame_header(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                head_motion={"state": head_motion},
            ),
            JPEG_BYTES,
        )
    )
    return json.loads(websocket.receive_text())


def test_websocket_sessions_have_isolated_event_ids():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person()]),
                PoseDetections(persons=[person((464.0, 100.0, 824.0, 620.0))]),
                PoseDetections(persons=[person((468.0, 100.0, 828.0, 620.0))]),
                PoseDetections(persons=[person((100.0, 100.0, 460.0, 620.0))]),
                PoseDetections(persons=[person((104.0, 100.0, 464.0, 620.0))]),
                PoseDetections(persons=[person((108.0, 100.0, 468.0, 620.0))]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as first_socket:
        visual_state_from(first_socket, frame_id=1, timestamp_ms=1000)
        visual_state_from(first_socket, frame_id=2, timestamp_ms=1200)
        first = visual_state_from(first_socket, frame_id=3, timestamp_ms=1400)

    with client.websocket_connect("/v1/stream") as second_socket:
        visual_state_from(second_socket, frame_id=1, timestamp_ms=5000)
        visual_state_from(second_socket, frame_id=2, timestamp_ms=5200)
        second = visual_state_from(second_socket, frame_id=3, timestamp_ms=5400)

    assert first["semantic_events"][0]["event_id"] == "front:evt_000001"
    assert second["semantic_events"][0]["event_id"] == "front:evt_000001"


def test_backend_error_does_not_advance_event_engine_or_consume_event_id():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person()]),
                PoseDetections(persons=[person((464.0, 100.0, 824.0, 620.0))]),
                RuntimeError("backend exploded"),
                PoseDetections(persons=[person((468.0, 100.0, 828.0, 620.0))]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        first = visual_state_from(websocket, frame_id=1, timestamp_ms=1000)
        second = visual_state_from(websocket, frame_id=2, timestamp_ms=1200)
        websocket.send_bytes(
            encode_frame_message(
                frame_header(frame_id=3, timestamp_ms=1300),
                JPEG_BYTES,
            )
        )
        error = json.loads(websocket.receive_text())
        recovered = visual_state_from(websocket, frame_id=4, timestamp_ms=1400)

    assert first["semantic_events"] == []
    assert second["semantic_events"] == []
    assert error["type"] == "error"
    assert error["code"] == "backend_unavailable"
    assert recovered["semantic_events"][0]["event_id"] == "front:evt_000001"
    assert recovered["semantic_events"][0]["event"] == "person_appeared"


@pytest.mark.parametrize(
    ("regression_header", "recovered_frames"),
    [
        pytest.param(
            {"frame_id": 4, "timestamp_ms": 1300},
            [(5, 1500), (6, 1700), (7, 1900)],
            id="timestamp",
        ),
        pytest.param(
            {"frame_id": 2, "timestamp_ms": 1600},
            [(3, 1800), (4, 2000), (5, 2200)],
            id="frame-id-only",
        ),
    ],
)
def test_regression_error_resets_connection_state_before_backend_inference(
    regression_header,
    recovered_frames,
):
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person()]),
                PoseDetections(persons=[person((464.0, 100.0, 824.0, 620.0))]),
                PoseDetections(persons=[person((468.0, 100.0, 828.0, 620.0))]),
                RuntimeError("backend exploded"),
                PoseDetections(persons=[person((472.0, 100.0, 832.0, 620.0))]),
                PoseDetections(persons=[person((476.0, 100.0, 836.0, 620.0))]),
                PoseDetections(persons=[person((480.0, 100.0, 840.0, 620.0))]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        visual_state_from(websocket, frame_id=1, timestamp_ms=1000)
        visual_state_from(websocket, frame_id=2, timestamp_ms=1200)
        stable = visual_state_from(websocket, frame_id=3, timestamp_ms=1400)

        websocket.send_bytes(
            encode_frame_message(
                frame_header(**regression_header),
                JPEG_BYTES,
            )
        )
        error = json.loads(websocket.receive_text())

        recovered = visual_state_from(
            websocket,
            frame_id=recovered_frames[0][0],
            timestamp_ms=recovered_frames[0][1],
        )
        visual_state_from(
            websocket,
            frame_id=recovered_frames[1][0],
            timestamp_ms=recovered_frames[1][1],
        )
        restabilized = visual_state_from(
            websocket,
            frame_id=recovered_frames[2][0],
            timestamp_ms=recovered_frames[2][1],
        )

    assert stable["semantic_events"][0]["event_id"] == "front:evt_000001"
    assert stable["semantic_events"][0]["event"] == "person_appeared"
    assert error["type"] == "error"
    assert error["code"] == "backend_unavailable"
    assert error["frame_id"] == regression_header["frame_id"]

    assert recovered["tracks"][0]["track_id"] == 1
    assert recovered["tracks"][0]["age_ms"] == 0
    assert recovered["tracks"][0]["lost_ms"] == 0
    assert recovered["attention"] is None
    assert recovered["scene_flags"]["largest_person_stable"] is False
    assert recovered["semantic_events"] == []

    assert restabilized["tracks"][0]["age_ms"] == 400
    assert restabilized["attention"]["target_track_id"] == 1
    assert restabilized["semantic_events"][0]["event_id"] == "front:evt_000002"
    assert restabilized["semantic_events"][0]["event"] == "person_appeared"


def test_semantic_events_have_protocol_fields_and_empty_frames_use_empty_list():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person()]),
                PoseDetections(persons=[person((464.0, 100.0, 824.0, 620.0))]),
                PoseDetections(persons=[person((468.0, 100.0, 828.0, 620.0))]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        first = visual_state_from(websocket, frame_id=1, timestamp_ms=1000)
        visual_state_from(websocket, frame_id=2, timestamp_ms=1200)
        third = visual_state_from(websocket, frame_id=3, timestamp_ms=1400)

    assert first["semantic_events"] == []
    assert set(third["semantic_events"][0]) == {
        "type",
        "event_id",
        "event",
        "camera",
        "track_id",
        "confidence",
        "duration_ms",
        "text",
    }


def test_head_motion_gates_motion_events_but_not_non_motion_events():
    stationary_processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person()]),
                PoseDetections(persons=[person((462.0, 100.0, 822.0, 620.0))]),
                PoseDetections(persons=[person((464.0, 100.0, 824.0, 620.0))]),
            ]
        )
    )
    moving_processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                PoseDetections(persons=[person(keypoints=wave_keypoints(590.0))]),
                PoseDetections(
                    persons=[
                        person(
                            (462.0, 100.0, 822.0, 620.0),
                            keypoints=wave_keypoints(700.0),
                        )
                    ]
                ),
                PoseDetections(
                    persons=[
                        person(
                            (464.0, 100.0, 824.0, 620.0),
                            keypoints=wave_keypoints(590.0),
                        )
                    ]
                ),
            ]
        )
    )

    stationary_client = TestClient(create_app(processor=stationary_processor))
    with stationary_client.websocket_connect("/v1/stream") as websocket:
        visual_state_from(websocket, frame_id=1, timestamp_ms=1000)
        visual_state_from(websocket, frame_id=2, timestamp_ms=1800)
        stationary = visual_state_from(websocket, frame_id=3, timestamp_ms=2500)

    moving_client = TestClient(create_app(processor=moving_processor))
    with moving_client.websocket_connect("/v1/stream") as websocket:
        visual_state_from(
            websocket,
            frame_id=1,
            timestamp_ms=1000,
            head_motion="moving",
        )
        visual_state_from(
            websocket,
            frame_id=2,
            timestamp_ms=1600,
            head_motion="unknown",
        )
        moving = visual_state_from(
            websocket,
            frame_id=3,
            timestamp_ms=2200,
            head_motion="moving",
        )

    stationary_events = {event["event"] for event in stationary["semantic_events"]}
    moving_events = {event["event"] for event in moving["semantic_events"]}
    assert "person_stopped_near_robot" in stationary_events
    assert "person_stopped_near_robot" not in moving_events
    assert "person_waving" in moving_events
