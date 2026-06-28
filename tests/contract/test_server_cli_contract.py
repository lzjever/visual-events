from __future__ import annotations

import json
import math

from fastapi.testclient import TestClient

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_server.app import create_app
from visual_events_server.inference.base import (
    PersonPoseDetection,
    PoseDetections,
    PoseKeypoint,
)
from visual_events_server.processor import BackendVisualFrameProcessor
from visual_events_server.protocol import encode_frame_message


JPEG_BYTES = JPEG_1280X720
EVENT_FIELDS = {
    "type",
    "event_id",
    "event",
    "camera",
    "track_id",
    "confidence",
    "duration_ms",
    "lifecycle_state",
    "evidence",
    "text",
}
BOTIFIED_PERSON_EVENTS = {
    "person_appeared",
    "person_left",
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
    "person_waving",
}
MOTION_SENSITIVE_EVENTS = {
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
}
REACQUIRE_KEYS = {
    "runtime_person_slot",
    "reacquired_from_track_id",
    "reacquired_to_track_id",
    "reacquire_elapsed_ms",
    "reacquire_center_distance_px",
    "reacquire_area_ratio",
}
STABLE_BBOXES = [
    (100.0, 100.0, 220.0, 460.0),
    (104.0, 100.0, 224.0, 460.0),
    (108.0, 100.0, 228.0, 460.0),
]
NEAR_STATIONARY_BBOXES = [
    (460.0, 100.0, 820.0, 620.0),
    (462.0, 100.0, 822.0, 620.0),
    (464.0, 100.0, 824.0, 620.0),
]
STOP_TIMESTAMPS_MS = [1000, 1800, 2500]


class SequenceBackend:
    def __init__(self, outcomes: list[PoseDetections]) -> None:
        self.outcomes = list(outcomes)

    async def infer(self, frame) -> PoseDetections:
        return self.outcomes.pop(0)


def frame_header(
    *,
    frame_id: int,
    timestamp_ms: int,
    head_motion: str | None = "stationary",
) -> dict[str, object]:
    header: dict[str, object] = {
        "type": "frame",
        "schema_version": 1,
        "camera": "front",
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "encoding": "jpeg",
        "width": 1280,
        "height": 720,
    }
    if head_motion is not None:
        header["head_motion"] = {"state": head_motion}
    return header


def person(
    bbox: tuple[float, float, float, float],
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


def detections_for(
    bboxes: list[tuple[float, float, float, float]],
) -> list[PoseDetections]:
    return [PoseDetections(persons=[person(bbox)]) for bbox in bboxes]


def visual_state_from(
    websocket,
    *,
    frame_id: int,
    timestamp_ms: int,
    head_motion: str | None = "stationary",
) -> dict:
    websocket.send_bytes(
        encode_frame_message(
            frame_header(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                head_motion=head_motion,
            ),
            JPEG_BYTES,
        )
    )
    return json.loads(websocket.receive_text())


def run_sequence(
    bboxes: list[tuple[float, float, float, float]],
    *,
    timestamps_ms: list[int],
    head_motion: str | None = "stationary",
) -> list[dict]:
    processor = BackendVisualFrameProcessor(SequenceBackend(detections_for(bboxes)))
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        return [
            visual_state_from(
                websocket,
                frame_id=index,
                timestamp_ms=timestamp_ms,
                head_motion=head_motion,
            )
            for index, timestamp_ms in enumerate(timestamps_ms, start=1)
        ]


def event_names(states: list[dict]) -> set[str]:
    return {
        event["event"]
        for state in states
        for event in state["semantic_events"]
    }


def assert_json_simple_and_finite(value: object) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        assert math.isfinite(value)
        return
    if isinstance(value, float):
        assert math.isfinite(value)
        return
    if isinstance(value, list):
        for item in value:
            assert_json_simple_and_finite(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            assert_json_simple_and_finite(item)
        return
    raise AssertionError(f"unsupported evidence value: {value!r}")


def test_visual_state_attention_contract_for_cli_gaze_mapper():
    states = run_sequence(
        STABLE_BBOXES,
        timestamps_ms=[1000, 1200, 1400],
        head_motion="stationary",
    )
    visual_state = states[-1]
    attention = visual_state["attention"]

    assert attention is not None
    assert set(attention) == {
        "target_track_id",
        "target_uv",
        "reason",
        "confidence",
    }
    assert attention["target_track_id"] in {
        track["track_id"] for track in visual_state["tracks"]
    }

    width, height = visual_state["image_size"]
    target_u, target_v = attention["target_uv"]
    assert math.isfinite(target_u)
    assert math.isfinite(target_v)
    assert 0.0 <= target_u <= width
    assert 0.0 <= target_v <= height
    assert attention["reason"] == "largest_stable_person"
    assert "scene_context" in visual_state
    assert "phase_latencies_ms" not in visual_state
    assert "resources" not in visual_state


def test_visual_state_scene_context_contract_for_available_target():
    states = run_sequence(
        NEAR_STATIONARY_BBOXES,
        timestamps_ms=STOP_TIMESTAMPS_MS,
        head_motion="stationary",
    )
    visual_state = states[-1]

    assert visual_state["scene_context"] == {
        "engagement_state": "available",
        "attention_available": True,
        "target_track_id": visual_state["attention"]["target_track_id"],
        "no_engage_reasons": [],
        "target_reacquired": None,
    }


def test_visual_state_scene_context_reacquired_contract_for_cli():
    states = run_sequence(
        [
            (460.0, 100.0, 820.0, 620.0),
            (560.0, 240.0, 720.0, 480.0),
        ],
        timestamps_ms=[1000, 2000],
        head_motion="stationary",
    )
    visual_state = states[-1]
    target_reacquired = visual_state["scene_context"]["target_reacquired"]

    assert isinstance(target_reacquired, dict)
    visible_track = next(track for track in visual_state["tracks"] if track["lost_ms"] == 0)
    assert set(target_reacquired) == REACQUIRE_KEYS
    assert target_reacquired["runtime_person_slot"] == 1
    assert target_reacquired["reacquired_from_track_id"] == 1
    assert target_reacquired["reacquired_to_track_id"] == visible_track["track_id"]
    assert target_reacquired["reacquired_to_track_id"] != target_reacquired[
        "reacquired_from_track_id"
    ]
    assert target_reacquired["reacquire_elapsed_ms"] == 1000
    assert_json_simple_and_finite(target_reacquired)


def test_semantic_event_contract_for_cli_botified_mapper():
    states = run_sequence(
        STABLE_BBOXES,
        timestamps_ms=[1000, 1200, 1400],
        head_motion="stationary",
    )
    event = states[-1]["semantic_events"][0]

    assert set(event) == EVENT_FIELDS
    assert event["type"] == "semantic_event"
    assert event["event_id"] == "front:evt_000001"
    assert event["camera"] == "front"
    assert event["event"] in BOTIFIED_PERSON_EVENTS
    assert event["event"] == "person_appeared"
    assert event["lifecycle_state"] == "confirmed"
    assert isinstance(event["evidence"], dict)
    assert {
        "runtime_person_slot",
        "visible_duration_ms",
        "bbox_area_ratio",
        "salient_reason",
    } <= set(event["evidence"])
    assert_json_simple_and_finite(event["evidence"])
    assert event["text"]


def test_missing_or_unknown_head_motion_suppresses_motion_sensitive_events():
    missing_head_states = run_sequence(
        NEAR_STATIONARY_BBOXES,
        timestamps_ms=STOP_TIMESTAMPS_MS,
        head_motion=None,
    )
    unknown_head_states = run_sequence(
        NEAR_STATIONARY_BBOXES,
        timestamps_ms=STOP_TIMESTAMPS_MS,
        head_motion="unknown",
    )
    stationary_states = run_sequence(
        NEAR_STATIONARY_BBOXES,
        timestamps_ms=STOP_TIMESTAMPS_MS,
        head_motion="stationary",
    )

    for states in (missing_head_states, unknown_head_states):
        names = event_names(states)
        assert "person_appeared" in names
        assert names.isdisjoint(MOTION_SENSITIVE_EVENTS)

    assert "person_stopped_near_robot" in event_names(stationary_states)
