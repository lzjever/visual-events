import json

from fastapi.testclient import TestClient

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_server.app import create_app
from visual_events_server.inference.base import (
    PersonPoseDetection,
    PoseDetections,
)
from visual_events_server.metrics import JsonlMetricsSink
from visual_events_server.processor import BackendVisualFrameProcessor
from visual_events_server.protocol import encode_frame_message


JPEG_BYTES = JPEG_1280X720


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


class SequenceBackend:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    async def infer(self, frame):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def person(confidence=0.88):
    return PersonPoseDetection(
        bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
        bbox_area=20000.0,
        confidence=confidence,
        keypoints=[],
    )


def test_fake_backend_detections_drive_visual_state_scene_flags():
    processor = BackendVisualFrameProcessor(
        SequenceBackend([PoseDetections(persons=[person()])])
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
        message = json.loads(websocket.receive_text())

    assert message["type"] == "visual_state"
    assert message["frame_id"] == 7
    assert len(message["tracks"]) == 1
    assert message["tracks"][0]["track_id"] == 1
    assert message["tracks"][0]["lost_ms"] == 0
    assert message["attention"] is None
    assert message["scene_flags"] == {
        "has_person": True,
        "person_count": 1,
        "largest_person_stable": False,
        "someone_near_center": False,
    }
    assert message["semantic_events"] == []


def test_metrics_jsonl_enabled_writes_processed_frame_without_wire_change(tmp_path):
    metrics_path = tmp_path / "metrics" / "frames.jsonl"
    processor = BackendVisualFrameProcessor(
        SequenceBackend([PoseDetections(persons=[person()])]),
        metrics_sink=JsonlMetricsSink(
            metrics_path,
            resource_sampler=lambda: {
                "rss": {"available": False, "reason": "rss_unavailable"},
                "vram": {"available": False, "reason": "torch_unavailable"},
            },
        ),
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
        message = json.loads(websocket.receive_text())

    assert message["type"] == "visual_state"
    assert "phase_latencies_ms" not in message
    assert "resources" not in message
    metrics_lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(metrics_lines) == 1
    metrics = json.loads(metrics_lines[0])
    assert metrics["type"] == "frame_metrics"
    assert metrics["camera"] == "front"
    assert metrics["frame_id"] == 7
    assert metrics["frame_timestamp_ms"] == 1710000000000
    assert set(metrics["phase_latencies_ms"]) == {
        "infer",
        "tracking",
        "attention",
        "events",
        "response",
        "total",
    }
    assert metrics["resources"] == {
        "rss": {"available": False, "reason": "rss_unavailable"},
        "vram": {"available": False, "reason": "torch_unavailable"},
    }


def test_backend_error_is_retryable_and_next_frame_can_continue_on_same_connection():
    processor = BackendVisualFrameProcessor(
        SequenceBackend(
            [
                RuntimeError("backend exploded"),
                PoseDetections(persons=[person(), person(confidence=0.74)]),
            ]
        )
    )
    client = TestClient(create_app(processor=processor))

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(frame_id=1), JPEG_BYTES))
        error = json.loads(websocket.receive_text())

        websocket.send_bytes(encode_frame_message(frame_header(frame_id=2), JPEG_BYTES))
        ok = json.loads(websocket.receive_text())

    assert error["type"] == "error"
    assert error["frame_id"] == 1
    assert error["code"] == "backend_unavailable"
    assert error["retryable"] is True
    assert ok["type"] == "visual_state"
    assert ok["frame_id"] == 2
    assert ok["scene_flags"]["has_person"] is True
    assert ok["scene_flags"]["person_count"] == 2
