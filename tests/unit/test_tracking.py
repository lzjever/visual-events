from __future__ import annotations

import pytest

from visual_events_server.inference.base import (
    PersonPoseDetection,
    PoseDetections,
    PoseKeypoint,
)
from visual_events_server.protocol import FrameMessage
from visual_events_server.tracking import ByteTrackStyleTracker, TrackingConfig


def frame(*, frame_id: int = 1, timestamp_ms: int = 1000) -> FrameMessage:
    return FrameMessage(
        camera="front",
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        width=100,
        height=100,
        jpeg_bytes=b"",
    )


def detections(*persons: PersonPoseDetection) -> PoseDetections:
    return PoseDetections(persons=list(persons))


def person(
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.9,
    keypoints: list[PoseKeypoint] | None = None,
) -> PersonPoseDetection:
    x1, y1, x2, y2 = bbox
    return PersonPoseDetection(
        bbox_xyxy=bbox,
        bbox_area=max(0.0, x2 - x1) * max(0.0, y2 - y1),
        confidence=confidence,
        keypoints=keypoints or [],
    )


def tracker(**overrides: float | int) -> ByteTrackStyleTracker:
    config = TrackingConfig(
        high_conf=0.6,
        low_conf=0.1,
        new_track_conf=0.7,
        match_iou=0.3,
        lost_ttl_ms=1000,
        history_ms=3000,
        velocity_window_ms=1000,
    )
    return ByteTrackStyleTracker(
        config=TrackingConfig(
            **{**config.__dict__, **overrides},
        )
    )


def protocol_track(track, source_frame: FrameMessage) -> dict:
    return track.to_protocol(image_width=source_frame.width, image_height=source_frame.height)


def test_single_person_keeps_id_reports_age_and_velocity():
    subject = tracker()

    first = subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        detections(person((10.0, 10.0, 30.0, 50.0))),
    )
    second_frame = frame(frame_id=2, timestamp_ms=1200)
    second = subject.update(
        second_frame,
        detections(person((20.0, 10.0, 40.0, 50.0))),
    )

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].track_id == first[0].track_id
    payload = protocol_track(second[0], second_frame)
    assert payload["age_ms"] == 200
    assert payload["lost_ms"] == 0
    assert payload["center_uv"] == [30.0, 30.0]
    assert payload["head_uv"] == [30.0, pytest.approx(21.2)]
    assert payload["velocity_uv_s"] == [pytest.approx(50.0), pytest.approx(0.0)]


def test_short_missed_detection_outputs_lost_track_then_recovers_same_id():
    subject = tracker(lost_ttl_ms=1000)
    first = subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        detections(person((10.0, 10.0, 30.0, 50.0))),
    )
    track_id = first[0].track_id

    lost = subject.update(frame(frame_id=2, timestamp_ms=1200), detections())
    recovered = subject.update(
        frame(frame_id=3, timestamp_ms=1300),
        detections(person((11.0, 10.0, 31.0, 50.0))),
    )

    assert [track.track_id for track in lost] == [track_id]
    assert lost[0].lost_ms == 200
    assert recovered[0].track_id == track_id
    assert recovered[0].lost_ms == 0


def test_detection_after_lost_ttl_gets_new_track_id():
    subject = tracker(lost_ttl_ms=1000)
    first = subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        detections(person((10.0, 10.0, 30.0, 50.0))),
    )

    subject.update(frame(frame_id=2, timestamp_ms=2101), detections())
    second = subject.update(
        frame(frame_id=3, timestamp_ms=2200),
        detections(person((10.0, 10.0, 30.0, 50.0))),
    )

    assert len(second) == 1
    assert second[0].track_id != first[0].track_id
    assert second[0].age_ms == 0


def test_two_people_keep_ids_when_detection_order_swaps():
    subject = tracker(match_iou=0.2)
    initial = subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        detections(
            person((10.0, 10.0, 30.0, 50.0)),
            person((60.0, 10.0, 80.0, 50.0)),
        ),
    )
    left_id = initial[0].track_id
    right_id = initial[1].track_id

    swapped = subject.update(
        frame(frame_id=2, timestamp_ms=1200),
        detections(
            person((61.0, 10.0, 81.0, 50.0)),
            person((11.0, 10.0, 31.0, 50.0)),
        ),
    )
    ids_by_center_x = {
        protocol_track(track, frame(timestamp_ms=1200))["center_uv"][0]: track.track_id
        for track in swapped
    }

    assert ids_by_center_x[21.0] == left_id
    assert ids_by_center_x[71.0] == right_id


def test_protocol_schema_fields_and_keypoint_head_priority_are_valid():
    subject = tracker()
    keypoints = [
        PoseKeypoint(name="nose", x=24.0, y=12.0, confidence=0.8),
        PoseKeypoint(name="left_eye", x=22.0, y=14.0, confidence=0.6),
        PoseKeypoint(name="right_eye", x=26.0, y=14.0, confidence=0.7),
        PoseKeypoint(name="left_ear", x=0.0, y=0.0, confidence=0.0),
    ]
    source_frame = frame(frame_id=1, timestamp_ms=1000)

    tracks = subject.update(
        source_frame,
        detections(person((10.0, 10.0, 30.0, 50.0), keypoints=keypoints)),
    )

    payload = protocol_track(tracks[0], source_frame)
    assert set(payload) == {
        "track_id",
        "class",
        "bbox_xyxy",
        "bbox_area_ratio",
        "center_uv",
        "head_uv",
        "velocity_uv_s",
        "age_ms",
        "lost_ms",
        "confidence",
        "pose_confidence",
    }
    assert payload["track_id"] == 1
    assert payload["class"] == "person"
    assert payload["bbox_xyxy"] == [10.0, 10.0, 30.0, 50.0]
    assert payload["bbox_area_ratio"] == pytest.approx(0.08)
    assert payload["center_uv"] == [20.0, 30.0]
    assert payload["head_uv"] == [24.0, pytest.approx(13.3333333333)]
    assert payload["velocity_uv_s"] == [0.0, 0.0]
    assert payload["age_ms"] == 0
    assert payload["lost_ms"] == 0
    assert payload["confidence"] == 0.9
    assert payload["pose_confidence"] == pytest.approx(0.525)


def test_low_confidence_detection_does_not_create_track_but_updates_existing_track():
    subject = tracker(low_conf=0.1, new_track_conf=0.7)

    assert subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        detections(person((10.0, 10.0, 30.0, 50.0), confidence=0.2)),
    ) == []
    created = subject.update(
        frame(frame_id=2, timestamp_ms=1100),
        detections(person((10.0, 10.0, 30.0, 50.0), confidence=0.9)),
    )
    updated = subject.update(
        frame(frame_id=3, timestamp_ms=1300),
        detections(person((12.0, 10.0, 32.0, 50.0), confidence=0.2)),
    )

    assert updated[0].track_id == created[0].track_id
    assert updated[0].lost_ms == 0
    assert updated[0].confidence == 0.2


def test_timestamp_or_frame_id_regression_resets_without_negative_values():
    subject = tracker()
    subject.update(
        frame(frame_id=10, timestamp_ms=1000),
        detections(person((10.0, 10.0, 30.0, 50.0))),
    )

    tracks = subject.update(
        frame(frame_id=9, timestamp_ms=900),
        detections(person((20.0, 10.0, 40.0, 50.0))),
    )
    payload = protocol_track(tracks[0], frame(frame_id=9, timestamp_ms=900))

    assert payload["age_ms"] == 0
    assert payload["lost_ms"] == 0
    assert payload["velocity_uv_s"] == [0.0, 0.0]
