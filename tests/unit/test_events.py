from __future__ import annotations

import math

import pytest

from visual_events_server.attention import AttentionResult
from visual_events_server.events import EventConfig, EventEngine, EventEngineResult
from visual_events_server.inference.base import PoseKeypoint
from visual_events_server.protocol import FrameMessage
from visual_events_server.tracking import TrackSnapshot


def frame(
    *,
    frame_id: int = 1,
    timestamp_ms: int = 1000,
    head_motion_state: str = "stationary",
    width: int = 1000,
    height: int = 1000,
) -> FrameMessage:
    return FrameMessage(
        camera="front",
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        jpeg_bytes=b"",
        head_motion_state=head_motion_state,
    )


def track(
    track_id: int = 1,
    *,
    timestamp_ms: int = 1000,
    first_seen_ms: int | None = None,
    last_seen_ms: int | None = None,
    bbox_xyxy: tuple[float, float, float, float] = (400.0, 100.0, 600.0, 500.0),
    confidence: float = 0.9,
    velocity_uv_s: tuple[float, float] = (0.0, 0.0),
    lost_ms: int = 0,
    hits: int = 2,
    keypoints: tuple[PoseKeypoint, ...] = (),
) -> TrackSnapshot:
    first_seen = timestamp_ms - 400 if first_seen_ms is None else first_seen_ms
    last_seen = timestamp_ms - lost_ms if last_seen_ms is None else last_seen_ms
    x1, y1, x2, _y2 = bbox_xyxy
    return TrackSnapshot(
        track_id=track_id,
        first_seen_ms=first_seen,
        last_seen_ms=last_seen,
        frame_timestamp_ms=timestamp_ms,
        bbox_xyxy=bbox_xyxy,
        confidence=confidence,
        pose_confidence=0.8 if keypoints else 0.0,
        head_uv=((x1 + x2) / 2.0, y1 + 40.0),
        velocity_uv_s=velocity_uv_s,
        lost_ms=lost_ms,
        hits=hits,
        misses=1 if lost_ms > 0 else 0,
        keypoints=keypoints,
    )


def attention(track_id: int) -> AttentionResult:
    return AttentionResult(
        target_track_id=track_id,
        target_uv=(500.0, 200.0),
        reason="largest_stable_person",
        confidence=0.9,
        largest_person_stable=True,
    )


def semantic_events(result: EventEngineResult | list[dict]) -> list[dict]:
    if isinstance(result, EventEngineResult):
        return result.semantic_events
    return result


def event_names(result: EventEngineResult | list[dict]) -> list[str]:
    events = semantic_events(result)
    return [str(event["event"]) for event in events]


def events_of(result: EventEngineResult | list[dict], name: str) -> list[dict]:
    events = semantic_events(result)
    return [event for event in events if event["event"] == name]


REQUIRED_EVIDENCE_KEYS = {
    "person_appeared": {
        "runtime_person_slot",
        "visible_duration_ms",
        "bbox_area_ratio",
        "salient_reason",
    },
    "person_left": {
        "runtime_person_slot",
        "lost_duration_ms",
        "last_bbox_area_ratio",
    },
    "person_passing_by": {
        "runtime_person_slot",
        "dx_ratio",
        "avg_vx_px_s",
        "crossed_side_bands",
        "camera_motion_state",
        "passing_speed_class",
    },
    "person_approaching_robot": {
        "runtime_person_slot",
        "bbox_area_ratio_start",
        "bbox_area_ratio_end",
        "area_growth_ratio",
        "area_delta",
        "camera_motion_state",
    },
    "person_stopped_near_robot": {
        "runtime_person_slot",
        "bbox_area_ratio",
        "speed_px_s_p95",
        "stationary_duration_ms",
        "camera_motion_state",
    },
    "person_waving": {
        "runtime_person_slot",
        "wrist_x_span_px",
        "wrist_x_span_bbox_ratio",
        "wrist_y_relative_to_shoulder_px",
        "wave_duration_ms",
        "keypoint_min_confidence",
    },
    "attention_target_changed": {
        "previous_track_id",
        "target_track_id",
        "switch_reason",
    },
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


def keypoints(
    *,
    wrist_x: float,
    wrist_y: float = 300.0,
    shoulder_y: float = 320.0,
    wrist_confidence: float = 0.9,
    side: str = "left",
) -> tuple[PoseKeypoint, ...]:
    return (
        PoseKeypoint(
            name=f"{side}_shoulder",
            x=500.0,
            y=shoulder_y,
            confidence=0.9,
        ),
        PoseKeypoint(
            name=f"{side}_wrist",
            x=wrist_x,
            y=wrist_y,
            confidence=wrist_confidence,
        ),
    )


def make_wave_history(
    subject: EventEngine,
    *,
    track_id: int = 1,
    start_ms: int = 1000,
    start_frame_id: int = 1,
    head_motion_state: str = "stationary",
    wrist_confidence: float = 0.9,
) -> list[dict]:
    emitted: list[dict] = []
    for index, wrist_x in enumerate((460.0, 545.0, 465.0)):
        timestamp_ms = start_ms + (index * 600)
        emitted = subject.update(
            frame(
                frame_id=start_frame_id + index,
                timestamp_ms=timestamp_ms,
                head_motion_state=head_motion_state,
            ),
            [
                track(
                    track_id,
                    timestamp_ms=timestamp_ms,
                    bbox_xyxy=(400.0, 150.0, 600.0, 450.0),
                    keypoints=keypoints(
                        wrist_x=wrist_x,
                        wrist_confidence=wrist_confidence,
                    ),
                )
            ],
            attention(track_id),
        ).semantic_events
    return emitted


def test_update_returns_event_engine_result_with_scene_context_no_target():
    result = EventEngine().update(frame(), [], None)

    assert isinstance(result, EventEngineResult)
    assert result.semantic_events == []
    assert result.scene_context == {
        "engagement_state": "no_target",
        "attention_available": False,
        "target_track_id": None,
        "no_engage_reasons": ["no_visible_person"],
        "target_reacquired": None,
    }


def test_scene_context_available_for_stable_near_stationary_attention_target():
    result = EventEngine().update(
        frame(),
        [track(7, bbox_xyxy=(350.0, 150.0, 650.0, 650.0))],
        attention(7),
    )

    assert result.scene_context == {
        "engagement_state": "available",
        "attention_available": True,
        "target_track_id": 7,
        "no_engage_reasons": [],
        "target_reacquired": None,
    }


def test_scene_context_marks_visible_unstable_person_as_no_engage_target():
    result = EventEngine().update(
        frame(),
        [track(1, hits=1)],
        None,
    )

    assert result.scene_context["engagement_state"] == "no_engage_target"
    assert result.scene_context["attention_available"] is False
    assert result.scene_context["target_track_id"] is None
    assert "unstable" in result.scene_context["no_engage_reasons"]


def test_scene_context_marks_stable_attention_target_too_far():
    result = EventEngine().update(
        frame(),
        [track(1, bbox_xyxy=(450.0, 100.0, 550.0, 300.0))],
        attention(1),
    )

    assert result.scene_context["engagement_state"] == "no_engage_target"
    assert result.scene_context["attention_available"] is False
    assert result.scene_context["target_track_id"] == 1
    assert "too_far" in result.scene_context["no_engage_reasons"]


def test_scene_context_marks_camera_motion_not_stationary():
    result = EventEngine().update(
        frame(head_motion_state="moving"),
        [track(1, bbox_xyxy=(350.0, 150.0, 650.0, 650.0))],
        attention(1),
    )

    assert result.scene_context["engagement_state"] == "no_engage_target"
    assert result.scene_context["attention_available"] is False
    assert result.scene_context["target_track_id"] == 1
    assert "camera_motion_not_stationary" in result.scene_context["no_engage_reasons"]


def test_scene_context_marks_confirmed_fast_passing_frame():
    subject = EventEngine()
    subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [
            track(
                1,
                timestamp_ms=1000,
                bbox_xyxy=(20.0, 200.0, 120.0, 520.0),
            )
        ],
        attention(1),
    )

    result = subject.update(
        frame(frame_id=2, timestamp_ms=2000),
        [
            track(
                1,
                timestamp_ms=2000,
                bbox_xyxy=(820.0, 200.0, 920.0, 520.0),
                velocity_uv_s=(800.0, 0.0),
            )
        ],
        attention(1),
    )

    assert "person_passing_by" in event_names(result)
    assert result.scene_context["engagement_state"] == "no_engage_target"
    assert "passing_fast" in result.scene_context["no_engage_reasons"]


def test_person_appeared_uses_salient_stable_target_and_rising_edge():
    subject = EventEngine()

    unstable = subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [
            track(1, timestamp_ms=1000, first_seen_ms=900, hits=1),
            track(
                2,
                timestamp_ms=1000,
                first_seen_ms=600,
                bbox_xyxy=(50.0, 100.0, 950.0, 900.0),
            ),
        ],
        attention(1),
    )
    appeared = subject.update(
        frame(frame_id=2, timestamp_ms=1300),
        [
            track(1, timestamp_ms=1300, first_seen_ms=900, hits=2),
            track(
                2,
                timestamp_ms=1300,
                first_seen_ms=600,
                bbox_xyxy=(50.0, 100.0, 950.0, 900.0),
            ),
        ],
        attention(1),
    )
    repeated = subject.update(
        frame(frame_id=3, timestamp_ms=1600),
        [track(1, timestamp_ms=1600, first_seen_ms=900, hits=3)],
        attention(1),
    )

    assert unstable.semantic_events == []
    assert event_names(appeared) == ["person_appeared"]
    assert appeared.semantic_events[0]["track_id"] == 1
    assert repeated.semantic_events == []


def test_person_appeared_falls_back_to_best_visible_stable_track_without_attention():
    subject = EventEngine()

    events = subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [
            track(1, timestamp_ms=1000, bbox_xyxy=(10.0, 10.0, 110.0, 210.0)),
            track(2, timestamp_ms=1000, bbox_xyxy=(300.0, 10.0, 700.0, 510.0)),
        ],
        None,
    )

    assert event_names(events) == ["person_appeared"]
    assert events.semantic_events[0]["track_id"] == 2


def test_person_left_triggers_once_after_appeared_track_expires_from_tracks():
    subject = EventEngine()
    subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [track(1, timestamp_ms=1000, first_seen_ms=600)],
        attention(1),
    )

    early = subject.update(frame(frame_id=2, timestamp_ms=2499), [], None)
    left = subject.update(frame(frame_id=3, timestamp_ms=2500), [], None)
    repeated = subject.update(frame(frame_id=4, timestamp_ms=3000), [], None)

    assert early.semantic_events == []
    assert event_names(left) == ["person_left"]
    assert left.semantic_events[0]["track_id"] == 1
    assert repeated.semantic_events == []


def test_person_passing_by_requires_stationary_lateral_history():
    subject = EventEngine()
    subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [
            track(
                1,
                timestamp_ms=1000,
                bbox_xyxy=(20.0, 200.0, 120.0, 520.0),
            )
        ],
        attention(1),
    )

    events = subject.update(
        frame(frame_id=2, timestamp_ms=2000),
        [
            track(
                1,
                timestamp_ms=2000,
                bbox_xyxy=(820.0, 200.0, 920.0, 520.0),
                velocity_uv_s=(800.0, 0.0),
            )
        ],
        attention(1),
    )

    assert "person_passing_by" in event_names(events)
    assert events_of(events, "person_passing_by")[0]["duration_ms"] == 1000


def test_person_passing_by_accepts_late_left_reference_crossing_side_bands():
    subject = EventEngine()
    subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [
            track(
                1,
                timestamp_ms=1000,
                bbox_xyxy=(268.0, 200.0, 428.0, 500.0),
            )
        ],
        attention(1),
    )

    events = subject.update(
        frame(frame_id=2, timestamp_ms=2000),
        [
            track(
                1,
                timestamp_ms=2000,
                bbox_xyxy=(692.0, 200.0, 852.0, 500.0),
                velocity_uv_s=(424.0, 0.0),
            )
        ],
        attention(1),
    )

    assert "person_passing_by" in event_names(events)


def test_person_passing_by_rejects_right_to_left_swept_bbox_before_center_threshold():
    subject = EventEngine()
    subject.update(
        frame(frame_id=1, timestamp_ms=1000, width=1280, height=720),
        [
            track(
                1,
                timestamp_ms=1000,
                bbox_xyxy=(843.0, 25.0, 947.0, 461.0),
            )
        ],
        attention(1),
    )

    early = subject.update(
        frame(frame_id=2, timestamp_ms=2200, width=1280, height=720),
        [
            track(
                1,
                timestamp_ms=2200,
                bbox_xyxy=(350.0, 25.0, 464.0, 455.0),
                velocity_uv_s=(-400.0, 0.0),
            )
        ],
        attention(1),
    )
    later = subject.update(
        frame(frame_id=3, timestamp_ms=2600, width=1280, height=720),
        [
            track(
                1,
                timestamp_ms=2600,
                bbox_xyxy=(230.0, 25.0, 370.0, 455.0),
                velocity_uv_s=(-400.0, 0.0),
            )
        ],
        attention(1),
    )

    assert "person_passing_by" not in event_names(early)
    assert "person_passing_by" in event_names(later)


def test_person_passing_by_rejects_short_or_near_stationary_motion():
    subject = EventEngine()
    subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [track(1, timestamp_ms=1000, bbox_xyxy=(100.0, 200.0, 200.0, 520.0))],
        attention(1),
    )

    short = subject.update(
        frame(frame_id=2, timestamp_ms=1500),
        [
            track(
                1,
                timestamp_ms=1500,
                bbox_xyxy=(820.0, 200.0, 920.0, 520.0),
                velocity_uv_s=(1440.0, 0.0),
            )
        ],
        attention(1),
    )

    assert "person_passing_by" not in event_names(short)

    near_stationary_subject = EventEngine()
    near_stationary_subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [track(1, timestamp_ms=1000, bbox_xyxy=(320.0, 200.0, 480.0, 500.0))],
        attention(1),
    )
    near_stationary = near_stationary_subject.update(
        frame(frame_id=2, timestamp_ms=2500),
        [
            track(
                1,
                timestamp_ms=2500,
                bbox_xyxy=(340.0, 200.0, 500.0, 500.0),
                velocity_uv_s=(13.0, 0.0),
            )
        ],
        attention(1),
    )

    assert "person_passing_by" not in event_names(near_stationary)


def test_person_approaching_robot_requires_area_growth_not_pure_sideways_motion():
    subject = EventEngine()
    subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [track(1, timestamp_ms=1000, bbox_xyxy=(450.0, 120.0, 550.0, 320.0))],
        attention(1),
    )

    events = subject.update(
        frame(frame_id=2, timestamp_ms=1600),
        [track(1, timestamp_ms=1600, bbox_xyxy=(425.0, 80.0, 575.0, 380.0))],
        attention(1),
    )

    sideways_subject = EventEngine()
    sideways_subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [track(1, timestamp_ms=1000, bbox_xyxy=(100.0, 120.0, 250.0, 420.0))],
        attention(1),
    )
    sideways = sideways_subject.update(
        frame(frame_id=2, timestamp_ms=1600),
        [track(1, timestamp_ms=1600, bbox_xyxy=(500.0, 120.0, 650.0, 420.0))],
        attention(1),
    )

    assert "person_approaching_robot" in event_names(events)
    assert "person_approaching_robot" not in event_names(sideways)


def test_person_approaching_robot_tolerates_small_bbox_area_jitter_before_stopped():
    subject = EventEngine()
    emitted: list[dict] = []
    boxes = (
        (405.0, 200.0, 595.0, 390.0),
        (385.0, 180.0, 615.0, 410.0),
        (390.0, 185.0, 610.0, 405.0),
        (370.0, 165.0, 630.0, 425.0),
        (375.0, 170.0, 625.0, 420.0),
        (355.0, 150.0, 645.0, 440.0),
        (355.0, 150.0, 645.0, 440.0),
        (355.0, 150.0, 645.0, 440.0),
        (355.0, 150.0, 645.0, 440.0),
    )
    for index, bbox_xyxy in enumerate(boxes):
        timestamp_ms = 1000 + (index * 100)
        if index >= 6:
            timestamp_ms += (index - 5) * 400
        emitted.extend(
            subject.update(
                frame(frame_id=index + 1, timestamp_ms=timestamp_ms),
                [
                    track(
                        1,
                        timestamp_ms=timestamp_ms,
                        bbox_xyxy=bbox_xyxy,
                        velocity_uv_s=(5.0, 0.0),
                    )
                ],
                attention(1),
            ).semantic_events
        )

    names = event_names(emitted)

    assert "person_approaching_robot" in names
    if "person_stopped_near_robot" in names:
        assert names.index("person_approaching_robot") < names.index(
            "person_stopped_near_robot"
        )


def test_person_approaching_robot_rejects_short_early_jitter_window():
    subject = EventEngine()
    samples = (
        (1000, (0.0, 214.8, 73.7, 456.5)),
        (1379, (0.0, 151.7, 105.9, 458.7)),
        (1421, (0.0, 152.2, 105.8, 457.6)),
        (1519, (0.0, 156.7, 99.0, 453.6)),
        (1594, (0.0, 144.6, 101.3, 475.1)),
        (1927, (0.0, 129.4, 132.4, 470.3)),
        (1949, (0.0, 125.8, 139.1, 472.5)),
        (1963, (0.0, 125.0, 129.6, 471.5)),
        (1997, (0.0, 125.7, 127.8, 471.3)),
        (2094, (0.0, 106.8, 153.7, 464.6)),
    )

    emitted_by_frame: list[tuple[int, list[str]]] = []
    for index, (timestamp_ms, bbox_xyxy) in enumerate(samples, start=1):
        events = subject.update(
            frame(
                frame_id=index,
                timestamp_ms=timestamp_ms,
                width=1280,
                height=720,
            ),
            [
                track(
                    1,
                    timestamp_ms=timestamp_ms,
                    first_seen_ms=1000,
                    bbox_xyxy=bbox_xyxy,
                    velocity_uv_s=(45.0, 0.0),
                )
            ],
            attention(1),
        )
        emitted_by_frame.append((index, event_names(events)))

    early_names = [
        name
        for frame_id, names in emitted_by_frame
        if frame_id <= 5
        for name in names
    ]
    all_names = [name for _, names in emitted_by_frame for name in names]

    assert "person_approaching_robot" not in early_names
    assert "person_approaching_robot" in all_names


def test_person_stopped_near_robot_requires_near_low_speed_duration_and_rising_edge():
    subject = EventEngine()
    for frame_id, timestamp_ms in enumerate((1000, 1800), start=1):
        assert "person_stopped_near_robot" not in event_names(
            subject.update(
                frame(frame_id=frame_id, timestamp_ms=timestamp_ms),
                [
                    track(
                        1,
                        timestamp_ms=timestamp_ms,
                        bbox_xyxy=(350.0, 160.0, 650.0, 620.0),
                        velocity_uv_s=(10.0, 0.0),
                    )
                ],
                attention(1),
            )
        )

    stopped = subject.update(
        frame(frame_id=3, timestamp_ms=2500),
        [
            track(
                1,
                timestamp_ms=2500,
                bbox_xyxy=(350.0, 160.0, 650.0, 620.0),
                velocity_uv_s=(10.0, 0.0),
            )
        ],
        attention(1),
    )
    repeated = subject.update(
        frame(frame_id=4, timestamp_ms=2600),
        [
            track(
                1,
                timestamp_ms=2600,
                bbox_xyxy=(350.0, 160.0, 650.0, 620.0),
                velocity_uv_s=(10.0, 0.0),
            )
        ],
        attention(1),
    )

    assert "person_stopped_near_robot" in event_names(stopped)
    assert "person_stopped_near_robot" not in event_names(repeated)


def test_person_waving_uses_keypoint_span_reversal_and_ignores_head_motion_gating():
    subject = EventEngine()

    events = make_wave_history(subject, head_motion_state="moving")

    assert event_names(events) == ["person_waving"]
    assert events[0]["duration_ms"] == 1200


def test_person_waving_rejects_low_confidence_keypoints():
    subject = EventEngine()

    events = make_wave_history(subject, wrist_confidence=0.2)

    assert "person_waving" not in event_names(events)


def test_person_waving_rejects_walking_arm_swing_near_or_below_shoulder():
    subject = EventEngine()
    emitted: list[dict] = []

    for index, (wrist_x, wrist_y) in enumerate(
        (
            (460.0, 318.0),
            (545.0, 330.0),
            (465.0, 319.0),
        )
    ):
        timestamp_ms = 1000 + (index * 600)
        emitted.extend(
            subject.update(
                frame(frame_id=1 + index, timestamp_ms=timestamp_ms),
                [
                    track(
                        1,
                        timestamp_ms=timestamp_ms,
                        bbox_xyxy=(400.0, 150.0, 600.0, 450.0),
                        keypoints=keypoints(wrist_x=wrist_x, wrist_y=wrist_y),
                    )
                ],
                attention(1),
            ).semantic_events
        )

    assert "person_waving" not in event_names(emitted)


def test_attention_target_changed_requires_previous_and_current_non_null_targets():
    subject = EventEngine()
    tracks = [
        track(1, timestamp_ms=1000, bbox_xyxy=(100.0, 100.0, 300.0, 500.0), hits=1),
        track(2, timestamp_ms=1000, bbox_xyxy=(600.0, 100.0, 800.0, 500.0), hits=1),
    ]

    assert (
        subject.update(frame(frame_id=1, timestamp_ms=1000), tracks, None).semantic_events
        == []
    )
    first = subject.update(frame(frame_id=2, timestamp_ms=1200), tracks, attention(1))
    changed = subject.update(
        frame(frame_id=3, timestamp_ms=1800),
        [
            track(1, timestamp_ms=1800, bbox_xyxy=(100.0, 100.0, 300.0, 500.0)),
            track(2, timestamp_ms=1800, bbox_xyxy=(600.0, 100.0, 800.0, 500.0), hits=1),
        ],
        attention(2),
    )

    assert "attention_target_changed" not in event_names(first)
    assert event_names(changed) == ["attention_target_changed"]
    assert changed.semantic_events[0]["track_id"] == 2


def test_cooldown_suppresses_same_event_type_at_4999ms_and_allows_at_5000ms():
    subject = EventEngine()

    first = make_wave_history(subject, track_id=1, start_ms=1000)
    suppressed: list[dict] = []
    for index, wrist_x in enumerate((60.0, 145.0, 65.0)):
        timestamp_ms = 5600 + (index * 600)
        suppressed = subject.update(
            frame(frame_id=4 + index, timestamp_ms=timestamp_ms),
            [
                track(
                    2,
                    timestamp_ms=timestamp_ms,
                    bbox_xyxy=(0.0, 150.0, 200.0, 450.0),
                    keypoints=keypoints(wrist_x=wrist_x),
                )
            ],
            attention(2),
        )
    allowed = subject.update(
        frame(frame_id=7, timestamp_ms=7200),
        [
            track(
                2,
                timestamp_ms=7200,
                bbox_xyxy=(0.0, 150.0, 200.0, 450.0),
                keypoints=keypoints(wrist_x=140.0),
            )
        ],
        attention(2),
    )

    assert event_names(first) == ["person_waving"]
    assert "person_waving" not in event_names(suppressed)
    assert event_names(allowed) == ["person_waving"]


def test_same_track_same_event_dedupes_inside_cooldown_window():
    subject = EventEngine()

    first = make_wave_history(subject, track_id=1, start_ms=1000)
    duplicate = make_wave_history(
        subject,
        track_id=1,
        start_ms=5600,
        start_frame_id=4,
    )

    assert event_names(first) == ["person_waving"]
    assert "person_waving" not in event_names(duplicate)


def test_nearby_reacquired_track_dedupes_greeting_after_global_cooldown():
    subject = EventEngine()

    first = make_wave_history(subject, track_id=1, start_ms=1000)
    reacquired: list[dict] = []
    for index, wrist_x in enumerate((460.0, 545.0, 465.0)):
        timestamp_ms = 7000 + (index * 600)
        reacquired.extend(
            subject.update(
                frame(frame_id=4 + index, timestamp_ms=timestamp_ms),
                [
                    track(
                        2,
                        timestamp_ms=timestamp_ms,
                        bbox_xyxy=(400.0, 150.0, 600.0, 450.0),
                        keypoints=keypoints(wrist_x=wrist_x),
                    )
                ],
                attention(2),
            ).semantic_events
        )

    assert event_names(first) == ["person_waving"]
    assert "person_appeared" not in event_names(reacquired)
    assert "person_waving" not in event_names(reacquired)


def test_nearby_aliased_track_can_emit_greeting_after_alias_window():
    subject = EventEngine()

    first = make_wave_history(subject, track_id=1, start_ms=1000)
    reacquired: list[dict] = []
    for index, wrist_x in enumerate((460.0, 545.0, 465.0)):
        timestamp_ms = 7000 + (index * 600)
        reacquired.extend(
            subject.update(
                frame(frame_id=4 + index, timestamp_ms=timestamp_ms),
                [
                    track(
                        2,
                        timestamp_ms=timestamp_ms,
                        bbox_xyxy=(400.0, 150.0, 600.0, 450.0),
                        keypoints=keypoints(wrist_x=wrist_x),
                    )
                ],
                attention(2),
            ).semantic_events
        )

    subject.update(
        frame(frame_id=7, timestamp_ms=9000),
        [
            track(
                2,
                timestamp_ms=9000,
                bbox_xyxy=(400.0, 150.0, 600.0, 450.0),
                keypoints=keypoints(wrist_x=500.0, wrist_confidence=0.1),
            )
        ],
        attention(2),
    )
    late: list[dict] = []
    for index, wrist_x in enumerate((460.0, 545.0, 465.0)):
        timestamp_ms = 12600 + (index * 600)
        late.extend(
            subject.update(
                frame(frame_id=8 + index, timestamp_ms=timestamp_ms),
                [
                    track(
                        2,
                        timestamp_ms=timestamp_ms,
                        bbox_xyxy=(400.0, 150.0, 600.0, 450.0),
                        keypoints=keypoints(wrist_x=wrist_x),
                    )
                ],
                attention(2),
            ).semantic_events
        )

    late_waves = events_of(late, "person_waving")

    assert event_names(first) == ["person_waving"]
    assert "person_waving" not in event_names(reacquired)
    assert len(late_waves) == 1
    assert late_waves[0]["track_id"] == 2


def test_far_reacquired_track_can_emit_greeting_after_global_cooldown():
    subject = EventEngine()

    first = make_wave_history(subject, track_id=1, start_ms=1000)
    emitted: list[dict] = []
    for index, wrist_x in enumerate((60.0, 145.0, 65.0)):
        timestamp_ms = 7000 + (index * 600)
        emitted.extend(
            subject.update(
                frame(frame_id=4 + index, timestamp_ms=timestamp_ms),
                [
                    track(
                        2,
                        timestamp_ms=timestamp_ms,
                        bbox_xyxy=(0.0, 150.0, 200.0, 450.0),
                        keypoints=keypoints(wrist_x=wrist_x),
                    )
                ],
                attention(2),
            ).semantic_events
        )

    assert event_names(first) == ["person_waving"]
    assert "person_appeared" in event_names(emitted)
    assert "person_waving" in event_names(emitted)
    assert {
        event["track_id"]
        for event in emitted
        if event["event"] in {"person_appeared", "person_waving"}
    } == {2}


def test_event_id_increments_and_reset_does_not_reuse_already_sent_ids():
    subject = EventEngine()
    first = subject.update(
        frame(frame_id=10, timestamp_ms=2000),
        [track(1, timestamp_ms=2000, first_seen_ms=1600)],
        attention(1),
    )
    after_reset = subject.update(
        frame(frame_id=9, timestamp_ms=1900),
        [track(2, timestamp_ms=1900, first_seen_ms=1500)],
        attention(2),
    )

    assert first.semantic_events[0]["event_id"] == "front:evt_000001"
    assert after_reset.semantic_events[0]["event_id"] == "front:evt_000002"


def emitted_event(name: str) -> tuple[dict, int]:
    subject = EventEngine()
    if name == "person_appeared":
        events = subject.update(
            frame(frame_id=1, timestamp_ms=1000),
            [track(1, timestamp_ms=1000, first_seen_ms=600)],
            attention(1),
        )
        return events.semantic_events[0], 400
    if name == "person_left":
        subject.update(
            frame(frame_id=1, timestamp_ms=1000),
            [track(1, timestamp_ms=1000, first_seen_ms=600)],
            attention(1),
        )
        events = subject.update(frame(frame_id=2, timestamp_ms=2500), [], None)
        return events.semantic_events[0], 1500
    if name == "person_passing_by":
        subject.update(
            frame(frame_id=1, timestamp_ms=1000),
            [track(1, timestamp_ms=1000, bbox_xyxy=(20.0, 200.0, 120.0, 520.0))],
            attention(1),
        )
        events = subject.update(
            frame(frame_id=2, timestamp_ms=2000),
            [
                track(
                    1,
                    timestamp_ms=2000,
                    bbox_xyxy=(820.0, 200.0, 920.0, 520.0),
                    velocity_uv_s=(800.0, 0.0),
                )
            ],
            attention(1),
        )
        return events_of(events, name)[0], 1000
    if name == "person_approaching_robot":
        subject.update(
            frame(frame_id=1, timestamp_ms=1000),
            [track(1, timestamp_ms=1000, bbox_xyxy=(450.0, 120.0, 550.0, 320.0))],
            attention(1),
        )
        events = subject.update(
            frame(frame_id=2, timestamp_ms=1600),
            [track(1, timestamp_ms=1600, bbox_xyxy=(425.0, 80.0, 575.0, 380.0))],
            attention(1),
        )
        return events_of(events, name)[0], 600
    if name == "person_stopped_near_robot":
        for frame_id, timestamp_ms in enumerate((1000, 1800), start=1):
            subject.update(
                frame(frame_id=frame_id, timestamp_ms=timestamp_ms),
                [
                    track(
                        1,
                        timestamp_ms=timestamp_ms,
                        bbox_xyxy=(350.0, 160.0, 650.0, 620.0),
                        velocity_uv_s=(10.0, 0.0),
                    )
                ],
                attention(1),
            )
        events = subject.update(
            frame(frame_id=3, timestamp_ms=2500),
            [
                track(
                    1,
                    timestamp_ms=2500,
                    bbox_xyxy=(350.0, 160.0, 650.0, 620.0),
                    velocity_uv_s=(10.0, 0.0),
                )
            ],
            attention(1),
        )
        return events_of(events, name)[0], 1500
    if name == "person_waving":
        events = make_wave_history(subject)
        return events_of(events, name)[0], 1200
    if name == "attention_target_changed":
        tracks = [
            track(1, timestamp_ms=1000, hits=1),
            track(2, timestamp_ms=1000, hits=1),
        ]
        subject.update(frame(frame_id=1, timestamp_ms=1000), tracks, attention(1))
        events = subject.update(
            frame(frame_id=2, timestamp_ms=1200),
            [
                track(1, timestamp_ms=1200, hits=1),
                track(2, timestamp_ms=1200, hits=1),
            ],
            attention(2),
        )
        return events.semantic_events[0], 0
    raise AssertionError(f"unsupported event fixture: {name}")


@pytest.mark.parametrize(
    "name",
    [
        "person_appeared",
        "person_left",
        "person_passing_by",
        "person_approaching_robot",
        "person_stopped_near_robot",
        "person_waving",
        "attention_target_changed",
    ],
)
def test_event_payload_schema_confidence_duration_and_text(name: str):
    event, expected_duration_ms = emitted_event(name)

    assert event["event"] == name
    assert set(event) == {
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
    assert event["type"] == "semantic_event"
    assert event["event_id"].startswith("front:evt_")
    assert 0.0 <= event["confidence"] <= 1.0
    assert isinstance(event["duration_ms"], int)
    assert event["duration_ms"] == expected_duration_ms
    assert event["lifecycle_state"] == "confirmed"
    assert isinstance(event["evidence"], dict)
    assert set(event["evidence"]) >= REQUIRED_EVIDENCE_KEYS[name]
    assert_json_simple_and_finite(event["evidence"])
    assert event["text"]


def test_head_motion_unknown_or_moving_suppresses_motion_events_without_accumulating():
    subject = EventEngine()
    subject.update(
        frame(frame_id=1, timestamp_ms=1000, head_motion_state="moving"),
        [track(1, timestamp_ms=1000, bbox_xyxy=(20.0, 200.0, 120.0, 520.0))],
        attention(1),
    )
    moving = subject.update(
        frame(frame_id=2, timestamp_ms=1800, head_motion_state="unknown"),
        [
            track(
                1,
                timestamp_ms=1800,
                bbox_xyxy=(460.0, 160.0, 760.0, 620.0),
                velocity_uv_s=(550.0, 0.0),
            )
        ],
        attention(1),
    )
    resumed = subject.update(
        frame(frame_id=3, timestamp_ms=2500, head_motion_state="stationary"),
        [
            track(
                1,
                timestamp_ms=2500,
                bbox_xyxy=(820.0, 160.0, 920.0, 520.0),
                velocity_uv_s=(514.0, 0.0),
            )
        ],
        attention(1),
    )

    assert not {"person_passing_by", "person_approaching_robot", "person_stopped_near_robot"} & set(
        event_names(moving)
    )
    assert not {"person_passing_by", "person_approaching_robot", "person_stopped_near_robot"} & set(
        event_names(resumed)
    )


def test_event_config_defaults_are_documented_thresholds():
    config = EventConfig()

    assert config.history_ms == 3000
    assert config.cooldown_ms == 5000
    assert config.stable_min_hits == 2
    assert config.stable_min_age_ms == 300
    assert config.left_lost_ms == 1500
    assert config.stop_duration_ms == 1500
    assert config.near_area_ratio == 0.075
    assert config.near_height_ratio == 0.55
    assert config.stop_max_speed_px_s == 35
    assert config.approach_duration_ms == 500
    assert config.approach_area_growth_ratio == 1.35
    assert config.approach_min_area_delta == 0.015
    assert config.approach_min_current_area == 0.035
    assert config.passing_duration_ms == 1000
    assert config.passing_min_dx_ratio == 0.45
    assert config.passing_min_abs_vx_px_s == 80
    assert config.keypoint_min_conf == 0.3
    assert config.wave_window_ms == 1800
    assert config.wave_min_x_span_px == 35
    assert config.wave_min_x_span_bbox_ratio == 0.12
    assert config.reacquire_alias_window_ms == 5000
    assert config.reacquire_center_distance_ratio == 0.08


def test_event_config_rejects_invalid_reacquire_alias_thresholds():
    with pytest.raises(ValueError, match="reacquire_alias_window_ms"):
        EventConfig(reacquire_alias_window_ms=-1)

    with pytest.raises(ValueError, match="reacquire_center_distance_ratio"):
        EventConfig(reacquire_center_distance_ratio=-0.01)
