from __future__ import annotations

import pytest

from visual_events_server.attention import AttentionResult
from visual_events_server.events import EventConfig, EventEngine
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


def event_names(events: list[dict]) -> list[str]:
    return [str(event["event"]) for event in events]


def events_of(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event["event"] == name]


def keypoints(
    *,
    wrist_x: float,
    wrist_confidence: float = 0.9,
    side: str = "left",
) -> tuple[PoseKeypoint, ...]:
    return (
        PoseKeypoint(
            name=f"{side}_shoulder",
            x=500.0,
            y=320.0,
            confidence=0.9,
        ),
        PoseKeypoint(
            name=f"{side}_wrist",
            x=wrist_x,
            y=300.0,
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
        )
    return emitted


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

    assert unstable == []
    assert event_names(appeared) == ["person_appeared"]
    assert appeared[0]["track_id"] == 1
    assert repeated == []


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
    assert events[0]["track_id"] == 2


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

    assert early == []
    assert event_names(left) == ["person_left"]
    assert left[0]["track_id"] == 1
    assert repeated == []


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


def test_attention_target_changed_requires_previous_and_current_non_null_targets():
    subject = EventEngine()
    tracks = [
        track(1, timestamp_ms=1000, bbox_xyxy=(100.0, 100.0, 300.0, 500.0), hits=1),
        track(2, timestamp_ms=1000, bbox_xyxy=(600.0, 100.0, 800.0, 500.0), hits=1),
    ]

    assert subject.update(frame(frame_id=1, timestamp_ms=1000), tracks, None) == []
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
    assert changed[0]["track_id"] == 2


def test_cooldown_suppresses_same_event_type_at_4999ms_and_allows_at_5000ms():
    subject = EventEngine()

    first = make_wave_history(subject, track_id=1, start_ms=1000)
    suppressed = make_wave_history(
        subject,
        track_id=2,
        start_ms=5600,
        start_frame_id=4,
    )
    allowed = subject.update(
        frame(frame_id=7, timestamp_ms=7200),
        [
            track(
                2,
                timestamp_ms=7200,
                bbox_xyxy=(400.0, 150.0, 600.0, 450.0),
                keypoints=keypoints(wrist_x=540.0),
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

    assert first[0]["event_id"] == "front:evt_000001"
    assert after_reset[0]["event_id"] == "front:evt_000002"


def emitted_event(name: str) -> tuple[dict, int]:
    subject = EventEngine()
    if name == "person_appeared":
        events = subject.update(
            frame(frame_id=1, timestamp_ms=1000),
            [track(1, timestamp_ms=1000, first_seen_ms=600)],
            attention(1),
        )
        return events[0], 400
    if name == "person_left":
        subject.update(
            frame(frame_id=1, timestamp_ms=1000),
            [track(1, timestamp_ms=1000, first_seen_ms=600)],
            attention(1),
        )
        events = subject.update(frame(frame_id=2, timestamp_ms=2500), [], None)
        return events[0], 1500
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
        return events[0], 0
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
        "text",
    }
    assert event["type"] == "semantic_event"
    assert event["event_id"].startswith("front:evt_")
    assert 0.0 <= event["confidence"] <= 1.0
    assert isinstance(event["duration_ms"], int)
    assert event["duration_ms"] == expected_duration_ms
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
