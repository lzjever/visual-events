from __future__ import annotations

import math

import pytest

from visual_events_server.attention import AttentionConfig, AttentionSelector
from visual_events_server.protocol import FrameMessage
from visual_events_server.tracking import TrackSnapshot


def frame(
    *,
    frame_id: int = 1,
    timestamp_ms: int = 1000,
    width: int = 100,
    height: int = 100,
) -> FrameMessage:
    return FrameMessage(
        camera="front",
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        jpeg_bytes=b"",
    )


def track(
    track_id: int,
    *,
    timestamp_ms: int = 1000,
    age_ms: int = 400,
    hits: int = 2,
    bbox_xyxy: tuple[float, float, float, float] = (10.0, 10.0, 30.0, 30.0),
    confidence: float = 0.9,
    head_uv: tuple[float, float] = (20.0, 15.0),
    lost_ms: int = 0,
    class_name: str = "person",
) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=track_id,
        first_seen_ms=timestamp_ms - age_ms,
        last_seen_ms=timestamp_ms - lost_ms,
        frame_timestamp_ms=timestamp_ms,
        bbox_xyxy=bbox_xyxy,
        confidence=confidence,
        pose_confidence=0.0,
        head_uv=head_uv,
        velocity_uv_s=(0.0, 0.0),
        lost_ms=lost_ms,
        hits=hits,
        misses=1 if lost_ms > 0 else 0,
        class_name=class_name,
    )


def selector(**overrides: int | float) -> AttentionSelector:
    config = AttentionConfig(**{**AttentionConfig().__dict__, **overrides})
    return AttentionSelector(config=config)


def test_no_tracks_outputs_null_attention():
    subject = AttentionSelector()

    assert subject.update(frame(), []) is None


def test_single_person_requires_stable_hits_and_age_before_output():
    subject = AttentionSelector()

    unstable_hits = subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [track(1, timestamp_ms=1000, hits=1, age_ms=400)],
    )
    unstable_age = subject.update(
        frame(frame_id=2, timestamp_ms=1200),
        [track(1, timestamp_ms=1200, hits=2, age_ms=200)],
    )
    stable = subject.update(
        frame(frame_id=3, timestamp_ms=1400),
        [track(1, timestamp_ms=1400, hits=3, age_ms=400)],
    )

    assert unstable_hits is None
    assert unstable_age is None
    assert stable is not None
    assert stable.target_track_id == 1


def test_target_uv_uses_head_uv_and_clamps_or_falls_back_to_bbox_center():
    clamped = selector(stable_min_hits=1, stable_min_age_ms=0).update(
        frame(width=100, height=80),
        [track(1, head_uv=(120.0, -10.0))],
    )
    fallback = selector(stable_min_hits=1, stable_min_age_ms=0).update(
        frame(width=100, height=80),
        [
            track(
                1,
                bbox_xyxy=(10.0, 20.0, 30.0, 60.0),
                head_uv=(math.nan, 5.0),
            )
        ],
    )

    assert clamped is not None
    assert clamped.target_uv == (100.0, 0.0)
    assert fallback is not None
    assert fallback.target_uv == (20.0, 40.0)


def test_multi_person_selects_largest_stable_person_by_area_confidence_score():
    result = selector(stable_min_hits=1, stable_min_age_ms=0).update(
        frame(),
        [
            track(1, bbox_xyxy=(0.0, 0.0, 10.0, 10.0), confidence=0.95),
            track(2, bbox_xyxy=(0.0, 0.0, 30.0, 30.0), confidence=0.50),
        ],
    )

    assert result is not None
    assert result.target_track_id == 2
    assert result.largest_person_stable is True


def test_current_target_holds_through_small_area_fluctuations():
    subject = selector(stable_min_hits=1, stable_min_age_ms=0)
    initial = subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [
            track(1, timestamp_ms=1000, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            track(2, timestamp_ms=1000, bbox_xyxy=(50.0, 0.0, 59.0, 10.0)),
        ],
    )
    held = subject.update(
        frame(frame_id=2, timestamp_ms=1200),
        [
            track(1, timestamp_ms=1200, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            track(2, timestamp_ms=1200, bbox_xyxy=(50.0, 0.0, 61.9, 10.0)),
        ],
    )

    assert initial is not None
    assert initial.target_track_id == 1
    assert held is not None
    assert held.target_track_id == 1


def test_challenger_must_exceed_area_ratio_for_confirm_duration_before_switch():
    subject = selector(
        stable_min_hits=1,
        stable_min_age_ms=0,
        switch_area_ratio=1.25,
        switch_confirm_ms=500,
    )

    initial = subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [
            track(1, timestamp_ms=1000, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            track(2, timestamp_ms=1000, bbox_xyxy=(50.0, 0.0, 59.0, 10.0)),
        ],
    )
    not_yet = subject.update(
        frame(frame_id=2, timestamp_ms=1200),
        [
            track(1, timestamp_ms=1200, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            track(2, timestamp_ms=1200, bbox_xyxy=(50.0, 0.0, 63.0, 10.0)),
        ],
    )
    switched = subject.update(
        frame(frame_id=3, timestamp_ms=1700),
        [
            track(1, timestamp_ms=1700, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            track(2, timestamp_ms=1700, bbox_xyxy=(50.0, 0.0, 63.0, 10.0)),
        ],
    )

    assert initial is not None
    assert initial.target_track_id == 1
    assert not_yet is not None
    assert not_yet.target_track_id == 1
    assert switched is not None
    assert switched.target_track_id == 2


def test_attention_config_750ms_switch_dwell_boundary():
    subject = AttentionSelector(
        config=AttentionConfig(
            stable_min_hits=1,
            stable_min_age_ms=0,
            switch_area_ratio=1.25,
            switch_confirm_ms=750,
        )
    )

    initial = subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [
            track(1, timestamp_ms=1000, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            track(2, timestamp_ms=1000, bbox_xyxy=(50.0, 0.0, 59.0, 10.0)),
        ],
    )
    challenger_seen = subject.update(
        frame(frame_id=2, timestamp_ms=1001),
        [
            track(1, timestamp_ms=1001, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            track(2, timestamp_ms=1001, bbox_xyxy=(50.0, 0.0, 70.0, 10.0)),
        ],
    )
    not_yet = subject.update(
        frame(frame_id=3, timestamp_ms=1750),
        [
            track(1, timestamp_ms=1750, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            track(2, timestamp_ms=1750, bbox_xyxy=(50.0, 0.0, 70.0, 10.0)),
        ],
    )
    switched = subject.update(
        frame(frame_id=4, timestamp_ms=1751),
        [
            track(1, timestamp_ms=1751, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            track(2, timestamp_ms=1751, bbox_xyxy=(50.0, 0.0, 70.0, 10.0)),
        ],
    )

    assert initial is not None
    assert initial.target_track_id == 1
    assert challenger_seen is not None
    assert challenger_seen.target_track_id == 1
    assert not_yet is not None
    assert not_yet.target_track_id == 1
    assert switched is not None
    assert switched.target_track_id == 2


def test_current_target_short_lost_hold_then_timeout_or_new_stable_target():
    held_subject = selector(stable_min_hits=1, stable_min_age_ms=0, lost_hold_ms=600)
    held_subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [track(1, timestamp_ms=1000)],
    )

    held = held_subject.update(
        frame(frame_id=2, timestamp_ms=1300),
        [track(1, timestamp_ms=1300, lost_ms=300)],
    )
    expired = held_subject.update(
        frame(frame_id=3, timestamp_ms=1701),
        [track(1, timestamp_ms=1701, lost_ms=701)],
    )

    new_target_subject = selector(
        stable_min_hits=1,
        stable_min_age_ms=0,
        lost_hold_ms=600,
    )
    new_target_subject.update(
        frame(frame_id=1, timestamp_ms=1000),
        [track(1, timestamp_ms=1000)],
    )
    replacement = new_target_subject.update(
        frame(frame_id=2, timestamp_ms=1701),
        [
            track(1, timestamp_ms=1701, lost_ms=701),
            track(2, timestamp_ms=1701, bbox_xyxy=(50.0, 0.0, 80.0, 30.0)),
        ],
    )

    assert held is not None
    assert held.target_track_id == 1
    assert held.reason == "held_lost_target"
    assert held.largest_person_stable is False
    assert expired is None
    assert replacement is not None
    assert replacement.target_track_id == 2


def test_lost_then_recovered_same_track_id_keeps_target():
    subject = selector(stable_min_hits=1, stable_min_age_ms=0, lost_hold_ms=600)
    subject.update(frame(frame_id=1, timestamp_ms=1000), [track(1, timestamp_ms=1000)])
    held = subject.update(
        frame(frame_id=2, timestamp_ms=1300),
        [track(1, timestamp_ms=1300, lost_ms=300)],
    )
    recovered = subject.update(
        frame(frame_id=3, timestamp_ms=1500),
        [
            track(1, timestamp_ms=1500, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            track(2, timestamp_ms=1500, bbox_xyxy=(50.0, 0.0, 63.0, 10.0)),
        ],
    )

    assert held is not None
    assert held.target_track_id == 1
    assert recovered is not None
    assert recovered.target_track_id == 1


def test_timestamp_or_frame_id_regression_resets_selector_state():
    subject = selector(stable_min_hits=1, stable_min_age_ms=0)
    subject.update(frame(frame_id=10, timestamp_ms=1000), [track(1, timestamp_ms=1000)])

    timestamp_reset = subject.update(
        frame(frame_id=11, timestamp_ms=900),
        [track(2, timestamp_ms=900)],
    )
    subject.update(frame(frame_id=12, timestamp_ms=1100), [track(2, timestamp_ms=1100)])
    frame_reset = subject.update(
        frame(frame_id=5, timestamp_ms=1200),
        [track(1, timestamp_ms=1200)],
    )

    assert timestamp_reset is not None
    assert timestamp_reset.target_track_id == 2
    assert frame_reset is not None
    assert frame_reset.target_track_id == 1


def test_attention_config_defaults_and_invalid_values():
    defaults = AttentionConfig()

    assert defaults.stable_min_hits == 2
    assert defaults.stable_min_age_ms == 300
    assert defaults.switch_area_ratio == 1.25
    assert defaults.switch_confirm_ms == 500
    assert defaults.lost_hold_ms == 600

    with pytest.raises(ValueError, match="stable_min_hits"):
        AttentionConfig(stable_min_hits=0)
    with pytest.raises(ValueError, match="stable_min_age_ms"):
        AttentionConfig(stable_min_age_ms=-1)
    with pytest.raises(ValueError, match="switch_area_ratio"):
        AttentionConfig(switch_area_ratio=0.99)
    with pytest.raises(ValueError, match="switch_confirm_ms"):
        AttentionConfig(switch_confirm_ms=-1)
    with pytest.raises(ValueError, match="lost_hold_ms"):
        AttentionConfig(lost_hold_ms=-1)
