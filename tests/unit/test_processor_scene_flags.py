from visual_events_server.processor import build_visual_state
from visual_events_server.protocol import FrameMessage
from visual_events_server.tracking import TrackSnapshot


def frame(*, width: int = 640, height: int = 480) -> FrameMessage:
    return FrameMessage(
        camera="front",
        frame_id=42,
        timestamp_ms=1710000000000,
        width=width,
        height=height,
        jpeg_bytes=b"",
    )


def person_track(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    lost_ms: int = 0,
) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=1,
        first_seen_ms=1710000000000,
        last_seen_ms=1710000000000 - lost_ms,
        frame_timestamp_ms=1710000000000,
        bbox_xyxy=bbox_xyxy,
        confidence=0.9,
        pose_confidence=0.8,
        head_uv=(0.5, 0.25),
        velocity_uv_s=(0.0, 0.0),
        lost_ms=lost_ms,
        hits=1,
        misses=1 if lost_ms > 0 else 0,
        class_name="person",
    )


def test_someone_near_center_true_for_visible_person_centered_bbox():
    visual_state = build_visual_state(
        frame(),
        [person_track((300.0, 220.0, 340.0, 260.0))],
    )

    assert visual_state["scene_flags"]["someone_near_center"] is True


def test_someone_near_center_false_for_visible_person_away_from_center():
    visual_state = build_visual_state(
        frame(),
        [person_track((10.0, 10.0, 50.0, 50.0))],
    )

    assert visual_state["scene_flags"]["someone_near_center"] is False


def test_someone_near_center_false_for_lost_person_centered_bbox():
    visual_state = build_visual_state(
        frame(),
        [person_track((300.0, 220.0, 340.0, 260.0), lost_ms=100)],
    )

    assert visual_state["scene_flags"]["someone_near_center"] is False
