from __future__ import annotations

import pytest

from visual_events_server.memory.frame_cache import FrameCache, FrameCacheError
from visual_events_server.protocol import FrameMessage


def frame(*, frame_id: int, camera: str = "front") -> FrameMessage:
    return FrameMessage(
        camera=camera,
        frame_id=frame_id,
        timestamp_ms=1_000 + frame_id,
        width=640,
        height=480,
        jpeg_bytes=b"jpeg",
        head_motion_state="stationary",
    )


def visual_state(frame_id: int) -> dict:
    return {
        "camera": "front",
        "frame_id": frame_id,
        "frame_timestamp_ms": 1_000 + frame_id,
        "image_size": [640, 480],
        "tracks": [],
    }


def test_frame_cache_latest_lookup_is_scoped_by_connection_and_camera() -> None:
    cache = FrameCache(max_age_ms=1_000, clock_ms=lambda: 2_000)
    cache.update(
        connection_id="ws_a",
        frame=frame(frame_id=1),
        visual_state=visual_state(1),
    )
    cache.update(
        connection_id="ws_b",
        frame=frame(frame_id=2),
        visual_state=visual_state(2),
    )

    assert cache.get_fresh_for_stream("ws_a", "front").frame.frame_id == 1
    assert cache.get_fresh_for_stream("ws_b", "front").frame.frame_id == 2
    assert cache.get_latest_for_camera("front").frame.frame_id == 2

    with pytest.raises(FrameCacheError) as exc:
        cache.get_fresh_for_stream("ws_missing", "front")

    assert exc.value.code == "no_active_frame"


def test_frame_cache_snapshot_window_is_scoped_by_connection_and_camera() -> None:
    cache = FrameCache(max_age_ms=1_000, clock_ms=lambda: 2_000)
    for connection_id, frame_id in (
        ("ws_a", 1),
        ("ws_b", 2),
        ("ws_a", 3),
        ("ws_b", 4),
    ):
        cache.update(
            connection_id=connection_id,
            frame=frame(frame_id=frame_id),
            visual_state=visual_state(frame_id),
        )

    first_window = cache.get_snapshot_window_for_stream("ws_a", "front")
    second_window = cache.get_snapshot_window_for_stream("ws_b", "front")
    camera_window = cache.get_snapshot_window_for_camera("front")

    assert [cached.frame.frame_id for cached in first_window.frames] == [1, 3]
    assert [cached.frame.frame_id for cached in second_window.frames] == [2, 4]
    assert [cached.frame.frame_id for cached in camera_window.frames] == [2, 3, 4]

    with pytest.raises(FrameCacheError):
        cache.get_snapshot_window_for_stream("ws_missing", "front")
