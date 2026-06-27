from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


_SCHEMA_VERSION = 1
_INVALID_STATES = {"lost", "stale", "disabled"}


@dataclass(frozen=True)
class GazeTargetPayload:
    schema_version: int
    camera: str
    frame_id: int
    frame_timestamp_ms: int
    publish_timestamp_ms: int
    valid: bool
    state: str
    target_track_id: int
    target_u: float
    target_v: float
    target_norm_x: float
    target_norm_y: float
    image_width: int
    image_height: int
    confidence: float
    reason: str
    stale_after_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def map_visual_state_to_gaze_target(
    visual_state: dict[str, Any],
    publish_timestamp_ms: int,
    stale_after_ms: int = 250,
    enabled: bool = True,
) -> GazeTargetPayload:
    camera = str(visual_state.get("camera", ""))
    frame_id = _as_int(visual_state.get("frame_id"), -1)
    frame_timestamp_ms = _as_int(visual_state.get("frame_timestamp_ms"), 0)
    image_size = _parse_image_size(visual_state.get("image_size"))

    if not enabled:
        return make_invalid_gaze_target(
            "disabled",
            camera=camera,
            frame_id=frame_id,
            frame_timestamp_ms=frame_timestamp_ms,
            image_size=image_size,
            publish_timestamp_ms=publish_timestamp_ms,
            stale_after_ms=stale_after_ms,
        )

    attention = visual_state.get("attention")
    if not isinstance(attention, dict):
        return make_invalid_gaze_target(
            "lost",
            camera=camera,
            frame_id=frame_id,
            frame_timestamp_ms=frame_timestamp_ms,
            image_size=image_size,
            publish_timestamp_ms=publish_timestamp_ms,
            stale_after_ms=stale_after_ms,
        )

    image_width, image_height = image_size
    target_track_id = attention.get("target_track_id")
    target_uv = _parse_target_uv(attention.get("target_uv"))

    if (
        image_width <= 0
        or image_height <= 0
        or not _track_exists(visual_state.get("tracks"), target_track_id)
        or target_uv is None
    ):
        return make_invalid_gaze_target(
            "lost",
            camera=camera,
            frame_id=frame_id,
            frame_timestamp_ms=frame_timestamp_ms,
            image_size=image_size,
            publish_timestamp_ms=publish_timestamp_ms,
            stale_after_ms=stale_after_ms,
        )

    target_u, target_v = target_uv
    if not (0.0 <= target_u <= float(image_width) and 0.0 <= target_v <= float(image_height)):
        return make_invalid_gaze_target(
            "lost",
            camera=camera,
            frame_id=frame_id,
            frame_timestamp_ms=frame_timestamp_ms,
            image_size=image_size,
            publish_timestamp_ms=publish_timestamp_ms,
            stale_after_ms=stale_after_ms,
        )

    return GazeTargetPayload(
        schema_version=_SCHEMA_VERSION,
        camera=camera,
        frame_id=frame_id,
        frame_timestamp_ms=frame_timestamp_ms,
        publish_timestamp_ms=int(publish_timestamp_ms),
        valid=True,
        state="tracking",
        target_track_id=_as_int(target_track_id, -1),
        target_u=target_u,
        target_v=target_v,
        target_norm_x=target_u / float(image_width) - 0.5,
        target_norm_y=target_v / float(image_height) - 0.5,
        image_width=image_width,
        image_height=image_height,
        confidence=_as_float(attention.get("confidence"), 0.0),
        reason=str(attention.get("reason", "")),
        stale_after_ms=int(stale_after_ms),
    )


def make_invalid_gaze_target(
    state: str,
    *,
    camera: str,
    frame_id: int = -1,
    frame_timestamp_ms: int = 0,
    image_size: tuple[int, int] = (0, 0),
    publish_timestamp_ms: int,
    stale_after_ms: int = 250,
) -> GazeTargetPayload:
    if state not in _INVALID_STATES:
        raise ValueError(f"unknown invalid gaze target state: {state}")

    image_width, image_height = _parse_image_size(image_size)
    return GazeTargetPayload(
        schema_version=_SCHEMA_VERSION,
        camera=str(camera),
        frame_id=int(frame_id),
        frame_timestamp_ms=int(frame_timestamp_ms),
        publish_timestamp_ms=int(publish_timestamp_ms),
        valid=False,
        state=state,
        target_track_id=-1,
        target_u=0.0,
        target_v=0.0,
        target_norm_x=0.0,
        target_norm_y=0.0,
        image_width=image_width,
        image_height=image_height,
        confidence=0.0,
        reason=state,
        stale_after_ms=int(stale_after_ms),
    )


def _parse_image_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return (0, 0)
    width = _as_int(value[0], 0)
    height = _as_int(value[1], 0)
    return (width, height)


def _parse_target_uv(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    target_u = _as_float(value[0], math.nan)
    target_v = _as_float(value[1], math.nan)
    if not (math.isfinite(target_u) and math.isfinite(target_v)):
        return None
    return (target_u, target_v)


def _track_exists(tracks: Any, target_track_id: Any) -> bool:
    if not isinstance(tracks, list):
        return False
    return any(
        isinstance(track, dict) and track.get("track_id") == target_track_id
        for track in tracks
    )


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
