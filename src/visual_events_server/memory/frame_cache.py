from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from visual_events_server.protocol import FrameMessage


@dataclass(frozen=True)
class CachedFrame:
    connection_id: str
    frame: FrameMessage
    visual_state: dict[str, Any]
    observed_at_ms: int


class FrameCacheError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class FrameCache:
    def __init__(
        self,
        *,
        max_age_ms: int,
        clock_ms: Callable[[], int],
    ) -> None:
        if max_age_ms <= 0:
            raise ValueError("max_age_ms must be positive")
        self.max_age_ms = max_age_ms
        self._clock_ms = clock_ms
        self._frames_by_camera: dict[str, CachedFrame] = {}

    def update(
        self,
        *,
        connection_id: str,
        frame: FrameMessage,
        visual_state: dict[str, Any],
    ) -> None:
        self._frames_by_camera[frame.camera] = CachedFrame(
            connection_id=connection_id,
            frame=frame,
            visual_state=_compact_visual_state(visual_state),
            observed_at_ms=self._clock_ms(),
        )

    def get_fresh(self, camera: str) -> CachedFrame:
        cached = self._frames_by_camera.get(camera)
        if cached is None:
            raise FrameCacheError(
                "no_active_frame",
                f"no fresh frame is cached for camera {camera}",
            )
        age_ms = self._clock_ms() - cached.observed_at_ms
        if age_ms > self.max_age_ms:
            raise FrameCacheError(
                "frame_cache_expired",
                f"cached frame for camera {camera} is expired",
            )
        return cached


def _compact_visual_state(visual_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "camera": visual_state.get("camera"),
        "frame_id": visual_state.get("frame_id"),
        "frame_timestamp_ms": visual_state.get("frame_timestamp_ms"),
        "image_size": list(visual_state.get("image_size") or []),
        "tracks": list(visual_state.get("tracks") or []),
        "attention": visual_state.get("attention"),
    }
