from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable

from visual_events_server.attention import AttentionResult
from visual_events_server.protocol import FrameMessage
from visual_events_server.tracking import TrackSnapshot


@dataclass(frozen=True)
class MemoryFrameSnapshot:
    connection_id: str
    frame: FrameMessage
    source_frame_ref: str
    snapshot_ref: str
    observed_at_ms: int
    image_size: tuple[int, int]
    tracks: list[TrackSnapshot]
    attention: AttentionResult | None
    scene_context: dict[str, Any]
    semantic_events: list[dict[str, Any]]


@dataclass(frozen=True)
class CachedFrame:
    connection_id: str
    frame: FrameMessage
    visual_state: dict[str, Any]
    observed_at_ms: int
    memory_snapshot: MemoryFrameSnapshot | None = None


@dataclass(frozen=True)
class MemoryFrameSnapshotWindow:
    camera: str
    frames: tuple[CachedFrame, ...]
    observed_at_ms: int
    frame_cache_ttl_ms: int


@dataclass(frozen=True)
class RequestInteractionSnapshot:
    selected: CachedFrame
    request_snapshot_ref: str
    source_frame_ref: str
    frame_timestamp_ms: int
    observed_at_ms: int
    frame_cache_ttl_ms: int
    stability_window: dict[str, Any]
    active_target_track_id: int


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
        self._frames_by_stream: dict[tuple[str, str], CachedFrame] = {}
        self._snapshot_windows_by_camera: dict[str, deque[CachedFrame]] = {}
        self._snapshot_windows_by_stream: dict[
            tuple[str, str],
            deque[CachedFrame],
        ] = {}
        self._snapshot_window_size = 3

    def update(
        self,
        *,
        connection_id: str,
        frame: FrameMessage,
        visual_state: dict[str, Any],
        memory_snapshot: MemoryFrameSnapshot | None = None,
    ) -> None:
        observed_at_ms = self._clock_ms()
        if memory_snapshot is not None:
            memory_snapshot = replace(
                memory_snapshot,
                connection_id=connection_id,
                frame=frame,
                observed_at_ms=observed_at_ms,
            )
        cached = CachedFrame(
            connection_id=connection_id,
            frame=frame,
            visual_state=_compact_visual_state(visual_state),
            observed_at_ms=observed_at_ms,
            memory_snapshot=memory_snapshot,
        )
        self._frames_by_camera[frame.camera] = cached
        self._frames_by_stream[(connection_id, frame.camera)] = cached
        window = self._snapshot_windows_by_camera.setdefault(
            frame.camera,
            deque(maxlen=self._snapshot_window_size),
        )
        window.append(cached)
        stream_window = self._snapshot_windows_by_stream.setdefault(
            (connection_id, frame.camera),
            deque(maxlen=self._snapshot_window_size),
        )
        stream_window.append(cached)

    def get_fresh_for_stream(self, connection_id: str, camera: str) -> CachedFrame:
        return self._fresh_or_error(
            self._frames_by_stream.get((connection_id, camera)),
            camera=camera,
        )

    def get_latest_for_camera(self, camera: str) -> CachedFrame:
        cached = self._frames_by_camera.get(camera)
        return self._fresh_or_error(cached, camera=camera)

    def _fresh_or_error(
        self,
        cached: CachedFrame | None,
        *,
        camera: str,
    ) -> CachedFrame:
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

    def get_snapshot_window_for_stream(
        self,
        connection_id: str,
        camera: str,
    ) -> MemoryFrameSnapshotWindow:
        frames = tuple(
            self._snapshot_windows_by_stream.get((connection_id, camera)) or ()
        )
        return self._snapshot_window_or_error(frames, camera=camera)

    def get_snapshot_window_for_camera(self, camera: str) -> MemoryFrameSnapshotWindow:
        frames = tuple(self._snapshot_windows_by_camera.get(camera) or ())
        return self._snapshot_window_or_error(frames, camera=camera)

    def _snapshot_window_or_error(
        self,
        frames: tuple[CachedFrame, ...],
        *,
        camera: str,
    ) -> MemoryFrameSnapshotWindow:
        if not frames:
            raise FrameCacheError(
                "no_active_frame",
                f"no fresh frame is cached for camera {camera}",
            )
        return MemoryFrameSnapshotWindow(
            camera=camera,
            frames=frames,
            observed_at_ms=self._clock_ms(),
            frame_cache_ttl_ms=self.max_age_ms,
        )


def _compact_visual_state(visual_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "camera": visual_state.get("camera"),
        "frame_id": visual_state.get("frame_id"),
        "frame_timestamp_ms": visual_state.get("frame_timestamp_ms"),
        "image_size": list(visual_state.get("image_size") or []),
        "tracks": list(visual_state.get("tracks") or []),
        "attention": visual_state.get("attention"),
    }
