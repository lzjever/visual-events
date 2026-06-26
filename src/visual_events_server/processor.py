from __future__ import annotations

import time
from typing import Any, Protocol

from .inference.base import InferBackend
from .protocol import FrameMessage, SCHEMA_VERSION
from .protocol import ProtocolError
from .tracking import ByteTrackStyleTracker, TrackSnapshot, TrackingConfig


class VisualFrameProcessor(Protocol):
    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        ...


class VisualStreamSession(VisualFrameProcessor, Protocol):
    pass


class VisualStreamSessionFactory(Protocol):
    def __call__(self) -> VisualStreamSession:
        ...


class BackendVisualFrameProcessor:
    def __init__(
        self,
        backend: InferBackend,
        *,
        tracking_config: TrackingConfig | None = None,
    ) -> None:
        self.backend = backend
        self.tracking_config = tracking_config or TrackingConfig()
        self._legacy_session: BackendVisualStreamSession | None = None

    def create_session(self) -> "BackendVisualStreamSession":
        return BackendVisualStreamSession(
            self.backend,
            tracker=ByteTrackStyleTracker(config=self.tracking_config),
        )

    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        if self._legacy_session is None:
            self._legacy_session = self.create_session()
        return await self._legacy_session.process_frame(frame)


class BackendVisualStreamSession:
    def __init__(
        self,
        backend: InferBackend,
        *,
        tracker: ByteTrackStyleTracker,
    ) -> None:
        self.backend = backend
        self.tracker = tracker

    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        try:
            detections = await self.backend.infer(frame)
        except Exception as exc:
            raise ProtocolError(
                "backend_unavailable",
                "inference backend unavailable for this frame",
                frame_id=frame.frame_id,
                retryable=True,
            ) from exc
        tracks = self.tracker.update(frame, detections)
        return build_visual_state(frame, tracks)


class MockVisualFrameProcessor:
    def __init__(self) -> None:
        from .inference.mock import MockInferBackend

        self._processor = BackendVisualFrameProcessor(MockInferBackend())

    def create_session(self) -> BackendVisualStreamSession:
        return self._processor.create_session()

    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        return await self._processor.process_frame(frame)


def build_visual_state(
    frame: FrameMessage,
    tracks: list[TrackSnapshot],
) -> dict[str, Any]:
    protocol_tracks = [
        track.to_protocol(image_width=frame.width, image_height=frame.height)
        for track in tracks
    ]
    visible_person_count = sum(1 for track in tracks if track.lost_ms == 0)
    return {
        "type": "visual_state",
        "schema_version": SCHEMA_VERSION,
        "camera": frame.camera,
        "frame_id": frame.frame_id,
        "frame_timestamp_ms": frame.timestamp_ms,
        "server_timestamp_ms": int(time.time() * 1000),
        "image_size": [frame.width, frame.height],
        "tracks": protocol_tracks,
        "attention": None,
        "scene_flags": {
            "has_person": visible_person_count > 0,
            "person_count": visible_person_count,
            "largest_person_stable": False,
            "someone_near_center": False,
        },
        "semantic_events": [],
    }
