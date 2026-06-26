from __future__ import annotations

import time
from typing import Any, Protocol

from .attention import AttentionConfig, AttentionResult, AttentionSelector
from .events import EventConfig, EventEngine
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
        attention_config: AttentionConfig | None = None,
        event_config: EventConfig | None = None,
    ) -> None:
        self.backend = backend
        self.tracking_config = tracking_config or TrackingConfig()
        self.attention_config = attention_config or AttentionConfig()
        self.event_config = event_config or EventConfig()
        self._legacy_session: BackendVisualStreamSession | None = None

    def create_session(self) -> "BackendVisualStreamSession":
        return BackendVisualStreamSession(
            self.backend,
            tracker=ByteTrackStyleTracker(config=self.tracking_config),
            attention_selector=AttentionSelector(config=self.attention_config),
            event_engine=EventEngine(config=self.event_config),
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
        attention_selector: AttentionSelector,
        event_engine: EventEngine,
    ) -> None:
        self.backend = backend
        self.tracker = tracker
        self.attention_selector = attention_selector
        self.event_engine = event_engine

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
        attention = self.attention_selector.update(frame, tracks)
        semantic_events = self.event_engine.update(frame, tracks, attention)
        return build_visual_state(
            frame,
            tracks,
            attention=attention,
            semantic_events=semantic_events,
        )


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
    *,
    attention: AttentionResult | None = None,
    semantic_events: list[dict[str, object]] | None = None,
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
        "attention": attention.to_protocol() if attention is not None else None,
        "scene_flags": {
            "has_person": visible_person_count > 0,
            "person_count": visible_person_count,
            "largest_person_stable": (
                attention.largest_person_stable if attention is not None else False
            ),
            "someone_near_center": False,
        },
        "semantic_events": semantic_events or [],
    }
