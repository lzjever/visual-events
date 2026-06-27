from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .attention import AttentionConfig, AttentionResult, AttentionSelector
from .events import EventConfig, EventEngine
from .inference.base import InferBackend
from .metrics import MetricsSink
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
        metrics_sink: MetricsSink | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.backend = backend
        self.tracking_config = tracking_config or TrackingConfig()
        self.attention_config = attention_config or AttentionConfig()
        self.event_config = event_config or EventConfig()
        self.metrics_sink = metrics_sink
        self._clock = clock or time.perf_counter
        self._legacy_session: BackendVisualStreamSession | None = None

    def create_session(self) -> "BackendVisualStreamSession":
        return BackendVisualStreamSession(
            self.backend,
            tracker=ByteTrackStyleTracker(config=self.tracking_config),
            attention_selector=AttentionSelector(config=self.attention_config),
            event_engine=EventEngine(config=self.event_config),
            metrics_sink=self.metrics_sink,
            clock=self._clock,
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
        metrics_sink: MetricsSink | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.backend = backend
        self.tracker = tracker
        self.attention_selector = attention_selector
        self.event_engine = event_engine
        self.metrics_sink = metrics_sink
        self._clock = clock or time.perf_counter
        self._last_frame_id: int | None = None
        self._last_timestamp_ms: int | None = None

    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        total_start = self._clock()
        phase_latencies_ms: dict[str, float] = {}
        if self._should_reset_for_regression(frame):
            self.reset()
        try:
            infer_start = self._clock()
            detections = await self.backend.infer(frame)
            infer_elapsed_ms = _elapsed_ms(infer_start, self._clock)
        except Exception as exc:
            raise ProtocolError(
                "backend_unavailable",
                "inference backend unavailable for this frame",
                frame_id=frame.frame_id,
                retryable=True,
            ) from exc

        backend_phase_latencies = _consume_backend_phase_metrics(self.backend)
        if backend_phase_latencies:
            phase_latencies_ms.update(backend_phase_latencies)
        else:
            phase_latencies_ms["infer"] = infer_elapsed_ms

        tracking_start = self._clock()
        tracks = self.tracker.update(frame, detections)
        phase_latencies_ms["tracking"] = _elapsed_ms(tracking_start, self._clock)

        attention_start = self._clock()
        attention = self.attention_selector.update(frame, tracks)
        phase_latencies_ms["attention"] = _elapsed_ms(attention_start, self._clock)

        events_start = self._clock()
        semantic_events = self.event_engine.update(frame, tracks, attention)
        phase_latencies_ms["events"] = _elapsed_ms(events_start, self._clock)

        response_start = self._clock()
        response = build_visual_state(
            frame,
            tracks,
            attention=attention,
            semantic_events=semantic_events,
        )
        phase_latencies_ms["response"] = _elapsed_ms(response_start, self._clock)
        phase_latencies_ms["total"] = _elapsed_ms(total_start, self._clock)
        self._emit_metrics(frame, phase_latencies_ms)
        self._last_frame_id = frame.frame_id
        self._last_timestamp_ms = frame.timestamp_ms
        return response

    def reset(self) -> None:
        self.tracker.reset()
        self.attention_selector.reset()
        self.event_engine.reset()
        self._last_frame_id = None
        self._last_timestamp_ms = None

    def _should_reset_for_regression(self, frame: FrameMessage) -> bool:
        if self._last_frame_id is not None and frame.frame_id < self._last_frame_id:
            return True
        return (
            self._last_timestamp_ms is not None
            and frame.timestamp_ms < self._last_timestamp_ms
        )

    def _emit_metrics(
        self,
        frame: FrameMessage,
        phase_latencies_ms: Mapping[str, float],
    ) -> None:
        if self.metrics_sink is None:
            return
        try:
            self.metrics_sink.write_frame_metrics(frame, phase_latencies_ms)
        except Exception:
            return


class MockVisualFrameProcessor:
    def __init__(self, *, metrics_sink: MetricsSink | None = None) -> None:
        from .inference.mock import MockInferBackend

        self._processor = BackendVisualFrameProcessor(
            MockInferBackend(),
            metrics_sink=metrics_sink,
        )

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


def _elapsed_ms(start: float, clock: Callable[[], float]) -> float:
    return max(0.0, (clock() - start) * 1000.0)


def _consume_backend_phase_metrics(backend: InferBackend) -> dict[str, float]:
    consume = getattr(backend, "consume_phase_metrics", None)
    if not callable(consume):
        return {}
    try:
        raw_phases = consume()
    except Exception:
        return {}
    if not isinstance(raw_phases, Mapping):
        return {}

    phases: dict[str, float] = {}
    for name, value in raw_phases.items():
        try:
            phases[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return phases
