from __future__ import annotations

import time
from typing import Any, Protocol

from .inference.base import InferBackend, PoseDetections
from .protocol import FrameMessage, SCHEMA_VERSION
from .protocol import ProtocolError


class VisualFrameProcessor(Protocol):
    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        ...


class BackendVisualFrameProcessor:
    def __init__(self, backend: InferBackend) -> None:
        self.backend = backend

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
        return build_visual_state(frame, detections)


class MockVisualFrameProcessor:
    def __init__(self) -> None:
        from .inference.mock import MockInferBackend

        self._processor = BackendVisualFrameProcessor(MockInferBackend())

    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        return await self._processor.process_frame(frame)


def build_visual_state(
    frame: FrameMessage,
    detections: PoseDetections,
) -> dict[str, Any]:
    person_count = len(detections.persons)
    return {
        "type": "visual_state",
        "schema_version": SCHEMA_VERSION,
        "camera": frame.camera,
        "frame_id": frame.frame_id,
        "frame_timestamp_ms": frame.timestamp_ms,
        "server_timestamp_ms": int(time.time() * 1000),
        "image_size": [frame.width, frame.height],
        "tracks": [],
        "attention": None,
        "scene_flags": {
            "has_person": person_count > 0,
            "person_count": person_count,
            "largest_person_stable": False,
            "someone_near_center": False,
        },
        "semantic_events": [],
    }
