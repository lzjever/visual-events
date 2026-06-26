from __future__ import annotations

from visual_events_server.protocol import FrameMessage

from .base import PoseDetections


class MockInferBackend:
    async def infer(self, frame: FrameMessage) -> PoseDetections:
        return PoseDetections(persons=[])
