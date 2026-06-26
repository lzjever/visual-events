from __future__ import annotations

import time
from typing import Any, Protocol

from .protocol import FrameMessage, SCHEMA_VERSION


class VisualFrameProcessor(Protocol):
    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        ...


class MockVisualFrameProcessor:
    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        server_timestamp_ms = int(time.time() * 1000)
        return {
            "type": "visual_state",
            "schema_version": SCHEMA_VERSION,
            "camera": frame.camera,
            "frame_id": frame.frame_id,
            "frame_timestamp_ms": frame.timestamp_ms,
            "server_timestamp_ms": server_timestamp_ms,
            "image_size": [frame.width, frame.height],
            "tracks": [],
            "attention": None,
            "scene_flags": {
                "has_person": False,
                "person_count": 0,
                "largest_person_stable": False,
                "someone_near_center": False,
            },
            "semantic_events": [],
        }
