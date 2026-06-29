from __future__ import annotations

import pytest

from visual_events_server.inference.base import (
    PersonPoseDetection,
    PoseDetections,
    PoseKeypoint,
)
from visual_events_server.processor import BackendVisualFrameProcessor
from visual_events_server.protocol import FrameMessage


class SinglePersonBackend:
    async def infer(self, frame: FrameMessage) -> PoseDetections:
        return PoseDetections(
            persons=[
                PersonPoseDetection(
                    bbox_xyxy=(300.0, 100.0, 700.0, 650.0),
                    bbox_area=220_000.0,
                    confidence=0.95,
                    keypoints=[
                        PoseKeypoint("nose", 500.0, 150.0, 0.91),
                        PoseKeypoint("left_wrist", 430.0, 260.0, 0.82),
                    ],
                )
            ]
        )


@pytest.mark.asyncio
async def test_processor_memory_snapshot_keeps_keypoints_out_of_public_response() -> None:
    session = BackendVisualFrameProcessor(SinglePersonBackend()).create_session()
    frame = FrameMessage(
        camera="front",
        frame_id=1,
        timestamp_ms=1_000,
        width=1280,
        height=720,
        jpeg_bytes=b"jpeg",
        head_motion_state="stationary",
    )

    response = await session.process_frame(frame)
    snapshot = session.take_memory_frame_snapshot()

    assert "keypoints" not in response["tracks"][0]
    assert snapshot is not None
    assert snapshot.frame is frame
    assert snapshot.tracks[0].keypoints[0].name == "nose"
    assert snapshot.tracks[0].keypoints[1].name == "left_wrist"
    assert snapshot.scene_context == response["scene_context"]
