from __future__ import annotations

import pytest

from visual_events_server.inference.base import PoseDetections
from visual_events_server.processor import BackendVisualFrameProcessor
from visual_events_server.protocol import FrameMessage, ProtocolError


def frame_message(*, frame_id: int = 7) -> FrameMessage:
    return FrameMessage(
        camera="front",
        frame_id=frame_id,
        timestamp_ms=1710000000000 + frame_id,
        width=640,
        height=480,
        jpeg_bytes=b"\xff\xd8fake\xff\xd9",
        head_motion_state="stationary",
    )


class EmptyBackend:
    async def infer(self, frame):
        return PoseDetections(persons=[])


class SplitPhaseBackend:
    async def infer(self, frame):
        return PoseDetections(persons=[])

    def consume_phase_metrics(self):
        return {
            "decode": 1.0,
            "infer": 2.0,
            "postprocess": 3.0,
        }


class FailingBackend:
    async def infer(self, frame):
        raise RuntimeError("backend down")


class RecordingMetricsSink:
    def __init__(self) -> None:
        self.records = []

    def write_frame_metrics(self, frame, phase_latencies_ms):
        self.records.append(
            {
                "camera": frame.camera,
                "frame_id": frame.frame_id,
                "frame_timestamp_ms": frame.timestamp_ms,
                "phase_latencies_ms": dict(phase_latencies_ms),
            }
        )


@pytest.mark.asyncio
async def test_processor_emits_phase_metrics_without_changing_visual_state():
    sink = RecordingMetricsSink()
    processor = BackendVisualFrameProcessor(EmptyBackend(), metrics_sink=sink)

    visual_state = await processor.process_frame(frame_message())

    assert visual_state["type"] == "visual_state"
    assert "phase_latencies_ms" not in visual_state
    assert "resources" not in visual_state
    assert sink.records[0]["frame_id"] == 7
    phases = sink.records[0]["phase_latencies_ms"]
    assert set(phases) == {
        "infer",
        "tracking",
        "attention",
        "events",
        "response",
        "total",
    }
    assert all(value >= 0.0 for value in phases.values())


@pytest.mark.asyncio
async def test_processor_preserves_backend_split_inference_phases():
    sink = RecordingMetricsSink()
    processor = BackendVisualFrameProcessor(SplitPhaseBackend(), metrics_sink=sink)

    await processor.process_frame(frame_message())

    phases = sink.records[0]["phase_latencies_ms"]
    assert phases["decode"] == 1.0
    assert phases["infer"] == 2.0
    assert phases["postprocess"] == 3.0
    assert phases["tracking"] >= 0.0
    assert phases["total"] >= 0.0


@pytest.mark.asyncio
async def test_processor_does_not_emit_metrics_for_failed_frames():
    sink = RecordingMetricsSink()
    processor = BackendVisualFrameProcessor(FailingBackend(), metrics_sink=sink)

    with pytest.raises(ProtocolError, match="inference backend unavailable"):
        await processor.process_frame(frame_message())

    assert sink.records == []
