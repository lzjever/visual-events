from __future__ import annotations

import asyncio

import pytest

from visual_events_server.inference.base import PoseDetections
from visual_events_server.processor import BackendVisualFrameProcessor
from visual_events_server.protocol import FrameMessage


def frame_message(*, frame_id: int) -> FrameMessage:
    return FrameMessage(
        camera="front",
        frame_id=frame_id,
        timestamp_ms=1710000000000 + frame_id,
        width=640,
        height=480,
        jpeg_bytes=b"\xff\xd8fake\xff\xd9",
        head_motion_state="stationary",
    )


class GatedBackend:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def infer(self, frame):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self.release.wait()
            return PoseDetections(persons=[])
        finally:
            self.active -= 1


class SharedPhaseBackend:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self._phase_latencies_ms: dict[str, float] = {}

    async def infer(self, frame):
        self._phase_latencies_ms = {"infer": float(frame.frame_id)}
        await self.release.wait()
        return PoseDetections(persons=[])

    def consume_phase_metrics(self):
        phases = dict(self._phase_latencies_ms)
        self._phase_latencies_ms = {}
        return phases


class RecordingMetricsSink:
    def __init__(self) -> None:
        self.records = []

    def write_frame_metrics(self, frame, phase_latencies_ms):
        self.records.append(
            {
                "frame_id": frame.frame_id,
                "phase_latencies_ms": dict(phase_latencies_ms),
            }
        )


async def allow_tasks_to_reach_blocking_points() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_shared_backend_inference_is_serialized_across_stream_sessions():
    backend = GatedBackend()
    processor = BackendVisualFrameProcessor(backend)
    first_session = processor.create_session()
    second_session = processor.create_session()

    first = asyncio.create_task(first_session.process_frame(frame_message(frame_id=1)))
    second = asyncio.create_task(second_session.process_frame(frame_message(frame_id=2)))

    await allow_tasks_to_reach_blocking_points()
    backend.release.set()
    await asyncio.gather(first, second)

    assert backend.max_active == 1


@pytest.mark.asyncio
async def test_backend_phase_metrics_are_consumed_in_same_serialized_slot():
    backend = SharedPhaseBackend()
    sink = RecordingMetricsSink()
    processor = BackendVisualFrameProcessor(backend, metrics_sink=sink)
    first_session = processor.create_session()
    second_session = processor.create_session()

    first = asyncio.create_task(first_session.process_frame(frame_message(frame_id=1)))
    second = asyncio.create_task(second_session.process_frame(frame_message(frame_id=2)))

    await allow_tasks_to_reach_blocking_points()
    backend.release.set()
    await asyncio.gather(first, second)

    infer_by_frame_id = {
        record["frame_id"]: record["phase_latencies_ms"]["infer"]
        for record in sink.records
    }
    assert infer_by_frame_id == {1: 1.0, 2: 2.0}
