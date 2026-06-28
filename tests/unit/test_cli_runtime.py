from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_cli.botified_output import BotifiedPipeClosed, format_botified_frame
from visual_events_cli.frame_pump import HeadMotion, InputFrame
from visual_events_cli.target_mapper import (
    make_invalid_gaze_target,
    map_visual_state_to_gaze_target,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VISUAL_STATE_TRACKING_SAMPLE = (
    REPO_ROOT / "common" / "schema" / "samples" / "visual_state_tracking.json"
)
TICK_TIMEOUT_SECONDS = 0.5


def import_runtime():
    try:
        import visual_events_cli.runtime as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.runtime module: {exc}")
    return module


def load_visual_state_tracking(**overrides: Any) -> dict[str, Any]:
    state = json.loads(VISUAL_STATE_TRACKING_SAMPLE.read_text(encoding="utf-8"))
    state.update(overrides)
    return state


def visual_state_lost(*, frame_id: int, timestamp_ms: int) -> dict[str, Any]:
    return load_visual_state_tracking(
        frame_id=frame_id,
        frame_timestamp_ms=timestamp_ms,
        attention=None,
        tracks=[],
        semantic_events=[],
    )


def service_result(visual_state: dict[str, Any] | None = None, error: Any = None) -> Any:
    return SimpleNamespace(visual_state=visual_state, error=error)


def make_frame(*, timestamp_ms: int, width: int = 1280, height: int = 720) -> InputFrame:
    return InputFrame(
        camera="front",
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        jpeg=JPEG_1280X720,
    )


class FakeClock:
    def __init__(self, now_ms: int):
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms


class FakeFrameSource:
    def __init__(self, frames: list[InputFrame]):
        self.frames = list(frames)
        self.poll_count = 0

    def poll_latest(self) -> InputFrame | None:
        self.poll_count += 1
        if not self.frames:
            return None
        return self.frames.pop(0)

    def read_latest(self) -> InputFrame | None:
        return self.poll_latest()


class FakeServiceClient:
    def __init__(self, results: list[Any]):
        self.requests: list[tuple[dict[str, Any], bytes]] = []
        self._results = list(results)
        self._request_event = asyncio.Event()

    async def request_frame(self, header: dict[str, Any], jpeg: bytes) -> Any:
        self.requests.append((header, jpeg))
        self._request_event.set()
        result = self._results.pop(0)
        if isinstance(result, asyncio.Future):
            return await result
        return result

    async def wait_for_requests(self, count: int) -> None:
        while len(self.requests) < count:
            await self._request_event.wait()
            self._request_event.clear()


class ShieldedFakeServiceClient(FakeServiceClient):
    async def request_frame(self, header: dict[str, Any], jpeg: bytes) -> Any:
        self.requests.append((header, jpeg))
        self._request_event.set()
        result = self._results.pop(0)
        if isinstance(result, asyncio.Future):
            return await asyncio.shield(result)
        return result


class FakeGazePublisher:
    def __init__(self):
        self.payloads: list[Any] = []
        self.closed = False

    def publish(self, payload: Any) -> None:
        if self.closed:
            raise AssertionError("published after close")
        self.payloads.append(payload)

    def dicts(self) -> list[dict[str, Any]]:
        return [payload.to_dict() for payload in self.payloads]


class FakeBotifiedWriter:
    def __init__(self):
        self.frames: list[str] = []
        self.drained_frames: list[str] = []
        self.drain_calls = 0

    def enqueue(self, frame: str) -> bool:
        self.frames.append(frame)
        return True

    def drain_available(self) -> None:
        self.drain_calls += 1
        self.drained_frames.extend(self.frames[len(self.drained_frames) :])


class BrokenOnQueuedDrainWriter(FakeBotifiedWriter):
    def drain_available(self) -> None:
        self.drain_calls += 1
        if self.frames:
            raise BotifiedPipeClosed("botified stdout closed")


class BlockingDrainWriter(FakeBotifiedWriter):
    def __init__(self, *, sleep_seconds: float):
        super().__init__()
        self.sleep_seconds = sleep_seconds
        self.started = threading.Event()

    def drain_available(self) -> None:
        self.drain_calls += 1
        self.started.set()
        time.sleep(self.sleep_seconds)


class BlockingFailingDrainWriter(FakeBotifiedWriter):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def drain_available(self) -> None:
        self.drain_calls += 1
        self.started.set()
        self.release.wait(timeout=TICK_TIMEOUT_SECONDS)
        raise BotifiedPipeClosed("botified stdout closed")


def head_motion() -> HeadMotion:
    return HeadMotion(
        state="stationary",
        yaw_vel_rad_s=0.0,
        pitch_vel_rad_s=0.0,
    )


def make_coordinator(
    runtime: Any,
    *,
    frame_source: FakeFrameSource,
    service: FakeServiceClient,
    gaze: FakeGazePublisher,
    botified: FakeBotifiedWriter | None = None,
    clock: FakeClock,
    stale_after_ms: int = 250,
) -> Any:
    return runtime.RuntimeCoordinator(
        frame_source=frame_source,
        service_client=service,
        gaze_publisher=gaze,
        head_motion_provider=head_motion,
        botified_writer=botified,
        stale_after_ms=stale_after_ms,
        clock_ms=clock,
    )


async def run_tick(coordinator: Any) -> Any:
    result = await asyncio.wait_for(
        coordinator.run_once(),
        timeout=TICK_TIMEOUT_SECONDS,
    )
    await asyncio.sleep(0)
    return result


@pytest.mark.asyncio
async def test_runtime_coordinator_pumps_frame_to_service_gaze_and_botified_without_stdout(
    capsys,
):
    runtime = import_runtime()
    clock = FakeClock(1710000000082)
    frame_source = FakeFrameSource([make_frame(timestamp_ms=1710000000000)])
    visual_state = load_visual_state_tracking(
        frame_id=1,
        frame_timestamp_ms=1710000000000,
    )
    service = FakeServiceClient([service_result(visual_state)])
    gaze = FakeGazePublisher()
    botified = FakeBotifiedWriter()
    coordinator = make_coordinator(
        runtime,
        frame_source=frame_source,
        service=service,
        gaze=gaze,
        botified=botified,
        clock=clock,
    )

    result = None
    for _ in range(3):
        result = await run_tick(coordinator)
        if service.requests and gaze.payloads and botified.drained_frames:
            break

    assert result in (None, 0)
    assert service.requests == [
        (
            {
                "type": "frame",
                "schema_version": 1,
                "camera": "front",
                "frame_id": 1,
                "timestamp_ms": 1710000000000,
                "encoding": "jpeg",
                "width": 1280,
                "height": 720,
                "head_motion": {
                    "state": "stationary",
                    "yaw_vel_rad_s": 0.0,
                    "pitch_vel_rad_s": 0.0,
                },
            },
            JPEG_1280X720,
        )
    ]
    assert gaze.dicts() == [
        map_visual_state_to_gaze_target(
            visual_state,
            publish_timestamp_ms=1710000000082,
            stale_after_ms=250,
        ).to_dict()
    ]
    assert botified.frames == [
        format_botified_frame(visual_state["semantic_events"][0])
    ]
    assert botified.drained_frames == botified.frames

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.asyncio
async def test_runtime_coordinator_stale_watchdog_fires_at_exact_deadline_while_service_hangs():
    runtime = import_runtime()
    clock = FakeClock(1710000000100)
    pending_response: asyncio.Future[Any] = asyncio.Future()
    frame_source = FakeFrameSource([make_frame(timestamp_ms=1710000000000)])
    service = FakeServiceClient([pending_response])
    gaze = FakeGazePublisher()
    coordinator = make_coordinator(
        runtime,
        frame_source=frame_source,
        service=service,
        gaze=gaze,
        clock=clock,
    )

    await run_tick(coordinator)
    await asyncio.wait_for(
        service.wait_for_requests(1),
        timeout=TICK_TIMEOUT_SECONDS,
    )
    clock.now_ms += 250

    result = await run_tick(coordinator)

    assert result in (None, 0)
    assert pending_response.done() is False
    assert gaze.dicts() == [
        make_invalid_gaze_target(
            "stale",
            camera="front",
            frame_id=1,
            frame_timestamp_ms=1710000000000,
            image_size=(1280, 720),
            publish_timestamp_ms=1710000000350,
            stale_after_ms=250,
        ).to_dict()
    ]

    pending_response.set_result(
        service_result(visual_state_lost(frame_id=1, timestamp_ms=1710000000000))
    )
    await run_tick(coordinator)


@pytest.mark.asyncio
async def test_runtime_coordinator_recovers_valid_gaze_after_camera_head_gap_publishes_stale():
    runtime = import_runtime()
    clock = FakeClock(1710000000100)
    stale_after_ms = 250
    first_frame_timestamp_ms = 1710000000000
    recovered_frame_timestamp_ms = 1710000000360
    frame_source = FakeFrameSource(
        [make_frame(timestamp_ms=first_frame_timestamp_ms)]
    )
    service = FakeServiceClient(
        [
            service_result(
                load_visual_state_tracking(
                    frame_id=1,
                    frame_timestamp_ms=first_frame_timestamp_ms,
                )
            ),
            service_result(
                load_visual_state_tracking(
                    frame_id=2,
                    frame_timestamp_ms=recovered_frame_timestamp_ms,
                )
            ),
        ]
    )
    gaze = FakeGazePublisher()
    head_state = {
        "motion": HeadMotion(
            state="stationary",
            yaw_vel_rad_s=0.0,
            pitch_vel_rad_s=0.0,
        )
    }

    def current_head_motion() -> HeadMotion:
        return head_state["motion"]

    coordinator = runtime.RuntimeCoordinator(
        frame_source=frame_source,
        service_client=service,
        gaze_publisher=gaze,
        head_motion_provider=current_head_motion,
        stale_after_ms=stale_after_ms,
        clock_ms=clock,
    )

    await run_tick(coordinator)
    assert [payload["state"] for payload in gaze.dicts()] == ["tracking"]

    clock.now_ms += stale_after_ms
    await run_tick(coordinator)
    assert [payload["state"] for payload in gaze.dicts()] == [
        "tracking",
        "stale",
    ]

    head_state["motion"] = HeadMotion(
        state="moving",
        yaw_vel_rad_s=0.2,
        pitch_vel_rad_s=-0.1,
    )
    clock.now_ms = recovered_frame_timestamp_ms
    frame_source.frames.append(make_frame(timestamp_ms=recovered_frame_timestamp_ms))

    await run_tick(coordinator)

    gaze_payloads = gaze.dicts()
    assert [payload["state"] for payload in gaze_payloads] == [
        "tracking",
        "stale",
        "tracking",
    ]
    assert gaze_payloads[-1]["valid"] is True
    assert service.requests[1][0]["head_motion"] == {
        "state": "moving",
        "yaw_vel_rad_s": 0.2,
        "pitch_vel_rad_s": -0.1,
    }


@pytest.mark.asyncio
async def test_runtime_coordinator_slow_botified_drain_does_not_block_stale_watchdog():
    runtime = import_runtime()
    clock = FakeClock(1710000000100)
    pending_response: asyncio.Future[Any] = asyncio.Future()
    frame_source = FakeFrameSource([make_frame(timestamp_ms=1710000000000)])
    service = FakeServiceClient([pending_response])
    gaze = FakeGazePublisher()
    botified = BlockingDrainWriter(sleep_seconds=0.1)
    coordinator = make_coordinator(
        runtime,
        frame_source=frame_source,
        service=service,
        gaze=gaze,
        botified=botified,
        clock=clock,
    )

    await run_tick(coordinator)
    await asyncio.wait_for(
        service.wait_for_requests(1),
        timeout=TICK_TIMEOUT_SECONDS,
    )
    assert botified.started.wait(timeout=TICK_TIMEOUT_SECONDS)
    assert pending_response.done() is False

    clock.now_ms += 250

    started_at = time.monotonic()
    result = await asyncio.wait_for(coordinator.run_once(), timeout=0.02)
    elapsed_seconds = time.monotonic() - started_at

    assert result in (None, 0)
    assert elapsed_seconds < 0.05
    assert pending_response.done() is False
    assert botified.drain_calls == 1
    assert gaze.dicts() == [
        make_invalid_gaze_target(
            "stale",
            camera="front",
            frame_id=1,
            frame_timestamp_ms=1710000000000,
            image_size=(1280, 720),
            publish_timestamp_ms=1710000000350,
            stale_after_ms=250,
        ).to_dict()
    ]

    pending_response.set_result(
        service_result(visual_state_lost(frame_id=1, timestamp_ms=1710000000000))
    )
    botified.sleep_seconds = 0.0
    await asyncio.sleep(0.11)
    await run_tick(coordinator)


@pytest.mark.asyncio
async def test_runtime_coordinator_broken_botified_pipe_publishes_stale_and_returns_nonzero():
    runtime = import_runtime()
    assert runtime.EXIT_BOTIFIED_PIPE_CLOSED != 0
    clock = FakeClock(1710000000082)
    frame_source = FakeFrameSource([make_frame(timestamp_ms=1710000000000)])
    visual_state = load_visual_state_tracking(
        frame_id=1,
        frame_timestamp_ms=1710000000000,
    )
    service = FakeServiceClient([service_result(visual_state)])
    gaze = FakeGazePublisher()
    botified = BrokenOnQueuedDrainWriter()
    coordinator = make_coordinator(
        runtime,
        frame_source=frame_source,
        service=service,
        gaze=gaze,
        botified=botified,
        clock=clock,
    )

    exit_code = None
    for _ in range(3):
        exit_code = await run_tick(coordinator)
        if exit_code not in (None, 0):
            break

    assert exit_code == runtime.EXIT_BOTIFIED_PIPE_CLOSED
    assert service.requests
    assert botified.frames == [
        format_botified_frame(visual_state["semantic_events"][0])
    ]
    stale_payloads = [
        payload for payload in gaze.dicts() if payload["state"] == "stale"
    ]
    assert len(stale_payloads) == 1
    assert stale_payloads[0] == make_invalid_gaze_target(
        "stale",
        camera="front",
        frame_id=1,
        frame_timestamp_ms=1710000000000,
        image_size=(1280, 720),
        publish_timestamp_ms=1710000000082,
        stale_after_ms=250,
    ).to_dict()


@pytest.mark.asyncio
async def test_runtime_coordinator_shutdown_cancels_pending_service_process_task():
    runtime = import_runtime()
    clock = FakeClock(1710000000100)
    pending_response: asyncio.Future[Any] = asyncio.Future()
    frame_source = FakeFrameSource([make_frame(timestamp_ms=1710000000000)])
    service = ShieldedFakeServiceClient([pending_response])
    gaze = FakeGazePublisher()
    coordinator = make_coordinator(
        runtime,
        frame_source=frame_source,
        service=service,
        gaze=gaze,
        clock=clock,
    )

    await run_tick(coordinator)
    await asyncio.wait_for(
        service.wait_for_requests(1),
        timeout=TICK_TIMEOUT_SECONDS,
    )

    await asyncio.wait_for(
        coordinator.shutdown(),
        timeout=TICK_TIMEOUT_SECONDS,
    )
    assert getattr(coordinator, "_process_task") is None

    gaze.closed = True
    pending_response.set_result(
        service_result(visual_state_lost(frame_id=1, timestamp_ms=1710000000000))
    )
    await asyncio.sleep(0)

    assert gaze.dicts() == []


@pytest.mark.asyncio
async def test_runtime_coordinator_shutdown_cancels_pending_botified_drain_task():
    runtime = import_runtime()
    clock = FakeClock(1710000000100)
    frame_source = FakeFrameSource([])
    service = FakeServiceClient([])
    gaze = FakeGazePublisher()
    botified = BlockingFailingDrainWriter()
    coordinator = make_coordinator(
        runtime,
        frame_source=frame_source,
        service=service,
        gaze=gaze,
        botified=botified,
        clock=clock,
    )

    await run_tick(coordinator)
    assert botified.started.wait(timeout=TICK_TIMEOUT_SECONDS)

    try:
        await asyncio.wait_for(
            coordinator.shutdown(),
            timeout=TICK_TIMEOUT_SECONDS,
        )
        assert getattr(coordinator, "_drain_task") is None
    finally:
        botified.release.set()

    await asyncio.sleep(0.05)
