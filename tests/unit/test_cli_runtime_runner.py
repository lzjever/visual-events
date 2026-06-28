from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_cli.botified_output import BotifiedPipeClosed
from visual_events_cli.config import default_config
from visual_events_cli.frame_pump import HeadMotion, InputFrame


def import_runtime():
    try:
        import visual_events_cli.runtime as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.runtime module: {exc}")
    return module


class FakeClock:
    def __init__(self, now_ms: int):
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms


class TrackingImageSubscriber:
    def __init__(self, events: list[str], frames: list[InputFrame] | None = None):
        self._events = events
        self._frames = list(frames or [])

    def start(self) -> None:
        self._events.append("image.start")

    def close(self) -> None:
        self._events.append("image.close")

    def poll_latest(self) -> InputFrame | None:
        if not self._frames:
            return None
        return self._frames.pop(0)


class TrackingHeadStateSubscriber:
    def __init__(
        self,
        events: list[str],
        *,
        motion: HeadMotion | None = None,
        fail_start: bool = False,
    ):
        self._events = events
        self._motion = motion or HeadMotion(
            state="stationary",
            yaw_vel_rad_s=0.0,
            pitch_vel_rad_s=0.0,
        )
        self._fail_start = fail_start
        self.current_motion_calls: list[int] = []

    def start(self) -> None:
        self._events.append("head.start")
        if self._fail_start:
            raise RuntimeError("head start failed")

    def close(self) -> None:
        self._events.append("head.close")

    def current_motion(self, now_ms: int) -> HeadMotion:
        self.current_motion_calls.append(now_ms)
        return self._motion


class TrackingGazePublisher:
    def __init__(self, events: list[str]):
        self._events = events
        self.payloads: list[Any] = []

    def start(self) -> None:
        self._events.append("gaze.start")

    def close(self) -> None:
        self._events.append("gaze.close")

    def publish(self, payload: Any) -> None:
        self.payloads.append(payload)

    def dicts(self) -> list[dict[str, Any]]:
        return [payload.to_dict() for payload in self.payloads]


class FakeServiceClient:
    def __init__(self, results: list[Any]):
        self.requests: list[tuple[dict[str, Any], bytes]] = []
        self._results = list(results)

    async def request_frame(self, header: dict[str, Any], jpeg: bytes) -> Any:
        self.requests.append((header, jpeg))
        await asyncio.sleep(0)
        return self._results.pop(0)


class ImmediateServiceClient(FakeServiceClient):
    async def request_frame(self, header: dict[str, Any], jpeg: bytes) -> Any:
        self.requests.append((header, jpeg))
        return self._results.pop(0)


class TrackingAsyncServiceClient(FakeServiceClient):
    def __init__(self, events: list[str], results: list[Any]):
        super().__init__(results)
        self._events = events

    async def close(self) -> None:
        self._events.append("service.close")


class TrackingPrivateCloseServiceClient(FakeServiceClient):
    def __init__(self, events: list[str], results: list[Any]):
        super().__init__(results)
        self._events = events

    async def _close_connection(self) -> None:
        self._events.append("service._close_connection")


class FailingServiceClient:
    def __init__(self, events: list[str]):
        self._events = events

    async def request_frame(self, header: dict[str, Any], jpeg: bytes) -> Any:
        self._events.append("service.request_frame")
        await asyncio.sleep(0)
        raise RuntimeError("service failed")

    async def close(self) -> None:
        self._events.append("service.close")


class TrackingBotifiedWriter:
    def __init__(self, events: list[str]):
        self._events = events

    def close(self) -> None:
        self._events.append("botified.close")


class BrokenDrainBotifiedWriter(TrackingBotifiedWriter):
    def __init__(self, events: list[str]):
        super().__init__(events)
        self.drain_calls = 0

    def drain_available(self) -> None:
        self.drain_calls += 1
        self._events.append("botified.drain")
        raise BotifiedPipeClosed("botified stdout closed")


class QueuedBrokenDrainBotifiedWriter(TrackingBotifiedWriter):
    def __init__(self, events: list[str]):
        super().__init__(events)
        self.frames: list[str] = []
        self.drain_calls = 0

    def enqueue(self, frame: str) -> bool:
        self.frames.append(frame)
        return True

    def drain_available(self) -> None:
        self.drain_calls += 1
        self._events.append("botified.drain")
        if self.frames:
            raise BotifiedPipeClosed("botified stdout closed")


class SlowQueuedDrainBotifiedWriter(TrackingBotifiedWriter):
    def __init__(self, events: list[str], *, wait_seconds: float):
        super().__init__(events)
        self.frames: list[str] = []
        self.drain_calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._wait_seconds = wait_seconds

    def enqueue(self, frame: str) -> bool:
        self.frames.append(frame)
        return True

    def drain_available(self) -> None:
        self.drain_calls += 1
        self._events.append("botified.drain")
        self.started.set()
        self.release.wait(timeout=self._wait_seconds)


def make_frame(*, timestamp_ms: int = 1710000000000) -> InputFrame:
    return InputFrame(
        camera="front",
        timestamp_ms=timestamp_ms,
        width=1280,
        height=720,
        jpeg=JPEG_1280X720,
    )


def service_result(visual_state: dict[str, Any] | None = None) -> Any:
    return SimpleNamespace(visual_state=visual_state, error=None)


def visual_state_lost(*, frame_id: int = 1, timestamp_ms: int = 1710000000000) -> dict[str, Any]:
    return {
        "type": "visual_state",
        "schema_version": 1,
        "camera": "front",
        "frame_id": frame_id,
        "frame_timestamp_ms": timestamp_ms,
        "image_size": [1280, 720],
        "attention": None,
        "tracks": [],
        "semantic_events": [],
    }


def visual_state_with_semantic_event(
    *,
    frame_id: int = 1,
    timestamp_ms: int = 1710000000000,
) -> dict[str, Any]:
    state = visual_state_lost(frame_id=frame_id, timestamp_ms=timestamp_ms)
    state["semantic_events"] = [
        {
            "type": "semantic_event",
            "event_id": "event-1",
            "event": "person_waving",
            "camera": "front",
            "track_id": 7,
            "confidence": 0.91,
            "duration_ms": 900,
            "evidence": {
                "runtime_person_slot": 3,
                "wrist_x_span_px": 84.0,
                "wrist_x_span_bbox_ratio": 0.42,
                "wrist_y_relative_to_shoulder_px": 18.0,
                "wave_duration_ms": 900,
                "keypoint_min_confidence": 0.72,
            },
            "text": "person waving",
        }
    ]
    return state


def test_default_runtime_runner_fails_fast_without_loading_dds_sdk_modules():
    runtime = import_runtime()
    denied_roots = {
        "cyclonedds",
        "fastdds",
        "rclpy",
        "rtidds",
        "unitree",
        "unitree_sdk2py",
    }
    script = f"""
import sys
from visual_events_cli.config import default_config
from visual_events_cli.runtime import RuntimeUnavailable, run_runtime

before = set(sys.modules)
try:
    run_runtime(default_config(), max_ticks=1, sleep_seconds=0)
except RuntimeUnavailable as exc:
    if "Step 4 DDS adapters not implemented" not in str(exc):
        raise
else:
    raise SystemExit("expected RuntimeUnavailable")
loaded = set(sys.modules) - before
violations = sorted(
    root for root in {sorted(denied_roots)!r}
    if any(name == root or name.startswith(root + ".") for name in loaded)
)
if violations:
    print("\\n".join(violations))
    raise SystemExit(1)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_default_runtime_factories_use_direct_unavailable_dds_factories():
    runtime = import_runtime()
    config = default_config()
    factories = runtime.default_runtime_factories()

    for factory in (
        factories.image_subscriber,
        factories.head_state_subscriber,
        factories.gaze_publisher,
    ):
        with pytest.raises(runtime.RuntimeUnavailable, match="Step 4 DDS"):
            factory(config)


def test_runtime_runner_starts_resources_runs_ticks_uses_head_provider_and_closes_reverse():
    runtime = import_runtime()
    events: list[str] = []
    clock = FakeClock(1710000000082)
    image = TrackingImageSubscriber(events, [make_frame()])
    head = TrackingHeadStateSubscriber(
        events,
        motion=HeadMotion(
            state="moving",
            yaw_vel_rad_s=0.12,
            pitch_vel_rad_s=-0.04,
        ),
    )
    gaze = TrackingGazePublisher(events)
    service = FakeServiceClient([service_result(visual_state_lost())])
    factories = runtime.RuntimeFactories(
        image_subscriber=lambda _config: image,
        head_state_subscriber=lambda _config: head,
        gaze_publisher=lambda _config: gaze,
        service_client=lambda _config: service,
        botified_writer=lambda _config: None,
    )

    result = runtime.run_runtime(
        default_config(),
        factories=factories,
        clock_ms=clock,
        max_ticks=2,
        sleep_seconds=0,
    )

    assert result == 0
    assert events[:3] == ["image.start", "head.start", "gaze.start"]
    assert events[-3:] == ["gaze.close", "head.close", "image.close"]
    assert head.current_motion_calls == [1710000000082]
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
                    "state": "moving",
                    "yaw_vel_rad_s": 0.12,
                    "pitch_vel_rad_s": -0.04,
                },
            },
            JPEG_1280X720,
        )
    ]
    assert len(gaze.dicts()) == 1
    assert gaze.dicts()[0]["camera"] == "front"
    assert gaze.dicts()[0]["frame_id"] == 1
    assert gaze.dicts()[0]["frame_timestamp_ms"] == 1710000000000
    assert gaze.dicts()[0]["publish_timestamp_ms"] == 1710000000082
    assert gaze.dicts()[0]["valid"] is False
    assert gaze.dicts()[0]["state"] == "lost"


def test_runtime_runner_shutdowns_coordinator_and_closes_service_botified_before_dds(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = import_runtime()
    events: list[str] = []
    image = TrackingImageSubscriber(events)
    head = TrackingHeadStateSubscriber(events)
    gaze = TrackingGazePublisher(events)
    service = TrackingAsyncServiceClient(events, [])
    botified = TrackingBotifiedWriter(events)
    original_coordinator = runtime.RuntimeCoordinator

    class TrackingCoordinator(original_coordinator):
        async def shutdown(self) -> None:
            events.append("coordinator.shutdown")
            shutdown = getattr(super(), "shutdown", None)
            if shutdown is not None:
                await shutdown()

    monkeypatch.setattr(runtime, "RuntimeCoordinator", TrackingCoordinator)
    factories = runtime.RuntimeFactories(
        image_subscriber=lambda _config: image,
        head_state_subscriber=lambda _config: head,
        gaze_publisher=lambda _config: gaze,
        service_client=lambda _config: service,
        botified_writer=lambda _config: botified,
    )

    result = runtime.run_runtime(
        default_config(),
        factories=factories,
        max_ticks=1,
        sleep_seconds=0,
    )

    assert result == 0
    assert events == [
        "image.start",
        "head.start",
        "gaze.start",
        "coordinator.shutdown",
        "botified.close",
        "service.close",
        "gaze.close",
        "head.close",
        "image.close",
    ]


@pytest.mark.parametrize("shutdown_trigger", ["max_ticks", "stop_requested"])
def test_runtime_runner_shutdown_observes_completed_botified_drain_pipe_closed(
    shutdown_trigger: str,
):
    runtime = import_runtime()
    events: list[str] = []
    clock = FakeClock(1710000000082)
    image = TrackingImageSubscriber(events, [make_frame()])
    head = TrackingHeadStateSubscriber(events)
    gaze = TrackingGazePublisher(events)
    service = TrackingAsyncServiceClient(events, [service_result(visual_state_lost())])
    botified = BrokenDrainBotifiedWriter(events)

    stop_requested = None
    max_ticks = 1
    if shutdown_trigger == "stop_requested":
        max_ticks = None
        stop_checks = 0

        def request_stop_after_first_tick() -> bool:
            nonlocal stop_checks
            stop_checks += 1
            return stop_checks > 1

        stop_requested = request_stop_after_first_tick

    factories = runtime.RuntimeFactories(
        image_subscriber=lambda _config: image,
        head_state_subscriber=lambda _config: head,
        gaze_publisher=lambda _config: gaze,
        service_client=lambda _config: service,
        botified_writer=lambda _config: botified,
    )

    result = runtime.run_runtime(
        default_config(),
        factories=factories,
        clock_ms=clock,
        stop_requested=stop_requested,
        max_ticks=max_ticks,
        sleep_seconds=0,
    )

    assert result == runtime.EXIT_BOTIFIED_PIPE_CLOSED
    assert botified.drain_calls == 1
    stale_payloads = [
        payload for payload in gaze.dicts() if payload["state"] == "stale"
    ]
    assert len(stale_payloads) == 1
    assert events.index("botified.drain") < events.index("botified.close")
    assert events[-5:] == [
        "botified.close",
        "service.close",
        "gaze.close",
        "head.close",
        "image.close",
    ]


def test_runtime_runner_shutdown_observes_daemon_thread_botified_pipe_closed_after_one_tick():
    runtime = import_runtime()
    events: list[str] = []
    clock = FakeClock(1710000000082)
    image = TrackingImageSubscriber(events, [make_frame()])
    head = TrackingHeadStateSubscriber(events)
    gaze = TrackingGazePublisher(events)
    service = ImmediateServiceClient([service_result(visual_state_with_semantic_event())])
    botified = QueuedBrokenDrainBotifiedWriter(events)
    factories = runtime.RuntimeFactories(
        image_subscriber=lambda _config: image,
        head_state_subscriber=lambda _config: head,
        gaze_publisher=lambda _config: gaze,
        service_client=lambda _config: service,
        botified_writer=lambda _config: botified,
    )

    result = runtime.run_runtime(
        default_config(),
        factories=factories,
        clock_ms=clock,
        max_ticks=1,
        sleep_seconds=0,
    )

    assert result == runtime.EXIT_BOTIFIED_PIPE_CLOSED
    assert botified.drain_calls == 1
    assert botified.frames
    stale_payloads = [
        payload for payload in gaze.dicts() if payload["state"] == "stale"
    ]
    assert len(stale_payloads) == 1
    assert stale_payloads[0]["frame_id"] == 1
    assert events.index("botified.drain") < events.index("botified.close")


def test_runtime_runner_shutdown_does_not_wait_for_slow_botified_drain():
    runtime = import_runtime()
    events: list[str] = []
    image = TrackingImageSubscriber(events, [make_frame()])
    head = TrackingHeadStateSubscriber(events)
    gaze = TrackingGazePublisher(events)
    service = ImmediateServiceClient([service_result(visual_state_with_semantic_event())])
    botified = SlowQueuedDrainBotifiedWriter(events, wait_seconds=0.25)
    factories = runtime.RuntimeFactories(
        image_subscriber=lambda _config: image,
        head_state_subscriber=lambda _config: head,
        gaze_publisher=lambda _config: gaze,
        service_client=lambda _config: service,
        botified_writer=lambda _config: botified,
    )

    started_at = time.monotonic()
    try:
        result = runtime.run_runtime(
            default_config(),
            factories=factories,
            max_ticks=1,
            sleep_seconds=0,
        )
    finally:
        botified.release.set()
    elapsed_seconds = time.monotonic() - started_at

    assert result == 0
    assert botified.drain_calls == 1
    assert elapsed_seconds < 0.15


def test_runtime_runner_uses_private_async_close_connection_when_no_public_close():
    runtime = import_runtime()
    events: list[str] = []
    image = TrackingImageSubscriber(events)
    head = TrackingHeadStateSubscriber(events)
    gaze = TrackingGazePublisher(events)
    service = TrackingPrivateCloseServiceClient(events, [])
    factories = runtime.RuntimeFactories(
        image_subscriber=lambda _config: image,
        head_state_subscriber=lambda _config: head,
        gaze_publisher=lambda _config: gaze,
        service_client=lambda _config: service,
        botified_writer=lambda _config: None,
    )

    result = runtime.run_runtime(
        default_config(),
        factories=factories,
        max_ticks=1,
        sleep_seconds=0,
    )

    assert result == 0
    assert events[-4:] == [
        "service._close_connection",
        "gaze.close",
        "head.close",
        "image.close",
    ]


def test_runtime_runner_closes_service_and_dds_when_runtime_loop_raises():
    runtime = import_runtime()
    events: list[str] = []
    image = TrackingImageSubscriber(events, [make_frame()])
    head = TrackingHeadStateSubscriber(events)
    gaze = TrackingGazePublisher(events)
    service = FailingServiceClient(events)
    factories = runtime.RuntimeFactories(
        image_subscriber=lambda _config: image,
        head_state_subscriber=lambda _config: head,
        gaze_publisher=lambda _config: gaze,
        service_client=lambda _config: service,
        botified_writer=lambda _config: None,
    )

    with pytest.raises(RuntimeError, match="service failed"):
        runtime.run_runtime(
            default_config(),
            factories=factories,
            max_ticks=2,
            sleep_seconds=0,
        )

    assert events == [
        "image.start",
        "head.start",
        "gaze.start",
        "service.request_frame",
        "service.close",
        "gaze.close",
        "head.close",
        "image.close",
    ]


def test_runtime_runner_closes_started_resources_when_later_factory_fails():
    runtime = import_runtime()
    events: list[str] = []
    image = TrackingImageSubscriber(events)
    head = TrackingHeadStateSubscriber(events)

    def fail_gaze_factory(_config: Any) -> Any:
        raise runtime.RuntimeUnavailable("gaze unavailable")

    factories = runtime.RuntimeFactories(
        image_subscriber=lambda _config: image,
        head_state_subscriber=lambda _config: head,
        gaze_publisher=fail_gaze_factory,
        service_client=lambda _config: FakeServiceClient([]),
        botified_writer=lambda _config: None,
    )

    with pytest.raises(runtime.RuntimeUnavailable, match="gaze unavailable"):
        runtime.run_runtime(
            default_config(),
            factories=factories,
            max_ticks=1,
            sleep_seconds=0,
        )

    assert events == ["image.start", "head.start", "head.close", "image.close"]


def test_runtime_runner_closes_started_resources_when_later_start_fails():
    runtime = import_runtime()
    events: list[str] = []
    image = TrackingImageSubscriber(events)
    head = TrackingHeadStateSubscriber(events, fail_start=True)

    factories = runtime.RuntimeFactories(
        image_subscriber=lambda _config: image,
        head_state_subscriber=lambda _config: head,
        gaze_publisher=lambda _config: TrackingGazePublisher(events),
        service_client=lambda _config: FakeServiceClient([]),
        botified_writer=lambda _config: None,
    )

    with pytest.raises(RuntimeError, match="head start failed"):
        runtime.run_runtime(
            default_config(),
            factories=factories,
            max_ticks=1,
            sleep_seconds=0,
        )

    assert events == ["image.start", "head.start", "head.close", "image.close"]
