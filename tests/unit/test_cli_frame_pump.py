from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_cli.target_mapper import (
    make_invalid_gaze_target,
    map_visual_state_to_gaze_target,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VISUAL_STATE_TRACKING_SAMPLE = (
    REPO_ROOT / "common" / "schema" / "samples" / "visual_state_tracking.json"
)


def import_frame_pump():
    try:
        import visual_events_cli.frame_pump as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.frame_pump module: {exc}")
    return module


def load_visual_state_tracking(**overrides: Any) -> dict[str, Any]:
    state = json.loads(VISUAL_STATE_TRACKING_SAMPLE.read_text(encoding="utf-8"))
    state.update(overrides)
    return state


def visual_state_lost(*, frame_id: int, timestamp_ms: int = 1710000000000) -> dict[str, Any]:
    return load_visual_state_tracking(
        frame_id=frame_id,
        frame_timestamp_ms=timestamp_ms,
        attention=None,
        tracks=[],
        semantic_events=[],
    )


def semantic_event(
    *,
    event_id: str,
    event: str = "person_waving",
) -> dict[str, Any]:
    return {
        "type": "semantic_event",
        "event_id": event_id,
        "event": event,
        "camera": "front",
        "track_id": 7,
        "confidence": 0.86,
        "duration_ms": 900,
        "evidence": {
            "runtime_person_slot": 3,
            "wrist_x_span_px": 84.0,
            "wrist_x_span_bbox_ratio": 0.42,
            "wrist_y_relative_to_shoulder_px": 18.0,
            "wave_duration_ms": 900,
            "keypoint_min_confidence": 0.72,
        },
        "text": "有人在机器人前方挥手",
    }


class FakeServiceClient:
    def __init__(self, results: list[Any] | None = None):
        self.requests: list[tuple[dict[str, Any], bytes]] = []
        self._results = list(results or [])
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


class FakeGazePublisher:
    def __init__(self):
        self.payloads: list[Any] = []

    def publish(self, payload: Any) -> None:
        self.payloads.append(payload)

    def dicts(self) -> list[dict[str, Any]]:
        return [payload.to_dict() for payload in self.payloads]


class FakeBotifiedWriter:
    def __init__(self):
        self.frames: list[str] = []

    def write(self, frame: str) -> None:
        self.frames.append(frame)


class FakeEnqueueBotifiedWriter:
    def __init__(self):
        self.frames: list[str] = []
        self.drain_called = False

    def enqueue(self, frame: str) -> bool:
        self.frames.append(frame)
        return True

    def drain_available(self) -> None:
        self.drain_called = True
        raise AssertionError("FramePump must not drain botified stdout")


def parse_botified_payload(frame: str) -> dict[str, Any]:
    assert frame.startswith("<botified>")
    assert frame.endswith("</botified>")
    return json.loads(frame[len("<botified>") : -len("</botified>")])


def service_result(visual_state: dict[str, Any] | None = None, error: Any = None) -> Any:
    return SimpleNamespace(visual_state=visual_state, error=error)


def make_frame(module: Any, *, timestamp_ms: int, width: int = 1280, height: int = 720) -> Any:
    return module.InputFrame(
        camera="front",
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        jpeg=JPEG_1280X720,
    )


def make_pump(
    module: Any,
    *,
    slot: Any,
    service: FakeServiceClient,
    gaze: FakeGazePublisher | None = None,
    botified: FakeBotifiedWriter | None = None,
    head_motion: Any | None = None,
    stale_after_ms: int = 250,
    clock_ms: Any | None = None,
) -> Any:
    current_head_motion = head_motion or module.HeadMotion(
        state="stationary",
        yaw_vel_rad_s=0.01,
        pitch_vel_rad_s=0.02,
    )
    return module.FramePump(
        latest_frame_slot=slot,
        service_client=service,
        gaze_publisher=gaze or FakeGazePublisher(),
        head_motion_provider=lambda: current_head_motion,
        botified_writer=botified,
        stale_after_ms=stale_after_ms,
        clock_ms=clock_ms,
    )


@pytest.mark.asyncio
async def test_latest_slot_keeps_only_newest_frame_while_request_is_in_flight():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    first_response: asyncio.Future[Any] = asyncio.Future()
    service = FakeServiceClient(
        [
            first_response,
            service_result(visual_state_lost(frame_id=2, timestamp_ms=1300)),
        ]
    )
    pump = make_pump(module, slot=slot, service=service)

    slot.push(make_frame(module, timestamp_ms=1000))
    in_flight = asyncio.create_task(pump.process_one(now_ms=1100))
    await service.wait_for_requests(1)

    slot.push(make_frame(module, timestamp_ms=1200))
    slot.push(make_frame(module, timestamp_ms=1300))
    first_response.set_result(service_result(visual_state_lost(frame_id=1, timestamp_ms=1000)))
    await in_flight
    await pump.process_one(now_ms=1400)

    sent_timestamps = [header["timestamp_ms"] for header, _jpeg in service.requests]
    assert sent_timestamps == [1000, 1300]


@pytest.mark.asyncio
async def test_process_one_sends_transport_header_with_frame_metadata_and_head_motion():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    service = FakeServiceClient([service_result(visual_state_lost(frame_id=1))])
    head_motion = module.HeadMotion(
        state="moving",
        yaw_vel_rad_s=0.125,
        pitch_vel_rad_s=-0.25,
    )
    pump = make_pump(module, slot=slot, service=service, head_motion=head_motion)

    slot.push(make_frame(module, timestamp_ms=1710000000000, width=1280, height=720))
    await pump.process_one(now_ms=1710000000082)

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
                    "yaw_vel_rad_s": 0.125,
                    "pitch_vel_rad_s": -0.25,
                },
            },
            JPEG_1280X720,
        )
    ]


@pytest.mark.asyncio
async def test_fresh_visual_state_without_attention_publishes_lost_invalid_gaze():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    gaze = FakeGazePublisher()
    state = visual_state_lost(frame_id=1, timestamp_ms=1710000000000)
    service = FakeServiceClient([service_result(state)])
    pump = make_pump(
        module,
        slot=slot,
        service=service,
        gaze=gaze,
        clock_ms=lambda: 1710000000082,
    )

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000082)

    expected = map_visual_state_to_gaze_target(
        state,
        publish_timestamp_ms=1710000000082,
        stale_after_ms=250,
    ).to_dict()
    assert gaze.dicts() == [expected]
    assert gaze.dicts()[0]["valid"] is False
    assert gaze.dicts()[0]["state"] == "lost"


@pytest.mark.asyncio
async def test_process_one_uses_clock_after_service_response_for_publish_timestamp():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    gaze = FakeGazePublisher()
    pending_response: asyncio.Future[Any] = asyncio.Future()
    service = FakeServiceClient([pending_response])
    current_clock_ms = 1710000000082
    pump = make_pump(
        module,
        slot=slot,
        service=service,
        gaze=gaze,
        clock_ms=lambda: current_clock_ms,
    )
    state = visual_state_lost(frame_id=1, timestamp_ms=1710000000000)

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    in_flight = asyncio.create_task(pump.process_one(now_ms=1710000000082))
    await service.wait_for_requests(1)

    current_clock_ms = 1710000000900
    pending_response.set_result(service_result(state))
    await in_flight

    expected = map_visual_state_to_gaze_target(
        state,
        publish_timestamp_ms=1710000000900,
        stale_after_ms=250,
    ).to_dict()
    assert gaze.dicts() == [expected]


@pytest.mark.asyncio
async def test_semantic_events_are_written_as_botified_frames_except_attention_changes():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    botified = FakeBotifiedWriter()
    allowed = semantic_event(event_id="front:evt_000456", event="person_waving")
    attention_change = semantic_event(
        event_id="front:evt_attn_001",
        event="attention_target_changed",
    )
    state = load_visual_state_tracking(
        frame_id=1,
        frame_timestamp_ms=1710000000000,
        semantic_events=[allowed, attention_change],
    )
    service = FakeServiceClient([service_result(state)])
    pump = make_pump(module, slot=slot, service=service, botified=botified)

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000082)

    assert len(botified.frames) == 1
    payload = parse_botified_payload(botified.frames[0])
    assert payload["id"] == f"visual:{allowed['event_id']}"
    assert "visual_context=" in payload["request"]


@pytest.mark.asyncio
async def test_frame_pump_enqueues_botified_frames_without_draining_writer():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    botified = FakeEnqueueBotifiedWriter()
    allowed = semantic_event(event_id="front:evt_000456", event="person_waving")
    state = load_visual_state_tracking(
        frame_id=1,
        frame_timestamp_ms=1710000000000,
        semantic_events=[allowed],
    )
    service = FakeServiceClient([service_result(state)])
    pump = make_pump(module, slot=slot, service=service, botified=botified)

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000082)

    assert len(botified.frames) == 1
    payload = parse_botified_payload(botified.frames[0])
    assert payload["id"] == f"visual:{allowed['event_id']}"
    assert "visual_context=" in payload["request"]
    assert botified.drain_called is False


@pytest.mark.asyncio
async def test_publish_stale_now_uses_last_metadata_and_does_not_duplicate_until_fresh_state_advances():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    gaze = FakeGazePublisher()
    first_state = load_visual_state_tracking(frame_id=1)
    second_state = load_visual_state_tracking(
        frame_id=2,
        frame_timestamp_ms=1710000000100,
    )
    service = FakeServiceClient(
        [
            service_result(first_state),
            service_result(second_state),
        ]
    )
    pump = make_pump(module, slot=slot, service=service, gaze=gaze)

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000082)
    pump.publish_stale_now(publish_timestamp_ms=1710000000300)
    pump.publish_stale_now(publish_timestamp_ms=1710000000400)
    slot.push(make_frame(module, timestamp_ms=1710000000100))
    await pump.process_one(now_ms=1710000000182)
    pump.publish_stale_now(publish_timestamp_ms=1710000000500)

    stale_payloads = [payload for payload in gaze.dicts() if payload["state"] == "stale"]
    assert stale_payloads == [
        make_invalid_gaze_target(
            "stale",
            camera="front",
            frame_id=1,
            frame_timestamp_ms=1710000000000,
            image_size=(1280, 720),
            publish_timestamp_ms=1710000000300,
            stale_after_ms=250,
        ).to_dict(),
        make_invalid_gaze_target(
            "stale",
            camera="front",
            frame_id=2,
            frame_timestamp_ms=1710000000100,
            image_size=(1280, 720),
            publish_timestamp_ms=1710000000500,
            stale_after_ms=250,
        ).to_dict(),
    ]


@pytest.mark.asyncio
async def test_stale_deadline_publishes_at_exact_boundary_after_fresh_target():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    gaze = FakeGazePublisher()
    fresh_publish_ms = 1710000000082
    state = load_visual_state_tracking(
        frame_id=1,
        frame_timestamp_ms=1710000000000,
    )
    service = FakeServiceClient([service_result(state)])
    pump = make_pump(
        module,
        slot=slot,
        service=service,
        gaze=gaze,
        stale_after_ms=250,
        clock_ms=lambda: fresh_publish_ms,
    )

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000000)

    assert pump.check_stale_deadline(now_ms=fresh_publish_ms + 250) is True
    assert gaze.dicts()[-1] == make_invalid_gaze_target(
        "stale",
        camera="front",
        frame_id=1,
        frame_timestamp_ms=1710000000000,
        image_size=(1280, 720),
        publish_timestamp_ms=fresh_publish_ms + 250,
        stale_after_ms=250,
    ).to_dict()


@pytest.mark.asyncio
async def test_first_pending_frame_can_publish_stale_from_sent_frame_metadata():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    gaze = FakeGazePublisher()
    pending_response: asyncio.Future[Any] = asyncio.Future()
    service = FakeServiceClient([pending_response])
    pump = make_pump(module, slot=slot, service=service, gaze=gaze, stale_after_ms=250)

    slot.push(
        make_frame(
            module,
            timestamp_ms=1710000000000,
            width=640,
            height=360,
        )
    )
    in_flight = asyncio.create_task(pump.process_one(now_ms=1710000000100))
    await service.wait_for_requests(1)

    assert pump.check_stale_deadline(now_ms=1710000000351) is True

    assert gaze.dicts() == [
        make_invalid_gaze_target(
            "stale",
            camera="front",
            frame_id=1,
            frame_timestamp_ms=1710000000000,
            image_size=(640, 360),
            publish_timestamp_ms=1710000000351,
            stale_after_ms=250,
        ).to_dict()
    ]
    assert in_flight.done() is False

    pending_response.set_result(
        service_result(visual_state_lost(frame_id=1, timestamp_ms=1710000000000))
    )
    await in_flight


@pytest.mark.asyncio
async def test_successful_response_clears_pending_sent_metadata_before_next_stale_deadline():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    gaze = FakeGazePublisher()
    pending_response: asyncio.Future[Any] = asyncio.Future()
    service = FakeServiceClient([pending_response])
    current_clock_ms = 1710000001000
    pump = make_pump(
        module,
        slot=slot,
        service=service,
        gaze=gaze,
        stale_after_ms=250,
        clock_ms=lambda: current_clock_ms,
    )

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    in_flight = asyncio.create_task(pump.process_one(now_ms=1710000000100))
    await service.wait_for_requests(1)

    assert pump.check_stale_deadline(now_ms=1710000000351) is True

    pending_response.set_result(
        service_result(visual_state_lost(frame_id=1, timestamp_ms=1710000000000))
    )
    await in_flight

    assert pump._last_sent_metadata is None
    assert pump._last_sent_timestamp_ms is None
    assert pump.check_stale_deadline(now_ms=1710000001100) is False

    stale_payloads = [payload for payload in gaze.dicts() if payload["state"] == "stale"]
    assert len(stale_payloads) == 1


@pytest.mark.asyncio
async def test_stale_watchdog_can_publish_while_service_request_is_still_in_flight():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    gaze = FakeGazePublisher()
    pending_response: asyncio.Future[Any] = asyncio.Future()
    service = FakeServiceClient(
        [
            service_result(load_visual_state_tracking(frame_id=1)),
            pending_response,
        ]
    )
    pump = make_pump(
        module,
        slot=slot,
        service=service,
        gaze=gaze,
        stale_after_ms=250,
        clock_ms=lambda: 1710000000000,
    )

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000000)
    slot.push(make_frame(module, timestamp_ms=1710000000100))
    in_flight = asyncio.create_task(pump.process_one(now_ms=1710000000100))
    await service.wait_for_requests(2)

    pump.check_stale_deadline(now_ms=1710000000251)

    stale_payloads = [payload for payload in gaze.dicts() if payload["state"] == "stale"]
    assert len(stale_payloads) == 1
    assert stale_payloads[0]["frame_id"] == 1
    assert in_flight.done() is False

    pending_response.set_result(service_result(visual_state_lost(frame_id=2)))
    await in_flight
