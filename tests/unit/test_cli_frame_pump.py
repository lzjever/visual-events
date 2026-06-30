from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from tests.unit.test_cli_botified_output import (
    assert_no_current_snapshot_forbidden_fields,
    familiar_unknown_identity_context,
    known_identity_context,
)
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


def known_person_event(*, event_id: str = "front:mem_evt_000001") -> dict[str, Any]:
    return {
        "type": "semantic_event",
        "event_id": event_id,
        "event": "known_person_present",
        "camera": "front",
        "track_id": 7,
        "confidence": 0.86,
        "duration_ms": 0,
        "lifecycle_state": "confirmed",
        "evidence": {
            "memory_match_id": "match_000001",
            "matched_type": "person",
            "matched_id": "person_000001",
            "embedding_id": "emb_face_000001",
            "match_type": "face",
            "match_score": 0.86,
            "top2_margin": 0.09,
            "source_target_mode": "track_id",
        },
        "memory_context": {
            "person": {
                "person_id": "person_000001",
                "display_name": "张三",
                "description": "店长，熟悉新品陈列和现场活动",
                "tags": ["staff", "manager"],
                "match_confidence": 0.86,
            }
        },
        "text": "看到已知人物：张三",
    }


class FakeServiceClient:
    def __init__(
        self,
        results: list[Any] | None = None,
        *,
        identify_responses: list[dict[str, Any]] | None = None,
        teach_responses: list[dict[str, Any]] | None = None,
    ):
        self.requests: list[tuple[dict[str, Any], bytes]] = []
        self.identify_requests: list[dict[str, Any]] = []
        self.teach_requests: list[dict[str, Any]] = []
        self._results = list(results or [])
        self._identify_responses = list(identify_responses or [])
        self._teach_responses = list(teach_responses or [])
        self._request_event = asyncio.Event()

    async def request_frame(self, header: dict[str, Any], jpeg: bytes) -> Any:
        self.requests.append((header, jpeg))
        self._request_event.set()
        result = self._results.pop(0)
        if isinstance(result, asyncio.Future):
            return await result
        return result

    async def identify_current(
        self,
        camera: str,
        stream_ref: str,
        timeout_ms: int = 500,
        target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.identify_requests.append(
            {
                "camera": camera,
                "stream_ref": stream_ref,
                "timeout_ms": timeout_ms,
                "target": target,
            }
        )
        return self._identify_responses.pop(0)

    async def teach_person(
        self,
        camera: str,
        stream_ref: str,
        profile: dict[str, Any],
        target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.teach_requests.append(
            {
                "camera": camera,
                "stream_ref": stream_ref,
                "profile": profile,
                "target": target,
            }
        )
        return self._teach_responses.pop(0)

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


def parse_visual_context(payload: dict[str, Any]) -> dict[str, Any]:
    marker = "visual_context="
    start = payload["request"].index(marker) + len(marker)
    wrapper, end = json.JSONDecoder().raw_decode(payload["request"][start:])
    assert payload["request"][start + end :].strip() == ""
    return wrapper["visual_context"]


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


def assert_no_active_memory_forbidden_fields(response: dict[str, Any]) -> None:
    serialized = json.dumps(response, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "stream_ref",
        "track_id",
        "bbox",
        "keypoints",
        "embedding",
        "embedding_id",
        "crop",
        "source_frame",
        "request_snapshot_ref",
        "evidence",
        "store_delta",
        "memory_match_id",
        "source_target_mode",
        "runtime_person_slot",
    ):
        assert forbidden not in serialized


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
async def test_identify_current_without_latest_visual_state_returns_business_failure():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    service = FakeServiceClient([])
    pump = make_pump(module, slot=slot, service=service)

    response = await pump.identify_current()

    assert response["ok"] is False
    assert response["status"] == "no_active_frame"
    assert service.identify_requests == []


@pytest.mark.asyncio
async def test_identify_current_without_latest_stream_ref_returns_business_failure():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    state = visual_state_lost(frame_id=1)
    state.pop("stream_ref", None)
    service = FakeServiceClient([service_result(state)])
    pump = make_pump(module, slot=slot, service=service)

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000082)
    response = await pump.identify_current()

    assert response["ok"] is False
    assert response["status"] == "no_latest_stream_ref"
    assert service.identify_requests == []


@pytest.mark.asyncio
async def test_identify_current_uses_latest_camera_stream_ref_and_redacts_response():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    state = load_visual_state_tracking(
        frame_id=1,
        frame_timestamp_ms=1710000000000,
        camera="front",
        stream_ref="private/state-stream",
    )
    service_response = {
        "ok": True,
        "status": "identified",
        "people": [
            {
                "target_ref": "current:front:active_target",
                "track_id": 7,
                "stream_ref": "private/track-stream",
                "memory_match_id": "match_000001",
                "source_target_mode": "track_id",
                "runtime_person_slot": 3,
                "identity_context": {
                    "status": "known_person",
                    "display_name": "张三",
                    "embedding_id": "emb_face_000001",
                    "embedding": [0.1, 0.2],
                },
            }
        ],
        "evidence": {
            "source_frame": {"request_snapshot_ref": "private/snapshot"},
            "bbox_xyxy": [1, 2, 3, 4],
        },
        "store_delta": {"person": 1},
    }
    service = FakeServiceClient(
        [service_result(state)],
        identify_responses=[service_response],
    )
    pump = make_pump(module, slot=slot, service=service)

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000082)
    response = await pump.identify_current(timeout_ms=650)

    assert service.identify_requests == [
        {
            "camera": "front",
            "stream_ref": "private/state-stream",
            "timeout_ms": 650,
            "target": None,
        }
    ]
    assert response == {
        "ok": True,
        "status": "identified",
        "people": [
            {
                "target_ref": "current:front:active_target",
                "identity_context": {
                    "status": "known_person",
                    "display_name": "张三",
                },
            }
        ],
    }
    assert_no_active_memory_forbidden_fields(response)


@pytest.mark.asyncio
async def test_teach_person_uses_self_introduction_default_or_explicit_target_and_redacts():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    state = load_visual_state_tracking(
        frame_id=1,
        frame_timestamp_ms=1710000000000,
        camera="front",
        stream_ref="private/state-stream",
    )
    leaky_response = {
        "ok": True,
        "person_id": "person_000001",
        "person": {
            "display_name": "张三",
            "source_frame_ref": "private/frame",
            "memory_match_id": "match_000001",
            "embedding_id": "emb_face_000001",
            "source_target_mode": "track_id",
            "runtime_person_slot": 3,
        },
        "evidence": {"crop_ref": "private/crop.jpg"},
        "store_delta": {"person": 1},
    }
    service = FakeServiceClient(
        [service_result(state)],
        teach_responses=[leaky_response, leaky_response],
    )
    pump = make_pump(module, slot=slot, service=service)
    explicit_target = {
        "kind": "person",
        "intent": "third_person_introduction",
        "referent_text": "左边的人",
    }

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000082)
    default_response = await pump.teach_person({"display_name": "张三"})
    explicit_response = await pump.teach_person(
        {"display_name": "李四"},
        target=explicit_target,
    )

    assert service.teach_requests == [
        {
            "camera": "front",
            "stream_ref": "private/state-stream",
            "profile": {"display_name": "张三"},
            "target": {
                "kind": "person",
                "intent": "self_introduction",
                "referent_text": "我",
            },
        },
        {
            "camera": "front",
            "stream_ref": "private/state-stream",
            "profile": {"display_name": "李四"},
            "target": explicit_target,
        },
    ]
    assert default_response == {
        "ok": True,
        "person_id": "person_000001",
        "person": {"display_name": "张三"},
    }
    assert explicit_response == default_response
    assert_no_active_memory_forbidden_fields(default_response)
    assert_no_active_memory_forbidden_fields(explicit_response)


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
async def test_current_visual_snapshot_uses_latest_service_visual_state_summary():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    event = semantic_event(event_id="front:evt_snapshot", event="person_waving")
    event["stream_ref"] = "private/event-stream"
    event["source_frame"] = {"request_snapshot_ref": "private/snapshot-ref"}
    state = load_visual_state_tracking(
        frame_id=1,
        frame_timestamp_ms=1710000000000,
        stream_ref="private/state-stream",
        identity_context={
            "overlay_status": "ready",
            "active_target": {"track_id": 8, "stream_ref": "private/target-stream"},
            "tracks": [
                {
                    "track_id": 7,
                    "identity": known_identity_context(),
                    "stream_ref": "private/identity-stream-7",
                },
                {
                    "track_id": 8,
                    "identity": familiar_unknown_identity_context(),
                    "stream_ref": "private/identity-stream-8",
                },
            ],
            "stream_ref": "private/overlay-stream",
        },
        tracks=[
            {
                "track_id": 7,
                "class": "person",
                "bbox_xyxy": [420.0, 90.0, 780.0, 690.0],
                "bbox_area_ratio": 0.24,
                "center_uv": [600.0, 390.0],
                "head_uv": [602.0, 180.0],
                "lost_ms": 0,
                "keypoints": [{"name": "nose"}],
                "crop_ref": "private/crop-7.jpg",
                "stream_ref": "private/track-stream-7",
            },
            {
                "track_id": 8,
                "class": "person",
                "bbox_xyxy": [860.0, 160.0, 1030.0, 610.0],
                "bbox_area_ratio": 0.083,
                "center_uv": [945.0, 385.0],
                "head_uv": [946.0, 230.0],
                "lost_ms": 0,
                "embedding": [0.1, 0.2],
                "stream_ref": "private/track-stream-8",
            },
        ],
        semantic_events=[event],
    )
    service = FakeServiceClient([service_result(state)])
    pump = make_pump(module, slot=slot, service=service)

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000082)

    snapshot = pump.current_visual_snapshot(now_ms=1710000000100)

    assert snapshot["type"] == "current_visual_snapshot"
    assert snapshot["overlay_status"] == "ready"
    assert snapshot["active_target_ref"] == "current:front:person:1"
    assert [person["target_ref"] for person in snapshot["people"]] == [
        "current:front:person:0",
        "current:front:person:1",
    ]
    assert snapshot["events"] == [
        {
            "event": "person_waving",
            "target_ref": "current:front:person:0",
            "confidence": 0.86,
            "identity_context": snapshot["people"][0]["identity_context"],
        }
    ]
    assert_no_current_snapshot_forbidden_fields(snapshot)


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
async def test_memory_events_are_written_with_projected_memory_context():
    module = import_frame_pump()
    slot = module.LatestFrameSlot()
    botified = FakeBotifiedWriter()
    event = known_person_event()
    state = load_visual_state_tracking(
        frame_id=1,
        frame_timestamp_ms=1710000000000,
        semantic_events=[event],
    )
    service = FakeServiceClient([service_result(state)])
    pump = make_pump(module, slot=slot, service=service, botified=botified)

    slot.push(make_frame(module, timestamp_ms=1710000000000))
    await pump.process_one(now_ms=1710000000082)

    assert len(botified.frames) == 1
    payload = parse_botified_payload(botified.frames[0])
    assert payload["id"] == "visual:front:mem_evt_000001"
    context = parse_visual_context(payload)
    assert context["memory_context"]["person"]["person_id"] == "person_000001"
    assert context["memory_context"]["person"]["display_name"] == "张三"
    assert set(payload) == {"id", "urgency", "timeout_secs", "request", "expect"}


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
