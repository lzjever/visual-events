from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

import pytest

from tests.jpeg_fixtures import JPEG_1280X720


def import_service_client():
    try:
        import visual_events_cli.service_client as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.service_client module: {exc}")
    return module


def frame_header(**overrides: Any) -> dict[str, Any]:
    header = {
        "type": "frame",
        "schema_version": 1,
        "camera": "front",
        "frame_id": 7,
        "timestamp_ms": 1710000000000,
        "encoding": "jpeg",
        "width": 1280,
        "height": 720,
        "head_motion": {"state": "stationary"},
    }
    header.update(overrides)
    return header


def visual_state(**overrides: Any) -> dict[str, Any]:
    state = {
        "type": "visual_state",
        "schema_version": 1,
        "camera": "front",
        "frame_id": 7,
        "frame_timestamp_ms": 1710000000000,
        "server_timestamp_ms": 1710000000082,
        "image_size": [1280, 720],
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
    state.update(overrides)
    return state


def error_response(**overrides: Any) -> dict[str, Any]:
    error = {
        "type": "error",
        "schema_version": 1,
        "frame_id": 7,
        "code": "invalid_frame",
        "message": "jpeg payload is invalid",
        "retryable": True,
    }
    error.update(overrides)
    return error


class FakeWebSocket:
    def __init__(
        self,
        responses: list[dict[str, Any] | str | bytes | asyncio.Future[Any]] | None = None,
    ):
        self.sent: list[bytes] = []
        self.closed = False
        self._responses = list(responses or [])

    async def send(self, message: bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        if not self._responses:
            future: asyncio.Future[str] = asyncio.Future()
            self._responses.append(future)

        response = self._responses.pop(0)
        if isinstance(response, asyncio.Future):
            return await response
        if isinstance(response, (str, bytes)):
            return response
        return json.dumps(response, separators=(",", ":"))

    async def close(self) -> None:
        self.closed = True


class FakeConnect:
    def __init__(self, *websockets: FakeWebSocket):
        self.websockets = list(websockets)
        self.urls: list[str] = []

    async def __call__(self, url: str) -> FakeWebSocket:
        self.urls.append(url)
        return self.websockets.pop(0)


class RaisingConnect:
    def __init__(self, exc: Exception):
        self.exc = exc
        self.urls: list[str] = []

    async def __call__(self, url: str) -> FakeWebSocket:
        self.urls.append(url)
        raise self.exc


class SendFailingWebSocket(FakeWebSocket):
    async def send(self, message: bytes) -> None:
        self.sent.append(message)
        raise OSError("send failed")


class RecordingPostJson:
    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, Any], float]] = []

    def __call__(self, url: str, payload: dict[str, Any], timeout_s: float) -> Any:
        self.requests.append((url, payload, timeout_s))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_pack_frame_message_uses_uint32_be_header_len_json_and_jpeg_roundtrip():
    module = import_service_client()
    header = frame_header()

    message = module.pack_frame_message(header, JPEG_1280X720)
    header_len = struct.unpack(">I", message[:4])[0]
    header_json = message[4 : 4 + header_len]

    assert json.loads(header_json.decode("utf-8")) == header
    assert message[4 + header_len :] == JPEG_1280X720
    assert module.unpack_frame_message(message) == (header, JPEG_1280X720)


@pytest.mark.asyncio
async def test_identify_current_posts_to_http_origin_with_agent_payload_and_returns_dict():
    module = import_service_client()
    response = {"ok": True, "status": "identified", "people": []}
    post_json = RecordingPostJson(response)
    client = module.VisualEventsServiceClient(
        "ws://service.local:8765/v1/stream",
        post_json=post_json,
    )

    result = await client.identify_current(
        camera="front",
        stream_ref="ws_abc",
        timeout_ms=750,
    )

    assert result == response
    assert post_json.requests[0][:2] == (
        "http://service.local:8765/v1/memory/identify-current",
        {
            "camera": "front",
            "stream_ref": "ws_abc",
            "target": {
                "kind": "person",
                "intent": "identify_current",
                "referent_text": "当前这个人",
            },
            "scope": "active_target",
            "timeout_ms": 750,
        },
    )
    assert post_json.requests[0][2] > 0


@pytest.mark.asyncio
async def test_identify_current_caps_timeout_to_server_schema_max():
    module = import_service_client()
    post_json = RecordingPostJson({"ok": True, "status": "identified"})
    client = module.VisualEventsServiceClient(
        "ws://service.local:8765/v1/stream",
        post_json=post_json,
    )

    await client.identify_current(
        camera="front",
        stream_ref="ws_abc",
        timeout_ms=1500,
    )

    assert post_json.requests[0][1]["timeout_ms"] == 1000


@pytest.mark.asyncio
async def test_teach_person_posts_to_https_origin_with_profile_and_target():
    module = import_service_client()
    response = {"ok": True, "person_id": "person_000001"}
    post_json = RecordingPostJson(response)
    client = module.VisualEventsServiceClient(
        "wss://memory.example.test/ws",
        post_json=post_json,
    )
    target = {
        "kind": "person",
        "intent": "third_person_introduction",
        "referent_text": "左边的人",
    }
    profile = {"display_name": "张三"}

    result = await client.teach_person(
        camera="front",
        stream_ref="ws_abc",
        profile=profile,
        target=target,
    )

    assert result == response
    assert post_json.requests[0][:2] == (
        "https://memory.example.test/v1/memory/teach/person",
        {
            "camera": "front",
            "stream_ref": "ws_abc",
            "target": target,
            "profile": profile,
        },
    )
    assert post_json.requests[0][2] > 0


@pytest.mark.asyncio
async def test_memory_post_timeout_and_non_dict_response_return_business_dicts():
    module = import_service_client()
    post_json = RecordingPostJson(TimeoutError("timed out"), ["not", "a", "dict"])
    client = module.VisualEventsServiceClient(
        "ws://service.local/v1/stream",
        post_json=post_json,
    )

    timeout = await client.identify_current(camera="front", stream_ref="ws_abc")
    non_dict = await client.identify_current(camera="front", stream_ref="ws_abc")

    assert timeout["ok"] is False
    assert timeout["status"] == "timeout"
    assert non_dict["ok"] is False
    assert non_dict["status"] == "invalid_response"


@pytest.mark.asyncio
async def test_visual_state_response_with_matching_frame_id_returns_result():
    module = import_service_client()
    websocket = FakeWebSocket([visual_state(frame_id=7)])
    connect = FakeConnect(websocket)
    client = module.VisualEventsServiceClient(
        "ws://service/v1/stream",
        response_timeout_ms=1000,
        connect=connect,
    )

    result = await client.request_frame(frame_header(frame_id=7), JPEG_1280X720)

    assert connect.urls == ["ws://service/v1/stream"]
    assert result.visual_state == visual_state(frame_id=7)
    assert result.error is None
    assert websocket.closed is False


@pytest.mark.asyncio
async def test_retryable_error_response_returns_error_and_keeps_connection_reusable():
    module = import_service_client()
    websocket = FakeWebSocket(
        [
            error_response(frame_id=7, retryable=True),
            visual_state(frame_id=8),
        ]
    )
    connect = FakeConnect(websocket)
    client = module.VisualEventsServiceClient(
        "ws://service/v1/stream",
        response_timeout_ms=1000,
        connect=connect,
    )

    error_result = await client.request_frame(frame_header(frame_id=7), JPEG_1280X720)
    ok_result = await client.request_frame(frame_header(frame_id=8), JPEG_1280X720)

    assert error_result.visual_state is None
    assert error_result.error == module.ServiceError(
        code="invalid_frame",
        message="jpeg payload is invalid",
        frame_id=7,
        retryable=True,
    )
    assert ok_result.visual_state == visual_state(frame_id=8)
    assert ok_result.error is None
    assert connect.urls == ["ws://service/v1/stream"]
    assert websocket.closed is False


@pytest.mark.asyncio
async def test_close_closes_cached_websocket_and_next_request_reconnects():
    module = import_service_client()
    first_ws = FakeWebSocket([visual_state(frame_id=7)])
    second_ws = FakeWebSocket([visual_state(frame_id=8)])
    connect = FakeConnect(first_ws, second_ws)
    client = module.VisualEventsServiceClient(
        "ws://service/v1/stream",
        response_timeout_ms=1000,
        connect=connect,
    )

    first_result = await client.request_frame(frame_header(frame_id=7), JPEG_1280X720)
    await client.close()
    await client.close()
    second_result = await client.request_frame(frame_header(frame_id=8), JPEG_1280X720)

    assert first_result.visual_state == visual_state(frame_id=7)
    assert first_ws.closed is True
    assert second_result.visual_state == visual_state(frame_id=8)
    assert second_ws.closed is False
    assert connect.urls == ["ws://service/v1/stream", "ws://service/v1/stream"]


@pytest.mark.asyncio
async def test_non_retryable_error_closes_connection_and_next_request_reconnects():
    module = import_service_client()
    first_ws = FakeWebSocket(
        [
            error_response(
                frame_id=7,
                code="invalid_header",
                message="camera switch is not allowed",
                retryable=False,
            )
        ]
    )
    second_ws = FakeWebSocket([visual_state(frame_id=8)])
    connect = FakeConnect(first_ws, second_ws)
    client = module.VisualEventsServiceClient(
        "ws://service/v1/stream",
        response_timeout_ms=1000,
        connect=connect,
    )

    error_result = await client.request_frame(frame_header(frame_id=7), JPEG_1280X720)
    ok_result = await client.request_frame(frame_header(frame_id=8), JPEG_1280X720)

    assert error_result.error == module.ServiceError(
        code="invalid_header",
        message="camera switch is not allowed",
        frame_id=7,
        retryable=False,
    )
    assert first_ws.closed is True
    assert ok_result.visual_state == visual_state(frame_id=8)
    assert ok_result.error is None
    assert connect.urls == ["ws://service/v1/stream", "ws://service/v1/stream"]


@pytest.mark.asyncio
async def test_response_timeout_closes_connection_and_returns_retryable_timeout_error():
    module = import_service_client()
    websocket = FakeWebSocket()
    connect = FakeConnect(websocket)
    client = module.VisualEventsServiceClient(
        "ws://service/v1/stream",
        response_timeout_ms=1,
        connect=connect,
    )

    result = await client.request_frame(frame_header(frame_id=7), JPEG_1280X720)

    assert result.visual_state is None
    assert result.error == module.ServiceError(
        code="response_timeout",
        message="timed out waiting for visual_state",
        frame_id=7,
        retryable=True,
        protocol_error=False,
    )
    assert websocket.closed is True


@pytest.mark.asyncio
async def test_visual_state_frame_id_mismatch_closes_connection_and_returns_protocol_error():
    module = import_service_client()
    websocket = FakeWebSocket([visual_state(frame_id=999)])
    connect = FakeConnect(websocket)
    client = module.VisualEventsServiceClient(
        "ws://service/v1/stream",
        response_timeout_ms=1000,
        connect=connect,
    )

    result = await client.request_frame(frame_header(frame_id=7), JPEG_1280X720)

    assert result.visual_state is None
    assert result.error == module.ServiceError(
        code="frame_id_mismatch",
        message="visual_state frame_id does not match requested frame",
        frame_id=7,
        retryable=True,
        protocol_error=True,
    )
    assert websocket.closed is True


@pytest.mark.asyncio
async def test_connect_failure_returns_retryable_connection_error_with_requested_frame_id():
    module = import_service_client()
    connect = RaisingConnect(OSError("dial failed"))
    client = module.VisualEventsServiceClient(
        "ws://service/v1/stream",
        response_timeout_ms=1000,
        connect=connect,
    )

    result = await client.request_frame(frame_header(frame_id=42), JPEG_1280X720)

    assert result.visual_state is None
    assert result.error == module.ServiceError(
        code="connection_error",
        message="dial failed",
        frame_id=42,
        retryable=True,
        protocol_error=False,
    )
    assert connect.urls == ["ws://service/v1/stream"]


@pytest.mark.asyncio
async def test_send_failure_closes_connection_and_returns_retryable_connection_error():
    module = import_service_client()
    websocket = SendFailingWebSocket()
    connect = FakeConnect(websocket)
    client = module.VisualEventsServiceClient(
        "ws://service/v1/stream",
        response_timeout_ms=1000,
        connect=connect,
    )

    result = await client.request_frame(frame_header(frame_id=43), JPEG_1280X720)

    assert result.visual_state is None
    assert result.error == module.ServiceError(
        code="connection_error",
        message="send failed",
        frame_id=43,
        retryable=True,
        protocol_error=False,
    )
    assert websocket.closed is True


@pytest.mark.parametrize(
    "raw_response",
    [
        "{",
        "[]",
        b"\xff",
    ],
)
@pytest.mark.asyncio
async def test_malformed_response_closes_connection_and_returns_retryable_protocol_error(
    raw_response: str | bytes,
):
    module = import_service_client()
    websocket = FakeWebSocket([raw_response])
    connect = FakeConnect(websocket)
    client = module.VisualEventsServiceClient(
        "ws://service/v1/stream",
        response_timeout_ms=1000,
        connect=connect,
    )

    result = await client.request_frame(frame_header(frame_id=44), JPEG_1280X720)

    assert result.visual_state is None
    assert result.error is not None
    assert result.error.code == "invalid_response"
    assert result.error.frame_id == 44
    assert result.error.retryable is True
    assert result.error.protocol_error is True
    assert websocket.closed is True
