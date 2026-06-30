from __future__ import annotations

import asyncio
import json
import socket
import struct
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


class _WebSocket(Protocol):
    async def send(self, message: bytes) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


ConnectCallable = Callable[[str], Awaitable[_WebSocket]]
PostJsonCallable = Callable[[str, dict[str, Any], float], Any]


@dataclass(frozen=True)
class ServiceError:
    code: str
    message: str
    frame_id: int
    retryable: bool
    protocol_error: bool = False


@dataclass(frozen=True)
class ServiceResult:
    visual_state: dict[str, Any] | None = None
    error: ServiceError | None = None


def pack_frame_message(header: dict[str, Any], jpeg: bytes) -> bytes:
    header_json = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return struct.pack(">I", len(header_json)) + header_json + bytes(jpeg)


def unpack_frame_message(message: bytes) -> tuple[dict[str, Any], bytes]:
    if len(message) < 4:
        raise ValueError("frame message is missing header length")

    header_len = struct.unpack(">I", message[:4])[0]
    header_end = 4 + header_len
    if len(message) < header_end:
        raise ValueError("frame message header is truncated")

    header = json.loads(message[4:header_end].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("frame message header must be a JSON object")
    return header, message[header_end:]


class VisualEventsServiceClient:
    def __init__(
        self,
        url: str,
        response_timeout_ms: int = 1000,
        connect: ConnectCallable | None = None,
        post_json: PostJsonCallable | None = None,
    ) -> None:
        self._url = url
        self._response_timeout_s = max(0, int(response_timeout_ms)) / 1000.0
        self._connect = connect or _default_connect
        self._post_json = post_json or _default_post_json
        self._http_origin = _http_origin_from_service_url(url)
        self._websocket: _WebSocket | None = None

    async def request_frame(self, header: dict[str, Any], jpeg: bytes) -> ServiceResult:
        frame_id = _as_int(header.get("frame_id"), -1)
        try:
            websocket = await self._ensure_connection()
            await websocket.send(pack_frame_message(header, jpeg))
        except Exception as exc:
            await self._close_connection()
            return _error_result(
                code="connection_error",
                message=str(exc),
                frame_id=frame_id,
                retryable=True,
            )

        try:
            raw_response = await asyncio.wait_for(
                websocket.recv(),
                timeout=self._response_timeout_s,
            )
        except TimeoutError:
            await self._close_connection()
            return ServiceResult(
                error=ServiceError(
                    code="response_timeout",
                    message="timed out waiting for visual_state",
                    frame_id=frame_id,
                    retryable=True,
                    protocol_error=False,
                )
            )
        except Exception as exc:
            await self._close_connection()
            return ServiceResult(
                error=ServiceError(
                    code="connection_error",
                    message=str(exc),
                    frame_id=frame_id,
                    retryable=True,
                )
            )

        try:
            response = _decode_response(raw_response)
        except Exception as exc:
            await self._close_connection()
            return _error_result(
                code="invalid_response",
                message=str(exc),
                frame_id=frame_id,
                retryable=True,
                protocol_error=True,
            )

        response_type = response.get("type")
        if response_type == "visual_state":
            if _as_int(response.get("frame_id"), -1) != frame_id:
                await self._close_connection()
                return ServiceResult(
                    error=ServiceError(
                        code="frame_id_mismatch",
                        message="visual_state frame_id does not match requested frame",
                        frame_id=frame_id,
                        retryable=True,
                        protocol_error=True,
                    )
                )
            return ServiceResult(visual_state=response)

        if response_type == "error":
            retryable = bool(response.get("retryable"))
            error = ServiceError(
                code=str(response.get("code", "error")),
                message=str(response.get("message", "")),
                frame_id=_as_int(response.get("frame_id"), frame_id),
                retryable=retryable,
            )
            if not retryable:
                await self._close_connection()
            return ServiceResult(error=error)

        await self._close_connection()
        return ServiceResult(
            error=ServiceError(
                code="unexpected_response",
                message="unexpected service response type",
                frame_id=frame_id,
                retryable=True,
                protocol_error=True,
            )
        )

    async def _ensure_connection(self) -> _WebSocket:
        if self._websocket is None:
            self._websocket = await self._connect(self._url)
        return self._websocket

    async def close(self) -> None:
        await self._close_connection()

    async def _close_connection(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is not None:
            await websocket.close()

    async def identify_current(
        self,
        camera: str,
        stream_ref: str,
        timeout_ms: int = 500,
        target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_timeout_ms = min(1000, max(1, int(timeout_ms)))
        payload = {
            "camera": str(camera),
            "stream_ref": str(stream_ref),
            "target": dict(target or _default_identify_current_target()),
            "scope": "active_target",
            "timeout_ms": safe_timeout_ms,
        }
        return await self._post_memory_json(
            "/v1/memory/identify-current",
            payload,
            timeout_s=max(safe_timeout_ms / 1000.0, self._response_timeout_s or 0.0),
        )

    async def teach_person(
        self,
        camera: str,
        stream_ref: str,
        profile: dict[str, Any],
        target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "camera": str(camera),
            "stream_ref": str(stream_ref),
            "target": dict(target or _default_self_introduction_target()),
            "profile": dict(profile),
        }
        return await self._post_memory_json(
            "/v1/memory/teach/person",
            payload,
            timeout_s=self._response_timeout_s or 1.0,
        )

    async def _post_memory_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        url = f"{self._http_origin}{endpoint}"
        try:
            response = await asyncio.to_thread(
                self._post_json,
                url,
                payload,
                float(timeout_s),
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                return _memory_business_failure(
                    status="timeout",
                    reason="timeout",
                    message="timed out calling memory service",
                )
            return _memory_business_failure(
                status="request_failed",
                reason="connection_error",
                message=str(exc),
            )

        if not isinstance(response, dict):
            return _memory_business_failure(
                status="invalid_response",
                reason="non_dict_response",
                message="memory service returned a non-object response",
            )
        return response


async def _default_connect(url: str) -> _WebSocket:
    import websockets

    return await websockets.connect(url)


def _default_post_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> Any:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    request = urllib_request.Request(
        url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=float(timeout_s)) as response:
            raw_response = response.read()
    except urllib_error.HTTPError as exc:
        raw_response = exc.read()
        parsed_error = _decode_http_json(raw_response)
        if parsed_error is None:
            raise
        return _http_error_business_response(exc.code, parsed_error)

    return json.loads(raw_response.decode("utf-8"))


def _decode_http_json(raw_response: bytes) -> dict[str, Any] | None:
    try:
        response = json.loads(raw_response.decode("utf-8"))
    except Exception:
        return None
    if isinstance(response, dict):
        return response
    return None


def _http_error_business_response(
    status_code: int,
    response: dict[str, Any],
) -> dict[str, Any]:
    detail = response.get("detail")
    if isinstance(detail, dict):
        status = str(detail.get("code") or "http_error")
        message = str(detail.get("message") or "")
        result = {
            "ok": False,
            "status": status,
            "reason": status,
            "message": message,
            "http_status": int(status_code),
        }
        for key, value in detail.items():
            if key not in result:
                result[key] = value
        return result
    return {
        "ok": False,
        "status": "http_error",
        "reason": "http_error",
        "message": "memory service returned an HTTP error",
        "http_status": int(status_code),
    }


def _http_origin_from_service_url(url: str) -> str:
    parsed = urllib_parse.urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    if parsed.netloc:
        return urllib_parse.urlunsplit((scheme, parsed.netloc, "", "", "")).rstrip("/")
    return url.rstrip("/")


def _default_identify_current_target() -> dict[str, str]:
    return {
        "kind": "person",
        "intent": "identify_current",
        "referent_text": "当前这个人",
    }


def _default_self_introduction_target() -> dict[str, str]:
    return {
        "kind": "person",
        "intent": "self_introduction",
        "referent_text": "我",
    }


def _decode_response(raw_response: str | bytes) -> dict[str, Any]:
    if isinstance(raw_response, bytes):
        raw_response = raw_response.decode("utf-8")
    response = json.loads(raw_response)
    if not isinstance(response, dict):
        raise ValueError("service response must be a JSON object")
    return response


def _error_result(
    *,
    code: str,
    message: str,
    frame_id: int,
    retryable: bool,
    protocol_error: bool = False,
) -> ServiceResult:
    return ServiceResult(
        error=ServiceError(
            code=code,
            message=message,
            frame_id=frame_id,
            retryable=retryable,
            protocol_error=protocol_error,
        )
    )


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (TimeoutError, socket.timeout))


def _memory_business_failure(
    *,
    status: str,
    reason: str,
    message: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "status": status,
        "reason": reason,
        "message": message,
    }
