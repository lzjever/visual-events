from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol


class _WebSocket(Protocol):
    async def send(self, message: bytes) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


ConnectCallable = Callable[[str], Awaitable[_WebSocket]]


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
    ) -> None:
        self._url = url
        self._response_timeout_s = max(0, int(response_timeout_ms)) / 1000.0
        self._connect = connect or _default_connect
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

    async def _close_connection(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is not None:
            await websocket.close()


async def _default_connect(url: str) -> _WebSocket:
    import websockets

    return await websockets.connect(url)


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
