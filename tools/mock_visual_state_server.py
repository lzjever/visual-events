from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from visual_events_server.protocol import (
    SCHEMA_VERSION,
    FrameMessage,
    ProtocolError,
    decode_frame_message,
    serialize_error,
    serialize_json_message,
    serialize_protocol_error,
)


_PROFILES = ("tracking", "lost", "event")
_TRACK_ID = 7


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mock-visual-state-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8767, type=int)
    parser.add_argument("--profile", default="tracking", choices=_PROFILES)
    parser.add_argument("--delay-ms", default=0, type=_non_negative_int)
    parser.add_argument("--disconnect-after", type=_positive_int)
    return parser.parse_args(argv)


def create_app(config: argparse.Namespace | None = None) -> FastAPI:
    config = config or parse_args([])
    app = FastAPI(title="mock-visual-state-server", version="0.1.0")
    app.state.config = config

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "profile": config.profile, "pid": os.getpid()}

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        stream_camera: str | None = None
        legal_frame_count = 0
        response_sequence_index = 0
        event_emitted = False

        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                return

            if message["type"] == "websocket.disconnect":
                return

            payload = message.get("bytes")
            if payload is None:
                await websocket.send_text(
                    serialize_error(
                        code="invalid_frame",
                        message="client message must be a binary frame",
                        retryable=True,
                    )
                )
                continue

            try:
                frame = decode_frame_message(payload)
            except ProtocolError as exc:
                await websocket.send_text(serialize_protocol_error(exc))
                continue

            if stream_camera is None:
                stream_camera = frame.camera
            elif frame.camera != stream_camera:
                await websocket.send_text(
                    serialize_error(
                        code="invalid_header",
                        message="camera cannot change within a WebSocket connection",
                        frame_id=frame.frame_id,
                        retryable=False,
                    )
                )
                await websocket.close(code=1008)
                return

            legal_frame_count += 1
            if (
                config.disconnect_after is not None
                and legal_frame_count >= config.disconnect_after
            ):
                await websocket.close()
                return

            response = build_visual_state(
                frame,
                config.profile,
                sequence_index=response_sequence_index,
                event_emitted=event_emitted,
                delay_ms=config.delay_ms,
            )
            if config.delay_ms:
                await asyncio.sleep(config.delay_ms / 1000.0)
            await websocket.send_text(serialize_json_message(response))

            if response["semantic_events"]:
                event_emitted = True
            response_sequence_index += 1

    return app


def build_visual_state(
    frame: FrameMessage,
    profile: str,
    *,
    sequence_index: int,
    event_emitted: bool,
    delay_ms: int = 0,
) -> dict[str, Any]:
    if profile not in _PROFILES:
        raise ValueError(f"unknown mock visual_state profile: {profile}")

    tracks: list[dict[str, Any]] = []
    attention: dict[str, Any] | None = None
    scene_flags = {
        "has_person": False,
        "person_count": 0,
        "largest_person_stable": False,
        "someone_near_center": False,
    }
    semantic_events: list[dict[str, Any]] = []

    if profile in {"tracking", "event"}:
        track = _person_track(frame, sequence_index=sequence_index)
        tracks = [track]
        attention = {
            "target_track_id": _TRACK_ID,
            "target_uv": track["head_uv"],
            "reason": "largest_stable_person",
            "confidence": track["confidence"],
        }
        scene_flags = {
            "has_person": True,
            "person_count": 1,
            "largest_person_stable": True,
            "someone_near_center": True,
        }
        if profile == "event" and not event_emitted:
            semantic_events = [_person_waving_event(frame, track)]

    return {
        "type": "visual_state",
        "schema_version": SCHEMA_VERSION,
        "camera": frame.camera,
        "frame_id": frame.frame_id,
        "frame_timestamp_ms": frame.timestamp_ms,
        "server_timestamp_ms": frame.timestamp_ms + delay_ms,
        "image_size": [frame.width, frame.height],
        "tracks": tracks,
        "attention": attention,
        "scene_flags": scene_flags,
        "semantic_events": semantic_events,
    }


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    uvicorn.run(create_app(config), host=config.host, port=config.port)


def _non_negative_int(value: str) -> int:
    parsed = _parse_int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _parse_int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc


def _person_track(
    frame: FrameMessage,
    *,
    sequence_index: int,
) -> dict[str, Any]:
    width = float(frame.width)
    height = float(frame.height)
    bbox = [
        round(width * 0.25, 1),
        round(height * 0.1667, 1),
        round(width * 0.40625, 1),
        round(height * 0.8333, 1),
    ]
    center_uv = [
        round((bbox[0] + bbox[2]) / 2.0, 1),
        round((bbox[1] + bbox[3]) / 2.0, 1),
    ]
    head_uv = [
        min(width, max(0.0, round(center_uv[0] + 1.0, 1))),
        min(height, max(0.0, round(height * 0.2847, 1))),
    ]
    bbox_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    image_area = max(1.0, width * height)

    return {
        "track_id": _TRACK_ID,
        "class": "person",
        "bbox_xyxy": bbox,
        "bbox_area_ratio": round(bbox_area / image_area, 4),
        "center_uv": center_uv,
        "head_uv": head_uv,
        "velocity_uv_s": [0.0, 0.0],
        "age_ms": 1000 + sequence_index * 33,
        "lost_ms": 0,
        "confidence": 0.86,
        "pose_confidence": 0.72,
    }


def _person_waving_event(
    frame: FrameMessage,
    track: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "semantic_event",
        "event_id": f"{frame.camera}:mock_evt_000001",
        "event": "person_waving",
        "camera": frame.camera,
        "track_id": track["track_id"],
        "confidence": track["confidence"],
        "duration_ms": 900,
        "text": "person waving",
    }


if __name__ == "__main__":
    main()
