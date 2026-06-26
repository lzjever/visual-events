from __future__ import annotations

import argparse
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .config import ServerConfig, load_config
from .inference.factory import create_infer_backend
from .processor import (
    BackendVisualFrameProcessor,
    MockVisualFrameProcessor,
    VisualFrameProcessor,
    VisualStreamSessionFactory,
)
from .protocol import (
    ProtocolError,
    decode_frame_message,
    serialize_error,
    serialize_json_message,
    serialize_protocol_error,
)


def create_app(
    *,
    processor: VisualFrameProcessor | None = None,
    session_factory: VisualStreamSessionFactory | None = None,
    config: ServerConfig | None = None,
) -> FastAPI:
    app = FastAPI(title="visual-events-server", version="0.1.0")
    app.state.config = config or ServerConfig()
    app.state.session_factory = session_factory or _session_factory_from_processor(
        processor or MockVisualFrameProcessor()
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        stream_camera: str | None = None
        processor_session = app.state.session_factory()
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
                response = await processor_session.process_frame(frame)
            except ProtocolError as exc:
                await websocket.send_text(serialize_protocol_error(exc))
                continue
            except Exception:
                await websocket.send_text(
                    serialize_error(
                        code="internal_error",
                        message="internal server error",
                        retryable=True,
                    )
                )
                continue

            await websocket.send_text(serialize_json_message(response))

    return app


def create_processor_from_config(config: ServerConfig) -> VisualFrameProcessor:
    backend = create_infer_backend(config.inference, runtime_dir=config.runtime_dir)
    return BackendVisualFrameProcessor(backend, tracking_config=config.tracking)


def _session_factory_from_processor(
    processor: VisualFrameProcessor,
) -> VisualStreamSessionFactory:
    create_session = getattr(processor, "create_session", None)
    if callable(create_session):
        return create_session
    return lambda: processor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visual-events-server")
    parser.add_argument("--config", help="Path to a JSON or TOML server config")
    parser.add_argument("--host", help="Override bind host")
    parser.add_argument("--port", type=int, help="Override bind port")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    host = args.host or config.host
    port = args.port if args.port is not None else config.port
    try:
        processor = create_processor_from_config(config)
    except Exception as exc:
        parser.exit(2, f"config error: {exc}\n")
    app = create_app(processor=processor, config=config)
    uvicorn.run(app, host=host, port=port)


app: Any = create_app()


if __name__ == "__main__":
    main()
