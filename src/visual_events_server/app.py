from __future__ import annotations

import argparse
import inspect
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from .config import MetricsConfig, ServerConfig, load_config
from .inference.factory import create_infer_backend
from .memory import (
    AppMemoryService,
    DisabledEmbeddingBackend,
    FakeEmbeddingBackend,
    LocalEmbeddingBackend,
    MemoryEmbeddingBackend,
    MemoryServiceError,
    MemoryStore,
)
from .memory.api_contract import (
    ConversationSummaryRequest,
    CorrectIdentityRequest,
    LinkExternalUserRequest,
    MergeAnonymousPersonRequest,
    ResolveTargetRequest,
    TeachPersonRequest,
    TeachSceneRequest,
    unsupported_target_kind_response,
    validate_resolve_target_response,
)
from .metrics import JsonlMetricsSink, MetricsSink
from .processor import (
    BackendVisualFrameProcessor,
    MockVisualFrameProcessor,
    VisualFrameProcessor,
    VisualStreamSessionFactory,
)
from .protocol import (
    FrameMessage,
    ProtocolError,
    decode_frame_message,
    serialize_error,
    serialize_json_message,
    serialize_protocol_error,
)


class MemoryService(Protocol):
    async def observe_visual_state(
        self,
        *,
        connection_id: str,
        frame: FrameMessage,
        visual_state: dict[str, Any],
        memory_snapshot: Any | None = None,
    ) -> None:
        ...

    async def drain_completed_events(
        self,
        *,
        camera: str,
        connection_id: str,
        frame_id: int,
        frame_timestamp_ms: int,
    ) -> list[dict[str, Any]]:
        ...

    async def teach_person(self, request: dict[str, Any]) -> dict[str, Any]:
        ...

    async def teach_scene(self, request: dict[str, Any]) -> dict[str, Any]:
        ...

    async def add_conversation_summary(
        self,
        person_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    async def link_external_user(self, request: dict[str, Any]) -> dict[str, Any]:
        ...

    async def get_person_by_external_user(
        self,
        external_user_ref: str,
    ) -> dict[str, Any]:
        ...

    async def merge_anonymous_person(self, request: dict[str, Any]) -> dict[str, Any]:
        ...

    async def correct_identity(self, request: dict[str, Any]) -> dict[str, Any]:
        ...

    async def resolve_target(self, request: dict[str, Any]) -> dict[str, Any]:
        ...


class DisabledMemoryService:
    async def observe_visual_state(
        self,
        *,
        connection_id: str,
        frame: FrameMessage,
        visual_state: dict[str, Any],
        memory_snapshot: Any | None = None,
    ) -> None:
        return None

    async def drain_completed_events(
        self,
        *,
        camera: str,
        connection_id: str,
        frame_id: int,
        frame_timestamp_ms: int,
    ) -> list[dict[str, Any]]:
        return []

    async def teach_person(self, request: dict[str, Any]) -> dict[str, Any]:
        raise self._disabled()

    async def teach_scene(self, request: dict[str, Any]) -> dict[str, Any]:
        raise self._disabled()

    async def add_conversation_summary(
        self,
        person_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        raise self._disabled()

    async def link_external_user(self, request: dict[str, Any]) -> dict[str, Any]:
        raise self._disabled()

    async def get_person_by_external_user(
        self,
        external_user_ref: str,
    ) -> dict[str, Any]:
        raise self._disabled()

    async def merge_anonymous_person(self, request: dict[str, Any]) -> dict[str, Any]:
        raise self._disabled()

    async def correct_identity(self, request: dict[str, Any]) -> dict[str, Any]:
        raise self._disabled()

    async def resolve_target(self, request: dict[str, Any]) -> dict[str, Any]:
        raise self._disabled()

    async def close(self) -> None:
        return None

    def _disabled(self) -> MemoryServiceError:
        return MemoryServiceError(
            "memory_disabled",
            "memory service is disabled",
            status_code=503,
        )


@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    try:
        yield
    finally:
        if not getattr(app.state, "_owns_memory_service", False):
            return
        close = getattr(app.state.memory_service, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


def create_app(
    *,
    processor: VisualFrameProcessor | None = None,
    session_factory: VisualStreamSessionFactory | None = None,
    config: ServerConfig | None = None,
    memory_service: MemoryService | None = None,
) -> FastAPI:
    app = FastAPI(
        title="visual-events-server",
        version="0.1.0",
        lifespan=_app_lifespan,
    )
    app_config = config or ServerConfig()
    owns_memory_service = memory_service is None
    app.state.config = app_config
    app.state.session_factory = session_factory or _session_factory_from_processor(
        processor or MockVisualFrameProcessor()
    )
    app.state.memory_service = (
        _memory_service_from_config(app_config)
        if memory_service is None
        else memory_service
    )
    app.state._owns_memory_service = owns_memory_service

    @app.get("/healthz")
    async def healthz() -> dict[str, bool | int]:
        return {"ok": True, "pid": os.getpid()}

    @app.post("/v1/memory/teach/person")
    async def teach_person(
        payload: TeachPersonRequest = Body(...),
    ) -> dict[str, Any]:
        return await _memory_response(
            app.state.memory_service.teach_person(payload.to_internal_request())
        )

    @app.post("/v1/memory/teach/scene")
    async def teach_scene(
        payload: TeachSceneRequest = Body(...),
    ) -> dict[str, Any]:
        return await _memory_response(
            app.state.memory_service.teach_scene(payload.to_internal_request())
        )

    @app.post("/v1/memory/person/{person_id}/conversation-summary")
    async def add_conversation_summary(
        person_id: str,
        payload: ConversationSummaryRequest = Body(...),
    ) -> dict[str, Any]:
        return await _memory_response(
            app.state.memory_service.add_conversation_summary(
                person_id,
                payload.to_internal_request(),
            )
        )

    @app.post("/v1/memory/link-external-user")
    async def link_external_user(
        payload: LinkExternalUserRequest = Body(...),
    ) -> dict[str, Any]:
        return await _memory_response(
            app.state.memory_service.link_external_user(payload.to_internal_request())
        )

    @app.get("/v1/memory/person/by-external-user/{external_user_ref}")
    async def get_person_by_external_user(
        external_user_ref: str,
    ) -> dict[str, Any]:
        return await _memory_response(
            app.state.memory_service.get_person_by_external_user(external_user_ref)
        )

    @app.post("/v1/memory/merge-anonymous-person")
    async def merge_anonymous_person(
        payload: MergeAnonymousPersonRequest = Body(...),
    ) -> dict[str, Any]:
        return await _memory_response(
            app.state.memory_service.merge_anonymous_person(
                payload.to_internal_request()
            )
        )

    @app.post("/v1/memory/correct-identity")
    async def correct_identity(
        payload: CorrectIdentityRequest = Body(...),
    ) -> dict[str, Any]:
        return await _memory_response(
            app.state.memory_service.correct_identity(payload.to_internal_request())
        )

    @app.post("/v1/memory/resolve-target")
    async def resolve_target(
        payload: ResolveTargetRequest = Body(...),
    ) -> dict[str, Any]:
        if payload.target.kind == "object":
            return unsupported_target_kind_response()
        response = await _memory_response(
            app.state.memory_service.resolve_target(payload.to_internal_request())
        )
        _validate_resolve_target_response(response)
        return response

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        connection_id = f"ws_{uuid.uuid4().hex}"
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
                memory_snapshot = _take_memory_frame_snapshot(processor_session)
                response = await _attach_memory_events(
                    app.state.memory_service,
                    connection_id=connection_id,
                    frame=frame,
                    visual_state=response,
                    memory_snapshot=memory_snapshot,
                )
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


async def _memory_response(awaitable: Any) -> dict[str, Any]:
    try:
        return await awaitable
    except MemoryServiceError as exc:
        detail = {"code": exc.code, "message": exc.message}
        detail.update(exc.details)
        raise HTTPException(
            status_code=exc.status_code,
            detail=detail,
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_memory_request", "message": str(exc)},
        ) from exc


def _validate_resolve_target_response(response: dict[str, Any]) -> None:
    try:
        validate_resolve_target_response(response)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "invalid_memory_response", "message": str(exc)},
        ) from exc


async def _attach_memory_events(
    memory_service: MemoryService,
    *,
    connection_id: str,
    frame: FrameMessage,
    visual_state: dict[str, Any],
    memory_snapshot: Any | None = None,
) -> dict[str, Any]:
    try:
        if memory_snapshot is None:
            await memory_service.observe_visual_state(
                connection_id=connection_id,
                frame=frame,
                visual_state=visual_state,
            )
        else:
            await memory_service.observe_visual_state(
                connection_id=connection_id,
                frame=frame,
                visual_state=visual_state,
                memory_snapshot=memory_snapshot,
            )
        completed_events = await memory_service.drain_completed_events(
            camera=frame.camera,
            connection_id=connection_id,
            frame_id=frame.frame_id,
            frame_timestamp_ms=frame.timestamp_ms,
        )
    except Exception:
        return visual_state

    if not completed_events:
        return visual_state

    semantic_events = visual_state.get("semantic_events")
    if not isinstance(semantic_events, list):
        visual_state["semantic_events"] = list(completed_events)
        return visual_state
    semantic_events.extend(completed_events)
    return visual_state


def _take_memory_frame_snapshot(processor_session: Any) -> Any | None:
    take_snapshot = getattr(processor_session, "take_memory_frame_snapshot", None)
    if not callable(take_snapshot):
        return None
    return take_snapshot()


def create_processor_from_config(config: ServerConfig) -> VisualFrameProcessor:
    backend = create_infer_backend(config.inference, runtime_dir=config.runtime_dir)
    return BackendVisualFrameProcessor(
        backend,
        tracking_config=config.tracking,
        attention_config=config.attention,
        event_config=config.events,
        metrics_sink=_metrics_sink_from_config(config),
    )


def _session_factory_from_processor(
    processor: VisualFrameProcessor,
) -> VisualStreamSessionFactory:
    create_session = getattr(processor, "create_session", None)
    if callable(create_session):
        return create_session
    return lambda: processor


def _metrics_sink_from_config(config: ServerConfig) -> MetricsSink | None:
    if config.metrics.jsonl_path is None:
        return None
    return JsonlMetricsSink(config.metrics.jsonl_path)


def _memory_service_from_config(config: ServerConfig) -> MemoryService:
    if not config.memory.enabled:
        return DisabledMemoryService()
    backend, person_dim, scene_dim = _embedding_backend_from_config(config)
    store = MemoryStore.open(
        config.memory.db_path,
        person_dim=person_dim,
        scene_dim=scene_dim,
    )
    return AppMemoryService(
        store=store,
        embedding_backend=backend,
        frame_cache_seconds=config.memory.frame_cache_seconds,
        query_interval_ms=config.memory.query_interval_ms,
        queue_size=config.memory.queue_size,
        known_person_threshold=config.memory.matching.known_person_threshold,
        known_person_margin=config.memory.matching.known_person_margin,
        anonymous_threshold=config.memory.matching.anonymous_threshold,
        anonymous_margin=config.memory.matching.anonymous_margin,
        familiar_seen_count=config.memory.matching.familiar_seen_count,
        familiar_threshold=config.memory.matching.familiar_threshold,
        scene_threshold=config.memory.matching.scene_threshold,
        event_cooldown_ms=config.memory.matching.event_cooldown_ms,
        teach_queue_size=config.memory.embedding.teach_queue_size,
        teach_queue_timeout_ms=config.memory.embedding.teach_queue_timeout_ms,
        artifact_dir=config.runtime_dir / "memory" / "artifacts",
    )


def _embedding_backend_from_config(
    config: ServerConfig,
) -> tuple[MemoryEmbeddingBackend, int, int]:
    backend_name = config.memory.embedding.backend
    if backend_name == "fake":
        backend = FakeEmbeddingBackend()
        return backend, backend.person_dim, backend.scene_dim
    if backend_name == "disabled":
        return DisabledEmbeddingBackend(), 32, 32
    if backend_name == "local":
        embedding_config = config.memory.embedding
        backend = LocalEmbeddingBackend(
            person_model_path=embedding_config.person_model_path,
            scene_model_path=embedding_config.scene_model_path,
        )
        return backend, backend.person_dim, backend.scene_dim
    raise ValueError(f"unsupported memory embedding backend {backend_name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visual-events-server")
    parser.add_argument("--config", help="Path to a JSON or TOML server config")
    parser.add_argument("--host", help="Override bind host")
    parser.add_argument("--port", type=int, help="Override bind port")
    parser.add_argument("--metrics-jsonl", help="Write per-frame metrics to JSONL")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.metrics_jsonl is not None:
        config = replace(
            config,
            metrics=MetricsConfig(jsonl_path=Path(args.metrics_jsonl)),
        )
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
