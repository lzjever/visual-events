import asyncio
import json
import time
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import visual_events_server.app as app_module
from visual_events_server.app import create_app
from visual_events_server.config import (
    MemoryConfig,
    MemoryEmbeddingConfig,
    MemoryMatchingConfig,
    ServerConfig,
)
from visual_events_server.attention import AttentionResult
from visual_events_server.inference.base import PoseKeypoint
from visual_events_server.memory.embedding import LocalEmbeddingBackend
from visual_events_server.memory.frame_cache import MemoryFrameSnapshot
from visual_events_server.protocol import FrameMessage, encode_frame_message
from visual_events_server.tracking import TrackSnapshot


def jpeg_1280x720() -> bytes:
    image = Image.new("RGB", (1280, 720), (28, 36, 46))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1279, 719), fill=(36, 58, 76))
    draw.rectangle((300, 100, 499, 374), fill=(35, 176, 99))
    draw.rectangle((500, 100, 699, 374), fill=(234, 188, 45))
    draw.rectangle((300, 375, 499, 649), fill=(46, 119, 214))
    draw.rectangle((500, 375, 699, 649), fill=(202, 74, 168))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


JPEG_BYTES = jpeg_1280x720()


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


def memory_event() -> dict[str, Any]:
    return {
        "type": "semantic_event",
        "event_id": "front:mem_evt_000001",
        "event": "known_person_present",
        "camera": "front",
        "track_id": 1,
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
            "source_frame_id": 6,
            "source_frame_timestamp_ms": 1709999999900,
            "embedding_model": "fake-face-v1",
        },
        "memory_context": {
            "person": {
                "person_id": "person_000001",
                "display_name": "张三",
                "match_confidence": 0.86,
            },
            "conversation_summaries": [],
        },
        "text": "看到已知人物：张三",
    }


class FakeMemoryService:
    def __init__(self) -> None:
        self.observed: list[dict[str, Any]] = []
        self.drain_calls: list[dict[str, Any]] = []
        self.completed_by_camera: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[tuple[str, Any]] = []

    async def observe_visual_state(
        self,
        *,
        connection_id: str,
        frame: FrameMessage,
        visual_state: dict[str, Any],
        memory_snapshot: Any | None = None,
    ) -> None:
        self.observed.append(
            {
                "connection_id": connection_id,
                "camera": frame.camera,
                "frame_id": frame.frame_id,
                "frame_timestamp_ms": frame.timestamp_ms,
                "jpeg_bytes": frame.jpeg_bytes,
                "visual_state_frame_id": visual_state["frame_id"],
                "memory_snapshot": memory_snapshot,
            }
        )

    async def drain_completed_events(
        self,
        *,
        camera: str,
        connection_id: str,
        frame_id: int,
        frame_timestamp_ms: int,
    ) -> list[dict[str, Any]]:
        self.drain_calls.append(
            {
                "camera": camera,
                "connection_id": connection_id,
                "frame_id": frame_id,
                "frame_timestamp_ms": frame_timestamp_ms,
            }
        )
        return list(self.completed_by_camera.pop(camera, []))

    async def teach_person(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("teach_person", request))
        return {"ok": True, "person_id": "person_000001"}

    async def teach_scene(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("teach_scene", request))
        return {"ok": True, "scene_id": "scene_000001"}

    async def identify_current(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("identify_current", request))
        return {
            "ok": True,
            "status": "identified",
            "people": [
                {
                    "target_ref": "current:front:active_target",
                    "identity_context": {"status": "known_person"},
                }
            ],
            "evidence": {"source_frame_ref": "front:7:1710000000000"},
        }

    async def add_conversation_summary(
        self,
        person_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(("add_conversation_summary", (person_id, request)))
        return {"ok": True, "summary_id": "summary_000001"}

    async def link_external_user(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("link_external_user", request))
        return {"ok": True, "person_id": request["person_id"]}

    async def get_person_by_external_user(
        self,
        external_user_ref: str,
    ) -> dict[str, Any]:
        self.calls.append(("get_person_by_external_user", external_user_ref))
        return {
            "ok": True,
            "external_user_ref": external_user_ref,
            "person": {"person_id": "person_000001"},
        }

    async def merge_anonymous_person(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("merge_anonymous_person", request))
        return {
            "ok": True,
            "anonymous_id": request["anonymous_id"],
            "person_id": request.get("person_id", "person_000002"),
            "copied_embedding_count": 2,
            "merge_id": "merge_000001",
        }

    async def correct_identity(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("correct_identity", request))
        return {
            "ok": True,
            "memory_match_id": request["memory_match_id"],
            "wrong_person_id": request["wrong_person_id"],
        }

    async def resolve_target(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("resolve_target", request))
        return {
            "ok": True,
            "status": "resolved",
            "candidates": [
                {
                    "target_type": "person",
                    "track_id": 7,
                    "bbox_xyxy": [300.0, 100.0, 700.0, 650.0],
                    "confidence": 0.9,
                    "reason": "track_id",
                }
            ],
        }


class ConflictResolveMemoryService(FakeMemoryService):
    async def resolve_target(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("resolve_target", request))
        return {"ok": True, "status": "conflict", "candidates": []}


class FailingMemoryService(FakeMemoryService):
    async def observe_visual_state(
        self,
        *,
        connection_id: str,
        frame: FrameMessage,
        visual_state: dict[str, Any],
    ) -> None:
        raise RuntimeError("memory worker unavailable")


def test_memory_http_endpoints_delegate_to_app_level_service():
    service = FakeMemoryService()
    client = TestClient(create_app(memory_service=service))

    person_request = {
        "camera": "front",
        "stream_ref": "ws_1",
        "target": {
            "kind": "person",
            "intent": "self_introduction",
            "referent_text": "我",
        },
        "profile": {"display_name": "张三"},
    }
    assert client.post("/v1/memory/teach/person", json=person_request).json() == {
        "ok": True,
        "person_id": "person_000001",
    }

    scene_request = {
        "camera": "front",
        "stream_ref": "ws_1",
        "target": {
            "kind": "scene",
            "intent": "teach_scene",
            "referent_text": "当前展示区",
        },
        "memory": {"title": "新品展示区"},
    }
    assert client.post("/v1/memory/teach/scene", json=scene_request).json() == {
        "ok": True,
        "scene_id": "scene_000001",
    }

    identify_request = {
        "camera": "front",
        "stream_ref": "ws_1",
        "target": {
            "kind": "person",
            "intent": "identify_current",
            "referent_text": "当前这个人",
        },
        "scope": "active_target",
        "timeout_ms": 500,
    }
    assert client.post(
        "/v1/memory/identify-current",
        json=identify_request,
    ).json() == {
        "ok": True,
        "status": "identified",
        "people": [
            {
                "target_ref": "current:front:active_target",
                "identity_context": {"status": "known_person"},
            }
        ],
        "evidence": {"source_frame_ref": "front:7:1710000000000"},
    }

    summary_request = {"summary": "上次问过新品尺码。", "source": "agent"}
    assert client.post(
        "/v1/memory/person/person_000001/conversation-summary",
        json=summary_request,
    ).json() == {"ok": True, "summary_id": "summary_000001"}

    link_request = {
        "person_id": "person_000001",
        "external_user_ref": "wechat:zhangsan",
    }
    assert client.post("/v1/memory/link-external-user", json=link_request).json() == {
        "ok": True,
        "person_id": "person_000001",
    }
    assert client.get(
        "/v1/memory/person/by-external-user/wechat:zhangsan"
    ).json() == {
        "ok": True,
        "external_user_ref": "wechat:zhangsan",
        "person": {"person_id": "person_000001"},
    }

    merge_request = {
        "anonymous_id": "anon_000001",
        "profile": {"display_name": "李四"},
    }
    assert client.post(
        "/v1/memory/merge-anonymous-person",
        json=merge_request,
    ).json() == {
        "ok": True,
        "anonymous_id": "anon_000001",
        "person_id": "person_000002",
        "copied_embedding_count": 2,
        "merge_id": "merge_000001",
    }

    correction_request = {
        "memory_match_id": "match_000001",
        "wrong_person_id": "person_000001",
    }
    assert client.post(
        "/v1/memory/correct-identity",
        json=correction_request,
    ).json() == {
        "ok": True,
        "memory_match_id": "match_000001",
        "wrong_person_id": "person_000001",
    }

    resolve_request = {
        "camera": "front",
        "stream_ref": "ws_1",
        "target": {
            "kind": "scene",
            "intent": "teach_scene",
            "referent_text": "当前展示区",
        },
    }
    assert client.post(
        "/v1/memory/resolve-target",
        json=resolve_request,
    ).json() == {
        "ok": True,
        "status": "resolved",
        "candidates": [
            {
                "target_type": "person",
                "track_id": 7,
                "bbox_xyxy": [300.0, 100.0, 700.0, 650.0],
                "confidence": 0.9,
                "reason": "track_id",
            }
        ],
    }

    assert service.calls == [
        (
            "teach_person",
            {
                "camera": "front",
                "stream_ref": "ws_1",
                "target": {
                    "kind": "person",
                    "intent": "self_introduction",
                    "referent_text": "我",
                },
                "profile": {"display_name": "张三"},
            },
        ),
        (
            "teach_scene",
            {
                "camera": "front",
                "stream_ref": "ws_1",
                "target": {"mode": "scene"},
                "memory": {"title": "新品展示区"},
            },
        ),
        ("identify_current", identify_request),
        ("add_conversation_summary", ("person_000001", summary_request)),
        ("link_external_user", link_request),
        ("get_person_by_external_user", "wechat:zhangsan"),
        ("merge_anonymous_person", merge_request),
        ("correct_identity", correction_request),
        (
            "resolve_target",
            {
                "camera": "front",
                "stream_ref": "ws_1",
                "target": {"mode": "scene"},
            },
        ),
    ]


def test_public_memory_routes_reject_low_level_agent_payload_fields():
    base_payloads = {
        "/v1/memory/resolve-target": {
            "camera": "front",
            "stream_ref": "ws_1",
            "target": {
                "kind": "scene",
                "intent": "teach_scene",
                "referent_text": "当前展示区",
            },
        },
        "/v1/memory/teach/person": {
            "camera": "front",
            "stream_ref": "ws_1",
            "target": {
                "kind": "person",
                "intent": "self_introduction",
                "referent_text": "我",
            },
            "profile": {"display_name": "张三"},
        },
        "/v1/memory/teach/scene": {
            "camera": "front",
            "stream_ref": "ws_1",
            "target": {
                "kind": "scene",
                "intent": "teach_scene",
                "referent_text": "当前展示区",
            },
            "memory": {"title": "新品展示区"},
        },
        "/v1/memory/identify-current": {
            "camera": "front",
            "stream_ref": "ws_1",
            "target": {
                "kind": "person",
                "intent": "identify_current",
                "referent_text": "当前这个人",
            },
            "scope": "active_target",
        },
    }
    forbidden_values = {
        "track_id": 7,
        "bbox": [0, 0, 10, 10],
        "bbox_xyxy": [0, 0, 10, 10],
        "point_uv": [0.5, 0.5],
        "keypoints": [{"name": "nose", "x": 0.5, "y": 0.5}],
        "embedding": [1.0, 0.0],
        "crop": "private-crop-bytes",
        "crop_ref": "runtime/private/crop.jpg",
        "test_hint": "fixture",
        "source_scene": "pic_teach_scene",
        "source_frame": 7,
        "source_frame_ref": "front:7:1710000000000",
        "request_snapshot_ref": "snapshot:front:7",
    }
    service = FakeMemoryService()
    client = TestClient(create_app(memory_service=service))

    for path, base_payload in base_payloads.items():
        for field, value in forbidden_values.items():
            payload = deepcopy(base_payload)
            if field.startswith("source_"):
                payload[field] = value
            else:
                payload["target"][field] = value
            response = client.post(path, json=payload)
            assert response.status_code == 422

    assert service.calls == []


def test_public_memory_routes_pass_required_stream_ref_without_exposing_low_level_targets():
    service = FakeMemoryService()
    client = TestClient(create_app(memory_service=service))

    response = client.post(
        "/v1/memory/resolve-target",
        json={
            "camera": "front",
            "stream_ref": "ws_stream_1",
            "target": {
                "kind": "person",
                "intent": "self_introduction",
                "referent_text": "我",
            },
        },
    )

    assert response.status_code == 200
    assert service.calls == [
        (
            "resolve_target",
            {
                "camera": "front",
                "stream_ref": "ws_stream_1",
                "target": {
                    "kind": "person",
                    "intent": "self_introduction",
                    "referent_text": "我",
                },
            },
        )
    ]


def test_public_frame_bound_memory_routes_require_stream_ref():
    service = FakeMemoryService()
    client = TestClient(create_app(memory_service=service))
    payloads = [
        (
            "/v1/memory/resolve-target",
            {
                "camera": "front",
                "target": {
                    "kind": "person",
                    "intent": "self_introduction",
                    "referent_text": "我",
                },
            },
        ),
        (
            "/v1/memory/teach/person",
            {
                "camera": "front",
                "target": {
                    "kind": "person",
                    "intent": "self_introduction",
                    "referent_text": "我",
                },
                "profile": {"display_name": "张三"},
            },
        ),
        (
            "/v1/memory/teach/scene",
            {
                "camera": "front",
                "target": {
                    "kind": "scene",
                    "intent": "teach_scene",
                    "referent_text": "当前展示区",
                },
                "memory": {"title": "新品展示区"},
            },
        ),
        (
            "/v1/memory/identify-current",
            {
                "camera": "front",
                "target": {
                    "kind": "person",
                    "intent": "identify_current",
                    "referent_text": "当前这个人",
                },
                "scope": "active_target",
            },
        ),
    ]

    for path, payload in payloads:
        response = client.post(path, json=payload)
        assert response.status_code == 422

    assert service.calls == []


def test_identify_current_disabled_memory_returns_business_unavailable():
    client = TestClient(create_app())

    response = client.post(
        "/v1/memory/identify-current",
        json={
            "camera": "front",
            "stream_ref": "ws_1",
            "target": {
                "kind": "person",
                "intent": "identify_current",
                "referent_text": "当前这个人",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "status": "unavailable",
        "reason": "memory_disabled",
        "people": [],
        "evidence": {},
    }


def test_identify_current_rejects_unsupported_scope_by_schema():
    service = FakeMemoryService()
    client = TestClient(create_app(memory_service=service))

    response = client.post(
        "/v1/memory/identify-current",
        json={
            "camera": "front",
            "stream_ref": "ws_1",
            "target": {
                "kind": "person",
                "intent": "identify_current",
                "referent_text": "当前这个人",
            },
            "scope": "visible_people",
        },
    )

    assert response.status_code == 422
    assert service.calls == []


def test_public_memory_management_routes_reject_unknown_and_low_level_fields_without_calling_service():
    base_payloads = {
        "/v1/memory/person/person_000001/conversation-summary": {
            "summary": "上次问过新品尺码。",
            "source": "agent",
        },
        "/v1/memory/link-external-user": {
            "person_id": "person_000001",
            "external_user_ref": "wechat:zhangsan",
        },
        "/v1/memory/merge-anonymous-person": {
            "anonymous_id": "anon_000001",
            "profile": {"display_name": "李四"},
        },
        "/v1/memory/correct-identity": {
            "memory_match_id": "match_000001",
            "wrong_person_id": "person_000001",
        },
    }
    forbidden_values = {
        "track_id": 7,
        "bbox": [0, 0, 10, 10],
        "bbox_xyxy": [0, 0, 10, 10],
        "point_uv": [0.5, 0.5],
        "keypoints": [{"name": "nose", "x": 0.5, "y": 0.5}],
        "embedding": [1.0, 0.0],
        "crop": "private-crop-bytes",
        "crop_ref": "runtime/private/crop.jpg",
        "test_hint": "fixture",
        "source_scene": "pic_teach_scene",
        "source_frame": 7,
        "source_frame_ref": "front:7:1710000000000",
        "request_snapshot_ref": "snapshot:front:7",
    }
    service = FakeMemoryService()
    client = TestClient(create_app(memory_service=service))

    for path, base_payload in base_payloads.items():
        unknown_payload = {**base_payload, "unexpected": "nope"}
        assert client.post(path, json=unknown_payload).status_code == 422

        for field, value in forbidden_values.items():
            top_level_payload = {**base_payload, field: value}
            assert client.post(path, json=top_level_payload).status_code == 422

    merge_payload = base_payloads["/v1/memory/merge-anonymous-person"]
    for field, value in forbidden_values.items():
        low_level_payload = deepcopy(merge_payload)
        low_level_payload["profile"] = {
            **merge_payload["profile"],
            field: value,
        }
        assert (
            client.post(
                "/v1/memory/merge-anonymous-person",
                json=low_level_payload,
            ).status_code
            == 422
        )

    assert service.calls == []


def test_public_resolve_target_rejects_object_without_calling_memory_service():
    service = FakeMemoryService()
    client = TestClient(create_app(memory_service=service))

    response = client.post(
        "/v1/memory/resolve-target",
        json={
            "camera": "front",
            "stream_ref": "ws_1",
            "target": {
                "kind": "object",
                "intent": "teach_object",
                "referent_text": "手机",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "status": "not_found",
        "error_code": "unsupported_target_kind",
        "retryable": False,
        "ask_user_hint": False,
        "ambiguity_type": "unsupported_target_kind",
        "candidates": [],
    }
    assert service.calls == []


def test_public_resolve_target_does_not_expose_conflict_as_resolver_status():
    client = TestClient(create_app(memory_service=ConflictResolveMemoryService()))

    response = client.post(
        "/v1/memory/resolve-target",
        json={
            "camera": "front",
            "stream_ref": "ws_1",
            "target": {
                "kind": "scene",
                "intent": "teach_scene",
                "referent_text": "当前展示区",
            },
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "invalid_memory_response"


def test_default_memory_endpoints_fail_fast_when_memory_service_is_disabled():
    client = TestClient(create_app())

    response = client.post(
        "/v1/memory/teach/person",
        json={
            "camera": "front",
            "stream_ref": "ws_1",
            "target": {
                "kind": "person",
                "intent": "self_introduction",
                "referent_text": "我",
            },
            "profile": {"display_name": "张三"},
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "memory_disabled"


def test_stream_observes_visual_state_and_drains_completed_memory_events():
    service = FakeMemoryService()
    event = memory_event()
    service.completed_by_camera["front"] = [event]
    client = TestClient(create_app(memory_service=service))

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
        message = json.loads(websocket.receive_text())

    assert message["type"] == "visual_state"
    assert message["semantic_events"] == [event]
    assert len(service.observed) == 1
    assert service.observed[0]["camera"] == "front"
    assert service.observed[0]["frame_id"] == 7
    assert service.observed[0]["frame_timestamp_ms"] == 1710000000000
    assert service.observed[0]["jpeg_bytes"] == JPEG_BYTES
    assert service.observed[0]["visual_state_frame_id"] == 7
    assert service.drain_calls == [
        {
            "camera": "front",
            "connection_id": service.observed[0]["connection_id"],
            "frame_id": 7,
            "frame_timestamp_ms": 1710000000000,
        }
    ]


def test_stream_passes_memory_snapshot_side_channel_to_memory_service():
    service = FakeMemoryService()
    client = TestClient(create_app(processor=MemoryVisualProcessor(), memory_service=service))

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
        message = json.loads(websocket.receive_text())

    assert message["tracks"][0].get("keypoints") is None
    snapshot = service.observed[0]["memory_snapshot"]
    assert snapshot is not None
    assert snapshot.tracks[0].track_id == 7
    assert snapshot.tracks[0].keypoints[0].name == "nose"
    assert snapshot.scene_context["engagement_state"] == "available"


def test_stream_snapshot_side_channel_exception_does_not_fallback_to_memory_service():
    service = FakeMemoryService()
    client = TestClient(
        create_app(processor=BrokenSnapshotProcessor(), memory_service=service)
    )

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
        message = json.loads(websocket.receive_text())

    assert message["type"] == "error"
    assert message["code"] == "internal_error"
    assert service.observed == []
    assert service.drain_calls == []


def test_memory_side_chain_failure_does_not_break_visual_stream():
    client = TestClient(create_app(memory_service=FailingMemoryService()))

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
        message = json.loads(websocket.receive_text())

    assert message["type"] == "visual_state"
    assert message["frame_id"] == 7
    assert message["semantic_events"] == []


def test_stream_includes_stream_ref_and_disabled_identity_context():
    client = TestClient(create_app())

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
        message = json.loads(websocket.receive_text())

    assert message["stream_ref"].startswith("ws_")
    assert message["identity_context"] == {
        "overlay_status": "unavailable",
        "reason": "memory_disabled",
        "tracks": [],
    }


def test_stream_ref_and_frame_cache_are_isolated_for_same_camera_connections(tmp_path):
    config = ServerConfig(
        memory=MemoryConfig(
            enabled=True,
            db_path=tmp_path / "memory.sqlite3",
            frame_cache_seconds=5,
            query_interval_ms=60_000,
            queue_size=4,
            embedding=MemoryEmbeddingConfig(backend="fake"),
            matching=MemoryMatchingConfig(
                known_person_threshold=0.95,
                known_person_margin=0.05,
                anonymous_threshold=0.95,
                anonymous_margin=0.05,
                familiar_seen_count=3,
                familiar_observed_duration_ms=0,
                familiar_threshold=0.95,
                scene_threshold=0.95,
                event_cooldown_ms=5_000,
            ),
        )
    )
    with TestClient(
        create_app(processor=MemoryVisualProcessor(), config=config)
    ) as client:
        with client.websocket_connect("/v1/stream") as first:
            first.send_bytes(
                encode_frame_message(
                    frame_header(frame_id=1, timestamp_ms=1710000000000),
                    JPEG_BYTES,
                )
            )
            first_message = json.loads(first.receive_text())
        with client.websocket_connect("/v1/stream") as second:
            second.send_bytes(
                encode_frame_message(
                    frame_header(frame_id=2, timestamp_ms=1710000000100),
                    JPEG_BYTES,
                )
            )
            second_message = json.loads(second.receive_text())

        memory_service = client.app.state.memory_service
        first_cached = memory_service._cache.get_fresh_for_stream(  # noqa: SLF001
            first_message["stream_ref"],
            "front",
        )
        second_cached = memory_service._cache.get_fresh_for_stream(  # noqa: SLF001
            second_message["stream_ref"],
            "front",
        )

    assert first_message["stream_ref"] != second_message["stream_ref"]
    assert first_cached.frame.frame_id == 1
    assert second_cached.frame.frame_id == 2
    assert first_message["identity_context"]["overlay_status"] == "ready"
    assert first_message["identity_context"]["tracks"][0]["identity"]["status"] in {
        "pending",
        "unknown",
    }


def test_configured_memory_rejects_unknown_stream_ref_without_camera_fallback(tmp_path):
    config = ServerConfig(
        memory=MemoryConfig(
            enabled=True,
            db_path=tmp_path / "memory.sqlite3",
            frame_cache_seconds=5,
            query_interval_ms=60_000,
            queue_size=4,
            embedding=MemoryEmbeddingConfig(backend="fake"),
        )
    )
    with TestClient(
        create_app(processor=MemoryVisualProcessor(), config=config)
    ) as client:
        with client.websocket_connect("/v1/stream") as websocket:
            websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
            message = json.loads(websocket.receive_text())
            assert message["stream_ref"] != "ws_missing"

            response = client.post(
                "/v1/memory/resolve-target",
                json={
                    "camera": "front",
                    "stream_ref": "ws_missing",
                    "target": {
                        "kind": "person",
                        "intent": "self_introduction",
                        "referent_text": "我",
                    },
                },
            )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "no_active_frame"


class MemoryVisualProcessor:
    def __init__(self) -> None:
        self._memory_snapshot: MemoryFrameSnapshot | None = None

    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        self._memory_snapshot = _memory_snapshot_for_frame(frame)
        return {
            "type": "visual_state",
            "schema_version": 1,
            "camera": frame.camera,
            "frame_id": frame.frame_id,
            "frame_timestamp_ms": frame.timestamp_ms,
            "server_timestamp_ms": frame.timestamp_ms,
            "image_size": [frame.width, frame.height],
            "tracks": [
                {
                    "track_id": 7,
                    "class": "person",
                    "bbox_xyxy": [300.0, 100.0, 700.0, 650.0],
                    "bbox_area_ratio": 0.2,
                    "center_uv": [500.0, 375.0],
                    "head_uv": [500.0, 150.0],
                    "velocity_uv_s": [0.0, 0.0],
                    "age_ms": 800,
                    "lost_ms": 0,
                    "confidence": 0.92,
                    "pose_confidence": 0.81,
                }
            ],
            "attention": {
                "target_track_id": 7,
                "target_uv": [500.0, 160.0],
                "reason": "largest_stable_person",
                "confidence": 0.9,
            },
            "scene_context": {
                "engagement_state": "available",
                "attention_available": True,
                "target_track_id": 7,
            },
            "scene_flags": {"has_person": True, "person_count": 1},
            "semantic_events": [],
        }

    def take_memory_frame_snapshot(self) -> MemoryFrameSnapshot | None:
        snapshot = self._memory_snapshot
        self._memory_snapshot = None
        return snapshot


class BrokenSnapshotProcessor(MemoryVisualProcessor):
    def take_memory_frame_snapshot(self) -> MemoryFrameSnapshot | None:
        raise RuntimeError("snapshot provider failed")


def _memory_snapshot_for_frame(frame: FrameMessage) -> MemoryFrameSnapshot:
    track = TrackSnapshot(
        track_id=7,
        first_seen_ms=frame.timestamp_ms - 800,
        last_seen_ms=frame.timestamp_ms,
        frame_timestamp_ms=frame.timestamp_ms,
        bbox_xyxy=(300.0, 100.0, 700.0, 650.0),
        confidence=0.92,
        pose_confidence=0.81,
        head_uv=(500.0, 150.0),
        velocity_uv_s=(0.0, 0.0),
        lost_ms=0,
        hits=2,
        misses=0,
        keypoints=(
            PoseKeypoint("nose", 500.0, 150.0, 0.9),
        ),
    )
    return MemoryFrameSnapshot(
        connection_id="ws_1",
        frame=frame,
        source_frame_ref=f"{frame.camera}:{frame.frame_id}:{frame.timestamp_ms}",
        snapshot_ref=f"snapshot:{frame.camera}:{frame.frame_id}",
        observed_at_ms=frame.timestamp_ms,
        image_size=(frame.width, frame.height),
        tracks=[track],
        attention=AttentionResult(
            target_track_id=7,
            target_uv=(500.0, 160.0),
            reason="largest_stable_person",
            confidence=0.9,
            largest_person_stable=True,
        ),
        scene_context={
            "engagement_state": "available",
            "attention_available": True,
            "target_track_id": 7,
        },
        semantic_events=[],
    )


def _send_stable_memory_window(
    websocket: Any,
    *,
    start_frame_id: int = 1,
    start_timestamp_ms: int = 1710000000000,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for offset in range(2):
        websocket.send_bytes(
            encode_frame_message(
                frame_header(
                    frame_id=start_frame_id + offset,
                    timestamp_ms=start_timestamp_ms + offset,
                ),
                JPEG_BYTES,
            )
        )
        messages.append(json.loads(websocket.receive_text()))
    return messages


def test_configured_memory_service_teaches_and_returns_memory_events(tmp_path):
    config = ServerConfig(
        memory=MemoryConfig(
            enabled=True,
            db_path=tmp_path / "memory.sqlite3",
            frame_cache_seconds=5,
            query_interval_ms=1,
            queue_size=4,
            embedding=MemoryEmbeddingConfig(backend="fake"),
            matching=MemoryMatchingConfig(
                known_person_threshold=0.95,
                known_person_margin=0.05,
                anonymous_threshold=0.95,
                anonymous_margin=0.05,
                familiar_seen_count=3,
                familiar_threshold=0.95,
                scene_threshold=0.95,
                event_cooldown_ms=5_000,
            ),
        )
    )
    with TestClient(
        create_app(processor=MemoryVisualProcessor(), config=config)
    ) as client:
        with client.websocket_connect("/v1/stream") as websocket:
            messages = _send_stable_memory_window(websocket)
            assert [message["semantic_events"] for message in messages] == [[], []]
            stream_ref = messages[-1]["stream_ref"]

            person_profile = {
                "display_name": "张三",
                "description": "店长",
                "tags": ["staff"],
            }
            person = client.post(
                "/v1/memory/teach/person",
                json={
                    "camera": "front",
                    "stream_ref": stream_ref,
                    "target": {
                        "kind": "person",
                        "intent": "self_introduction",
                        "referent_text": "我",
                    },
                    "profile": person_profile,
                },
            )
            assert person.status_code == 200
            taught_person = person.json()
            assert taught_person["ok"] is True
            assert taught_person["outcome"] == "merged_anonymous_person"
            assert taught_person["merged_anonymous_id"].startswith("anon_")
            assert taught_person["copied_embedding_count"] >= 1
            assert taught_person["embedding_id"].startswith("emb_person_")
            assert taught_person["merge_id"].startswith("merge_")
            assert taught_person["store_delta"]["delta"]["person_profiles"] == 1
            assert taught_person["store_delta"]["delta"]["person_embeddings"] >= 2
            assert (
                taught_person["store_delta"]["delta"]["profile_merge_history"] == 1
            )
            person_id = taught_person["person_id"]

            scene = client.post(
                "/v1/memory/teach/scene",
                json={
                    "camera": "front",
                    "stream_ref": stream_ref,
                    "target": {
                        "kind": "scene",
                        "intent": "teach_scene",
                        "referent_text": "当前展示区",
                    },
                    "memory": {
                        "title": "新品展示区",
                        "description": "夏季外套区域",
                        "activation_hint": "介绍新品活动",
                        "region_id": "display_zone",
                    },
                },
            )
            assert scene.status_code == 200

            assert client.post(
                f"/v1/memory/person/{person_id}/conversation-summary",
                json={
                    "summary": "上次问过新品尺码，偏好浅色外套。",
                    "source": "agent",
                    "source_conversation_id": "conv-1",
                },
            ).json()["ok"] is True
            assert client.post(
                "/v1/memory/link-external-user",
                json={
                    "person_id": person_id,
                    "external_user_ref": "wechat:zhangsan",
                },
            ).json() == {"ok": True, "person_id": person_id}
            external = client.get(
                "/v1/memory/person/by-external-user/wechat:zhangsan"
            ).json()
            assert external["person"]["display_name"] == "张三"
            assert external["conversation_summaries"] == [
                "上次问过新品尺码，偏好浅色外套。"
            ]

            websocket.send_bytes(
                encode_frame_message(
                    frame_header(frame_id=2, timestamp_ms=1710000000600),
                    JPEG_BYTES,
                )
            )
            message = json.loads(websocket.receive_text())
            events_by_type = {
                event["event"]: event
                for event in message["semantic_events"]
                if event["event"] in {"known_person_present", "scene_activated"}
            }
            collected_event_names = [
                event["event"] for event in message["semantic_events"]
            ]
            for frame_id in range(3, 8):
                if {"known_person_present", "scene_activated"}.issubset(events_by_type):
                    break
                time.sleep(0.05)
                websocket.send_bytes(
                    encode_frame_message(
                        frame_header(
                            frame_id=frame_id,
                            timestamp_ms=1710000000600 + frame_id,
                        ),
                        JPEG_BYTES,
                    )
                )
                message = json.loads(websocket.receive_text())
                for event in message["semantic_events"]:
                    collected_event_names.append(event["event"])
                    if event["event"] in {"known_person_present", "scene_activated"}:
                        events_by_type[event["event"]] = event

    assert sorted(collected_event_names) == [
        "known_person_present",
        "scene_activated",
    ]
    events = [
        events_by_type["known_person_present"],
        events_by_type["scene_activated"],
    ]
    assert [event["event"] for event in events] == [
        "known_person_present",
        "scene_activated",
    ]
    assert events[0]["memory_context"]["person"]["person_id"] == person_id
    assert events[0]["memory_context"]["conversation_summaries"] == [
        "上次问过新品尺码，偏好浅色外套。"
    ]
    assert events[0]["evidence"]["source_frame_id"] == 2
    assert events[0]["evidence"]["match_score"] >= 0.95
    assert events[0]["evidence"]["top2_margin"] >= 0.05
    assert events[1]["memory_context"]["scene"]["title"] == "新品展示区"
    assert events[1]["memory_context"]["scene"]["region_id"] == "display_zone"


def test_local_memory_config_uses_bundle_dimensions_for_store(
    tmp_path,
    monkeypatch,
):
    person_bundle = _write_local_bundle(
        tmp_path,
        "person-local",
        dim=3,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_local_bundle(
        tmp_path,
        "scene-local",
        dim=5,
        files={"model": "scene.onnx"},
    )

    class StubLocalLoader:
        def embed_person(self, image_crop: bytes) -> list[float]:
            return [1.0, 0.0, 0.0]

        def embed_scene(self, image_or_crop: bytes) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0, 0.0]

    def local_backend_factory(*, person_model_path, scene_model_path):
        return LocalEmbeddingBackend(
            person_model_path=person_model_path,
            scene_model_path=scene_model_path,
            loader=StubLocalLoader(),
        )

    monkeypatch.setattr(
        app_module,
        "LocalEmbeddingBackend",
        local_backend_factory,
        raising=False,
    )
    config = ServerConfig(
        memory=MemoryConfig(
            enabled=True,
            db_path=tmp_path / "memory.sqlite3",
            embedding=MemoryEmbeddingConfig(
                backend="local",
                person_model_path=person_bundle,
                scene_model_path=scene_bundle,
            ),
        )
    )

    app = create_app(processor=MemoryVisualProcessor(), config=config)

    assert app.state.memory_service.store.person_dim == 3
    assert app.state.memory_service.store.scene_dim == 5
    asyncio.run(app.state.memory_service.close())


def test_configured_memory_resolve_target_endpoint_previews_without_writing(tmp_path):
    config = ServerConfig(
        memory=MemoryConfig(
            enabled=True,
            db_path=tmp_path / "memory.sqlite3",
            frame_cache_seconds=5,
            query_interval_ms=1,
            queue_size=4,
            embedding=MemoryEmbeddingConfig(backend="fake"),
            matching=MemoryMatchingConfig(
                known_person_threshold=0.95,
                known_person_margin=0.05,
                anonymous_threshold=0.95,
                anonymous_margin=0.05,
                familiar_seen_count=3,
                familiar_threshold=0.95,
                scene_threshold=0.95,
                event_cooldown_ms=5_000,
            ),
        )
    )
    with TestClient(
        create_app(processor=MemoryVisualProcessor(), config=config)
    ) as client:
        with client.websocket_connect("/v1/stream") as websocket:
            messages = _send_stable_memory_window(websocket)
            stream_ref = messages[-1]["stream_ref"]

            response = client.post(
                "/v1/memory/resolve-target",
                json={
                    "camera": "front",
                    "stream_ref": stream_ref,
                    "target": {
                        "kind": "person",
                        "intent": "self_introduction",
                        "referent_text": "我",
                    },
                },
            )

    assert response.status_code == 200
    assert response.json()["status"] == "resolved"
    assert response.json()["candidates"][0]["track_id"] == 7


def _write_local_bundle(
    tmp_path: Path,
    name: str,
    *,
    dim: int,
    files: dict[str, str],
) -> Path:
    bundle_path = tmp_path / name
    bundle_path.mkdir()
    for relative_path in files.values():
        (bundle_path / relative_path).write_bytes(b"dummy onnx")
    manifest: dict[str, Any] = {
        "model_name": name,
        "version": "v1",
        "dim": dim,
        "runtime": "onnxruntime",
        "files": files,
    }
    if set(files) == {"detector", "recognizer"}:
        manifest["input_size"] = {
            "detector": [640, 640],
            "recognizer": [112, 112],
        }
    else:
        manifest.update(
            {
                "input_size": [224, 224],
                "input_name": "image",
                "output_name": "embedding",
                "preprocess": {
                    "mean": [0.48145466, 0.4578275, 0.40821073],
                    "resize_mode": "resize_shorter_center_crop",
                    "std": [0.26862954, 0.26130258, 0.27577711],
                },
            }
        )
    (bundle_path / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")),
        encoding="utf-8",
    )
    return bundle_path
