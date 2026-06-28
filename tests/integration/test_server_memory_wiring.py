import json
import time
from typing import Any

from fastapi.testclient import TestClient

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_server.app import create_app
from visual_events_server.config import (
    MemoryConfig,
    MemoryEmbeddingConfig,
    MemoryMatchingConfig,
    ServerConfig,
)
from visual_events_server.protocol import FrameMessage, encode_frame_message


JPEG_BYTES = JPEG_1280X720


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
    ) -> None:
        self.observed.append(
            {
                "connection_id": connection_id,
                "camera": frame.camera,
                "frame_id": frame.frame_id,
                "frame_timestamp_ms": frame.timestamp_ms,
                "jpeg_bytes": frame.jpeg_bytes,
                "visual_state_frame_id": visual_state["frame_id"],
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
        "target": {"mode": "attention_target"},
        "profile": {"display_name": "张三"},
    }
    assert client.post("/v1/memory/teach/person", json=person_request).json() == {
        "ok": True,
        "person_id": "person_000001",
    }

    scene_request = {
        "camera": "front",
        "target": {"mode": "scene"},
        "memory": {"title": "新品展示区"},
    }
    assert client.post("/v1/memory/teach/scene", json=scene_request).json() == {
        "ok": True,
        "scene_id": "scene_000001",
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
        "target": {"mode": "track_id", "track_id": 7},
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
        ("teach_person", person_request),
        ("teach_scene", scene_request),
        ("add_conversation_summary", ("person_000001", summary_request)),
        ("link_external_user", link_request),
        ("get_person_by_external_user", "wechat:zhangsan"),
        ("merge_anonymous_person", merge_request),
        ("correct_identity", correction_request),
        ("resolve_target", resolve_request),
    ]


def test_default_memory_endpoints_fail_fast_when_memory_service_is_disabled():
    client = TestClient(create_app())

    response = client.post(
        "/v1/memory/teach/person",
        json={"camera": "front", "target": {"mode": "attention_target"}},
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


def test_memory_side_chain_failure_does_not_break_visual_stream():
    client = TestClient(create_app(memory_service=FailingMemoryService()))

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
        message = json.loads(websocket.receive_text())

    assert message["type"] == "visual_state"
    assert message["frame_id"] == 7
    assert message["semantic_events"] == []


class MemoryVisualProcessor:
    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
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
            "scene_context": {"attention_available": True, "target_track_id": 7},
            "scene_flags": {"has_person": True, "person_count": 1},
            "semantic_events": [],
        }


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
    client = TestClient(create_app(processor=MemoryVisualProcessor(), config=config))

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(
            encode_frame_message(
                frame_header(frame_id=1, timestamp_ms=1710000000000),
                JPEG_BYTES,
            )
        )
        assert json.loads(websocket.receive_text())["semantic_events"] == []

        person = client.post(
            "/v1/memory/teach/person",
            json={
                "camera": "front",
                "target": {"mode": "track_id", "track_id": 7},
                "profile": {
                    "display_name": "张三",
                    "description": "店长",
                    "tags": ["staff"],
                },
            },
        )
        assert person.status_code == 200
        person_id = person.json()["person_id"]

        scene = client.post(
            "/v1/memory/teach/scene",
            json={
                "camera": "front",
                "target": {"mode": "scene"},
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
        assert external["conversation_summaries"] == ["上次问过新品尺码，偏好浅色外套。"]

        websocket.send_bytes(
            encode_frame_message(
                frame_header(frame_id=2, timestamp_ms=1710000000600),
                JPEG_BYTES,
            )
        )
        message = json.loads(websocket.receive_text())
        for frame_id in range(3, 8):
            if message["semantic_events"]:
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

    events = message["semantic_events"]
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
    client = TestClient(create_app(processor=MemoryVisualProcessor(), config=config))

    with client.websocket_connect("/v1/stream") as websocket:
        websocket.send_bytes(encode_frame_message(frame_header(), JPEG_BYTES))
        json.loads(websocket.receive_text())

        response = client.post(
            "/v1/memory/resolve-target",
            json={"camera": "front", "target": {"mode": "track_id", "track_id": 7}},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "resolved"
    assert response.json()["candidates"][0]["track_id"] == 7
