from __future__ import annotations

import asyncio
import threading
import time

import pytest

from tests.jpeg_fixtures import JPEG_1280X720
from visual_events_server.memory import AppMemoryService, MemoryServiceError
from visual_events_server.memory.embedding import EmbeddingResult
from visual_events_server.memory.embedding import FakeEmbeddingBackend
from visual_events_server.memory.store import MemoryStore
from visual_events_server.protocol import FrameMessage


class BlockingQueryEmbeddingBackend(FakeEmbeddingBackend):
    def __init__(self, *, person_dim: int, scene_dim: int) -> None:
        super().__init__(person_dim=person_dim, scene_dim=scene_dim)
        self.block_queries = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def embed_person(self, image_crop: bytes):
        if self.block_queries:
            self.entered.set()
            if not self.release.wait(timeout=2.0):
                raise AssertionError("blocking embedding backend was not released")
        return super().embed_person(image_crop)


class SequencePersonEmbeddingBackend(FakeEmbeddingBackend):
    def __init__(
        self,
        *,
        person_vectors: list[tuple[float, ...]],
        scene_dim: int = 8,
    ) -> None:
        super().__init__(person_dim=len(person_vectors[0]), scene_dim=scene_dim)
        self._person_vectors = list(person_vectors)

    def embed_person(self, image_crop: bytes):
        if not self._person_vectors:
            raise AssertionError("no person embedding vector queued")
        return EmbeddingResult(
            vector=self._person_vectors.pop(0),
            embedding_type="face",
            embedding_model=self.person_model,
            embedding_version=self.model_version,
            quality=1.0,
        )


def frame(*, frame_id: int = 1, timestamp_ms: int = 1_000) -> FrameMessage:
    return FrameMessage(
        camera="front",
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        width=1280,
        height=720,
        jpeg_bytes=JPEG_1280X720,
        head_motion_state="stationary",
    )


def visual_state(*, frame_id: int = 1, timestamp_ms: int = 1_000) -> dict:
    return {
        "type": "visual_state",
        "camera": "front",
        "frame_id": frame_id,
        "frame_timestamp_ms": timestamp_ms,
        "image_size": [1280, 720],
        "tracks": [
            {
                "track_id": 7,
                "class": "person",
                "bbox_xyxy": [300.0, 100.0, 700.0, 650.0],
                "confidence": 0.92,
                "pose_confidence": 0.81,
                "head_uv": [500.0, 150.0],
                "velocity_uv_s": [0.0, 0.0],
                "age_ms": 800,
                "lost_ms": 0,
            }
        ],
        "attention": {
            "target_track_id": 7,
            "target_uv": [500.0, 160.0],
            "reason": "largest_stable_person",
            "confidence": 0.9,
        },
        "semantic_events": [],
    }


def service(tmp_path, *, now_ms: int = 10_000) -> AppMemoryService:
    clock = lambda: now_ms
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=8, scene_dim=8)
    return AppMemoryService(
        store=store,
        embedding_backend=FakeEmbeddingBackend(person_dim=8, scene_dim=8),
        frame_cache_seconds=1,
        query_interval_ms=500,
        queue_size=4,
        known_person_threshold=0.95,
        known_person_margin=0.05,
        anonymous_threshold=0.95,
        anonymous_margin=0.05,
        familiar_seen_count=3,
        familiar_threshold=0.95,
        scene_threshold=0.95,
        event_cooldown_ms=5_000,
        clock_ms=clock,
    )


@pytest.mark.asyncio
async def test_observe_visual_state_does_not_wait_for_blocking_memory_query(tmp_path):
    now = 10_000
    clock = lambda: now
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=8, scene_dim=8)
    backend = BlockingQueryEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = AppMemoryService(
        store=store,
        embedding_backend=backend,
        frame_cache_seconds=5,
        query_interval_ms=500,
        queue_size=4,
        known_person_threshold=0.95,
        known_person_margin=0.05,
        anonymous_threshold=0.95,
        anonymous_margin=0.05,
        familiar_seen_count=3,
        familiar_threshold=0.95,
        scene_threshold=0.95,
        event_cooldown_ms=5_000,
        clock_ms=clock,
    )

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=1, timestamp_ms=1_000),
        visual_state=visual_state(frame_id=1, timestamp_ms=1_000),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    person = await subject.teach_person(
        {
            "camera": "front",
            "target": {"mode": "track_id", "track_id": 7},
            "profile": {"display_name": "张三"},
        }
    )
    backend.block_queries = True

    started = time.perf_counter()
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=2, timestamp_ms=1_500),
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.1
    assert await asyncio.to_thread(backend.entered.wait, 1.0)
    assert (
        await subject.drain_completed_events(
            camera="front",
            connection_id="ws_1",
            frame_id=2,
            frame_timestamp_ms=1_500,
        )
        == []
    )

    backend.release.set()
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_500,
    )

    assert [event["event"] for event in events] == ["known_person_present"]
    assert events[0]["memory_context"]["person"]["person_id"] == person["person_id"]
    assert events[0]["evidence"]["source_frame_id"] == 2


async def _wait_for_memory_query_idle(
    subject: AppMemoryService,
    *,
    camera: str,
) -> None:
    for _ in range(20):
        pending = subject._pending_queries_by_camera.get(camera)  # noqa: SLF001
        if pending is None or pending.done():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("memory query did not become idle")


async def _drain_memory_events(
    subject: AppMemoryService,
    *,
    camera: str,
    connection_id: str,
    frame_id: int,
    frame_timestamp_ms: int,
) -> list[dict]:
    for _ in range(20):
        events = await subject.drain_completed_events(
            camera=camera,
            connection_id=connection_id,
            frame_id=frame_id,
            frame_timestamp_ms=frame_timestamp_ms,
        )
        if events:
            return events
        await asyncio.sleep(0.05)
    return []


@pytest.mark.asyncio
async def test_teach_person_requires_fresh_cached_target_and_does_not_write_on_error(tmp_path):
    now = 10_000
    subject = service(tmp_path, now_ms=now)

    with pytest.raises(MemoryServiceError) as missing:
        await subject.teach_person(
            {
                "camera": "front",
                "target": {"mode": "track_id", "track_id": 7},
                "profile": {"display_name": "张三"},
            }
        )
    assert missing.value.code == "no_active_frame"

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )

    with pytest.raises(MemoryServiceError) as absent:
        await subject.teach_person(
            {
                "camera": "front",
                "target": {"mode": "track_id", "track_id": 99},
                "profile": {"display_name": "张三"},
            }
        )
    assert absent.value.code == "target_not_visible"
    assert subject.store.get_person_profile("person_000001") is None


@pytest.mark.asyncio
async def test_teach_person_rejects_expired_cached_frame_without_writing(tmp_path):
    now = 10_000
    clock = lambda: now
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=8, scene_dim=8)
    subject = AppMemoryService(
        store=store,
        embedding_backend=FakeEmbeddingBackend(person_dim=8, scene_dim=8),
        frame_cache_seconds=1,
        query_interval_ms=500,
        queue_size=4,
        known_person_threshold=0.95,
        known_person_margin=0.05,
        anonymous_threshold=0.95,
        anonymous_margin=0.05,
        familiar_seen_count=3,
        familiar_threshold=0.95,
        scene_threshold=0.95,
        event_cooldown_ms=5_000,
        clock_ms=clock,
    )
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    now = 11_001

    with pytest.raises(MemoryServiceError) as expired:
        await subject.teach_person(
            {
                "camera": "front",
                "target": {"mode": "track_id", "track_id": 7},
                "profile": {"display_name": "张三"},
            }
        )

    assert expired.value.code == "frame_cache_expired"
    assert subject.store.get_person_profile("person_000001") is None


@pytest.mark.asyncio
async def test_teach_summary_link_and_low_frequency_memory_events(tmp_path):
    now = 10_000
    clock = lambda: now
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=8, scene_dim=8)
    subject = AppMemoryService(
        store=store,
        embedding_backend=FakeEmbeddingBackend(person_dim=8, scene_dim=8),
        frame_cache_seconds=5,
        query_interval_ms=500,
        queue_size=4,
        known_person_threshold=0.95,
        known_person_margin=0.05,
        anonymous_threshold=0.95,
        anonymous_margin=0.05,
        familiar_seen_count=3,
        familiar_threshold=0.95,
        scene_threshold=0.95,
        event_cooldown_ms=5_000,
        clock_ms=clock,
    )

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=1, timestamp_ms=1_000),
        visual_state=visual_state(frame_id=1, timestamp_ms=1_000),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    person = await subject.teach_person(
        {
            "camera": "front",
            "target": {"mode": "track_id", "track_id": 7},
            "profile": {
                "display_name": "张三",
                "description": "店长",
                "tags": ["staff"],
            },
        }
    )
    scene = await subject.teach_scene(
        {
            "camera": "front",
            "target": {"mode": "scene"},
            "memory": {
                "title": "新品展示区",
                "description": "夏季外套区域",
                "activation_hint": "介绍新品活动",
            },
        }
    )
    await subject.add_conversation_summary(
        person["person_id"],
        {
            "summary": "上次问过新品尺码，偏好浅色外套。" + ("很长" * 200),
            "source": "agent",
            "source_conversation_id": "conv-1",
        },
    )
    await subject.link_external_user(
        {
            "person_id": person["person_id"],
            "external_user_ref": "wechat:zhangsan",
        }
    )

    external = await subject.get_person_by_external_user("wechat:zhangsan")
    assert external["person"]["person_id"] == person["person_id"]
    assert external["conversation_summaries"][0].startswith("上次问过新品尺码")
    assert len(external["conversation_summaries"][0]) <= 240

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=2, timestamp_ms=1_500),
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
    )
    first = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_500,
    )
    assert [event["event"] for event in first] == [
        "known_person_present",
        "scene_activated",
    ]
    assert first[0]["memory_context"]["person"]["display_name"] == "张三"
    assert first[0]["memory_context"]["conversation_summaries"][0].startswith(
        "上次问过新品尺码"
    )
    assert first[0]["evidence"]["memory_match_id"].startswith("match_")
    assert first[0]["evidence"]["source_frame_id"] == 2
    assert first[1]["memory_context"]["scene"]["scene_id"] == scene["scene_id"]

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=3, timestamp_ms=1_600),
        visual_state=visual_state(frame_id=3, timestamp_ms=1_600),
    )
    assert (
        await subject.drain_completed_events(
            camera="front",
            connection_id="ws_1",
            frame_id=3,
            frame_timestamp_ms=1_600,
        )
        == []
    )


@pytest.mark.asyncio
async def test_unknown_person_becomes_familiar_unknown_then_merge_suppresses_anonymous_event(
    tmp_path,
):
    now = 10_000

    def clock() -> int:
        return now

    backend = SequencePersonEmbeddingBackend(
        person_vectors=[
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        ],
        scene_dim=4,
    )
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    subject = AppMemoryService(
        store=store,
        embedding_backend=backend,
        frame_cache_seconds=5,
        query_interval_ms=500,
        queue_size=4,
        known_person_threshold=0.95,
        known_person_margin=0.05,
        anonymous_threshold=0.95,
        anonymous_margin=0.0,
        familiar_seen_count=2,
        familiar_threshold=0.95,
        scene_threshold=1.1,
        event_cooldown_ms=0,
        clock_ms=clock,
    )

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=1, timestamp_ms=1_000),
        visual_state=visual_state(frame_id=1, timestamp_ms=1_000),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    assert store.search_anonymous_embeddings(backend.embed_person(b"unused"), limit=2)
    assert (
        await subject.drain_completed_events(
            camera="front",
            connection_id="ws_1",
            frame_id=1,
            frame_timestamp_ms=1_000,
        )
        == []
    )

    now = 10_600
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=2, timestamp_ms=1_600),
        visual_state=visual_state(frame_id=2, timestamp_ms=1_600),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_600,
    )

    assert [event["event"] for event in events] == ["familiar_unknown_present"]
    anonymous_id = events[0]["memory_context"]["anonymous_person"]["anonymous_id"]
    assert events[0]["memory_context"]["anonymous_person"]["seen_count"] == 2

    merged = await subject.merge_anonymous_person(
        {
            "anonymous_id": anonymous_id,
            "profile": {"display_name": "新客户", "description": "常客"},
        }
    )
    assert merged["ok"] is True
    assert merged["person_id"].startswith("person_")
    assert merged["copied_embedding_count"] == 1
    assert store.get_active_anonymous_profile(anonymous_id) is None

    now = 11_200
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=3, timestamp_ms=2_200),
        visual_state=visual_state(frame_id=3, timestamp_ms=2_200),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=3,
        frame_timestamp_ms=2_200,
    )

    assert [event["event"] for event in events] == ["known_person_present"]
    assert events[0]["memory_context"]["person"]["person_id"] == merged["person_id"]


@pytest.mark.asyncio
async def test_correct_identity_blocks_same_wrong_candidate_without_deleting_profile(
    tmp_path,
):
    now = 10_000
    backend = SequencePersonEmbeddingBackend(
        person_vectors=[
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        ],
        scene_dim=4,
    )
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    subject = AppMemoryService(
        store=store,
        embedding_backend=backend,
        frame_cache_seconds=5,
        query_interval_ms=500,
        queue_size=4,
        known_person_threshold=0.95,
        known_person_margin=0.0,
        anonymous_threshold=0.95,
        anonymous_margin=0.0,
        familiar_seen_count=3,
        familiar_threshold=0.95,
        scene_threshold=1.1,
        event_cooldown_ms=0,
        clock_ms=lambda: now,
    )
    store.upsert_person_profile(
        person_id="person_wrong",
        display_name="错的人",
        description="",
        tags=(),
        now_ms=now,
    )
    store.add_person_embedding(
        person_id="person_wrong",
        result=EmbeddingResult(
            vector=(1.0, 0.0, 0.0, 0.0),
            embedding_type="face",
            embedding_model=backend.person_model,
            embedding_version=backend.model_version,
            quality=1.0,
        ),
        source_target_type="track_id",
        now_ms=now,
    )

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=1, timestamp_ms=1_000),
        visual_state=visual_state(frame_id=1, timestamp_ms=1_000),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=1,
        frame_timestamp_ms=1_000,
    )
    assert [event["event"] for event in events] == ["known_person_present"]
    memory_match_id = events[0]["evidence"]["memory_match_id"]

    corrected = await subject.correct_identity(
        {
            "memory_match_id": memory_match_id,
            "wrong_person_id": "person_wrong",
        }
    )

    assert corrected == {
        "ok": True,
        "memory_match_id": memory_match_id,
        "wrong_person_id": "person_wrong",
    }
    assert store.get_person_profile("person_wrong") is not None

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=2, timestamp_ms=1_600),
        visual_state=visual_state(frame_id=2, timestamp_ms=1_600),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_600,
    )

    assert [event["event"] for event in events] == []


@pytest.mark.asyncio
async def test_resolve_target_preview_and_point_teach_rejects_ambiguous_write(tmp_path):
    subject = service(tmp_path)
    state = visual_state()
    state["tracks"].append(
        {
            "track_id": 8,
            "class": "person",
            "bbox_xyxy": [250.0, 90.0, 750.0, 670.0],
            "confidence": 0.9,
            "pose_confidence": 0.8,
            "head_uv": [530.0, 150.0],
            "velocity_uv_s": [0.0, 0.0],
            "age_ms": 800,
            "lost_ms": 0,
        }
    )
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=state,
    )

    preview = await subject.resolve_target(
        {
            "camera": "front",
            "target": {"mode": "point_uv", "point_uv": [500.0, 160.0]},
        }
    )

    assert preview["ok"] is True
    assert preview["status"] == "ambiguous"
    assert [candidate["track_id"] for candidate in preview["candidates"]] == [7, 8]

    with pytest.raises(MemoryServiceError) as exc:
        await subject.teach_person(
            {
                "camera": "front",
                "target": {"mode": "point_uv", "point_uv": [500.0, 160.0]},
                "profile": {"display_name": "张三"},
            }
        )

    assert exc.value.code == "target_ambiguous"
    assert subject.store.search_person_embeddings(
        subject.embedding_backend.embed_person(b"unused"),
        limit=1,
    ) == []


@pytest.mark.asyncio
async def test_teach_scene_persists_region_id_and_emits_it_in_scene_context(tmp_path):
    subject = service(tmp_path)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=1, timestamp_ms=1_000),
        visual_state=visual_state(frame_id=1, timestamp_ms=1_000),
    )
    await _wait_for_memory_query_idle(subject, camera="front")

    scene = await subject.teach_scene(
        {
            "camera": "front",
            "target": {"mode": "scene"},
            "memory": {
                "title": "门口",
                "description": "入口区域",
                "activation_hint": "问候进门客户",
                "region_id": "front_door",
            },
        }
    )
    assert subject.store.get_scene_memory(scene["scene_id"])["region_id"] == "front_door"

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(frame_id=2, timestamp_ms=1_500),
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_500,
    )

    scene_events = [event for event in events if event["event"] == "scene_activated"]
    assert scene_events[0]["memory_context"]["scene"]["region_id"] == "front_door"
