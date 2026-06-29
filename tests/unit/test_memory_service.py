from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from visual_events_server.memory import AppMemoryService, MemoryServiceError
from visual_events_server.memory.embedding import EmbeddingResult
from visual_events_server.memory.embedding import FakeEmbeddingBackend
from visual_events_server.memory.service import _target_bytes
from visual_events_server.memory.store import MemoryStore
from visual_events_server.memory.target_resolver import ResolvedTarget
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


class RecordingEmbeddingBackend(FakeEmbeddingBackend):
    def __init__(self, *, person_dim: int, scene_dim: int) -> None:
        super().__init__(person_dim=person_dim, scene_dim=scene_dim)
        self.person_inputs: list[bytes] = []
        self.scene_inputs: list[bytes] = []

    def embed_person(self, image_crop: bytes):
        self.person_inputs.append(image_crop)
        return super().embed_person(image_crop)

    def embed_scene(self, image_or_crop: bytes):
        self.scene_inputs.append(image_or_crop)
        return super().embed_scene(image_or_crop)


def _jpeg_1280x720() -> bytes:
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


JPEG_1280X720 = _jpeg_1280x720()


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


def service(
    tmp_path,
    *,
    now_ms: int = 10_000,
    embedding_backend: FakeEmbeddingBackend | None = None,
) -> AppMemoryService:
    clock = lambda: now_ms
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=8, scene_dim=8)
    return AppMemoryService(
        store=store,
        embedding_backend=embedding_backend
        if embedding_backend is not None
        else FakeEmbeddingBackend(person_dim=8, scene_dim=8),
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


def _decoded_jpeg(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as image:
        assert image.format == "JPEG"
        return image.convert("RGB")


def _assert_rgb_close(
    actual: tuple[int, int, int],
    expected: tuple[int, int, int],
    *,
    tolerance: int = 35,
) -> None:
    assert all(abs(left - right) <= tolerance for left, right in zip(actual, expected))


@pytest.mark.asyncio
async def test_teach_person_sends_decodable_face_safe_person_crop(tmp_path):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()
    backend.scene_inputs.clear()

    await subject.teach_person(
        {
            "camera": "front",
            "target": {"mode": "track_id", "track_id": 7},
            "profile": {"display_name": "张三"},
        }
    )

    assert len(backend.person_inputs) == 1
    crop = _decoded_jpeg(backend.person_inputs[0])
    original = _decoded_jpeg(JPEG_1280X720)
    assert crop.size == (480, 660)
    assert crop.width < original.width
    assert crop.height < original.height
    _assert_rgb_close(crop.getpixel((20, 100)), original.getpixel((280, 145)))
    _assert_rgb_close(crop.getpixel((80, 20)), original.getpixel((340, 65)))
    _assert_rgb_close(crop.getpixel((460, 100)), original.getpixel((720, 145)))
    _assert_rgb_close(crop.getpixel((80, 640)), original.getpixel((340, 685)))
    _assert_rgb_close(crop.getpixel((60, 75)), original.getpixel((320, 120)))


@pytest.mark.asyncio
async def test_teach_person_persists_embedding_provenance_for_resolved_target(tmp_path):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()
    backend.scene_inputs.clear()

    person = await subject.teach_person(
        {
            "camera": "front",
            "target": {"mode": "track_id", "track_id": 7},
            "profile": {"display_name": "张三"},
        }
    )

    assert len(backend.person_inputs) == 1
    expected_embedding = FakeEmbeddingBackend(
        person_dim=8,
        scene_dim=8,
    ).embed_person(backend.person_inputs[0])
    matches = subject.store.search_person_embeddings(expected_embedding, limit=1)
    assert len(matches) == 1
    assert matches[0].matched_id == person["person_id"]
    assert subject.store.get_embedding_provenance(matches[0].embedding_id) == {
        "embedding_id": matches[0].embedding_id,
        "owner_type": "person",
        "owner_id": person["person_id"],
        "source_track_ref": "front:track:7",
        "source_frame_ref": "front:1:1000",
        "crop_hash": hashlib.sha256(backend.person_inputs[0]).hexdigest(),
        "crop_path_or_artifact_ref": None,
        "resolver_target_ref": "front:track:7",
        "resolution_reason": "track_id",
        "embedding_type": "face",
        "embedding_model": "fake-face",
        "embedding_version": "test-v1",
        "embedding_dim": 8,
        "created_at_ms": 10000,
    }


@pytest.mark.asyncio
async def test_teach_person_bbox_target_uses_face_safe_person_crop(tmp_path):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()
    backend.scene_inputs.clear()

    await subject.teach_person(
        {
            "camera": "front",
            "target": {
                "mode": "bbox",
                "bbox_xyxy": [300.0, 100.0, 700.0, 650.0],
            },
            "profile": {"display_name": "张三"},
        }
    )

    assert len(backend.person_inputs) == 1
    crop = _decoded_jpeg(backend.person_inputs[0])
    original = _decoded_jpeg(JPEG_1280X720)
    assert crop.size == (480, 660)
    _assert_rgb_close(crop.getpixel((20, 100)), original.getpixel((280, 145)))
    _assert_rgb_close(crop.getpixel((80, 20)), original.getpixel((340, 65)))
    _assert_rgb_close(crop.getpixel((460, 100)), original.getpixel((720, 145)))
    _assert_rgb_close(crop.getpixel((80, 640)), original.getpixel((340, 685)))


def test_target_bytes_region_uses_exact_bbox_crop():
    target = ResolvedTarget(
        source_target_mode="bbox",
        target_type="region",
        bbox_xyxy=(300.0, 100.0, 700.0, 650.0),
        track_id=None,
        quality="usable",
    )

    crop = _decoded_jpeg(_target_bytes(JPEG_1280X720, target))
    original = _decoded_jpeg(JPEG_1280X720)

    assert crop.size == (400, 550)
    _assert_rgb_close(crop.getpixel((50, 50)), original.getpixel((350, 150)))
    _assert_rgb_close(crop.getpixel((350, 500)), original.getpixel((650, 600)))


@pytest.mark.asyncio
async def test_teach_person_rejects_invalid_bbox_without_embedding_or_write(tmp_path):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()
    backend.scene_inputs.clear()

    with pytest.raises(MemoryServiceError) as exc:
        await subject.teach_person(
            {
                "camera": "front",
                "target": {
                    "mode": "bbox",
                    "bbox_xyxy": [300.0, 100.0, 305.0, 105.0],
                },
                "profile": {"display_name": "张三"},
            }
        )

    assert exc.value.code == "target_too_small"
    assert backend.person_inputs == []
    assert subject.store.search_person_embeddings(
        FakeEmbeddingBackend(person_dim=8, scene_dim=8).embed_person(b"unused"),
        limit=1,
    ) == []


@pytest.mark.asyncio
async def test_teach_scene_scene_mode_sends_original_decodable_jpeg_to_backend(tmp_path):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()
    backend.scene_inputs.clear()

    await subject.teach_scene(
        {
            "camera": "front",
            "target": {"mode": "scene"},
            "memory": {"title": "新品展示区"},
        }
    )

    assert backend.scene_inputs == [JPEG_1280X720]
    scene_image = _decoded_jpeg(backend.scene_inputs[0])
    assert scene_image.size == (1280, 720)


@pytest.mark.asyncio
async def test_teach_scene_persists_embedding_provenance_for_full_frame(tmp_path):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()
    backend.scene_inputs.clear()

    scene = await subject.teach_scene(
        {
            "camera": "front",
            "target": {"mode": "scene"},
            "memory": {"title": "新品展示区"},
        }
    )

    assert backend.scene_inputs == [JPEG_1280X720]
    expected_embedding = FakeEmbeddingBackend(
        person_dim=8,
        scene_dim=8,
    ).embed_scene(backend.scene_inputs[0])
    matches = subject.store.search_scene_embeddings(expected_embedding, limit=1)
    assert len(matches) == 1
    assert matches[0].matched_id == scene["scene_id"]
    assert subject.store.get_embedding_provenance(matches[0].embedding_id) == {
        "embedding_id": matches[0].embedding_id,
        "owner_type": "scene",
        "owner_id": scene["scene_id"],
        "source_track_ref": None,
        "source_frame_ref": "front:1:1000",
        "crop_hash": hashlib.sha256(JPEG_1280X720).hexdigest(),
        "crop_path_or_artifact_ref": None,
        "resolver_target_ref": "scene",
        "resolution_reason": "scene",
        "embedding_type": "scene",
        "embedding_model": "fake-scene",
        "embedding_version": "test-v1",
        "embedding_dim": 8,
        "created_at_ms": 10000,
    }


@pytest.mark.asyncio
async def test_teach_scene_rejects_bbox_target_without_writing_scene(tmp_path):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()
    backend.scene_inputs.clear()

    with pytest.raises(MemoryServiceError) as exc:
        await subject.teach_scene(
            {
                "camera": "front",
                "target": {
                    "mode": "bbox",
                    "bbox_xyxy": [300.0, 100.0, 700.0, 650.0],
                },
                "memory": {"title": "新品展示区局部"},
            }
        )

    assert exc.value.code == "unsupported_scene_target"
    assert "target.mode=scene" in exc.value.message
    assert backend.scene_inputs == []
    assert subject.store.search_scene_embeddings(
        FakeEmbeddingBackend(person_dim=8, scene_dim=8).embed_scene(b"unused"),
        limit=1,
    ) == []


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
