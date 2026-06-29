from __future__ import annotations

import asyncio
import hashlib
import inspect
import threading
import time
from contextlib import suppress
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from visual_events_server.memory import AppMemoryService, MemoryServiceError
from visual_events_server.memory.embedding import EmbeddingResult, EmbeddingUnavailable
from visual_events_server.memory.embedding import FakeEmbeddingBackend
from visual_events_server.memory.frame_cache import MemoryFrameSnapshot
from visual_events_server.memory.service import (
    _person_embedding_input_candidates,
    _target_bytes,
)
from visual_events_server.memory.store import MemoryStore
from visual_events_server.memory.target_resolver import ResolvedTarget
from visual_events_server.attention import AttentionResult
from visual_events_server.inference.base import PoseKeypoint
from visual_events_server.tracking import TrackSnapshot
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


class BlockingTeachEmbeddingBackend(FakeEmbeddingBackend):
    def __init__(
        self,
        *,
        person_dim: int,
        scene_dim: int,
        block_person: bool = False,
        block_scene: bool = False,
        wait_timeout_s: float = 0.5,
    ) -> None:
        super().__init__(person_dim=person_dim, scene_dim=scene_dim)
        self.block_person = block_person
        self.block_scene = block_scene
        self.wait_timeout_s = wait_timeout_s
        self.person_entered = threading.Event()
        self.scene_entered = threading.Event()
        self.release = threading.Event()

    def embed_person(self, image_crop: bytes):
        if self.block_person:
            self.person_entered.set()
            if not self.release.wait(timeout=self.wait_timeout_s):
                raise AssertionError(
                    "blocking person embedding backend was not released"
                )
        return super().embed_person(image_crop)

    def embed_scene(self, image_or_crop: bytes):
        if self.block_scene:
            self.scene_entered.set()
            if not self.release.wait(timeout=self.wait_timeout_s):
                raise AssertionError(
                    "blocking scene embedding backend was not released"
                )
        return super().embed_scene(image_or_crop)


class SequencePersonEmbeddingBackend(FakeEmbeddingBackend):
    def __init__(
        self,
        *,
        person_vectors: list[tuple[float, ...]],
        scene_dim: int = 8,
    ) -> None:
        super().__init__(person_dim=len(person_vectors[0]), scene_dim=scene_dim)
        self._person_vectors = list(person_vectors)
        self.person_inputs: list[bytes] = []

    def embed_person(self, image_crop: bytes):
        self.person_inputs.append(image_crop)
        if not self._person_vectors:
            raise AssertionError("no person embedding vector queued")
        return EmbeddingResult(
            vector=self._person_vectors.pop(0),
            embedding_type="face",
            embedding_model=self.person_model,
            embedding_version=self.model_version,
            quality=1.0,
        )


class ScriptedPersonEmbeddingBackend(FakeEmbeddingBackend):
    def __init__(
        self,
        *,
        person_outcomes: list[tuple[float, ...] | EmbeddingUnavailable],
        scene_dim: int = 8,
    ) -> None:
        super().__init__(
            person_dim=len(_first_vector(person_outcomes)),
            scene_dim=scene_dim,
        )
        self._person_outcomes = list(person_outcomes)
        self.person_inputs: list[bytes] = []
        self.scene_inputs: list[bytes] = []

    def embed_person(self, image_crop: bytes):
        self.person_inputs.append(image_crop)
        if not self._person_outcomes:
            raise AssertionError("no person embedding outcome queued")
        outcome = self._person_outcomes.pop(0)
        if isinstance(outcome, EmbeddingUnavailable):
            raise outcome
        return EmbeddingResult(
            vector=outcome,
            embedding_type="face",
            embedding_model=self.person_model,
            embedding_version=self.model_version,
            quality=1.0,
        )

    def embed_scene(self, image_or_crop: bytes):
        self.scene_inputs.append(image_or_crop)
        return super().embed_scene(image_or_crop)

    def set_person_outcomes(
        self,
        outcomes: list[tuple[float, ...] | EmbeddingUnavailable],
    ) -> None:
        self._person_outcomes = list(outcomes)


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


def _first_vector(
    outcomes: list[tuple[float, ...] | EmbeddingUnavailable],
) -> tuple[float, ...]:
    for outcome in outcomes:
        if not isinstance(outcome, EmbeddingUnavailable):
            return outcome
    return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _no_usable_face() -> EmbeddingUnavailable:
    return EmbeddingUnavailable("no_usable_face", "no usable face detected")


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
ALLOWED_AMBIGUITY_TYPES = {
    "introducer_unclear",
    "target_unclear",
    "pose_unclear",
    "multiple_candidates",
    "stale_interaction",
    "no_active_interaction_target",
    "unsupported_target_kind",
    "quality_too_low",
}
_CREATED_SERVICES: list[AppMemoryService] = []


@pytest.fixture(autouse=True)
async def close_created_memory_services():
    _CREATED_SERVICES.clear()
    try:
        yield
    finally:
        while _CREATED_SERVICES:
            await _close_service(_CREATED_SERVICES.pop())


def _track_service(subject: AppMemoryService) -> AppMemoryService:
    _CREATED_SERVICES.append(subject)
    return subject


async def _close_service(subject: AppMemoryService) -> None:
    result = subject.close()
    if inspect.isawaitable(result):
        await result


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


def kp(name: str, x: float, y: float, confidence: float = 0.9) -> PoseKeypoint:
    return PoseKeypoint(name=name, x=x, y=y, confidence=confidence)


def pointing_right_keypoints() -> tuple[PoseKeypoint, ...]:
    return (
        kp("left_shoulder", 420.0, 240.0),
        kp("left_elbow", 520.0, 260.0),
        kp("left_wrist", 620.0, 275.0),
    )


def memory_snapshot(
    *,
    frame_message: FrameMessage | None = None,
    engagement_state: str = "available",
    attention_available: bool = True,
    scene_target_track_id: int | None = 7,
    attention_track_id: int | None = 7,
    lost_ms: int = 0,
) -> MemoryFrameSnapshot:
    source_frame = frame_message or frame()
    track = TrackSnapshot(
        track_id=7,
        first_seen_ms=source_frame.timestamp_ms - 800,
        last_seen_ms=source_frame.timestamp_ms - lost_ms,
        frame_timestamp_ms=source_frame.timestamp_ms,
        bbox_xyxy=(300.0, 100.0, 700.0, 650.0),
        confidence=0.92,
        pose_confidence=0.81,
        head_uv=(500.0, 150.0),
        velocity_uv_s=(0.0, 0.0),
        lost_ms=lost_ms,
        hits=2,
        misses=0,
        keypoints=(),
    )
    attention = (
        AttentionResult(
            target_track_id=attention_track_id,
            target_uv=(500.0, 160.0),
            reason="largest_stable_person",
            confidence=0.9,
            largest_person_stable=True,
        )
        if attention_track_id is not None
        else None
    )
    return MemoryFrameSnapshot(
        connection_id="ws_1",
        frame=source_frame,
        source_frame_ref=f"{source_frame.camera}:{source_frame.frame_id}:{source_frame.timestamp_ms}",
        snapshot_ref=f"snapshot:{source_frame.camera}:{source_frame.frame_id}",
        observed_at_ms=10_000,
        image_size=(source_frame.width, source_frame.height),
        tracks=[track],
        attention=attention,
        scene_context={
            "engagement_state": engagement_state,
            "attention_available": attention_available,
            "target_track_id": scene_target_track_id,
        },
        semantic_events=[],
    )


def memory_snapshot_with_tracks(
    tracks: list[TrackSnapshot],
    *,
    frame_message: FrameMessage | None = None,
    attention_track_id: int | None = 7,
    scene_target_track_id: int | None = 7,
) -> MemoryFrameSnapshot:
    source_frame = frame_message or frame()
    attention = (
        AttentionResult(
            target_track_id=attention_track_id,
            target_uv=(500.0, 160.0),
            reason="largest_stable_person",
            confidence=0.9,
            largest_person_stable=True,
        )
        if attention_track_id is not None
        else None
    )
    return MemoryFrameSnapshot(
        connection_id="ws_1",
        frame=source_frame,
        source_frame_ref=f"{source_frame.camera}:{source_frame.frame_id}:{source_frame.timestamp_ms}",
        snapshot_ref=f"snapshot:{source_frame.camera}:{source_frame.frame_id}",
        observed_at_ms=10_000,
        image_size=(source_frame.width, source_frame.height),
        tracks=tracks,
        attention=attention,
        scene_context={
            "engagement_state": "available",
            "attention_available": attention is not None,
            "target_track_id": scene_target_track_id,
        },
        semantic_events=[],
    )


def person_track(
    track_id: int,
    *,
    bbox_xyxy: tuple[float, float, float, float] | None = None,
    class_name: str = "person",
    lost_ms: int = 0,
    hits: int = 2,
    frame_message: FrameMessage | None = None,
    keypoints: tuple[PoseKeypoint, ...] = (),
) -> TrackSnapshot:
    source_frame = frame_message or frame()
    bbox = bbox_xyxy or (
        300.0 + float(track_id),
        100.0,
        700.0 + float(track_id),
        650.0,
    )
    return TrackSnapshot(
        track_id=track_id,
        first_seen_ms=source_frame.timestamp_ms - 800,
        last_seen_ms=source_frame.timestamp_ms - lost_ms,
        frame_timestamp_ms=source_frame.timestamp_ms,
        bbox_xyxy=bbox,
        confidence=0.92,
        pose_confidence=0.81,
        head_uv=((bbox[0] + bbox[2]) / 2.0, bbox[1] + 50.0),
        velocity_uv_s=(0.0, 0.0),
        lost_ms=lost_ms,
        hits=hits,
        misses=0,
        class_name=class_name,
        keypoints=keypoints,
    )


def service(
    tmp_path,
    *,
    now_ms: int = 10_000,
    embedding_backend: FakeEmbeddingBackend | None = None,
    teach_queue_size: int = 2,
    teach_queue_timeout_ms: int = 500,
) -> AppMemoryService:
    clock = lambda: now_ms
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=8, scene_dim=8)
    return _track_service(
        AppMemoryService(
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
            teach_queue_size=teach_queue_size,
            teach_queue_timeout_ms=teach_queue_timeout_ms,
        )
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
async def test_teach_person_embedding_does_not_block_event_loop_when_backend_blocks(
    tmp_path,
):
    backend = BlockingTeachEmbeddingBackend(
        person_dim=8,
        scene_dim=8,
        block_person=True,
    )
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")

    teach = asyncio.create_task(
        subject.teach_person(
            {
                "camera": "front",
                "target": {"mode": "track_id", "track_id": 7},
                "profile": {"display_name": "张三"},
            }
        )
    )

    try:
        started = time.perf_counter()
        await asyncio.sleep(0)
        assert time.perf_counter() - started < 0.1
        assert await asyncio.to_thread(backend.person_entered.wait, 1.0)
    finally:
        backend.release.set()
        with suppress(Exception):
            await asyncio.wait_for(teach, 1.0)


@pytest.mark.asyncio
async def test_teach_scene_embedding_does_not_block_event_loop_when_backend_blocks(
    tmp_path,
):
    backend = BlockingTeachEmbeddingBackend(
        person_dim=8,
        scene_dim=8,
        block_scene=True,
    )
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")

    teach = asyncio.create_task(
        subject.teach_scene(
            {
                "camera": "front",
                "target": {"mode": "scene"},
                "memory": {"title": "新品展示区"},
            }
        )
    )

    try:
        started = time.perf_counter()
        await asyncio.sleep(0)
        assert time.perf_counter() - started < 0.1
        assert await asyncio.to_thread(backend.scene_entered.wait, 1.0)
    finally:
        backend.release.set()
        with suppress(Exception):
            await asyncio.wait_for(teach, 1.0)


@pytest.mark.asyncio
async def test_teach_embedding_backpressure_rejects_second_write_and_first_persists(
    tmp_path,
):
    backend = BlockingTeachEmbeddingBackend(
        person_dim=8,
        scene_dim=8,
        block_person=True,
        wait_timeout_s=2.0,
    )
    subject = service(
        tmp_path,
        embedding_backend=backend,
        teach_queue_size=0,
        teach_queue_timeout_ms=25,
    )
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")

    first = asyncio.create_task(
        subject.teach_person(
            {
                "camera": "front",
                "target": {"mode": "track_id", "track_id": 7},
                "profile": {"display_name": "第一位"},
            }
        )
    )
    assert await asyncio.to_thread(backend.person_entered.wait, 1.0)

    try:
        with pytest.raises(MemoryServiceError) as exc:
            await subject.teach_person(
                {
                    "camera": "front",
                    "target": {"mode": "track_id", "track_id": 7},
                    "profile": {"display_name": "第二位"},
                }
            )

        assert exc.value.code == "embedding_queue_full"
        assert exc.value.status_code == 503
        assert _count_rows(subject.store, "person_profiles") == 0
        assert _count_rows(subject.store, "person_embeddings") == 0
        assert _count_rows(subject.store, "embedding_provenance") == 0
    finally:
        backend.release.set()

    created = await asyncio.wait_for(first, 1.0)

    assert created["person_id"].startswith("person_")
    assert _count_rows(subject.store, "person_profiles") == 1
    assert _count_rows(subject.store, "person_embeddings") == 1
    assert _count_rows(subject.store, "embedding_provenance") == 1


@pytest.mark.asyncio
async def test_teach_embedding_slot_survives_cancelled_request_until_job_finishes(
    tmp_path,
):
    backend = BlockingTeachEmbeddingBackend(
        person_dim=8,
        scene_dim=8,
        block_person=True,
        wait_timeout_s=2.0,
    )
    subject = service(
        tmp_path,
        embedding_backend=backend,
        teach_queue_size=0,
        teach_queue_timeout_ms=25,
    )
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")

    first = asyncio.create_task(
        subject.teach_person(
            {
                "camera": "front",
                "target": {"mode": "track_id", "track_id": 7},
                "profile": {"display_name": "第一位"},
            }
        )
    )
    assert await asyncio.to_thread(backend.person_entered.wait, 1.0)

    try:
        first.cancel()
        with suppress(asyncio.CancelledError):
            await first

        with pytest.raises(MemoryServiceError) as exc:
            await asyncio.wait_for(
                subject.teach_person(
                    {
                        "camera": "front",
                        "target": {"mode": "track_id", "track_id": 7},
                        "profile": {"display_name": "第二位"},
                    }
                ),
                timeout=0.5,
            )

        assert exc.value.code == "embedding_queue_full"
        assert _count_rows(subject.store, "person_profiles") == 0
        assert _count_rows(subject.store, "person_embeddings") == 0
        assert _count_rows(subject.store, "embedding_provenance") == 0
    finally:
        backend.release.set()


@pytest.mark.asyncio
async def test_close_waits_for_in_flight_teach_before_closing_store(tmp_path):
    backend = BlockingTeachEmbeddingBackend(
        person_dim=8,
        scene_dim=8,
        block_person=True,
        wait_timeout_s=2.0,
    )
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")

    teach = asyncio.create_task(
        subject.teach_person(
            {
                "camera": "front",
                "target": {"mode": "track_id", "track_id": 7},
                "profile": {"display_name": "第一位"},
            }
        )
    )
    assert await asyncio.to_thread(backend.person_entered.wait, 1.0)

    close_task = asyncio.create_task(_close_service(subject))
    try:
        await asyncio.sleep(0.05)
        assert not close_task.done()
        assert _count_rows(subject.store, "person_profiles") == 0
    finally:
        backend.release.set()

    await asyncio.wait_for(close_task, 1.0)
    await asyncio.wait_for(teach, 1.0)


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
async def test_teach_person_falls_back_to_target_masked_context_on_no_usable_face(
    tmp_path,
):
    success_vector = _unit_vector(0)
    backend = ScriptedPersonEmbeddingBackend(
        person_outcomes=[_no_usable_face(), success_vector],
    )
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

    target = ResolvedTarget(
        source_target_mode="track_id",
        target_type="person",
        bbox_xyxy=(300.0, 100.0, 700.0, 650.0),
        track_id=7,
        quality="usable",
    )
    assert len(backend.person_inputs) == 2
    assert backend.person_inputs[0] == _target_bytes(JPEG_1280X720, target)
    fallback = _decoded_jpeg(backend.person_inputs[1])
    original = _decoded_jpeg(JPEG_1280X720)
    assert fallback.size == original.size
    _assert_rgb_close(fallback.getpixel((20, 20)), (0, 0, 0), tolerance=10)
    _assert_rgb_close(fallback.getpixel((500, 150)), original.getpixel((500, 150)))

    matches = subject.store.search_person_embeddings(
        EmbeddingResult(
            vector=success_vector,
            embedding_type="face",
            embedding_model=backend.person_model,
            embedding_version=backend.model_version,
            quality=1.0,
        ),
        limit=1,
    )
    assert len(matches) == 1
    assert matches[0].matched_id == person["person_id"]
    provenance = subject.store.get_embedding_provenance(matches[0].embedding_id)
    assert provenance["crop_hash"] == hashlib.sha256(
        backend.person_inputs[1],
    ).hexdigest()
    assert person["evidence"]["crop_hash"] == provenance["crop_hash"]
    assert _count_rows(subject.store, "person_profiles") == 1
    assert _count_rows(subject.store, "person_embeddings") == 1
    assert _count_rows(subject.store, "embedding_provenance") == 1


@pytest.mark.asyncio
async def test_teach_person_no_usable_face_all_candidates_does_not_write(tmp_path):
    backend = ScriptedPersonEmbeddingBackend(
        person_outcomes=[_no_usable_face(), _no_usable_face()],
    )
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
                "target": {"mode": "track_id", "track_id": 7},
                "profile": {"display_name": "张三"},
            }
        )

    assert exc.value.code == "no_usable_face"
    assert len(backend.person_inputs) == 2
    assert _count_rows(subject.store, "person_profiles") == 0
    assert _count_rows(subject.store, "person_embeddings") == 0
    assert _count_rows(subject.store, "embedding_provenance") == 0


@pytest.mark.asyncio
async def test_self_introduction_requires_active_interaction_target_and_does_not_fallback(
    tmp_path,
):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()

    preview = await subject.resolve_target(
        {
            "camera": "front",
            "target": {
                "kind": "person",
                "intent": "self_introduction",
                "referent_text": "我",
            },
        }
    )
    assert preview["ok"] is True
    assert preview["status"] == "ambiguous"
    assert preview["retryable"] is True
    assert preview["ask_user_hint"] is True
    assert preview["ambiguity_type"] == "no_active_interaction_target"
    assert preview["candidates"] == []
    _assert_allowed_ambiguity(preview)

    with pytest.raises(MemoryServiceError) as exc:
        await subject.teach_person(
            {
                "camera": "front",
                "target": {
                    "kind": "person",
                    "intent": "self_introduction",
                    "referent_text": "我",
                },
                "profile": {"display_name": "张三"},
            }
        )

    assert exc.value.code == "target_ambiguous"
    assert exc.value.status_code == 409
    assert exc.value.details["ambiguity_type"] == "no_active_interaction_target"
    _assert_allowed_ambiguity(exc.value.details)
    assert backend.person_inputs == []
    assert subject.store.search_person_embeddings(
        FakeEmbeddingBackend(person_dim=8, scene_dim=8).embed_person(b"unused"),
        limit=1,
    ) == []
    assert _count_rows(subject.store, "person_profiles") == 0


@pytest.mark.asyncio
async def test_self_introduction_uses_active_interaction_target_for_write(tmp_path):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    source_frame = frame()
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(),
        memory_snapshot=memory_snapshot(frame_message=source_frame),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()

    person = await subject.teach_person(
        {
            "camera": "front",
            "target": {
                "kind": "person",
                "intent": "self_introduction",
                "referent_text": "我",
            },
            "profile": {"display_name": "张三"},
        }
    )

    assert len(backend.person_inputs) == 1
    expected_embedding = FakeEmbeddingBackend(
        person_dim=8,
        scene_dim=8,
    ).embed_person(backend.person_inputs[0])
    matches = subject.store.search_person_embeddings(expected_embedding, limit=1)
    assert matches[0].matched_id == person["person_id"]
    provenance = subject.store.get_embedding_provenance(matches[0].embedding_id)
    assert person["evidence"]["source_frame_ref"] == provenance["source_frame_ref"]
    assert person["evidence"]["request_snapshot_ref"] == "snapshot:front:1"
    assert person["evidence"]["source_track_ref"] == provenance["source_track_ref"]
    assert person["evidence"]["resolver_target_ref"] == provenance["resolver_target_ref"]
    assert person["evidence"]["resolution_reason"] == provenance["resolution_reason"]
    assert person["evidence"]["crop_hash"] == provenance["crop_hash"]
    assert provenance["source_track_ref"] == "front:track:7"
    assert provenance["resolver_target_ref"] == "front:track:7"
    assert provenance["resolution_reason"] == "active_interaction_target"
    row = subject.store.connection.execute(
        "SELECT source_target_type FROM person_embeddings WHERE embedding_id = ?",
        (matches[0].embedding_id,),
    ).fetchone()
    assert row["source_target_type"] == "active_interaction_target"


@pytest.mark.asyncio
async def test_resolve_self_introduction_returns_snapshot_evidence_and_zero_store_delta(
    tmp_path,
):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    source_frame = frame()
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(),
        memory_snapshot=memory_snapshot(frame_message=source_frame),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    before_counts = _store_counts(subject.store)
    backend.person_inputs.clear()
    backend.scene_inputs.clear()

    preview = await subject.resolve_target(
        {
            "camera": "front",
            "target": {
                "kind": "person",
                "intent": "self_introduction",
                "referent_text": "我",
            },
        }
    )

    assert preview["status"] == "resolved"
    assert preview["candidates"][0]["track_id"] == 7
    assert preview["evidence"] == {
        "request_snapshot_ref": "snapshot:front:1",
        "source_frame_ref": "front:1:1000",
        "frame_id": 1,
        "frame_timestamp_ms": 1000,
        "observed_at_ms": 10000,
        "frame_cache_ttl_ms": 1000,
        "resolution_reason": "active_interaction_target",
        "source_track_ref": "front:track:7",
        "resolver_target_ref": "front:track:7",
    }
    _assert_zero_store_delta(preview["store_delta"])
    assert _store_counts(subject.store) == before_counts
    assert backend.person_inputs == []
    assert backend.scene_inputs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot_kwargs", "expected_ambiguity_type"),
    [
        (
            {"engagement_state": "no_target"},
            "no_active_interaction_target",
        ),
        (
            {"attention_available": False},
            "no_active_interaction_target",
        ),
        (
            {"scene_target_track_id": 7, "attention_track_id": 8},
            "no_active_interaction_target",
        ),
        (
            {"lost_ms": 100},
            "no_active_interaction_target",
        ),
    ],
)
async def test_self_introduction_inactive_snapshot_uses_allowed_ambiguity_type(
    tmp_path,
    snapshot_kwargs,
    expected_ambiguity_type,
):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    source_frame = frame()
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(),
        memory_snapshot=memory_snapshot(
            frame_message=source_frame,
            **snapshot_kwargs,
        ),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()

    request = {
        "camera": "front",
        "target": {
            "kind": "person",
            "intent": "self_introduction",
            "referent_text": "我",
        },
    }
    preview = await subject.resolve_target(request)

    assert preview["status"] == "ambiguous"
    assert preview["ambiguity_type"] == expected_ambiguity_type
    _assert_allowed_ambiguity(preview)

    with pytest.raises(MemoryServiceError) as exc:
        await subject.teach_person(
            {
                **request,
                "profile": {"display_name": "张三"},
            }
        )

    assert exc.value.code == "target_ambiguous"
    assert exc.value.details["ambiguity_type"] == expected_ambiguity_type
    _assert_allowed_ambiguity(exc.value.details)
    assert backend.person_inputs == []
    assert _count_rows(subject.store, "person_profiles") == 0


@pytest.mark.asyncio
async def test_self_introduction_stale_snapshot_uses_allowed_ambiguity_type(tmp_path):
    now = 10_000

    def clock() -> int:
        return now

    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=8, scene_dim=8)
    subject = _track_service(
        AppMemoryService(
            store=store,
            embedding_backend=backend,
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
    )
    source_frame = frame()
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(),
        memory_snapshot=memory_snapshot(frame_message=source_frame),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()
    now = 11_001

    request = {
        "camera": "front",
        "target": {
            "kind": "person",
            "intent": "self_introduction",
            "referent_text": "我",
        },
    }
    preview = await subject.resolve_target(request)

    assert preview["status"] == "ambiguous"
    assert preview["ambiguity_type"] == "stale_interaction"
    _assert_allowed_ambiguity(preview)

    with pytest.raises(MemoryServiceError) as exc:
        await subject.teach_person(
            {
                **request,
                "profile": {"display_name": "张三"},
            }
        )

    assert exc.value.code == "target_ambiguous"
    assert exc.value.details["ambiguity_type"] == "stale_interaction"
    _assert_allowed_ambiguity(exc.value.details)
    assert backend.person_inputs == []
    assert _count_rows(subject.store, "person_profiles") == 0


@pytest.mark.asyncio
async def test_third_person_introduction_uses_active_person_pose_pointing_for_write(
    tmp_path,
):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    source_frame = frame()
    introducer = person_track(
        7,
        bbox_xyxy=(300.0, 100.0, 500.0, 650.0),
        keypoints=pointing_right_keypoints(),
        frame_message=source_frame,
    )
    introduced = person_track(
        8,
        bbox_xyxy=(720.0, 190.0, 920.0, 390.0),
        frame_message=source_frame,
    )
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(),
        memory_snapshot=memory_snapshot_with_tracks(
            [introducer, introduced],
            frame_message=source_frame,
            attention_track_id=7,
            scene_target_track_id=7,
        ),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()

    request = {
        "camera": "front",
        "target": {
            "kind": "person",
            "intent": "third_person_introduction",
            "referent_text": "他",
        },
    }
    preview = await subject.resolve_target(request)

    assert preview["status"] == "resolved"
    assert preview["candidates"][0]["track_id"] == 8
    assert preview["candidates"][0]["reason"] == "pose_pointing_to_person"
    assert preview["evidence"]["resolver_target_ref"] == "front:track:8"
    assert preview["evidence"]["introducer_ref"] == "front:track:7"

    person = await subject.teach_person(
        {
            **request,
            "profile": {"display_name": "李四"},
        }
    )

    assert len(backend.person_inputs) == 1
    expected_embedding = FakeEmbeddingBackend(
        person_dim=8,
        scene_dim=8,
    ).embed_person(backend.person_inputs[0])
    matches = subject.store.search_person_embeddings(expected_embedding, limit=1)
    assert matches[0].matched_id == person["person_id"]
    provenance = subject.store.get_embedding_provenance(matches[0].embedding_id)
    assert provenance["source_track_ref"] == "front:track:8"
    assert provenance["resolver_target_ref"] == "front:track:8"
    assert provenance["resolution_reason"] == "pose_pointing_to_person"
    assert person["evidence"]["resolver_target_ref"] == "front:track:8"
    assert person["evidence"]["introducer_ref"] == "front:track:7"
    row = subject.store.connection.execute(
        "SELECT source_target_type FROM person_embeddings WHERE embedding_id = ?",
        (matches[0].embedding_id,),
    ).fetchone()
    assert row["source_target_type"] == "pose_pointing_to_person"


@pytest.mark.asyncio
async def test_third_person_fallback_keeps_introduced_person_evidence(tmp_path):
    success_vector = _unit_vector(0)
    backend = ScriptedPersonEmbeddingBackend(
        person_outcomes=[_no_usable_face(), success_vector],
    )
    subject = service(tmp_path, embedding_backend=backend)
    source_frame = frame()
    introducer = person_track(
        7,
        bbox_xyxy=(620.0, 100.0, 780.0, 650.0),
        keypoints=(
            kp("left_shoulder", 650.0, 240.0),
            kp("left_elbow", 720.0, 260.0),
            kp("left_wrist", 790.0, 275.0),
        ),
        frame_message=source_frame,
    )
    introduced = person_track(
        8,
        bbox_xyxy=(820.0, 190.0, 1020.0, 390.0),
        frame_message=source_frame,
    )
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(),
        memory_snapshot=memory_snapshot_with_tracks(
            [introducer, introduced],
            frame_message=source_frame,
            attention_track_id=7,
            scene_target_track_id=7,
        ),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()
    backend.set_person_outcomes([_no_usable_face(), success_vector])

    person = await subject.teach_person(
        {
            "camera": "front",
            "target": {
                "kind": "person",
                "intent": "third_person_introduction",
                "referent_text": "他",
            },
            "profile": {"display_name": "李四"},
        }
    )

    assert len(backend.person_inputs) == 2
    fallback = _decoded_jpeg(backend.person_inputs[1])
    original = _decoded_jpeg(JPEG_1280X720)
    _assert_rgb_close(fallback.getpixel((750, 200)), (0, 0, 0), tolerance=10)
    _assert_rgb_close(fallback.getpixel((900, 220)), original.getpixel((900, 220)))
    matches = subject.store.search_person_embeddings(
        EmbeddingResult(
            vector=success_vector,
            embedding_type="face",
            embedding_model=backend.person_model,
            embedding_version=backend.model_version,
            quality=1.0,
        ),
        limit=1,
    )
    provenance = subject.store.get_embedding_provenance(matches[0].embedding_id)
    assert provenance["source_track_ref"] == "front:track:8"
    assert provenance["resolver_target_ref"] == "front:track:8"
    assert person["evidence"]["resolver_target_ref"] == "front:track:8"
    assert person["evidence"]["introducer_ref"] == "front:track:7"


@pytest.mark.asyncio
async def test_third_person_introduction_pose_unclear_does_not_write(tmp_path):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    source_frame = frame()
    introducer = person_track(
        7,
        bbox_xyxy=(300.0, 100.0, 500.0, 650.0),
        frame_message=source_frame,
    )
    introduced = person_track(
        8,
        bbox_xyxy=(720.0, 190.0, 920.0, 390.0),
        frame_message=source_frame,
    )
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(),
        memory_snapshot=memory_snapshot_with_tracks(
            [introducer, introduced],
            frame_message=source_frame,
            attention_track_id=7,
            scene_target_track_id=7,
        ),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()

    request = {
        "camera": "front",
        "target": {
            "kind": "person",
            "intent": "third_person_introduction",
            "referent_text": "他",
        },
    }
    preview = await subject.resolve_target(request)

    assert preview["status"] == "ambiguous"
    assert preview["ambiguity_type"] == "pose_unclear"
    _assert_zero_store_delta(preview["store_delta"])
    _assert_allowed_ambiguity(preview)

    with pytest.raises(MemoryServiceError) as exc:
        await subject.teach_person(
            {
                **request,
                "profile": {"display_name": "李四"},
            }
        )

    assert exc.value.code == "target_ambiguous"
    assert exc.value.details["ambiguity_type"] == "pose_unclear"
    _assert_zero_store_delta(exc.value.details["store_delta"])
    _assert_allowed_ambiguity(exc.value.details)
    assert backend.person_inputs == []
    assert _count_rows(subject.store, "person_profiles") == 0
    assert _count_rows(subject.store, "person_embeddings") == 0


@pytest.mark.asyncio
async def test_third_person_introduction_multiple_pointing_candidates_does_not_write(
    tmp_path,
):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    source_frame = frame()
    introducer = person_track(
        7,
        bbox_xyxy=(300.0, 100.0, 500.0, 650.0),
        keypoints=pointing_right_keypoints(),
        frame_message=source_frame,
    )
    near_candidate = person_track(
        8,
        bbox_xyxy=(650.0, 230.0, 750.0, 430.0),
        frame_message=source_frame,
    )
    far_candidate = person_track(
        9,
        bbox_xyxy=(1060.0, 280.0, 1200.0, 560.0),
        frame_message=source_frame,
    )
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(),
        memory_snapshot=memory_snapshot_with_tracks(
            [introducer, near_candidate, far_candidate],
            frame_message=source_frame,
            attention_track_id=7,
            scene_target_track_id=7,
        ),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()

    request = {
        "camera": "front",
        "target": {
            "kind": "person",
            "intent": "third_person_introduction",
            "referent_text": "他",
        },
    }
    preview = await subject.resolve_target(request)

    assert preview["status"] == "ambiguous"
    assert preview["ambiguity_type"] == "multiple_candidates"
    _assert_allowed_ambiguity(preview)

    with pytest.raises(MemoryServiceError) as exc:
        await subject.teach_person(
            {
                **request,
                "profile": {"display_name": "李四"},
            }
        )

    assert exc.value.code == "target_ambiguous"
    assert exc.value.details["ambiguity_type"] == "multiple_candidates"
    _assert_allowed_ambiguity(exc.value.details)
    assert backend.person_inputs == []
    assert _count_rows(subject.store, "person_profiles") == 0


@pytest.mark.asyncio
async def test_non_self_public_person_intents_do_not_invent_ambiguity_types(tmp_path):
    backend = RecordingEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = service(tmp_path, embedding_backend=backend)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=frame(),
        visual_state=visual_state(),
    )
    await _wait_for_memory_query_idle(subject, camera="front")
    backend.person_inputs.clear()

    third_person = {
        "camera": "front",
        "target": {
            "kind": "person",
            "intent": "third_person_introduction",
            "referent_text": "他",
        },
    }
    third_preview = await subject.resolve_target(third_person)

    assert third_preview["status"] == "ambiguous"
    assert third_preview["ambiguity_type"] == "no_active_interaction_target"
    _assert_allowed_ambiguity(third_preview)

    with pytest.raises(MemoryServiceError) as third_error:
        await subject.teach_person(
            {
                **third_person,
                "profile": {"display_name": "李四"},
            }
        )
    assert third_error.value.code == "target_ambiguous"
    assert third_error.value.details["ambiguity_type"] == "no_active_interaction_target"
    _assert_allowed_ambiguity(third_error.value.details)

    unsupported = {
        "camera": "front",
        "target": {
            "kind": "person",
            "intent": "identify_person",
            "referent_text": "这个人",
        },
    }
    unsupported_preview = await subject.resolve_target(unsupported)

    assert unsupported_preview["status"] == "ambiguous"
    assert unsupported_preview["error_code"] == "unsupported_person_intent"
    assert unsupported_preview["ambiguity_type"] == "target_unclear"
    _assert_allowed_ambiguity(unsupported_preview)

    with pytest.raises(MemoryServiceError) as unsupported_error:
        await subject.teach_person(
            {
                **unsupported,
                "profile": {"display_name": "王五"},
            }
        )
    assert unsupported_error.value.code == "unsupported_person_intent"
    assert "ambiguity_type" not in unsupported_error.value.details
    assert backend.person_inputs == []


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


def test_person_embedding_input_candidates_are_lazy_and_primary_first():
    target = ResolvedTarget(
        source_target_mode="track_id",
        target_type="person",
        bbox_xyxy=(300.0, 100.0, 700.0, 650.0),
        track_id=7,
        quality="usable",
    )

    candidates = _person_embedding_input_candidates(JPEG_1280X720, target)

    assert iter(candidates) is candidates
    assert next(candidates) == _target_bytes(JPEG_1280X720, target)


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
async def test_memory_query_uses_snapshot_tracks_not_public_visual_state(tmp_path):
    match_vector = _unit_vector(0)
    backend = SequencePersonEmbeddingBackend(person_vectors=[match_vector])
    subject = service(tmp_path, embedding_backend=backend)
    _seed_person(subject, backend, vector=match_vector, display_name="Snapshot Person")
    source_frame = frame(frame_id=2, timestamp_ms=1_500)

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
        memory_snapshot=memory_snapshot_with_tracks(
            [person_track(8, frame_message=source_frame)],
            frame_message=source_frame,
            attention_track_id=7,
            scene_target_track_id=7,
        ),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_500,
    )

    assert [event["event"] for event in events] == ["known_person_present"]
    assert events[0]["track_id"] == 8
    assert events[0]["evidence"]["source_target_mode"] == "recognition_track"
    assert events[0]["memory_context"]["person"]["display_name"] == "Snapshot Person"
    assert len(backend.person_inputs) == 1


@pytest.mark.asyncio
async def test_memory_query_falls_back_to_target_masked_context_for_known_person(
    tmp_path,
):
    match_vector = _unit_vector(0)
    backend = ScriptedPersonEmbeddingBackend(
        person_outcomes=[_no_usable_face(), match_vector],
    )
    subject = service(tmp_path, embedding_backend=backend)
    _seed_person(subject, backend, vector=match_vector, display_name="Snapshot Person")
    source_frame = frame(frame_id=2, timestamp_ms=1_500)

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
        memory_snapshot=memory_snapshot_with_tracks(
            [person_track(8, frame_message=source_frame)],
            frame_message=source_frame,
            attention_track_id=7,
            scene_target_track_id=7,
        ),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_500,
    )

    assert [event["event"] for event in events] == ["known_person_present"]
    assert events[0]["track_id"] == 8
    assert events[0]["memory_context"]["person"]["display_name"] == "Snapshot Person"
    assert len(backend.person_inputs) == 2


@pytest.mark.asyncio
async def test_memory_query_without_snapshot_does_not_emit_known_person_event(tmp_path):
    match_vector = _unit_vector(0)
    backend = SequencePersonEmbeddingBackend(person_vectors=[match_vector])
    subject = service(tmp_path, embedding_backend=backend)
    _seed_person(subject, backend, vector=match_vector, display_name="张三")

    query_frame = frame(frame_id=2, timestamp_ms=1_500)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=query_frame,
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_500,
    )

    assert events == []
    assert backend.person_inputs == []


@pytest.mark.asyncio
async def test_memory_query_scans_multiple_eligible_person_tracks_not_attention_only(
    tmp_path,
):
    miss_vector = _unit_vector(1)
    match_vector = _unit_vector(0)
    backend = SequencePersonEmbeddingBackend(
        person_vectors=[miss_vector, match_vector],
    )
    subject = service(tmp_path, embedding_backend=backend)
    _seed_person(subject, backend, vector=match_vector, display_name="Track Eight")
    source_frame = frame(frame_id=2, timestamp_ms=1_500)

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
        memory_snapshot=memory_snapshot_with_tracks(
            [
                person_track(7, frame_message=source_frame),
                person_track(8, frame_message=source_frame),
            ],
            frame_message=source_frame,
            attention_track_id=7,
            scene_target_track_id=7,
        ),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_500,
    )

    assert [event["event"] for event in events] == ["known_person_present"]
    assert events[0]["track_id"] == 8
    assert events[0]["evidence"]["source_target_mode"] == "recognition_track"
    assert len(backend.person_inputs) == 2


@pytest.mark.asyncio
async def test_memory_query_respects_max_person_tracks_bound(tmp_path):
    match_vector = _unit_vector(5)
    backend = SequencePersonEmbeddingBackend(
        person_vectors=[
            _unit_vector(0),
            _unit_vector(1),
            _unit_vector(2),
            _unit_vector(3),
            match_vector,
        ],
    )
    subject = service(tmp_path, embedding_backend=backend)
    _seed_person(subject, backend, vector=match_vector, display_name="Fifth Person")
    source_frame = frame(frame_id=2, timestamp_ms=1_500)

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
        memory_snapshot=memory_snapshot_with_tracks(
            [person_track(track_id, frame_message=source_frame) for track_id in range(1, 6)],
            frame_message=source_frame,
            attention_track_id=1,
            scene_target_track_id=1,
        ),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_500,
    )

    assert events == []
    assert len(backend.person_inputs) == 4


@pytest.mark.asyncio
async def test_memory_query_skips_ineligible_person_tracks(tmp_path):
    match_vector = _unit_vector(0)
    backend = SequencePersonEmbeddingBackend(person_vectors=[match_vector])
    subject = service(tmp_path, embedding_backend=backend)
    _seed_person(subject, backend, vector=match_vector, display_name="Valid Track")
    source_frame = frame(frame_id=2, timestamp_ms=1_500)

    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=source_frame,
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
        memory_snapshot=memory_snapshot_with_tracks(
            [
                person_track(7, lost_ms=100, frame_message=source_frame),
                person_track(8, class_name="bag", frame_message=source_frame),
                person_track(9, hits=0, frame_message=source_frame),
                person_track(
                    10,
                    bbox_xyxy=(100.0, 100.0, 105.0, 105.0),
                    frame_message=source_frame,
                ),
                person_track(11, frame_message=source_frame),
            ],
            frame_message=source_frame,
            attention_track_id=7,
            scene_target_track_id=7,
        ),
    )
    events = await _drain_memory_events(
        subject,
        camera="front",
        connection_id="ws_1",
        frame_id=2,
        frame_timestamp_ms=1_500,
    )

    assert [event["event"] for event in events] == ["known_person_present"]
    assert events[0]["track_id"] == 11
    assert events[0]["evidence"]["source_target_mode"] == "recognition_track"
    assert len(backend.person_inputs) == 1


@pytest.mark.asyncio
async def test_observe_visual_state_does_not_wait_for_blocking_memory_query(tmp_path):
    now = 10_000
    clock = lambda: now
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=8, scene_dim=8)
    backend = BlockingQueryEmbeddingBackend(person_dim=8, scene_dim=8)
    subject = _track_service(
        AppMemoryService(
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
    )

    first_frame = frame(frame_id=1, timestamp_ms=1_000)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=first_frame,
        visual_state=visual_state(frame_id=1, timestamp_ms=1_000),
        memory_snapshot=memory_snapshot(frame_message=first_frame),
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

    query_frame = frame(frame_id=2, timestamp_ms=1_500)
    started = time.perf_counter()
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=query_frame,
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
        memory_snapshot=memory_snapshot(frame_message=query_frame),
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


@pytest.mark.asyncio
async def test_close_waits_for_in_flight_query_before_closing_store(tmp_path):
    backend = BlockingQueryEmbeddingBackend(person_dim=8, scene_dim=8)
    backend.block_queries = True
    subject = service(tmp_path, embedding_backend=backend)

    query_frame = frame(frame_id=1, timestamp_ms=1_000)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=query_frame,
        visual_state=visual_state(frame_id=1, timestamp_ms=1_000),
        memory_snapshot=memory_snapshot(frame_message=query_frame),
    )
    assert await asyncio.to_thread(backend.entered.wait, 1.0)

    close_task = asyncio.create_task(_close_service(subject))
    try:
        await asyncio.sleep(0.05)
        assert not close_task.done()
        assert _count_rows(subject.store, "anonymous_profiles") == 0
    finally:
        backend.release.set()

    await asyncio.wait_for(close_task, 1.0)


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


def _count_rows(store: MemoryStore, table: str) -> int:
    row = store.connection.execute(
        f"SELECT COUNT(*) AS count FROM {table}",
    ).fetchone()
    return int(row["count"])


def _store_counts(store: MemoryStore) -> dict[str, int]:
    return store.memory_table_counts()


def _assert_zero_store_delta(store_delta: dict) -> None:
    assert {
        "embedding_provenance",
        "conversation_summaries",
        "external_user_links",
        "memory_match_records",
        "profile_merge_history",
        "negative_identity_matches",
        "person_embedding_vectors",
        "scene_embedding_vectors",
        "anonymous_embedding_vectors",
    } <= set(store_delta["delta"])
    assert store_delta["before"] == store_delta["after"]
    assert all(value == 0 for value in store_delta["delta"].values())


def _assert_allowed_ambiguity(payload: dict) -> None:
    assert payload["ambiguity_type"] in ALLOWED_AMBIGUITY_TYPES


def _unit_vector(index: int, *, dim: int = 8) -> tuple[float, ...]:
    vector = [0.0] * dim
    vector[index] = 1.0
    return tuple(vector)


def _seed_person(
    subject: AppMemoryService,
    backend: FakeEmbeddingBackend,
    *,
    vector: tuple[float, ...],
    display_name: str,
    person_id: str = "person_seeded",
    now_ms: int = 10_000,
) -> None:
    subject.store.upsert_person_profile(
        person_id=person_id,
        display_name=display_name,
        description="",
        tags=(),
        now_ms=now_ms,
    )
    subject.store.add_person_embedding(
        person_id=person_id,
        result=EmbeddingResult(
            vector=vector,
            embedding_type="face",
            embedding_model=backend.person_model,
            embedding_version=backend.model_version,
            quality=1.0,
        ),
        source_target_type="test_seed",
        now_ms=now_ms,
    )


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
    subject = _track_service(
        AppMemoryService(
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
    subject = _track_service(
        AppMemoryService(
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
    )

    first_frame = frame(frame_id=1, timestamp_ms=1_000)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=first_frame,
        visual_state=visual_state(frame_id=1, timestamp_ms=1_000),
        memory_snapshot=memory_snapshot(frame_message=first_frame),
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

    query_frame = frame(frame_id=2, timestamp_ms=1_500)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=query_frame,
        visual_state=visual_state(frame_id=2, timestamp_ms=1_500),
        memory_snapshot=memory_snapshot(frame_message=query_frame),
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
    subject = _track_service(
        AppMemoryService(
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
    )

    first_frame = frame(frame_id=1, timestamp_ms=1_000)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=first_frame,
        visual_state=visual_state(frame_id=1, timestamp_ms=1_000),
        memory_snapshot=memory_snapshot(frame_message=first_frame),
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
    second_frame = frame(frame_id=2, timestamp_ms=1_600)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=second_frame,
        visual_state=visual_state(frame_id=2, timestamp_ms=1_600),
        memory_snapshot=memory_snapshot(frame_message=second_frame),
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
    third_frame = frame(frame_id=3, timestamp_ms=2_200)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=third_frame,
        visual_state=visual_state(frame_id=3, timestamp_ms=2_200),
        memory_snapshot=memory_snapshot(frame_message=third_frame),
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
    subject = _track_service(
        AppMemoryService(
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

    first_frame = frame(frame_id=1, timestamp_ms=1_000)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=first_frame,
        visual_state=visual_state(frame_id=1, timestamp_ms=1_000),
        memory_snapshot=memory_snapshot(frame_message=first_frame),
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

    second_frame = frame(frame_id=2, timestamp_ms=1_600)
    await subject.observe_visual_state(
        connection_id="ws_1",
        frame=second_frame,
        visual_state=visual_state(frame_id=2, timestamp_ms=1_600),
        memory_snapshot=memory_snapshot(frame_message=second_frame),
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
