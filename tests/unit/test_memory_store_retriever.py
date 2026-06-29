from __future__ import annotations

import hashlib
import sqlite3

import pytest

from visual_events_server.memory.embedding import EmbeddingResult
from visual_events_server.memory.retriever import MemoryRetriever
from visual_events_server.memory.store import MemoryStore, MemoryStoreError


def embedding(
    vector: list[float],
    *,
    embedding_type: str,
    model: str,
    version: str = "v1",
) -> EmbeddingResult:
    return EmbeddingResult(
        vector=tuple(vector),
        embedding_type=embedding_type,
        embedding_model=model,
        embedding_version=version,
        quality=0.9,
    )


def test_person_profile_embedding_and_sqlite_vec_retrieval_round_trip(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    store.upsert_person_profile(
        person_id="person_1",
        display_name="张三",
        description="店长",
        tags=("staff",),
        now_ms=1000,
    )
    embedding_id = store.add_person_embedding(
        person_id="person_1",
        result=embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        source_target_type="track_id",
        now_ms=1000,
    )
    store.add_person_embedding(
        person_id="person_1",
        result=embedding([0.0, 1.0, 0.0, 0.0], embedding_type="face", model="other-face"),
        source_target_type="track_id",
        now_ms=1000,
    )

    match = MemoryRetriever(store).query_person(
        embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        threshold=0.95,
        margin=0.05,
    )

    assert match is not None
    assert match.matched_id == "person_1"
    assert match.embedding_id == embedding_id
    assert match.match_type == "face"
    assert match.match_score >= 0.99
    assert match.top2_margin >= 0.99
    assert store.get_person_profile("person_1") == {
        "person_id": "person_1",
        "display_name": "张三",
        "description": "店长",
        "tags": ["staff"],
    }


def test_create_person_with_embedding_commits_profile_embedding_vector_and_provenance(
    tmp_path,
) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)

    result = store.create_person_with_embedding(
        person_id="person_atomic",
        display_name="张三",
        description="店长",
        tags=("staff",),
        embedding=embedding(
            [1.0, 0.0, 0.0, 0.0],
            embedding_type="face",
            model="fake-face",
        ),
        source_target_type="track_id",
        source_track_ref="front:track:7",
        source_frame_ref="front:1:1000",
        crop_hash=hashlib.sha256(b"person-crop").hexdigest(),
        crop_path_or_artifact_ref=None,
        resolver_target_ref="front:track:7",
        resolution_reason="track_id",
        now_ms=1000,
    )

    assert result["person_id"] == "person_atomic"
    embedding_id = result["embedding_id"]
    assert store.get_person_profile("person_atomic") == {
        "person_id": "person_atomic",
        "display_name": "张三",
        "description": "店长",
        "tags": ["staff"],
    }
    match = MemoryRetriever(store).query_person(
        embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        threshold=0.95,
        margin=0.0,
    )
    assert match is not None
    assert match.embedding_id == embedding_id
    assert store.get_embedding_provenance(embedding_id) == {
        "embedding_id": embedding_id,
        "owner_type": "person",
        "owner_id": "person_atomic",
        "source_track_ref": "front:track:7",
        "source_frame_ref": "front:1:1000",
        "crop_hash": hashlib.sha256(b"person-crop").hexdigest(),
        "crop_path_or_artifact_ref": None,
        "resolver_target_ref": "front:track:7",
        "resolution_reason": "track_id",
        "embedding_type": "face",
        "embedding_model": "fake-face",
        "embedding_version": "v1",
        "embedding_dim": 4,
        "created_at_ms": 1000,
    }


def test_create_person_with_embedding_rolls_back_orphan_rows_on_late_failure(
    tmp_path,
) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)

    with pytest.raises(sqlite3.IntegrityError):
        store.create_person_with_embedding(
            person_id="person_orphan",
            display_name="张三",
            description="店长",
            tags=("staff",),
            embedding=embedding(
                [1.0, 0.0, 0.0, 0.0],
                embedding_type="face",
                model="fake-face",
            ),
            source_target_type="track_id",
            source_track_ref="front:track:7",
            source_frame_ref="front:1:1000",
            crop_hash=None,
            crop_path_or_artifact_ref=None,
            resolver_target_ref="front:track:7",
            resolution_reason="track_id",
            now_ms=1000,
        )

    assert store.get_person_profile("person_orphan") is None
    assert _count_rows(store, "person_embeddings", "person_id = ?", "person_orphan") == 0
    assert (
        _count_rows(
            store,
            "person_embedding_vectors",
            "person_id = ?",
            "person_orphan",
        )
        == 0
    )
    assert _count_rows(store, "embedding_provenance", "owner_id = ?", "person_orphan") == 0


def test_retriever_rejects_low_score_or_small_margin(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    for index, vector in enumerate(([1.0, 0.0, 0.0, 0.0], [0.99, 0.01, 0.0, 0.0])):
        person_id = f"person_{index}"
        store.upsert_person_profile(
            person_id=person_id,
            display_name=person_id,
            description="",
            tags=(),
            now_ms=1000,
        )
        store.add_person_embedding(
            person_id=person_id,
            result=embedding(list(vector), embedding_type="face", model="fake-face"),
            source_target_type="track_id",
            now_ms=1000,
        )

    retriever = MemoryRetriever(store)

    assert (
        retriever.query_person(
            embedding([0.0, 1.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
            threshold=0.95,
            margin=0.0,
        )
        is None
    )
    assert (
        retriever.query_person(
            embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
            threshold=0.95,
            margin=0.2,
        )
        is None
    )


def test_sqlite_vec_filters_model_metadata_inside_knn_query(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    for index in range(6):
        person_id = f"wrong_model_{index}"
        store.upsert_person_profile(
            person_id=person_id,
            display_name=person_id,
            description="",
            tags=(),
            now_ms=1000,
        )
        store.add_person_embedding(
            person_id=person_id,
            result=embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="other-face"),
            source_target_type="track_id",
            now_ms=1000,
        )
    store.upsert_person_profile(
        person_id="target",
        display_name="目标",
        description="",
        tags=(),
        now_ms=1000,
    )
    store.add_person_embedding(
        person_id="target",
        result=embedding([0.99, 0.01, 0.0, 0.0], embedding_type="face", model="fake-face"),
        source_target_type="track_id",
        now_ms=1000,
    )

    match = MemoryRetriever(store).query_person(
        embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        threshold=0.95,
        margin=0.0,
        top_k=2,
    )

    assert match is not None
    assert match.matched_id == "target"


def test_scene_memory_uses_separate_sqlite_vec_table(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=6)
    store.create_scene_memory(
        scene_id="scene_1",
        title="货柜",
        description="门口货柜",
        activation_hint="有人停留时介绍货品",
        target_type="scene",
        now_ms=1000,
    )
    embedding_id = store.add_scene_embedding(
        scene_id="scene_1",
        result=embedding(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            embedding_type="scene",
            model="fake-scene",
        ),
        source_target_type="scene",
        now_ms=1000,
    )

    match = MemoryRetriever(store).query_scene(
        embedding(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            embedding_type="scene",
            model="fake-scene",
        ),
        threshold=0.95,
        margin=0.0,
    )

    assert match is not None
    assert match.matched_id == "scene_1"
    assert match.embedding_id == embedding_id
    assert match.match_type == "scene"


def test_create_scene_with_embedding_commits_memory_embedding_vector_and_provenance(
    tmp_path,
) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=6)

    result = store.create_scene_with_embedding(
        scene_id="scene_atomic",
        title="货柜",
        description="门口货柜",
        activation_hint="有人停留时介绍货品",
        target_type="scene",
        region_id=None,
        embedding=embedding(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            embedding_type="scene",
            model="fake-scene",
        ),
        source_target_type="scene",
        source_track_ref=None,
        source_frame_ref="front:2:2000",
        crop_hash=hashlib.sha256(b"scene-frame").hexdigest(),
        crop_path_or_artifact_ref=None,
        resolver_target_ref="scene",
        resolution_reason="scene",
        now_ms=2000,
    )

    assert result["scene_id"] == "scene_atomic"
    embedding_id = result["embedding_id"]
    assert store.get_scene_memory("scene_atomic") == {
        "scene_id": "scene_atomic",
        "title": "货柜",
        "description": "门口货柜",
        "activation_hint": "有人停留时介绍货品",
        "target_type": "scene",
        "region_id": None,
    }
    match = MemoryRetriever(store).query_scene(
        embedding(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            embedding_type="scene",
            model="fake-scene",
        ),
        threshold=0.95,
        margin=0.0,
    )
    assert match is not None
    assert match.embedding_id == embedding_id
    assert store.get_embedding_provenance(embedding_id) == {
        "embedding_id": embedding_id,
        "owner_type": "scene",
        "owner_id": "scene_atomic",
        "source_track_ref": None,
        "source_frame_ref": "front:2:2000",
        "crop_hash": hashlib.sha256(b"scene-frame").hexdigest(),
        "crop_path_or_artifact_ref": None,
        "resolver_target_ref": "scene",
        "resolution_reason": "scene",
        "embedding_type": "scene",
        "embedding_model": "fake-scene",
        "embedding_version": "v1",
        "embedding_dim": 6,
        "created_at_ms": 2000,
    }


def test_create_scene_with_embedding_rolls_back_orphan_rows_on_late_failure(
    tmp_path,
) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=6)

    with pytest.raises(sqlite3.IntegrityError):
        store.create_scene_with_embedding(
            scene_id="scene_orphan",
            title="货柜",
            description="门口货柜",
            activation_hint="有人停留时介绍货品",
            target_type="scene",
            region_id=None,
            embedding=embedding(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                embedding_type="scene",
                model="fake-scene",
            ),
            source_target_type="scene",
            source_track_ref=None,
            source_frame_ref="front:2:2000",
            crop_hash=None,
            crop_path_or_artifact_ref=None,
            resolver_target_ref="scene",
            resolution_reason="scene",
            now_ms=2000,
        )

    assert store.get_scene_memory("scene_orphan") is None
    assert _count_rows(store, "scene_embeddings", "scene_id = ?", "scene_orphan") == 0
    assert (
        _count_rows(
            store,
            "scene_embedding_vectors",
            "scene_id = ?",
            "scene_orphan",
        )
        == 0
    )
    assert _count_rows(store, "embedding_provenance", "owner_id = ?", "scene_orphan") == 0


def test_scene_memory_can_store_optional_region_id(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    store.create_scene_memory(
        scene_id="scene_1",
        title="门口",
        description="入口区域",
        activation_hint="有人进门时问候",
        target_type="scene",
        region_id="front_door",
        now_ms=1000,
    )

    assert store.get_scene_memory("scene_1") == {
        "scene_id": "scene_1",
        "title": "门口",
        "description": "入口区域",
        "activation_hint": "有人进门时问候",
        "target_type": "scene",
        "region_id": "front_door",
    }


def test_anonymous_profile_embedding_and_retrieval_round_trip(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    store.create_anonymous_profile(
        anonymous_id="anon_1",
        seen_count=1,
        first_seen_at_ms=1000,
        last_seen_at_ms=1000,
        familiar_score=0.25,
    )
    store.update_anonymous_profile(
        anonymous_id="anon_1",
        seen_count=3,
        last_seen_at_ms=1300,
        familiar_score=0.72,
    )
    embedding_id = store.add_anonymous_embedding(
        anonymous_id="anon_1",
        result=embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        source_target_type="track_id",
        now_ms=1300,
    )
    store.create_anonymous_profile(
        anonymous_id="anon_inactive",
        seen_count=5,
        first_seen_at_ms=900,
        last_seen_at_ms=1200,
        familiar_score=0.9,
        status="merged",
        merged_person_id="person_9",
    )
    store.add_anonymous_embedding(
        anonymous_id="anon_inactive",
        result=embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        source_target_type="track_id",
        now_ms=1200,
    )

    match = MemoryRetriever(store).query_anonymous_person(
        embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        threshold=0.95,
        margin=0.0,
    )

    assert match is not None
    assert match.matched_type == "anonymous_person"
    assert match.matched_id == "anon_1"
    assert match.embedding_id == embedding_id
    assert store.get_active_anonymous_profile("anon_1") == {
        "anonymous_id": "anon_1",
        "seen_count": 3,
        "first_seen_at_ms": 1000,
        "last_seen_at_ms": 1300,
        "familiar_score": 0.72,
        "status": "active",
        "merged_person_id": None,
    }
    assert store.get_active_anonymous_profile("anon_inactive") is None


def test_memory_table_counts_include_sqlite_vec_tables_after_embedding_writes(
    tmp_path,
) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    store.upsert_person_profile(
        person_id="person_1",
        display_name="张三",
        description="店长",
        tags=(),
        now_ms=1000,
    )
    store.add_person_embedding(
        person_id="person_1",
        result=embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        source_target_type="track_id",
        now_ms=1000,
    )
    store.create_anonymous_profile(
        anonymous_id="anon_1",
        seen_count=1,
        first_seen_at_ms=1000,
        last_seen_at_ms=1000,
        familiar_score=0.25,
    )
    store.add_anonymous_embedding(
        anonymous_id="anon_1",
        result=embedding([0.0, 1.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        source_target_type="track_id",
        now_ms=1000,
    )
    store.create_scene_memory(
        scene_id="scene_1",
        title="货柜",
        description="门口货柜",
        activation_hint="有人停留时介绍货品",
        target_type="scene",
        now_ms=1000,
    )
    store.add_scene_embedding(
        scene_id="scene_1",
        result=embedding([0.0, 0.0, 1.0, 0.0], embedding_type="scene", model="fake-scene"),
        source_target_type="scene",
        now_ms=1000,
    )

    counts = store.memory_table_counts()

    assert set(counts) == {
        "person_profiles",
        "person_embeddings",
        "person_embedding_vectors",
        "scene_memories",
        "scene_embeddings",
        "scene_embedding_vectors",
        "anonymous_profiles",
        "anonymous_embeddings",
        "anonymous_embedding_vectors",
        "embedding_provenance",
        "conversation_summaries",
        "external_user_links",
        "memory_match_records",
        "profile_merge_history",
        "negative_identity_matches",
    }
    assert len(counts) == 15
    assert counts["person_embeddings"] == 1
    assert counts["person_embedding_vectors"] == 1
    assert counts["anonymous_embeddings"] == 1
    assert counts["anonymous_embedding_vectors"] == 1
    assert counts["scene_embeddings"] == 1
    assert counts["scene_embedding_vectors"] == 1


def test_anonymous_merge_low_level_methods_copy_and_mark_profile(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    store.upsert_person_profile(
        person_id="person_1",
        display_name="张三",
        description="店长",
        tags=(),
        now_ms=1000,
    )
    store.create_anonymous_profile(
        anonymous_id="anon_1",
        seen_count=4,
        first_seen_at_ms=1000,
        last_seen_at_ms=2000,
        familiar_score=0.8,
    )
    anon_embedding_id = store.add_anonymous_embedding(
        anonymous_id="anon_1",
        result=embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        source_target_type="track_id",
        now_ms=2000,
    )

    anonymous_embeddings = store.list_anonymous_embeddings("anon_1")
    copied_ids = store.copy_anonymous_embeddings_to_person(
        anonymous_id="anon_1",
        person_id="person_1",
        now_ms=2100,
    )
    history_id = store.add_profile_merge_history(
        anonymous_id="anon_1",
        person_id="person_1",
        merge_reason="agent_confirmed",
        now_ms=2200,
    )
    store.mark_anonymous_profile_merged(
        anonymous_id="anon_1",
        person_id="person_1",
        now_ms=2200,
    )

    assert anonymous_embeddings[0]["embedding_id"] == anon_embedding_id
    assert len(copied_ids) == 1
    assert copied_ids[0].startswith("emb_person_")
    assert history_id.startswith("merge_")
    assert store.get_profile_merge_history(history_id) == {
        "merge_id": history_id,
        "anonymous_id": "anon_1",
        "person_id": "person_1",
        "merge_reason": "agent_confirmed",
        "created_at_ms": 2200,
    }
    assert store.get_active_anonymous_profile("anon_1") is None
    match = MemoryRetriever(store).query_person(
        embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        threshold=0.95,
        margin=0.0,
    )
    assert match is not None
    assert match.matched_id == "person_1"
    assert match.embedding_id == copied_ids[0]


def test_query_person_filters_explicit_negative_identity_match(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    for person_id, vector in (
        ("person_wrong", [1.0, 0.0, 0.0, 0.0]),
        ("person_right", [0.99, 0.01, 0.0, 0.0]),
    ):
        store.upsert_person_profile(
            person_id=person_id,
            display_name=person_id,
            description="",
            tags=(),
            now_ms=1000,
        )
        embedding_id = store.add_person_embedding(
            person_id=person_id,
            result=embedding(vector, embedding_type="face", model="fake-face"),
            source_target_type="track_id",
            now_ms=1000,
        )
        if person_id == "person_wrong":
            wrong_embedding_id = embedding_id

    store.add_negative_identity_match(
        memory_match_id="match_wrong",
        wrong_person_id="person_wrong",
        embedding_id=wrong_embedding_id,
        now_ms=2000,
    )

    match = MemoryRetriever(store).query_person(
        embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
        threshold=0.95,
        margin=0.0,
    )

    assert store.get_negative_identity_match(
        wrong_person_id="person_wrong",
        embedding_id=wrong_embedding_id,
    ) == {
        "memory_match_id": "match_wrong",
        "wrong_person_id": "person_wrong",
        "embedding_id": wrong_embedding_id,
        "created_at_ms": 2000,
    }
    assert match is not None
    assert match.matched_id == "person_right"


def test_store_fails_fast_when_sqlite_vec_dependency_is_unavailable(monkeypatch, tmp_path) -> None:
    import visual_events_server.memory.store as store_module

    monkeypatch.setattr(store_module, "sqlite_vec", None)

    with pytest.raises(MemoryStoreError) as exc:
        MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)

    assert exc.value.code == "sqlite_vec_unavailable"


def test_store_requires_vector_dimension_to_match_sqlite_vec_table(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    store.upsert_person_profile(
        person_id="person_1",
        display_name="张三",
        description="",
        tags=(),
        now_ms=1000,
    )

    with pytest.raises(ValueError, match="expected 4"):
        store.add_person_embedding(
            person_id="person_1",
            result=embedding([1.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
            source_target_type="track_id",
            now_ms=1000,
        )


def test_store_enables_foreign_keys(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)

    with pytest.raises(sqlite3.IntegrityError):
        store.add_person_embedding(
            person_id="missing",
            result=embedding([1.0, 0.0, 0.0, 0.0], embedding_type="face", model="fake-face"),
            source_target_type="track_id",
            now_ms=1000,
        )


def test_store_persists_summary_external_link_and_match_record(tmp_path) -> None:
    store = MemoryStore.open(tmp_path / "memory.sqlite3", person_dim=4, scene_dim=4)
    store.upsert_person_profile(
        person_id="person_1",
        display_name="张三",
        description="店长",
        tags=("staff",),
        now_ms=1000,
    )
    summary_id = store.add_conversation_summary(
        person_id="person_1",
        summary="上次问过新品尺码，偏好浅色外套。" + ("很长" * 200),
        source="agent",
        source_conversation_id="conv-1",
        now_ms=1100,
    )
    store.link_external_user(
        person_id="person_1",
        external_user_ref="wechat:zhangsan",
        now_ms=1200,
    )
    match_id = store.add_memory_match_record(
        event_id="front:mem_evt_000001",
        matched_type="person",
        matched_id="person_1",
        embedding_id="emb_1",
        match_score=0.98,
        top2_margin=0.12,
        source_target_mode="track_id",
        camera="front",
        frame_id=7,
        frame_timestamp_ms=1300,
        now_ms=1301,
        memory_match_id="match_1",
    )

    assert summary_id.startswith("summary_")
    summaries = store.get_conversation_summaries("person_1", limit=2)
    assert summaries[0].startswith("上次问过新品尺码，偏好浅色外套。")
    assert len(summaries[0]) == 240
    assert store.get_person_by_external_user("wechat:zhangsan") == {
        "person_id": "person_1",
        "display_name": "张三",
        "description": "店长",
        "tags": ["staff"],
    }
    assert match_id == "match_1"
    assert store.get_memory_match_record("match_1") == {
        "memory_match_id": "match_1",
        "event_id": "front:mem_evt_000001",
        "matched_type": "person",
        "matched_id": "person_1",
        "embedding_id": "emb_1",
        "match_score": 0.98,
        "top2_margin": 0.12,
        "source_target_mode": "track_id",
        "camera": "front",
        "frame_id": 7,
        "frame_timestamp_ms": 1300,
    }


def _count_rows(
    store: MemoryStore,
    table: str,
    where: str,
    *params: object,
) -> int:
    row = store.connection.execute(
        f"SELECT COUNT(*) AS count FROM {table} WHERE {where}",
        params,
    ).fetchone()
    return int(row["count"])
