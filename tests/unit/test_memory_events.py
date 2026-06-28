from __future__ import annotations

from visual_events_server.memory.events import (
    MemoryEventGate,
    MemoryMatch,
    SourceFrameRef,
    build_familiar_unknown_event,
    build_known_person_event,
    build_scene_event,
)


def match(**overrides) -> MemoryMatch:
    values = {
        "memory_match_id": "match_1",
        "matched_type": "person",
        "matched_id": "person_1",
        "embedding_id": "emb_1",
        "match_type": "face",
        "match_score": 0.87,
        "top2_margin": 0.12,
        "embedding_model": "fake-face",
        "embedding_version": "v1",
    }
    values.update(overrides)
    return MemoryMatch(**values)


def source_frame() -> SourceFrameRef:
    return SourceFrameRef(
        camera="front",
        frame_id=42,
        frame_timestamp_ms=1710000000000,
        source_target_mode="track_id",
        track_id=7,
    )


def test_known_person_event_contains_stable_evidence_and_compact_context() -> None:
    event = build_known_person_event(
        event_id="front:mem_evt_1",
        match=match(),
        source=source_frame(),
        person_profile={
            "person_id": "person_1",
            "display_name": "张三",
            "description": "店长",
            "tags": ["staff"],
        },
        conversation_summaries=("上次问过浅色外套。",),
    )

    assert event["event"] == "known_person_present"
    assert event["lifecycle_state"] == "confirmed"
    assert event["track_id"] == 7
    assert event["confidence"] == 0.87
    assert event["evidence"] == {
        "memory_match_id": "match_1",
        "matched_type": "person",
        "matched_id": "person_1",
        "embedding_id": "emb_1",
        "match_type": "face",
        "match_score": 0.87,
        "top2_margin": 0.12,
        "source_target_mode": "track_id",
        "source_frame_id": 42,
        "source_frame_timestamp_ms": 1710000000000,
        "embedding_model": "fake-face",
        "embedding_version": "v1",
    }
    assert event["memory_context"]["person"]["display_name"] == "张三"
    assert event["memory_context"]["conversation_summaries"] == ["上次问过浅色外套。"]


def test_scene_event_uses_same_memory_evidence_contract() -> None:
    event = build_scene_event(
        event_id="front:mem_evt_2",
        match=match(
            memory_match_id="match_2",
            matched_type="scene",
            matched_id="scene_1",
            embedding_id="emb_scene_1",
            match_type="scene",
            embedding_model="fake-scene",
        ),
        source=SourceFrameRef(
            camera="front",
            frame_id=43,
            frame_timestamp_ms=1710000000100,
            source_target_mode="scene",
            track_id=None,
        ),
        scene_memory={
            "scene_id": "scene_1",
            "title": "货柜",
            "description": "门口货柜",
            "activation_hint": "介绍货品",
        },
    )

    assert event["event"] == "scene_activated"
    assert event["lifecycle_state"] == "confirmed"
    assert "track_id" not in event
    assert event["evidence"]["matched_type"] == "scene"
    assert event["evidence"]["source_target_mode"] == "scene"
    assert event["memory_context"]["scene"]["title"] == "货柜"


def test_familiar_unknown_event_contains_anonymous_person_context() -> None:
    event = build_familiar_unknown_event(
        event_id="front:mem_evt_3",
        match=match(
            memory_match_id="match_3",
            matched_type="anonymous_person",
            matched_id="anon_1",
            embedding_id="emb_anon_1",
            match_score=0.81,
            top2_margin=0.08,
        ),
        source=source_frame(),
        anonymous_profile={
            "anonymous_id": "anon_1",
            "seen_count": 5,
            "familiar_score": 0.81,
            "last_seen_at_ms": 1710000000000,
        },
    )

    assert event["event"] == "familiar_unknown_present"
    assert event["track_id"] == 7
    assert event["confidence"] == 0.81
    assert event["evidence"]["memory_match_id"] == "match_3"
    assert event["evidence"]["matched_type"] == "anonymous_person"
    assert event["evidence"]["matched_id"] == "anon_1"
    assert event["evidence"]["match_score"] == 0.81
    assert event["evidence"]["top2_margin"] == 0.08
    assert event["evidence"]["source_target_mode"] == "track_id"
    assert event["memory_context"]["anonymous_person"] == {
        "anonymous_id": "anon_1",
        "seen_count": 5,
        "familiar_score": 0.81,
        "last_seen_at_ms": 1710000000000,
    }
    assert event["text"] == "看到一个经常出现但尚未命名的人"


def test_memory_event_gate_applies_cooldown_by_event_camera_and_subject() -> None:
    gate = MemoryEventGate(cooldown_ms=60_000)

    assert gate.allow("front", "known_person_present", "person_1", now_ms=1_000)
    assert not gate.allow("front", "known_person_present", "person_1", now_ms=30_000)
    assert gate.allow("front", "known_person_present", "person_2", now_ms=30_000)
    assert gate.allow("front", "known_person_present", "person_1", now_ms=61_000)
