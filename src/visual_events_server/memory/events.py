from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryMatch:
    memory_match_id: str
    matched_type: str
    matched_id: str
    embedding_id: str
    match_type: str
    match_score: float
    top2_margin: float
    embedding_model: str
    embedding_version: str


@dataclass(frozen=True)
class SourceFrameRef:
    camera: str
    frame_id: int
    frame_timestamp_ms: int
    source_target_mode: str
    track_id: int | None = None


class MemoryEventGate:
    def __init__(self, *, cooldown_ms: int) -> None:
        if cooldown_ms < 0:
            raise ValueError("cooldown_ms must be non-negative")
        self.cooldown_ms = cooldown_ms
        self._last_emit_ms: dict[tuple[str, str, str], int] = {}

    def allow(
        self,
        camera: str,
        event: str,
        subject_id: str,
        *,
        now_ms: int,
    ) -> bool:
        key = (camera, event, subject_id)
        last_emit_ms = self._last_emit_ms.get(key)
        if last_emit_ms is not None and now_ms - last_emit_ms < self.cooldown_ms:
            return False
        self._last_emit_ms[key] = now_ms
        return True


def build_known_person_event(
    *,
    event_id: str,
    match: MemoryMatch,
    source: SourceFrameRef,
    person_profile: dict[str, Any],
    conversation_summaries: tuple[str, ...] = (),
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "semantic_event",
        "event_id": event_id,
        "event": "known_person_present",
        "camera": source.camera,
        "confidence": _rounded(match.match_score),
        "duration_ms": 0,
        "lifecycle_state": "confirmed",
        "evidence": _evidence(match, source),
        "memory_context": {
            "person": {
                "person_id": person_profile["person_id"],
                "display_name": person_profile["display_name"],
                "description": person_profile.get("description", ""),
                "tags": list(person_profile.get("tags", [])),
                "match_confidence": _rounded(match.match_score),
            },
            "conversation_summaries": list(conversation_summaries),
        },
        "text": f"看到已知人物：{person_profile['display_name']}",
    }
    if source.track_id is not None:
        event["track_id"] = source.track_id
    return event


def build_scene_event(
    *,
    event_id: str,
    match: MemoryMatch,
    source: SourceFrameRef,
    scene_memory: dict[str, Any],
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "semantic_event",
        "event_id": event_id,
        "event": "scene_activated",
        "camera": source.camera,
        "confidence": _rounded(match.match_score),
        "duration_ms": 0,
        "lifecycle_state": "confirmed",
        "evidence": _evidence(match, source),
        "memory_context": {
            "scene": {
                "scene_id": scene_memory["scene_id"],
                "title": scene_memory["title"],
                "description": scene_memory.get("description", ""),
                "activation_hint": scene_memory.get("activation_hint", ""),
                "region_id": scene_memory.get("region_id"),
                "match_confidence": _rounded(match.match_score),
            }
        },
        "text": f"场景已激活：{scene_memory['title']}",
    }
    if source.track_id is not None:
        event["track_id"] = source.track_id
    return event


def build_familiar_unknown_event(
    *,
    event_id: str,
    match: MemoryMatch,
    source: SourceFrameRef,
    anonymous_profile: dict[str, Any],
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "semantic_event",
        "event_id": event_id,
        "event": "familiar_unknown_present",
        "camera": source.camera,
        "confidence": _rounded(match.match_score),
        "duration_ms": 0,
        "lifecycle_state": "confirmed",
        "evidence": _evidence(match, source),
        "memory_context": {
            "anonymous_person": {
                "anonymous_id": anonymous_profile["anonymous_id"],
                "seen_count": int(anonymous_profile["seen_count"]),
                "familiar_score": _rounded(anonymous_profile["familiar_score"]),
                "last_seen_at_ms": int(anonymous_profile["last_seen_at_ms"]),
            }
        },
        "text": "看到一个经常出现但尚未命名的人",
    }
    if source.track_id is not None:
        event["track_id"] = source.track_id
    return event


def _evidence(match: MemoryMatch, source: SourceFrameRef) -> dict[str, Any]:
    return {
        "memory_match_id": match.memory_match_id,
        "matched_type": match.matched_type,
        "matched_id": match.matched_id,
        "embedding_id": match.embedding_id,
        "match_type": match.match_type,
        "match_score": _rounded(match.match_score),
        "top2_margin": _rounded(match.top2_margin),
        "source_target_mode": source.source_target_mode,
        "source_frame_id": source.frame_id,
        "source_frame_timestamp_ms": source.frame_timestamp_ms,
        "embedding_model": match.embedding_model,
        "embedding_version": match.embedding_version,
    }


def _rounded(value: float) -> float:
    return round(float(value), 6)
