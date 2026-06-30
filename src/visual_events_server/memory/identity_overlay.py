from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .events import MemoryMatch


@dataclass(frozen=True)
class _OverlayRecord:
    status: str
    source: str
    observed_at_ms: int
    confidence: float | None = None
    person: dict[str, Any] | None = None
    anonymous_person: dict[str, Any] | None = None
    reason: str | None = None


class IdentityOverlay:
    def __init__(
        self,
        *,
        ttl_ms: int,
        clock_ms: Callable[[], int],
    ) -> None:
        if ttl_ms <= 0:
            raise ValueError("ttl_ms must be positive")
        self.ttl_ms = int(ttl_ms)
        self._clock_ms = clock_ms
        self._records: dict[tuple[str, str, int], _OverlayRecord] = {}
        self._lock = threading.RLock()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._records)

    def put_known_person(
        self,
        *,
        connection_id: str,
        camera: str,
        track_id: int | None,
        person_profile: dict[str, Any],
        match: MemoryMatch,
        source: str = "background_recall",
    ) -> None:
        if track_id is None:
            return
        with self._lock:
            self._records[(connection_id, camera, int(track_id))] = _OverlayRecord(
                status="known_person",
                source=source,
                observed_at_ms=self._clock_ms(),
                confidence=_rounded(match.match_score),
                person=_public_person(person_profile),
            )

    def put_familiar_unknown(
        self,
        *,
        connection_id: str,
        camera: str,
        track_id: int | None,
        anonymous_profile: dict[str, Any],
        match: MemoryMatch,
        source: str = "background_recall",
    ) -> None:
        if track_id is None:
            return
        with self._lock:
            self._records[(connection_id, camera, int(track_id))] = _OverlayRecord(
                status="familiar_unknown",
                source=source,
                observed_at_ms=self._clock_ms(),
                confidence=_rounded(match.match_score),
                anonymous_person=_public_anonymous(anonymous_profile),
            )

    def put_unknown(
        self,
        *,
        connection_id: str,
        camera: str,
        track_id: int | None,
        reason: str,
        source: str = "background_recall",
    ) -> None:
        if track_id is None:
            return
        with self._lock:
            self._records[(connection_id, camera, int(track_id))] = _OverlayRecord(
                status="unknown",
                source=source,
                observed_at_ms=self._clock_ms(),
                reason=reason,
            )

    def put_unavailable(
        self,
        *,
        connection_id: str,
        camera: str,
        track_id: int | None,
        reason: str,
        source: str = "background_recall",
    ) -> None:
        if track_id is None:
            return
        with self._lock:
            self._records[(connection_id, camera, int(track_id))] = _OverlayRecord(
                status="unavailable",
                source=source,
                observed_at_ms=self._clock_ms(),
                reason=reason,
            )

    def project(
        self,
        *,
        connection_id: str,
        camera: str,
        visual_state: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            self.purge()
            now_ms = self._clock_ms()
            visible_track_ids = _visible_person_track_ids(visual_state)
            self._purge_missing_tracks(connection_id, camera, visible_track_ids)
            projected_tracks: list[dict[str, Any]] = []
            for track_id in visible_track_ids:
                record = self._records.get((connection_id, camera, track_id))
                projected_tracks.append(
                    {
                        "track_id": track_id,
                        "identity": (
                            _record_to_public(record, now_ms=now_ms)
                            if record is not None
                            else _pending_identity()
                        ),
                    }
                )
        context: dict[str, Any] = {
            "overlay_status": "ready",
            "tracks": projected_tracks,
        }
        active_target = _active_target(visual_state, visible_track_ids)
        if active_target is not None:
            context["active_target"] = {"track_id": active_target}
        return context

    def identity_for_track(
        self,
        *,
        connection_id: str,
        camera: str,
        track_id: int | None,
        source: str = "cache",
    ) -> dict[str, Any] | None:
        if track_id is None:
            return None
        with self._lock:
            self.purge()
            now_ms = self._clock_ms()
            record = self._records.get((connection_id, camera, int(track_id)))
            if record is None:
                return None
            if record.status not in {"known_person", "familiar_unknown"}:
                return None
            identity = _identity_base(
                status=record.status,
                source=source,
                confidence=record.confidence,
                person=record.person,
                anonymous_person=record.anonymous_person,
                reason=record.reason,
            )
            identity["fresh_ms"] = max(0, int(now_ms - record.observed_at_ms))
            return identity

    def purge(self) -> None:
        with self._lock:
            now_ms = self._clock_ms()
            expired = [
                key
                for key, record in self._records.items()
                if now_ms - record.observed_at_ms > self.ttl_ms
            ]
            for key in expired:
                self._records.pop(key, None)

    def _purge_missing_tracks(
        self,
        connection_id: str,
        camera: str,
        visible_track_ids: list[int],
    ) -> None:
        with self._lock:
            visible = set(visible_track_ids)
            stale = [
                key
                for key in self._records
                if key[0] == connection_id
                and key[1] == camera
                and key[2] not in visible
            ]
            for key in stale:
                self._records.pop(key, None)


def unavailable_identity_context(reason: str) -> dict[str, Any]:
    return {
        "overlay_status": "unavailable",
        "reason": reason,
        "tracks": [],
    }


def known_person_identity_context(
    person_profile: dict[str, Any],
    match: MemoryMatch,
    *,
    source: str,
) -> dict[str, Any]:
    return _identity_base(
        status="known_person",
        source=source,
        confidence=_rounded(match.match_score),
        person=_public_person(person_profile),
    )


def familiar_unknown_identity_context(
    anonymous_profile: dict[str, Any],
    match: MemoryMatch,
    *,
    source: str,
) -> dict[str, Any]:
    return _identity_base(
        status="familiar_unknown",
        source=source,
        confidence=_rounded(match.match_score),
        anonymous_person=_public_anonymous(anonymous_profile),
    )


def unknown_identity_context(*, reason: str, source: str) -> dict[str, Any]:
    return _identity_base(status="unknown", source=source, reason=reason)


def unavailable_person_identity_context(*, reason: str, source: str) -> dict[str, Any]:
    return _identity_base(status="unavailable", source=source, reason=reason)


def _record_to_public(
    record: _OverlayRecord,
    *,
    now_ms: int,
) -> dict[str, Any]:
    identity = _identity_base(
        status=record.status,
        source=record.source,
        confidence=record.confidence,
        person=record.person,
        anonymous_person=record.anonymous_person,
        reason=record.reason,
    )
    identity["fresh_ms"] = max(0, int(now_ms - record.observed_at_ms))
    return identity


def _identity_base(
    *,
    status: str,
    source: str,
    confidence: float | None = None,
    person: dict[str, Any] | None = None,
    anonymous_person: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "status": status,
        "source": source,
    }
    if confidence is not None:
        identity["confidence"] = confidence
    if person is not None:
        identity["person"] = dict(person)
    if anonymous_person is not None:
        identity["anonymous_person"] = dict(anonymous_person)
    if reason is not None:
        identity["reason"] = reason
    return identity


def _pending_identity() -> dict[str, Any]:
    return {
        "status": "pending",
        "source": "none",
    }


def _public_person(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "person_id": str(profile["person_id"]),
        "display_name": str(profile["display_name"]),
        "description": str(profile.get("description", "")),
        "tags": [str(tag) for tag in profile.get("tags", [])],
    }


def _public_anonymous(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "anonymous_id": str(profile["anonymous_id"]),
        "seen_count": int(profile["seen_count"]),
        "observed_duration_ms": int(profile.get("observed_duration_ms", 0)),
        "familiar_score": _rounded(float(profile["familiar_score"])),
    }


def _visible_person_track_ids(visual_state: dict[str, Any]) -> list[int]:
    track_ids: list[int] = []
    tracks = visual_state.get("tracks")
    if not isinstance(tracks, list):
        return track_ids
    for track in tracks:
        if not isinstance(track, dict):
            continue
        class_name = track.get("class", track.get("class_name", "person"))
        if class_name != "person":
            continue
        if int(track.get("lost_ms") or 0) != 0:
            continue
        track_id = track.get("track_id")
        if isinstance(track_id, int):
            track_ids.append(track_id)
    return track_ids


def _active_target(
    visual_state: dict[str, Any],
    visible_track_ids: list[int],
) -> int | None:
    attention = visual_state.get("attention")
    if not isinstance(attention, dict):
        return None
    track_id = attention.get("target_track_id")
    if isinstance(track_id, int) and track_id in set(visible_track_ids):
        return track_id
    return None


def _rounded(value: float) -> float:
    return round(float(value), 6)
