from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

from visual_events_server.attention import AttentionResult
from visual_events_server.protocol import FrameMessage
from visual_events_server.tracking import TrackSnapshot

from .embedding import EmbeddingResult, EmbeddingUnavailable, MemoryEmbeddingBackend
from .events import (
    MemoryEventGate,
    MemoryMatch,
    SourceFrameRef,
    build_familiar_unknown_event,
    build_known_person_event,
    build_scene_event,
)
from .frame_cache import (
    CachedFrame,
    FrameCache,
    FrameCacheError,
    MemoryFrameSnapshot,
    MemoryFrameSnapshotWindow,
    RequestInteractionSnapshot,
)
from .identity_overlay import (
    IdentityOverlay,
    _visible_person_track_ids,
    familiar_unknown_identity_context,
    known_person_identity_context,
    unavailable_person_identity_context,
    unknown_identity_context,
)
from .retriever import MemoryRetriever
from .store import MemoryStore, MemoryStoreError
from .target_resolver import (
    ResolvedTarget,
    TargetRequest,
    TargetResolveError,
    TargetResolver,
)


_LOGGER = logging.getLogger(__name__)

_PERSON_EMBEDDING_CROP_MARGIN_RATIO = 0.10
_PERSON_EMBEDDING_CONTEXT_MARGIN_RATIO = 0.50
_MAX_PERSON_QUERY_TRACKS = 4
_REQUIRED_POSE_SNAPSHOT_COUNT = 2
_MEMORY_SEMANTIC_EVENTS = frozenset(
    {
        "known_person_present",
        "familiar_unknown_present",
        "scene_activated",
    }
)
_EVENT_IDENTITY_RECALL_EVENTS = frozenset(
    {
        "person_appeared",
        "person_passing_by",
        "person_approaching_robot",
        "person_stopped_near_robot",
        "person_waving",
    }
)
_RECOGNITION_ELIGIBILITY_POLICY = (
    "class_name == 'person' and lost_ms == 0 and hits > 0"
)
_StreamKey = tuple[str, str]
_EventRecallKey = tuple[str, str, int, str]


class MemoryServiceError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


@dataclass(frozen=True)
class _QueryPlan:
    cached: CachedFrame


@dataclass(frozen=True)
class _PersonIdentifyResult:
    status: str
    identity_context: dict[str, Any]
    reason: str | None = None
    match: MemoryMatch | None = None
    profile: dict[str, Any] | None = None
    embedding: EmbeddingResult | None = None
    embedding_bytes: bytes | None = None


@dataclass(frozen=True)
class _RecognitionTickReport:
    camera: str
    frame_id: int
    frame_timestamp_ms: int
    source_frame_ref: str
    tracks_seen: int
    tracks_eligible: int
    tracks_candidates: int
    candidate_track_ids: tuple[int, ...]
    tracks_queried: int
    tracks_skipped_reason: dict[str, int]
    queried_track_ids: tuple[int, ...]
    attention_target_track_id: int | None
    attention_target_only: bool
    max_tracks_per_tick: int
    query_interval_ms: int
    event_cooldown_ms: int
    recognition_runs_in_executor: bool
    eligibility_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "frame_id": self.frame_id,
            "frame_timestamp_ms": self.frame_timestamp_ms,
            "source_frame_ref": self.source_frame_ref,
            "tracks_seen": self.tracks_seen,
            "tracks_eligible": self.tracks_eligible,
            "tracks_candidates": self.tracks_candidates,
            "candidate_track_ids": list(self.candidate_track_ids),
            "tracks_queried": self.tracks_queried,
            "tracks_skipped_reason": dict(self.tracks_skipped_reason),
            "queried_track_ids": list(self.queried_track_ids),
            "attention_target_track_id": self.attention_target_track_id,
            "attention_target_only": self.attention_target_only,
            "max_tracks_per_tick": self.max_tracks_per_tick,
            "query_interval_ms": self.query_interval_ms,
            "event_cooldown_ms": self.event_cooldown_ms,
            "recognition_runs_in_executor": self.recognition_runs_in_executor,
            "eligibility_policy": self.eligibility_policy,
        }


@dataclass(frozen=True)
class _ExternalRefDecision:
    external_user_ref: str
    linked_person_id: str | None
    conflict: bool
    same_person: bool
    should_link: bool


@dataclass(frozen=True)
class _PersonEmbeddingInputCandidate:
    payload: bytes
    crop_box_xyxy: tuple[float, float, float, float]
    crop_box_coordinate_space: str


class AppMemoryService:
    def __init__(
        self,
        *,
        store: MemoryStore,
        embedding_backend: MemoryEmbeddingBackend,
        frame_cache_seconds: int,
        query_interval_ms: int,
        queue_size: int,
        known_person_threshold: float,
        known_person_margin: float,
        anonymous_threshold: float,
        anonymous_margin: float,
        familiar_seen_count: int,
        familiar_observed_duration_ms: int,
        familiar_threshold: float,
        scene_threshold: float,
        event_cooldown_ms: int,
        clock_ms: Callable[[], int] | None = None,
        target_resolver: TargetResolver | None = None,
        teach_queue_size: int = 2,
        teach_queue_timeout_ms: int = 500,
        artifact_dir: str | Path | None = None,
    ) -> None:
        if query_interval_ms <= 0:
            raise ValueError("query_interval_ms must be positive")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if teach_queue_size < 0:
            raise ValueError("teach_queue_size must be non-negative")
        if teach_queue_timeout_ms <= 0:
            raise ValueError("teach_queue_timeout_ms must be positive")
        self.store = store
        self.embedding_backend = embedding_backend
        self.query_interval_ms = int(query_interval_ms)
        self.known_person_threshold = float(known_person_threshold)
        self.known_person_margin = float(known_person_margin)
        self.anonymous_threshold = float(anonymous_threshold)
        self.anonymous_margin = float(anonymous_margin)
        self.familiar_seen_count = int(familiar_seen_count)
        if familiar_observed_duration_ms < 0:
            raise ValueError("familiar_observed_duration_ms must be non-negative")
        self.familiar_observed_duration_ms = int(familiar_observed_duration_ms)
        self.familiar_threshold = float(familiar_threshold)
        self.scene_threshold = float(scene_threshold)
        self.event_cooldown_ms = int(event_cooldown_ms)
        self._clock_ms = clock_ms or _system_time_ms
        self._cache = FrameCache(
            max_age_ms=int(frame_cache_seconds) * 1000,
            clock_ms=self._clock_ms,
        )
        self._identity_overlay = IdentityOverlay(
            ttl_ms=int(frame_cache_seconds) * 1000,
            clock_ms=self._clock_ms,
        )
        self._resolver = target_resolver or TargetResolver()
        self._retriever = MemoryRetriever(store)
        self._gate = MemoryEventGate(cooldown_ms=self.event_cooldown_ms)
        self._artifact_dir = (
            Path(artifact_dir)
            if artifact_dir is not None
            else Path("runtime") / "memory" / "artifacts"
        )
        self._completed_by_stream: dict[_StreamKey, deque[dict[str, Any]]] = {}
        self._queue_size = int(queue_size)
        self._last_query_frame_timestamp_ms_by_stream: dict[_StreamKey, int] = {}
        self._event_counters: dict[str, int] = {}
        self._pending_queries_by_stream: dict[
            _StreamKey,
            asyncio.Future[list[dict[str, Any]]],
        ] = {}
        self._pending_event_identity_recalls: dict[
            _EventRecallKey,
            asyncio.Future[None],
        ] = {}
        self._query_state_lock = threading.Lock()
        self._recognition_report_lock = threading.Lock()
        self._latest_recognition_report: _RecognitionTickReport | None = None
        self._latest_recognition_reports_by_camera: dict[
            str,
            _RecognitionTickReport,
        ] = {}
        self._teach_embedding_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="memory-teach-embedding",
        )
        self._teach_embedding_futures: set[Future[EmbeddingResult]] = set()
        self._teach_embedding_futures_lock = threading.Lock()
        self._teach_embedding_slots = asyncio.BoundedSemaphore(
            1 + int(teach_queue_size)
        )
        self._teach_queue_timeout_s = int(teach_queue_timeout_ms) / 1000.0
        self._lifecycle_condition = asyncio.Condition()
        self._active_teach_requests = 0
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._store_closed = False

    async def close(self) -> None:
        async with self._close_lock:
            if self._store_closed:
                return
            async with self._lifecycle_condition:
                self._closed = True
            await self._wait_for_active_teach_requests()
            await self._wait_for_pending_queries()
            await self._wait_for_pending_event_identity_recalls()
            self._teach_embedding_executor.shutdown(wait=False, cancel_futures=True)
            await self._wait_for_teach_embedding_futures()
            self._teach_embedding_executor.shutdown(wait=True, cancel_futures=True)
            self.store.close()
            self._store_closed = True

    async def observe_visual_state(
        self,
        *,
        connection_id: str,
        frame: FrameMessage,
        visual_state: dict[str, Any],
        memory_snapshot: MemoryFrameSnapshot | None = None,
    ) -> None:
        if self._closed:
            return
        self._cache.update(
            connection_id=connection_id,
            frame=frame,
            visual_state=visual_state,
            memory_snapshot=memory_snapshot,
        )
        stream_key = _stream_key(connection_id, frame.camera)
        if not self._query_due(stream_key, frame):
            return
        pending = self._pending_queries_by_stream.get(stream_key)
        if pending is not None:
            if pending.done():
                self._collect_completed_query(stream_key, pending)
            else:
                return
        self._last_query_frame_timestamp_ms_by_stream[stream_key] = frame.timestamp_ms
        plan = _QueryPlan(
            cached=self._cache.get_fresh_for_stream(connection_id, frame.camera),
        )
        loop = asyncio.get_running_loop()
        self._pending_queries_by_stream[stream_key] = loop.run_in_executor(
            None,
            self._run_query,
            plan,
        )

    async def drain_completed_events(
        self,
        *,
        camera: str,
        connection_id: str,
        frame_id: int,
        frame_timestamp_ms: int,
    ) -> list[dict[str, Any]]:
        stream_key = _stream_key(connection_id, camera)
        pending = self._pending_queries_by_stream.get(stream_key)
        if pending is not None and pending.done():
            self._collect_completed_query(stream_key, pending)
        queue = self._completed_by_stream.get(stream_key)
        if not queue:
            return []
        events = list(queue)
        queue.clear()
        return events

    def latest_recognition_report(
        self,
        camera: str | None = None,
    ) -> dict[str, Any] | None:
        with self._recognition_report_lock:
            report = (
                self._latest_recognition_report
                if camera is None
                else self._latest_recognition_reports_by_camera.get(camera)
            )
        return None if report is None else report.to_dict()

    def identity_context_for_visual_state(
        self,
        *,
        connection_id: str,
        visual_state: dict[str, Any],
    ) -> dict[str, Any]:
        camera = str(visual_state.get("camera") or "")
        return self._identity_overlay.project(
            connection_id=connection_id,
            camera=camera,
            visual_state=visual_state,
        )

    def enrich_visual_state_event_identities(
        self,
        *,
        connection_id: str,
        visual_state: dict[str, Any],
    ) -> None:
        semantic_events = visual_state.get("semantic_events")
        if not isinstance(semantic_events, list):
            return
        visible_track_ids = set(_visible_person_track_ids(visual_state))
        if not visible_track_ids:
            return
        camera = str(visual_state.get("camera") or "")
        for event in semantic_events:
            if not isinstance(event, dict):
                continue
            if event.get("event") in _MEMORY_SEMANTIC_EVENTS:
                continue
            track_id = event.get("track_id")
            if type(track_id) is not int:
                continue
            if track_id not in visible_track_ids:
                continue
            identity = self._identity_overlay.identity_for_track(
                connection_id=connection_id,
                camera=camera,
                track_id=track_id,
                source="cache",
            )
            if identity is not None:
                event["identity_context"] = identity
                continue
            if event.get("event") not in _EVENT_IDENTITY_RECALL_EVENTS:
                continue
            self._schedule_event_identity_recall(
                connection_id=connection_id,
                camera=camera,
                track_id=track_id,
            )

    def _schedule_event_identity_recall(
        self,
        *,
        connection_id: str,
        camera: str,
        track_id: int,
    ) -> None:
        stream_key = _stream_key(connection_id, camera)
        pending_query = self._pending_queries_by_stream.get(stream_key)
        if pending_query is not None and not pending_query.done():
            return

        try:
            cached = self._cache.get_fresh_for_stream(connection_id, camera)
        except FrameCacheError:
            return
        source_frame_ref = _source_frame_ref(cached)
        recall_key = (connection_id, camera, track_id, source_frame_ref)
        pending_recall = self._pending_event_identity_recalls.get(recall_key)
        if pending_recall is not None:
            if pending_recall.done():
                self._finish_event_identity_recall(recall_key, pending_recall)
            else:
                return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._identity_overlay.register_pending(
            connection_id=connection_id,
            camera=camera,
            track_id=track_id,
            source="event_recall",
            reason="cache_miss",
            source_frame_ref=source_frame_ref,
        )
        future = loop.run_in_executor(
            None,
            self._run_event_identity_recall,
            cached,
            track_id,
            source_frame_ref,
        )
        self._pending_event_identity_recalls[recall_key] = future
        future.add_done_callback(
            lambda completed, key=recall_key: self._finish_event_identity_recall(
                key,
                completed,
            )
        )

    def _finish_event_identity_recall(
        self,
        recall_key: _EventRecallKey,
        future: asyncio.Future[None],
    ) -> None:
        try:
            future.result()
        except Exception:
            _LOGGER.exception(
                "event identity recall failed for camera %s track %s frame %s",
                recall_key[1],
                recall_key[2],
                recall_key[3],
            )
        finally:
            self._identity_overlay.clear_pending(
                connection_id=recall_key[0],
                camera=recall_key[1],
                track_id=recall_key[2],
                source_frame_ref=recall_key[3],
            )
        if self._pending_event_identity_recalls.get(recall_key) is future:
            self._pending_event_identity_recalls.pop(recall_key, None)

    def _run_event_identity_recall(
        self,
        cached: CachedFrame,
        track_id: int,
        source_frame_ref: str,
    ) -> None:
        try:
            target = self._resolve_event_recall_target(cached, track_id)
            if target is None:
                return
            result = self._identify_person_target(
                cached,
                target,
                source="event_recall",
                include_anonymous=True,
            )
            latest = self._cache.get_fresh_for_stream(
                cached.connection_id,
                cached.frame.camera,
            )
            if _source_frame_ref(latest) != source_frame_ref:
                return
            if not _cached_visible_person_track(latest, track_id):
                return
            self._put_identity_overlay_result(
                latest,
                target,
                result,
                source="event_recall",
            )
        except FrameCacheError:
            return

    def _resolve_event_recall_target(
        self,
        cached: CachedFrame,
        track_id: int,
    ) -> ResolvedTarget | None:
        snapshot = cached.memory_snapshot
        if snapshot is None:
            return None
        try:
            resolved = self._resolver.resolve(
                TargetRequest(mode="track_id", track_id=track_id),
                image_width=snapshot.image_size[0],
                image_height=snapshot.image_size[1],
                tracks=snapshot.tracks,
                attention=snapshot.attention,
            )
        except TargetResolveError:
            return None
        if resolved.target_type != "person":
            return None
        return ResolvedTarget(
            source_target_mode="track_id",
            target_type=resolved.target_type,
            bbox_xyxy=resolved.bbox_xyxy,
            track_id=resolved.track_id,
            quality=resolved.quality,
        )

    async def identify_current(self, request: dict[str, Any]) -> dict[str, Any]:
        camera = _required_text(request, "camera")
        scope = _optional_text(request.get("scope")) or "active_target"
        if scope != "active_target":
            raise MemoryServiceError(
                "invalid_memory_request",
                "identify-current only supports scope=active_target",
            )
        timeout_ms = int(request.get("timeout_ms") or 500)
        timeout_ms = max(1, min(timeout_ms, 1000))
        try:
            cached = self._fresh_cached_frame(request)
        except MemoryServiceError as exc:
            if exc.code == "frame_cache_expired":
                return _identify_current_response(
                    status="stale_interaction",
                    reason="frame_cache_expired",
                )
            if exc.code == "no_active_frame":
                return _identify_current_response(status="no_active_frame")
            raise

        target, ambiguity_type = self._active_interaction_target_from_frame(cached)
        if target is None:
            status = (
                "stale_interaction"
                if ambiguity_type == "stale_interaction"
                else "ambiguous"
            )
            return _identify_current_response(
                status=status,
                reason=ambiguity_type,
                evidence=_identify_current_evidence(cached),
            )

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None,
            partial(
                self._identify_person_target,
                cached,
                target,
                source="active_identify",
                include_anonymous=True,
            ),
        )
        try:
            result = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=timeout_ms / 1000.0,
            )
        except TimeoutError:
            future.add_done_callback(_consume_identify_future_exception)
            return _identify_current_response(
                status="timeout",
                evidence=_identify_current_evidence(cached),
            )

        self._put_identity_overlay_result(
            cached,
            target,
            result,
            source="active_identify",
        )
        status = (
            "identified"
            if result.status in {"known_person", "familiar_unknown"}
            else result.status
        )
        response = _identify_current_response(
            status=status,
            reason=result.reason,
            people=[
                {
                    "target_ref": f"current:{camera}:active_target",
                    "identity_context": result.identity_context,
                }
            ],
            evidence=_identify_current_evidence(cached),
        )
        return response

    async def teach_person(self, request: dict[str, Any]) -> dict[str, Any]:
        await self._enter_teach_request()
        try:
            camera = _required_text(request, "camera")
            stream_ref = _required_text(request, "stream_ref")
            await self._drain_pending_query_for_stream(stream_ref, camera)
            store_before = self._store_count_snapshot()
            try:
                cached = self._fresh_cached_frame(request)
            except MemoryServiceError as exc:
                if (
                    _public_person_target(request) is not None
                    and exc.code == "frame_cache_expired"
                ):
                    error = _target_ambiguous_error(
                        "stale_interaction",
                        evidence=self._interaction_window_evidence_for_request(request),
                    )
                    _attach_store_delta_to_error(error, self._store_delta(store_before))
                    raise error from exc
                raise
            try:
                cached, target, interaction_snapshot = self._resolve_person_teach_target(
                    cached,
                    request,
                )
            except MemoryServiceError as exc:
                if exc.code == "target_ambiguous":
                    _attach_store_delta_to_error(exc, self._store_delta(store_before))
                raise
            if target.target_type not in {"person", "region"}:
                raise MemoryServiceError(
                    "invalid_target_type",
                    "teach_person target must resolve to a person or region",
                )
            profile = _required_mapping(request, "profile")
            display_name = _required_text(profile, "display_name")
            description_provided = "description" in profile
            tags_provided = "tags" in profile
            description = (
                _optional_text(profile.get("description"))
                if description_provided
                else ""
            )
            tags = (
                tuple(str(tag) for tag in profile.get("tags", []) if str(tag))
                if tags_provided
                else ()
            )
            embedding: EmbeddingResult | None = None
            embedding_bytes: bytes | None = None
            embedding_candidate: _PersonEmbeddingInputCandidate | None = None
            first_no_usable_face: MemoryServiceError | None = None
            for candidate in _person_embedding_input_candidate_records(
                cached.frame.jpeg_bytes,
                target,
                tracks=(
                    cached.memory_snapshot.tracks
                    if cached.memory_snapshot is not None
                    else None
                ),
            ):
                candidate_bytes = candidate.payload
                try:
                    embedding = await self._run_teach_embedding(
                        self.embedding_backend.embed_person,
                        candidate_bytes,
                    )
                except MemoryServiceError as exc:
                    if exc.code != "no_usable_face":
                        raise
                    if first_no_usable_face is None:
                        first_no_usable_face = exc
                    continue
                embedding_bytes = candidate_bytes
                embedding_candidate = candidate
                break
            if (
                embedding is None
                or embedding_bytes is None
                or embedding_candidate is None
            ):
                if first_no_usable_face is not None:
                    raise first_no_usable_face
                raise MemoryServiceError(
                    "embedding_unavailable",
                    "person embedding is unavailable",
                    status_code=503,
                )

            existing_person = self._retriever.query_person(
                embedding,
                threshold=self.known_person_threshold,
                margin=self.known_person_margin,
            )
            if existing_person is not None:
                existing_profile = self.store.get_person_profile(
                    existing_person.matched_id,
                )
                if existing_profile is not None:
                    external_ref = self._external_ref_decision(
                        profile,
                        existing_person.matched_id,
                    )
                    evidence = self._teach_match_evidence(
                        existing_person,
                        cached,
                        target,
                        interaction_snapshot=interaction_snapshot,
                    )
                    if external_ref.conflict:
                        raise self._teach_person_conflict_error(
                            matched_person_id=existing_person.matched_id,
                            evidence=evidence,
                            store_before=store_before,
                            external_user_ref=external_ref.external_user_ref,
                            external_user_person_id=external_ref.linked_person_id,
                        )
                    if (
                        existing_profile["display_name"] == display_name
                        or external_ref.same_person
                    ):
                        next_description = (
                            description
                            if description_provided
                            else str(existing_profile["description"])
                        )
                        next_tags = (
                            tags
                            if tags_provided
                            else tuple(str(tag) for tag in existing_profile["tags"])
                        )
                        now_ms = self._clock_ms()
                        self.store.upsert_person_profile(
                            person_id=existing_person.matched_id,
                            display_name=display_name,
                            description=next_description,
                            tags=next_tags,
                            now_ms=now_ms,
                        )
                        if external_ref.should_link:
                            self.store.link_external_user(
                                person_id=existing_person.matched_id,
                                external_user_ref=external_ref.external_user_ref,
                                now_ms=now_ms,
                            )
                        self._refresh_teach_person_identity_overlay(
                            cached,
                            target,
                            existing_person.matched_id,
                        )
                        return {
                            "ok": True,
                            "person_id": existing_person.matched_id,
                            "outcome": "updated_existing_person",
                            "matched_person_id": existing_person.matched_id,
                            "target_quality": target.quality,
                            "evidence": evidence,
                            "store_delta": self._store_delta(store_before),
                        }
                    raise self._teach_person_conflict_error(
                        matched_person_id=existing_person.matched_id,
                        evidence=evidence,
                        store_before=store_before,
                    )

            anonymous_match = self._retriever.query_anonymous_person(
                embedding,
                threshold=self.anonymous_threshold,
                margin=self.anonymous_margin,
            )
            if anonymous_match is not None:
                now_ms = self._clock_ms()
                crop_hash = _sha256_hex(embedding_bytes)
                person_id = _public_id("person")
                crop_path_or_artifact_ref = self._write_embedding_artifact(
                    owner_type="person",
                    owner_id=person_id,
                    crop_hash=crop_hash,
                    payload=embedding_bytes,
                )
                external_user_ref = _optional_text(profile.get("external_user_ref"))
                try:
                    promoted = self.store.promote_anonymous_to_person(
                        anonymous_id=anonymous_match.matched_id,
                        person_id=person_id,
                        display_name=display_name,
                        description=description,
                        tags=tags,
                        external_user_ref=external_user_ref,
                        fresh_embedding=embedding,
                        source_target_type=target.source_target_mode,
                        source_track_ref=_source_track_ref(cached, target),
                        source_frame_ref=_source_frame_ref(cached),
                        crop_hash=crop_hash,
                        crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                        resolver_target_ref=_resolver_target_ref(cached, target),
                        resolution_reason=target.source_target_mode,
                        merge_reason="teach_person",
                        now_ms=now_ms,
                    )
                except MemoryStoreError as exc:
                    self._delete_embedding_artifact(crop_path_or_artifact_ref)
                    if exc.code == "external_user_ref_conflict":
                        raise self._teach_anonymous_external_ref_conflict_error(
                            matched_anonymous_id=anonymous_match.matched_id,
                            evidence=self._teach_match_evidence(
                                anonymous_match,
                                cached,
                                target,
                                crop_hash=crop_hash,
                                embedding=embedding,
                                person_embedding_candidate=embedding_candidate,
                                interaction_snapshot=interaction_snapshot,
                            ),
                            store_before=store_before,
                            external_user_ref=str(
                                exc.details.get("external_user_ref")
                                or external_user_ref
                                or ""
                            ),
                            external_user_person_id=str(
                                exc.details.get("external_user_person_id") or ""
                            ),
                        ) from exc
                    if exc.code == "anonymous_not_found":
                        raise MemoryServiceError(
                            "anonymous_not_found",
                            "active anonymous profile not found",
                            status_code=404,
                            details={
                                "anonymous_id": anonymous_match.matched_id,
                                "store_delta": self._store_delta(store_before),
                            },
                        ) from exc
                    raise
                except Exception:
                    self._delete_embedding_artifact(crop_path_or_artifact_ref)
                    raise
                self._refresh_teach_person_identity_overlay(
                    cached,
                    target,
                    person_id,
                )
                return {
                    "ok": True,
                    "person_id": person_id,
                    "outcome": "merged_anonymous_person",
                    "merged_anonymous_id": anonymous_match.matched_id,
                    "copied_embedding_count": len(
                        promoted["copied_embedding_ids"],
                    ),
                    "embedding_id": promoted["embedding_id"],
                    "merge_id": promoted["merge_id"],
                    "target_quality": target.quality,
                    "evidence": self._teach_match_evidence(
                        anonymous_match,
                        cached,
                        target,
                        crop_hash=crop_hash,
                        crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                        embedding=embedding,
                        person_embedding_candidate=embedding_candidate,
                        interaction_snapshot=interaction_snapshot,
                    ),
                    "store_delta": self._store_delta(store_before),
                }

            now_ms = self._clock_ms()
            crop_hash = _sha256_hex(embedding_bytes)
            person_id = _public_id("person")
            crop_path_or_artifact_ref = self._write_embedding_artifact(
                owner_type="person",
                owner_id=person_id,
                crop_hash=crop_hash,
                payload=embedding_bytes,
            )
            external_user_ref = _optional_text(profile.get("external_user_ref"))
            evidence = self._target_evidence(
                cached,
                target,
                crop_hash=crop_hash,
                crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                embedding=embedding,
                person_embedding_candidate=embedding_candidate,
                interaction_snapshot=interaction_snapshot,
            )
            try:
                created = self.store.create_person_with_embedding(
                    person_id=person_id,
                    display_name=display_name,
                    description=description,
                    tags=tags,
                    external_user_ref=external_user_ref,
                    embedding=embedding,
                    source_target_type=target.source_target_mode,
                    source_track_ref=_source_track_ref(cached, target),
                    source_frame_ref=_source_frame_ref(cached),
                    crop_hash=crop_hash,
                    crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                    resolver_target_ref=_resolver_target_ref(cached, target),
                    resolution_reason=target.source_target_mode,
                    now_ms=now_ms,
                )
            except MemoryStoreError as exc:
                self._delete_embedding_artifact(crop_path_or_artifact_ref)
                if exc.code == "external_user_ref_conflict":
                    raise self._teach_created_external_ref_conflict_error(
                        evidence=evidence,
                        store_before=store_before,
                        external_user_ref=str(
                            exc.details.get("external_user_ref")
                            or external_user_ref
                            or ""
                        ),
                        external_user_person_id=str(
                            exc.details.get("external_user_person_id") or ""
                        ),
                    ) from exc
                raise
            except Exception:
                self._delete_embedding_artifact(crop_path_or_artifact_ref)
                raise
            person_profile = self._refresh_teach_person_identity_overlay(
                cached,
                target,
                created["person_id"],
            )
            if person_profile is None:
                person_profile = {
                    "person_id": created["person_id"],
                    "display_name": display_name,
                    "description": description,
                    "tags": list(tags),
                }
            return {
                "ok": True,
                "outcome": "created_person",
                "person_id": created["person_id"],
                "embedding_id": created["embedding_id"],
                "embedding_count": 1,
                "target_quality": target.quality,
                "evidence": evidence,
                "profile": person_profile,
                "store_delta": self._store_delta(store_before),
            }
        finally:
            await self._exit_teach_request()

    async def teach_scene(self, request: dict[str, Any]) -> dict[str, Any]:
        await self._enter_teach_request()
        try:
            target_request = _target_request(request)
            if target_request.mode != "scene":
                raise MemoryServiceError(
                    "unsupported_scene_target",
                    "teach_scene only supports target.mode=scene until region scene "
                    "queries are supported",
                )
            cached = self._fresh_cached_frame(request)
            target = self._resolve_cached_target(cached, target_request)
            memory = _required_mapping(request, "memory")
            title = _required_text(memory, "title")
            description = _optional_text(memory.get("description"))
            activation_hint = _optional_text(memory.get("activation_hint"))
            region_id = _optional_text(memory.get("region_id")) or None
            embedding_bytes = _target_bytes(cached.frame.jpeg_bytes, target)
            embedding = await self._run_teach_embedding(
                self.embedding_backend.embed_scene,
                embedding_bytes,
            )

            now_ms = self._clock_ms()
            crop_hash = _sha256_hex(embedding_bytes)
            scene_id = _public_id("scene")
            crop_path_or_artifact_ref = self._write_embedding_artifact(
                owner_type="scene",
                owner_id=scene_id,
                crop_hash=crop_hash,
                payload=embedding_bytes,
            )
            try:
                created = self.store.create_scene_with_embedding(
                    scene_id=scene_id,
                    title=title,
                    description=description,
                    activation_hint=activation_hint,
                    target_type=target.target_type,
                    region_id=region_id,
                    embedding=embedding,
                    source_target_type=target.source_target_mode,
                    source_track_ref=_source_track_ref(cached, target),
                    source_frame_ref=_source_frame_ref(cached),
                    crop_hash=crop_hash,
                    crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                    resolver_target_ref=_resolver_target_ref(cached, target),
                    resolution_reason=target.source_target_mode,
                    now_ms=now_ms,
                )
            except Exception:
                self._delete_embedding_artifact(crop_path_or_artifact_ref)
                raise
            return {
                "ok": True,
                "scene_id": created["scene_id"],
                "embedding_id": created["embedding_id"],
                "embedding_count": 1,
                "target_quality": target.quality,
                "evidence": self._target_evidence(
                    cached,
                    target,
                    crop_hash=crop_hash,
                    crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                ),
            }
        finally:
            await self._exit_teach_request()

    async def resolve_target(self, request: dict[str, Any]) -> dict[str, Any]:
        store_before = self._store_count_snapshot()
        public_person_target = _public_person_target(request)
        try:
            cached = self._fresh_cached_frame(request)
        except MemoryServiceError as exc:
            if public_person_target is not None and exc.code == "frame_cache_expired":
                return {
                    **_ambiguous_target_response("stale_interaction"),
                    "evidence": self._interaction_window_evidence_for_request(request),
                    "store_delta": self._store_delta(store_before),
                }
            raise
        if public_person_target is not None:
            response = self._preview_public_person_target(cached, public_person_target)
            response["store_delta"] = self._store_delta(store_before)
            return response
        preview = self._preview_cached_target(cached, _target_request(request))
        response = {
            "ok": True,
            "status": preview.status,
            "candidates": [_candidate_to_dict(candidate) for candidate in preview.candidates],
        }
        evidence = self._preview_evidence(cached, preview)
        if evidence is not None:
            response["evidence"] = evidence
        response["store_delta"] = self._store_delta(store_before)
        return response

    async def merge_anonymous_person(self, request: dict[str, Any]) -> dict[str, Any]:
        anonymous_id = _required_text(request, "anonymous_id")
        anonymous_profile = self.store.get_active_anonymous_profile(anonymous_id)
        if anonymous_profile is None:
            raise MemoryServiceError(
                "anonymous_not_found",
                "active anonymous profile not found",
                status_code=404,
            )

        person_id = _optional_text(request.get("person_id"))
        profile = request.get("profile")
        now_ms = self._clock_ms()
        merge_reason = _optional_text(request.get("merge_reason")) or "manual_merge"
        if person_id:
            existing_profile = self.store.get_person_profile(person_id)
            if existing_profile is None:
                raise MemoryServiceError(
                    "person_not_found",
                    "person profile not found",
                    status_code=404,
                )
            display_name = str(existing_profile["display_name"])
            description = str(existing_profile["description"])
            tags = tuple(str(tag) for tag in existing_profile["tags"])
            external_user_ref = None
        else:
            if not isinstance(profile, dict):
                raise MemoryServiceError(
                    "invalid_memory_request",
                    "profile is required when person_id is absent",
                )
            person_id = _public_id("person")
            display_name = _required_text(profile, "display_name")
            description = _optional_text(profile.get("description"))
            tags = tuple(str(tag) for tag in profile.get("tags", []) if str(tag))
            external_user_ref = _optional_text(profile.get("external_user_ref"))

        try:
            promoted = self.store.promote_anonymous_to_person(
                anonymous_id=anonymous_id,
                person_id=person_id,
                display_name=display_name,
                description=description,
                tags=tags,
                external_user_ref=external_user_ref,
                merge_reason=merge_reason,
                now_ms=now_ms,
            )
        except MemoryStoreError as exc:
            if exc.code == "anonymous_not_found":
                raise MemoryServiceError(
                    "anonymous_not_found",
                    "active anonymous profile not found",
                    status_code=404,
                ) from exc
            if exc.code == "external_user_ref_conflict":
                raise MemoryServiceError(
                    "person_teach_conflict",
                    "external_user_ref is already linked to a different person",
                    status_code=409,
                    details={
                        "error_code": "person_teach_conflict",
                        "outcome": "conflict",
                        "external_user_ref": exc.details.get("external_user_ref"),
                        "external_user_person_id": exc.details.get(
                            "external_user_person_id",
                        ),
                    },
                ) from exc
            raise
        return {
            "ok": True,
            "anonymous_id": anonymous_id,
            "person_id": person_id,
            "copied_embedding_count": len(promoted["copied_embedding_ids"]),
            "merge_id": promoted["merge_id"],
        }

    async def correct_identity(self, request: dict[str, Any]) -> dict[str, Any]:
        memory_match_id = _required_text(request, "memory_match_id")
        wrong_person_id = _required_text(request, "wrong_person_id")
        record = self.store.get_memory_match_record(memory_match_id)
        if record is None:
            raise MemoryServiceError(
                "memory_match_not_found",
                "memory match record not found",
                status_code=404,
            )
        if record["matched_type"] != "person" or record["matched_id"] != wrong_person_id:
            raise MemoryServiceError(
                "memory_match_person_mismatch",
                "memory match record does not match wrong_person_id",
            )
        if self.store.get_person_profile(wrong_person_id) is None:
            raise MemoryServiceError(
                "person_not_found",
                "person profile not found",
                status_code=404,
            )
        self.store.add_negative_identity_match(
            memory_match_id=memory_match_id,
            wrong_person_id=wrong_person_id,
            embedding_id=record["embedding_id"],
            now_ms=self._clock_ms(),
        )
        return {
            "ok": True,
            "memory_match_id": memory_match_id,
            "wrong_person_id": wrong_person_id,
        }

    async def add_conversation_summary(
        self,
        person_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        if self.store.get_person_profile(person_id) is None:
            raise MemoryServiceError(
                "person_not_found",
                "person profile not found",
                status_code=404,
            )
        summary_id = self.store.add_conversation_summary(
            person_id=person_id,
            summary=_required_text(request, "summary"),
            source=_optional_text(request.get("source")) or "agent",
            source_conversation_id=_optional_text(
                request.get("source_conversation_id")
            )
            or None,
            now_ms=self._clock_ms(),
        )
        return {"ok": True, "summary_id": summary_id}

    async def link_external_user(self, request: dict[str, Any]) -> dict[str, Any]:
        person_id = _required_text(request, "person_id")
        external_user_ref = _required_text(request, "external_user_ref")
        if self.store.get_person_profile(person_id) is None:
            raise MemoryServiceError(
                "person_not_found",
                "person profile not found",
                status_code=404,
            )
        self.store.link_external_user(
            person_id=person_id,
            external_user_ref=external_user_ref,
            now_ms=self._clock_ms(),
        )
        return {"ok": True, "person_id": person_id}

    async def get_person_by_external_user(
        self,
        external_user_ref: str,
    ) -> dict[str, Any]:
        person = self.store.get_person_by_external_user(external_user_ref)
        if person is None:
            raise MemoryServiceError(
                "external_user_not_found",
                "external user is not linked to a person",
                status_code=404,
            )
        return {
            "ok": True,
            "external_user_ref": external_user_ref,
            "person": person,
            "conversation_summaries": self.store.get_conversation_summaries(
                person["person_id"],
                limit=3,
            ),
        }

    def _fresh_cached_frame(self, request: dict[str, Any]) -> CachedFrame:
        camera = _required_text(request, "camera")
        stream_ref = _required_text(request, "stream_ref")
        try:
            return self._cache.get_fresh_for_stream(stream_ref, camera)
        except FrameCacheError as exc:
            raise MemoryServiceError(exc.code, exc.message, status_code=409) from exc

    def _legacy_latest_cached_frame_for_camera(self, camera: str) -> CachedFrame:
        try:
            return self._cache.get_latest_for_camera(camera)
        except FrameCacheError as exc:
            raise MemoryServiceError(exc.code, exc.message, status_code=409) from exc

    async def _enter_teach_request(self) -> None:
        async with self._lifecycle_condition:
            if self._closed:
                raise MemoryServiceError(
                    "memory_service_closed",
                    "memory service is closed",
                    status_code=503,
                )
            self._active_teach_requests += 1

    async def _exit_teach_request(self) -> None:
        async with self._lifecycle_condition:
            self._active_teach_requests -= 1
            if self._active_teach_requests == 0:
                self._lifecycle_condition.notify_all()

    async def _wait_for_active_teach_requests(self) -> None:
        async with self._lifecycle_condition:
            while self._active_teach_requests:
                await self._lifecycle_condition.wait()

    async def _wait_for_pending_queries(self) -> None:
        pending = list(self._pending_queries_by_stream.values())
        if pending:
            await asyncio.gather(
                *(asyncio.shield(future) for future in pending),
                return_exceptions=True,
            )
        self._pending_queries_by_stream.clear()

    async def _wait_for_pending_event_identity_recalls(self) -> None:
        pending = list(self._pending_event_identity_recalls.values())
        if pending:
            await asyncio.gather(
                *(asyncio.shield(future) for future in pending),
                return_exceptions=True,
            )
        self._pending_event_identity_recalls.clear()

    async def _drain_pending_query_for_stream(
        self,
        connection_id: str,
        camera: str,
    ) -> None:
        stream_key = _stream_key(connection_id, camera)
        pending = self._pending_queries_by_stream.get(stream_key)
        if pending is None:
            return
        if not pending.done():
            try:
                await asyncio.shield(pending)
            except Exception:
                pass
        if pending.done():
            self._collect_completed_query(stream_key, pending)

    async def _wait_for_teach_embedding_futures(self) -> None:
        while True:
            with self._teach_embedding_futures_lock:
                futures = list(self._teach_embedding_futures)
            if not futures:
                return
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=True,
            )

    async def _run_teach_embedding(
        self,
        embed: Callable[[bytes], EmbeddingResult],
        payload: bytes,
    ) -> EmbeddingResult:
        try:
            await asyncio.wait_for(
                self._teach_embedding_slots.acquire(),
                timeout=self._teach_queue_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise MemoryServiceError(
                "embedding_queue_full",
                "memory teach embedding queue is full",
                status_code=503,
            ) from exc

        try:
            loop = asyncio.get_running_loop()
            try:
                future = self._teach_embedding_executor.submit(embed, payload)
            except RuntimeError as exc:
                self._teach_embedding_slots.release()
                raise MemoryServiceError(
                    "memory_service_closed",
                    "memory service is closed",
                    status_code=503,
                ) from exc
            self._track_teach_embedding_future(future, loop)
            return await asyncio.wrap_future(future)
        except EmbeddingUnavailable as exc:
            raise MemoryServiceError(exc.code, exc.message, status_code=503) from exc

    def _write_embedding_artifact(
        self,
        *,
        owner_type: str,
        owner_id: str,
        crop_hash: str,
        payload: bytes,
    ) -> str:
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self._artifact_dir / f"{owner_type}-{owner_id}-{crop_hash[:16]}.jpg"
        path.write_bytes(payload)
        return _readable_artifact_ref(path)

    def _delete_embedding_artifact(self, crop_path_or_artifact_ref: str) -> None:
        path = Path(crop_path_or_artifact_ref)
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            resolved = path.resolve()
            artifact_root = self._artifact_dir.resolve()
            if resolved != artifact_root and artifact_root not in resolved.parents:
                return
            resolved.unlink()
        except FileNotFoundError:
            return
        except OSError:
            _LOGGER.warning(
                "failed to delete memory embedding artifact after store write failure",
                exc_info=True,
            )

    def _track_teach_embedding_future(
        self,
        future: Future[EmbeddingResult],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        with self._teach_embedding_futures_lock:
            self._teach_embedding_futures.add(future)

        def on_done(done: Future[EmbeddingResult]) -> None:
            with self._teach_embedding_futures_lock:
                self._teach_embedding_futures.discard(done)
            try:
                loop.call_soon_threadsafe(self._teach_embedding_slots.release)
            except RuntimeError:
                self._teach_embedding_slots.release()

        future.add_done_callback(on_done)

    def _resolve_cached_target(
        self,
        cached: CachedFrame,
        target_request: TargetRequest,
    ) -> ResolvedTarget:
        if target_request.mode == "point_uv":
            preview = self._preview_cached_target(cached, target_request)
            if preview.status != "resolved":
                code = "target_ambiguous"
                message = "target point is ambiguous"
                if preview.status == "not_found":
                    code = "target_not_found"
                    message = "target point did not resolve to a writable target"
                raise MemoryServiceError(
                    code,
                    message,
                    details={
                        "status": preview.status,
                        "candidates": [
                            _candidate_to_dict(candidate)
                            for candidate in preview.candidates
                        ],
                        "evidence": self._request_evidence(cached),
                    },
                )
        try:
            return self._resolver.resolve(
                target_request,
                image_width=cached.frame.width,
                image_height=cached.frame.height,
                tracks=_tracks_from_visual_state(
                    cached.visual_state,
                    frame=cached.frame,
                ),
                attention=_attention_from_visual_state(cached.visual_state),
            )
        except TargetResolveError as exc:
            raise MemoryServiceError(exc.code, exc.message) from exc

    def _preview_cached_target(
        self,
        cached: CachedFrame,
        target_request: TargetRequest,
    ):
        try:
            return self._resolver.preview(
                target_request,
                image_width=cached.frame.width,
                image_height=cached.frame.height,
                tracks=_tracks_from_visual_state(
                    cached.visual_state,
                    frame=cached.frame,
                ),
                attention=_attention_from_visual_state(cached.visual_state),
            )
        except TargetResolveError as exc:
            raise MemoryServiceError(exc.code, exc.message) from exc

    def _resolve_person_teach_target(
        self,
        cached: CachedFrame,
        request: dict[str, Any],
    ) -> tuple[CachedFrame, ResolvedTarget, RequestInteractionSnapshot | None]:
        public_target = _public_person_target(request)
        if public_target is None:
            target = self._resolve_cached_target(cached, _target_request(request))
            return cached, target, None

        intent = _required_text(public_target, "intent")
        if intent == "self_introduction":
            interaction_snapshot, ambiguity_type, evidence = (
                self._request_interaction_snapshot(
                    cached.connection_id,
                    cached.frame.camera,
                )
            )
            target = (
                self._active_interaction_target_from_frame(
                    interaction_snapshot.selected,
                    check_stale=False,
                )[0]
                if interaction_snapshot is not None
                else None
            )
            if target is not None:
                return interaction_snapshot.selected, target, interaction_snapshot
            raise _target_ambiguous_error(
                ambiguity_type,
                evidence=evidence,
            )
        if intent == "third_person_introduction":
            target, ambiguity_type, interaction_snapshot, evidence = (
                self._third_person_introduction_target(
                    cached.connection_id,
                    cached.frame.camera,
                )
            )
            if target is not None and interaction_snapshot is not None:
                return interaction_snapshot.selected, target, interaction_snapshot
            raise _target_ambiguous_error(
                ambiguity_type,
                evidence=evidence,
            )
        raise MemoryServiceError(
            "unsupported_person_intent",
            f"unsupported person target intent {intent}",
        )

    def _preview_public_person_target(
        self,
        cached: CachedFrame,
        target: dict[str, Any],
    ) -> dict[str, Any]:
        intent = _required_text(target, "intent")
        if intent == "self_introduction":
            interaction_snapshot, ambiguity_type, evidence = (
                self._request_interaction_snapshot(
                    cached.connection_id,
                    cached.frame.camera,
                )
            )
            resolved = (
                self._active_interaction_target_from_frame(
                    interaction_snapshot.selected,
                    check_stale=False,
                )[0]
                if interaction_snapshot is not None
                else None
            )
            if resolved is None:
                return _ambiguous_target_response(
                    ambiguity_type,
                    evidence=evidence,
                )
            return {
                "ok": True,
                "status": "resolved",
                "candidates": [
                    _resolved_target_to_candidate_dict(
                        resolved,
                        reason="active_interaction_target",
                    )
                ],
                "evidence": self._target_evidence(
                    interaction_snapshot.selected,
                    resolved,
                    interaction_snapshot=interaction_snapshot,
                ),
            }
        if intent == "third_person_introduction":
            resolved, ambiguity_type, interaction_snapshot, evidence = (
                self._third_person_introduction_target(
                    cached.connection_id,
                    cached.frame.camera,
                )
            )
            if resolved is None or interaction_snapshot is None:
                return _ambiguous_target_response(
                    ambiguity_type,
                    evidence=evidence,
                )
            return {
                "ok": True,
                "status": "resolved",
                "candidates": [
                    _resolved_target_to_candidate_dict(
                        resolved,
                        reason="pose_pointing_to_person",
                    )
                ],
                "evidence": self._target_evidence(
                    interaction_snapshot.selected,
                    resolved,
                    interaction_snapshot=interaction_snapshot,
                ),
            }
        return {
            **_ambiguous_target_response(
                "target_unclear",
                evidence=self._request_evidence(cached),
            ),
            "error_code": "unsupported_person_intent",
        }

    def _request_interaction_snapshot(
        self,
        connection_id: str,
        camera: str,
    ) -> tuple[RequestInteractionSnapshot | None, str, dict[str, Any]]:
        try:
            window = self._cache.get_snapshot_window_for_stream(connection_id, camera)
        except FrameCacheError as exc:
            raise MemoryServiceError(exc.code, exc.message, status_code=409) from exc

        now_ms = self._clock_ms()
        snapshot_count = 0
        fresh_snapshot_count = 0
        active_entries: list[tuple[CachedFrame, ResolvedTarget]] = []
        active_track_ids: list[int] = []
        latest_fresh_cached: CachedFrame | None = None
        latest_fresh_target: ResolvedTarget | None = None

        for cached in window.frames:
            snapshot = cached.memory_snapshot
            if snapshot is None:
                continue
            snapshot_count += 1
            if now_ms - snapshot.observed_at_ms > self._cache.max_age_ms:
                continue
            fresh_snapshot_count += 1
            latest_fresh_cached = cached
            latest_fresh_target = None
            target, _ambiguity_type = self._active_interaction_target_from_frame(
                cached,
                check_stale=False,
            )
            if target is None or target.track_id is None:
                continue
            active_entries.append((cached, target))
            active_track_ids.append(target.track_id)
            latest_fresh_target = target

        track_counts = Counter(active_track_ids)
        stable_track_id: int | None = None
        for track_id, count in track_counts.most_common():
            if count >= 2:
                stable_track_id = track_id
                break

        selected_cached: CachedFrame | None = None
        if (
            stable_track_id is not None
            and latest_fresh_cached is not None
            and latest_fresh_target is not None
            and latest_fresh_target.track_id == stable_track_id
        ):
            selected_cached = latest_fresh_cached

        summary = {
            "size": len(window.frames),
            "snapshot_count": snapshot_count,
            "fresh_snapshot_count": fresh_snapshot_count,
            "active_snapshot_count": len(active_entries),
            "required_active_snapshot_count": 2,
            "active_track_ids": active_track_ids,
        }
        if stable_track_id is not None and selected_cached is not None:
            selected_snapshot = selected_cached.memory_snapshot
            if selected_snapshot is not None:
                summary.update(
                    {
                        "active_target_track_id": stable_track_id,
                        "selected_snapshot_ref": selected_snapshot.snapshot_ref,
                        "selected_frame_ref": _source_frame_ref(selected_cached),
                    }
                )
                interaction_snapshot = RequestInteractionSnapshot(
                    selected=selected_cached,
                    request_snapshot_ref=selected_snapshot.snapshot_ref,
                    source_frame_ref=selected_snapshot.source_frame_ref,
                    frame_timestamp_ms=selected_cached.frame.timestamp_ms,
                    observed_at_ms=selected_cached.observed_at_ms,
                    frame_cache_ttl_ms=self._cache.max_age_ms,
                    stability_window=summary,
                    active_target_track_id=stable_track_id,
                )
                return interaction_snapshot, "", self._request_evidence(
                    selected_cached,
                    interaction_snapshot=interaction_snapshot,
                )

        evidence_cached = self._newest_snapshot_frame(window.frames) or window.frames[-1]
        ambiguity_type = (
            "stale_interaction"
            if snapshot_count > 0 and fresh_snapshot_count == 0
            else "no_active_interaction_target"
        )
        return (
            None,
            ambiguity_type,
            self._request_evidence(
                evidence_cached,
                stability_window=summary,
            ),
        )

    def _active_interaction_target_from_frame(
        self,
        cached: CachedFrame,
        *,
        check_stale: bool = True,
    ) -> tuple[ResolvedTarget | None, str]:
        snapshot = cached.memory_snapshot
        if snapshot is None:
            return None, "no_active_interaction_target"
        if (
            check_stale
            and self._clock_ms() - snapshot.observed_at_ms > self._cache.max_age_ms
        ):
            return None, "stale_interaction"

        scene_context = snapshot.scene_context or {}
        if scene_context.get("engagement_state") != "available":
            return None, "no_active_interaction_target"
        if scene_context.get("attention_available") is not True:
            return None, "no_active_interaction_target"

        scene_track_id = scene_context.get("target_track_id")
        if not isinstance(scene_track_id, int):
            return None, "no_active_interaction_target"

        attention = snapshot.attention
        if attention is None or attention.target_track_id != scene_track_id:
            return None, "no_active_interaction_target"
        if attention.largest_person_stable is not True:
            return None, "no_active_interaction_target"

        track = _visible_person_track(snapshot.tracks, scene_track_id)
        if track is None:
            return None, "no_active_interaction_target"

        try:
            resolved = self._resolver.resolve(
                TargetRequest(mode="track_id", track_id=scene_track_id),
                image_width=snapshot.image_size[0],
                image_height=snapshot.image_size[1],
                tracks=snapshot.tracks,
                attention=attention,
            )
        except TargetResolveError:
            return None, "no_active_interaction_target"
        return (
            ResolvedTarget(
                source_target_mode="active_interaction_target",
                target_type=resolved.target_type,
                bbox_xyxy=resolved.bbox_xyxy,
                track_id=resolved.track_id,
                quality=resolved.quality,
            ),
            "",
        )

    def _third_person_introduction_target(
        self,
        connection_id: str,
        camera: str,
    ) -> tuple[
        ResolvedTarget | None,
        str,
        RequestInteractionSnapshot | None,
        dict[str, Any],
    ]:
        interaction_snapshot, ambiguity_type, evidence = (
            self._request_interaction_snapshot(connection_id, camera)
        )
        if interaction_snapshot is None:
            return None, ambiguity_type, None, evidence

        cached = interaction_snapshot.selected
        introducer, ambiguity_type = self._active_interaction_target_from_frame(
            cached,
            check_stale=False,
        )
        if introducer is None or introducer.track_id is None:
            return None, ambiguity_type, interaction_snapshot, evidence

        snapshot = cached.memory_snapshot
        if snapshot is None:
            return None, "no_active_interaction_target", interaction_snapshot, evidence
        return self._stable_third_person_pose_target(
            connection_id=connection_id,
            camera=camera,
            interaction_snapshot=interaction_snapshot,
            introducer_track_id=introducer.track_id,
            base_evidence=evidence,
        )

    def _stable_third_person_pose_target(
        self,
        *,
        connection_id: str,
        camera: str,
        interaction_snapshot: RequestInteractionSnapshot,
        introducer_track_id: int,
        base_evidence: dict[str, Any],
    ) -> tuple[
        ResolvedTarget | None,
        str,
        RequestInteractionSnapshot,
        dict[str, Any],
    ]:
        try:
            window = self._cache.get_snapshot_window_for_stream(connection_id, camera)
        except FrameCacheError:
            return None, "target_unclear", interaction_snapshot, base_evidence

        (
            selected_candidate,
            selected_preview_evidence,
            ambiguity_type,
            pose_window,
        ) = self._third_person_pose_window_decision(
            window=window,
            interaction_snapshot=interaction_snapshot,
            introducer_track_id=introducer_track_id,
        )
        evidence = dict(base_evidence)
        evidence.update(selected_preview_evidence)
        evidence["pose_stability_window"] = pose_window
        introducer_ref = f"{camera}:track:{introducer_track_id}"
        evidence = _with_pose_visual_service_evidence(
            evidence,
            introducer_ref=introducer_ref,
            target_ref=None,
            pose_stability_window=pose_window,
            clear_target=selected_candidate is None,
        )
        if selected_candidate is None:
            return None, ambiguity_type, interaction_snapshot, evidence

        target_evidence = dict(selected_candidate.evidence or selected_preview_evidence)
        target_evidence["pose_stability_window"] = pose_window
        target_ref = (
            f"{camera}:track:{selected_candidate.track_id}"
            if selected_candidate.track_id is not None
            else None
        )
        target_evidence = _with_pose_visual_service_evidence(
            target_evidence,
            introducer_ref=introducer_ref,
            target_ref=target_ref,
            pose_stability_window=pose_window,
        )
        return (
            ResolvedTarget(
                source_target_mode="pose_pointing_to_person",
                target_type=selected_candidate.target_type,
                bbox_xyxy=selected_candidate.bbox_xyxy,
                track_id=selected_candidate.track_id,
                quality="usable",
                evidence=target_evidence,
            ),
            "",
            interaction_snapshot,
            evidence,
        )

    def _third_person_pose_window_decision(
        self,
        *,
        window: MemoryFrameSnapshotWindow,
        interaction_snapshot: RequestInteractionSnapshot,
        introducer_track_id: int,
    ) -> tuple[Any | None, dict[str, Any], str, dict[str, Any]]:
        now_ms = self._clock_ms()
        fresh_snapshot_count = 0
        same_introducer_snapshot_count = 0
        status_counts: Counter[str] = Counter()
        pair_counts: Counter[tuple[int, str]] = Counter()
        pair_snapshot_refs: dict[tuple[int, str], list[str]] = {}
        selected_pair: tuple[int, str] | None = None
        selected_candidate: Any | None = None
        selected_target_track_id: int | None = None
        selected_arm_side: str | None = None
        selected_preview_evidence: dict[str, Any] = {}
        selected_ambiguity_type = "target_unclear"

        for cached in window.frames:
            snapshot = cached.memory_snapshot
            if snapshot is None:
                continue
            if now_ms - snapshot.observed_at_ms > self._cache.max_age_ms:
                continue
            fresh_snapshot_count += 1
            introducer, _ambiguity_type = self._active_interaction_target_from_frame(
                cached,
                check_stale=False,
            )
            if introducer is None or introducer.track_id != introducer_track_id:
                continue
            same_introducer_snapshot_count += 1
            is_selected = (
                cached is interaction_snapshot.selected
                or snapshot.snapshot_ref == interaction_snapshot.request_snapshot_ref
            )
            try:
                preview = self._resolver.preview_pose_pointing_person(
                    introducer_track_id=introducer_track_id,
                    image_width=snapshot.image_size[0],
                    image_height=snapshot.image_size[1],
                    tracks=snapshot.tracks,
                )
            except TargetResolveError:
                status_counts["resolver_error"] += 1
                if is_selected:
                    selected_ambiguity_type = "target_unclear"
                continue

            status_counts[preview.status] += 1
            preview_evidence = dict(preview.evidence or {})
            if is_selected:
                selected_preview_evidence = preview_evidence
                selected_arm_side = _pose_preview_arm_side(preview_evidence)
                selected_ambiguity_type = preview.ambiguity_type or "target_unclear"

            if preview.status != "resolved" or not preview.candidates:
                continue

            candidate = preview.candidates[0]
            arm_side = _pose_preview_arm_side(candidate.evidence or preview_evidence)
            if candidate.track_id is None or arm_side is None:
                if is_selected:
                    selected_candidate = candidate
                    selected_target_track_id = candidate.track_id
                    selected_arm_side = arm_side
                continue

            pair = (int(candidate.track_id), arm_side)
            pair_counts[pair] += 1
            pair_snapshot_refs.setdefault(pair, []).append(snapshot.snapshot_ref)
            if is_selected:
                selected_pair = pair
                selected_candidate = candidate
                selected_target_track_id = int(candidate.track_id)
                selected_arm_side = arm_side

        selected_count = pair_counts[selected_pair] if selected_pair is not None else 0
        failure_reason: str | None = None
        if selected_candidate is None:
            failure_reason = selected_ambiguity_type
        elif selected_pair is None:
            failure_reason = "pose_target_unclear"
            selected_ambiguity_type = "target_unclear"
        elif selected_count < _REQUIRED_POSE_SNAPSHOT_COUNT:
            failure_reason = "pose_target_unstable"
            selected_ambiguity_type = "target_unclear"

        pose_window = {
            "size": len(window.frames),
            "fresh_snapshot_count": fresh_snapshot_count,
            "same_introducer_snapshot_count": same_introducer_snapshot_count,
            "required_pose_snapshot_count": _REQUIRED_POSE_SNAPSHOT_COUNT,
            "resolved_pose_snapshot_count": sum(pair_counts.values()),
            "selected_snapshot_ref": interaction_snapshot.request_snapshot_ref,
            "selected_target_track_id": selected_target_track_id,
            "selected_arm_side": selected_arm_side,
            "selected_count": selected_count,
            "candidate_arm_counts": [
                {
                    "target_track_id": track_id,
                    "arm_side": arm_side,
                    "count": count,
                    "snapshot_refs": pair_snapshot_refs[(track_id, arm_side)],
                }
                for (track_id, arm_side), count in sorted(
                    pair_counts.items(),
                    key=lambda item: (-item[1], item[0][0], item[0][1]),
                )
            ],
            "status_counts": dict(sorted(status_counts.items())),
            "failure_reason": failure_reason,
        }
        if failure_reason is not None:
            return None, selected_preview_evidence, selected_ambiguity_type, pose_window
        return selected_candidate, selected_preview_evidence, "", pose_window

    def _query_due(self, stream_key: _StreamKey, frame: FrameMessage) -> bool:
        last = self._last_query_frame_timestamp_ms_by_stream.get(stream_key)
        return last is None or frame.timestamp_ms - last >= self.query_interval_ms

    def _collect_completed_query(
        self,
        stream_key: _StreamKey,
        future: asyncio.Future[list[dict[str, Any]]],
    ) -> None:
        try:
            completed = future.result()
        except Exception:
            _LOGGER.exception(
                "memory query failed for camera %s",
                stream_key[1],
            )
            completed = []
        for event in completed:
            self._enqueue(stream_key, event)
        if self._pending_queries_by_stream.get(stream_key) is future:
            self._pending_queries_by_stream.pop(stream_key, None)

    def _run_query(self, plan: _QueryPlan) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        queried_person_targets: list[ResolvedTarget] = []
        updated_anonymous_ids: set[str] = set()
        for person_target in self._query_person_targets(plan):
            queried_person_targets.append(person_target)
            self._record_recognition_queried_targets(plan, queried_person_targets)
            person_event = self._query_person(
                plan,
                person_target,
                updated_anonymous_ids=updated_anonymous_ids,
            )
            if person_event is not None:
                completed.append(person_event)
                break
        scene_event = self._query_scene(plan)
        if scene_event is not None:
            completed.append(scene_event)
        return completed

    def _query_person_targets(
        self,
        plan: _QueryPlan,
    ) -> list[ResolvedTarget]:
        snapshot = plan.cached.memory_snapshot
        if snapshot is None:
            self._record_recognition_report(
                _RecognitionTickReport(
                    camera=plan.cached.frame.camera,
                    frame_id=plan.cached.frame.frame_id,
                    frame_timestamp_ms=plan.cached.frame.timestamp_ms,
                    source_frame_ref=_source_frame_ref(plan.cached),
                    tracks_seen=0,
                    tracks_eligible=0,
                    tracks_candidates=0,
                    candidate_track_ids=(),
                    tracks_queried=0,
                    tracks_skipped_reason={},
                    queried_track_ids=(),
                    attention_target_track_id=None,
                    attention_target_only=False,
                    max_tracks_per_tick=_MAX_PERSON_QUERY_TRACKS,
                    query_interval_ms=self.query_interval_ms,
                    event_cooldown_ms=self.event_cooldown_ms,
                    recognition_runs_in_executor=True,
                    eligibility_policy=_RECOGNITION_ELIGIBILITY_POLICY,
                )
            )
            return []

        targets: list[ResolvedTarget] = []
        skipped_reason: Counter[str] = Counter()
        tracks_eligible = 0
        for track in snapshot.tracks:
            skip_reason = _recognition_track_skip_reason(track)
            if skip_reason:
                skipped_reason[skip_reason] += 1
                continue
            tracks_eligible += 1
            if len(targets) >= _MAX_PERSON_QUERY_TRACKS:
                skipped_reason["max_tracks_per_tick"] += 1
                continue
            try:
                resolved = self._resolver.resolve(
                    TargetRequest(mode="track_id", track_id=track.track_id),
                    image_width=snapshot.image_size[0],
                    image_height=snapshot.image_size[1],
                    tracks=snapshot.tracks,
                    attention=snapshot.attention,
                )
            except TargetResolveError:
                skipped_reason["target_resolve_failed"] += 1
                continue
            targets.append(
                ResolvedTarget(
                    source_target_mode="recognition_track",
                    target_type=resolved.target_type,
                    bbox_xyxy=resolved.bbox_xyxy,
                    track_id=resolved.track_id,
                    quality=resolved.quality,
                )
            )
        self._record_recognition_report(
            _RecognitionTickReport(
                camera=plan.cached.frame.camera,
                frame_id=plan.cached.frame.frame_id,
                frame_timestamp_ms=plan.cached.frame.timestamp_ms,
                source_frame_ref=_source_frame_ref(plan.cached),
                tracks_seen=len(snapshot.tracks),
                tracks_eligible=tracks_eligible,
                tracks_candidates=len(targets),
                candidate_track_ids=tuple(
                    int(target.track_id)
                    for target in targets
                    if target.track_id is not None
                ),
                tracks_queried=0,
                tracks_skipped_reason=dict(sorted(skipped_reason.items())),
                queried_track_ids=(),
                attention_target_track_id=(
                    int(snapshot.attention.target_track_id)
                    if snapshot.attention is not None
                    and snapshot.attention.target_track_id is not None
                    else None
                ),
                attention_target_only=False,
                max_tracks_per_tick=_MAX_PERSON_QUERY_TRACKS,
                query_interval_ms=self.query_interval_ms,
                event_cooldown_ms=self.event_cooldown_ms,
                recognition_runs_in_executor=True,
                eligibility_policy=_RECOGNITION_ELIGIBILITY_POLICY,
            )
        )
        return targets

    def _record_recognition_report(self, report: _RecognitionTickReport) -> None:
        with self._recognition_report_lock:
            self._latest_recognition_report = report
            self._latest_recognition_reports_by_camera[report.camera] = report

    def _record_recognition_queried_targets(
        self,
        plan: _QueryPlan,
        targets: list[ResolvedTarget],
    ) -> None:
        camera = plan.cached.frame.camera
        queried_track_ids = tuple(
            int(target.track_id) for target in targets if target.track_id is not None
        )
        with self._recognition_report_lock:
            report = self._latest_recognition_reports_by_camera.get(camera)
            if (
                report is None
                or report.frame_id != plan.cached.frame.frame_id
                or report.frame_timestamp_ms != plan.cached.frame.timestamp_ms
            ):
                return
            updated = replace(
                report,
                tracks_queried=len(queried_track_ids),
                queried_track_ids=queried_track_ids,
            )
            self._latest_recognition_reports_by_camera[camera] = updated
            if self._latest_recognition_report is report:
                self._latest_recognition_report = updated

    def _query_person(
        self,
        plan: _QueryPlan,
        target: ResolvedTarget,
        *,
        updated_anonymous_ids: set[str],
    ) -> dict[str, Any] | None:
        result = self._identify_person_target(
            plan.cached,
            target,
            source="background_recall",
            include_anonymous=False,
        )
        if result.status == "unavailable":
            if result.reason == "no_usable_face":
                self._put_identity_overlay_result(
                    plan.cached,
                    target,
                    result,
                    source="background_recall",
                )
            return None
        if result.status != "known_person":
            if result.embedding is None or result.embedding_bytes is None:
                return None
            return self._query_anonymous_person(
                plan,
                target,
                result.embedding,
                result.embedding_bytes,
                updated_anonymous_ids=updated_anonymous_ids,
            )
        if result.match is None or result.profile is None:
            return None
        conversation_summaries = tuple(
            self.store.get_conversation_summaries(result.match.matched_id, limit=3)
        )
        self._put_identity_overlay_result(
            plan.cached,
            target,
            result,
            source="background_recall",
        )
        with self._query_state_lock:
            if not self._gate.allow(
                _event_gate_scope(plan.cached),
                "known_person_present",
                result.match.matched_id,
                now_ms=plan.cached.frame.timestamp_ms,
            ):
                return None
            event_id = self._next_event_id(plan.cached.frame.camera)
        source = SourceFrameRef(
            camera=plan.cached.frame.camera,
            frame_id=plan.cached.frame.frame_id,
            frame_timestamp_ms=plan.cached.frame.timestamp_ms,
            source_target_mode=target.source_target_mode,
            track_id=target.track_id,
        )
        self._record_match(result.match, event_id=event_id, source=source)
        return build_known_person_event(
            event_id=event_id,
            match=result.match,
            source=source,
            person_profile=result.profile,
            conversation_summaries=conversation_summaries,
        )

    def _identify_person_target(
        self,
        cached: CachedFrame,
        target: ResolvedTarget,
        *,
        source: str,
        include_anonymous: bool,
    ) -> _PersonIdentifyResult:
        embedding: EmbeddingResult | None = None
        embedding_bytes: bytes | None = None
        saw_no_usable_face = False
        for candidate_bytes in _person_embedding_input_candidates(
            cached.frame.jpeg_bytes,
            target,
            tracks=(
                cached.memory_snapshot.tracks
                if cached.memory_snapshot is not None
                else None
            ),
        ):
            try:
                embedding = self.embedding_backend.embed_person(candidate_bytes)
            except EmbeddingUnavailable as exc:
                if exc.code == "no_usable_face":
                    saw_no_usable_face = True
                    continue
                return _PersonIdentifyResult(
                    status="unavailable",
                    reason=exc.code or "embedding_unavailable",
                    identity_context=unavailable_person_identity_context(
                        reason=exc.code or "embedding_unavailable",
                        source=source,
                    ),
                )
            embedding_bytes = candidate_bytes
            break
        if embedding is None or embedding_bytes is None:
            reason = "no_usable_face" if saw_no_usable_face else "embedding_unavailable"
            return _PersonIdentifyResult(
                status="unavailable",
                reason=reason,
                identity_context=unavailable_person_identity_context(
                    reason=reason,
                    source=source,
                ),
            )

        match = self._retriever.query_person(
            embedding,
            threshold=self.known_person_threshold,
            margin=self.known_person_margin,
        )
        if match is not None:
            profile = self.store.get_person_profile(match.matched_id)
            if profile is not None:
                return _PersonIdentifyResult(
                    status="known_person",
                    identity_context=known_person_identity_context(
                        profile,
                        match,
                        source=source,
                    ),
                    match=match,
                    profile=profile,
                    embedding=embedding,
                    embedding_bytes=embedding_bytes,
                )

        if include_anonymous:
            anonymous_result = self._identify_existing_anonymous_person(
                target,
                embedding,
                source=source,
            )
            return replace(
                anonymous_result,
                embedding=embedding,
                embedding_bytes=embedding_bytes,
            )

        return _PersonIdentifyResult(
            status="unknown",
            reason="no_known_person_match",
            identity_context=unknown_identity_context(
                reason="no_known_person_match",
                source=source,
            ),
            embedding=embedding,
            embedding_bytes=embedding_bytes,
        )

    def _identify_existing_anonymous_person(
        self,
        target: ResolvedTarget,
        embedding: EmbeddingResult,
        *,
        source: str,
    ) -> _PersonIdentifyResult:
        if target.quality != "usable":
            return _PersonIdentifyResult(
                status="unknown",
                reason="target_quality_not_usable",
                identity_context=unknown_identity_context(
                    reason="target_quality_not_usable",
                    source=source,
                ),
            )
        match = self._retriever.query_anonymous_person(
            embedding,
            threshold=self.anonymous_threshold,
            margin=self.anonymous_margin,
        )
        if match is None:
            return _PersonIdentifyResult(
                status="unknown",
                reason="no_public_identity_match",
                identity_context=unknown_identity_context(
                    reason="no_public_identity_match",
                    source=source,
                ),
            )
        profile = self.store.get_active_anonymous_profile(match.matched_id)
        if profile is None:
            return _PersonIdentifyResult(
                status="unknown",
                reason="anonymous_profile_unavailable",
                identity_context=unknown_identity_context(
                    reason="anonymous_profile_unavailable",
                    source=source,
                ),
            )
        if not self._anonymous_profile_is_familiar(profile):
            return _PersonIdentifyResult(
                status="unknown",
                reason="not_familiar",
                identity_context=unknown_identity_context(
                    reason="not_familiar",
                    source=source,
                ),
                match=match,
                profile=profile,
            )
        return _PersonIdentifyResult(
            status="familiar_unknown",
            identity_context=familiar_unknown_identity_context(
                profile,
                match,
                source=source,
            ),
            match=match,
            profile=profile,
        )

    def _anonymous_profile_is_familiar(self, profile: dict[str, Any]) -> bool:
        return (
            int(profile["seen_count"]) >= self.familiar_seen_count
            and int(profile.get("observed_duration_ms", 0))
            >= self.familiar_observed_duration_ms
            and float(profile["familiar_score"]) >= self.familiar_threshold
        )

    def _put_identity_overlay_result(
        self,
        cached: CachedFrame,
        target: ResolvedTarget,
        result: _PersonIdentifyResult,
        *,
        source: str,
    ) -> None:
        if result.status == "known_person" and result.match is not None:
            if result.profile is not None:
                self._identity_overlay.put_known_person(
                    connection_id=cached.connection_id,
                    camera=cached.frame.camera,
                    track_id=target.track_id,
                    person_profile=result.profile,
                    match=result.match,
                    source=source,
                )
            return
        if result.status == "familiar_unknown" and result.match is not None:
            if result.profile is not None:
                self._identity_overlay.put_familiar_unknown(
                    connection_id=cached.connection_id,
                    camera=cached.frame.camera,
                    track_id=target.track_id,
                    anonymous_profile=result.profile,
                    match=result.match,
                    source=source,
                )
            return
        if result.status == "unavailable":
            self._identity_overlay.put_unavailable(
                connection_id=cached.connection_id,
                camera=cached.frame.camera,
                track_id=target.track_id,
                reason=result.reason or "embedding_unavailable",
                source=source,
            )
            return
        self._identity_overlay.put_unknown(
            connection_id=cached.connection_id,
            camera=cached.frame.camera,
            track_id=target.track_id,
            reason=result.reason or "no_public_identity_match",
            source=source,
        )

    def _refresh_teach_person_identity_overlay(
        self,
        cached: CachedFrame,
        target: ResolvedTarget,
        person_id: str,
    ) -> dict[str, Any] | None:
        person_profile = self.store.get_person_profile(person_id)
        if person_profile is None:
            return None
        self._identity_overlay.put_known_person_profile(
            connection_id=cached.connection_id,
            camera=cached.frame.camera,
            track_id=target.track_id,
            person_profile=person_profile,
            source="teach",
        )
        return person_profile

    def _query_anonymous_person(
        self,
        plan: _QueryPlan,
        target: ResolvedTarget,
        embedding,
        embedding_bytes: bytes,
        *,
        updated_anonymous_ids: set[str],
    ) -> dict[str, Any] | None:
        if target.quality != "usable":
            return None
        match = self._retriever.query_anonymous_person(
            embedding,
            threshold=self.anonymous_threshold,
            margin=self.anonymous_margin,
        )
        now_ms = self._clock_ms()
        if match is None:
            anonymous_id = _public_id("anon")
            self.store.create_anonymous_profile(
                anonymous_id=anonymous_id,
                seen_count=1,
                first_seen_at_ms=plan.cached.frame.timestamp_ms,
                last_seen_at_ms=plan.cached.frame.timestamp_ms,
                familiar_score=0.0,
                observed_duration_ms=0,
            )
            self.store.add_anonymous_embedding(
                anonymous_id=anonymous_id,
                result=embedding,
                source_target_type=target.source_target_mode,
                source_track_ref=_source_track_ref(plan.cached, target),
                source_frame_ref=_source_frame_ref(plan.cached),
                crop_hash=_sha256_hex(embedding_bytes),
                crop_path_or_artifact_ref=None,
                resolver_target_ref=_resolver_target_ref(plan.cached, target),
                resolution_reason=target.source_target_mode,
                now_ms=now_ms,
            )
            updated_anonymous_ids.add(anonymous_id)
            self._identity_overlay.put_unknown(
                connection_id=plan.cached.connection_id,
                camera=plan.cached.frame.camera,
                track_id=target.track_id,
                reason="new_anonymous",
            )
            return None

        profile = self.store.get_active_anonymous_profile(match.matched_id)
        if profile is None:
            return None
        if match.matched_id in updated_anonymous_ids:
            self._identity_overlay.put_unknown(
                connection_id=plan.cached.connection_id,
                camera=plan.cached.frame.camera,
                track_id=target.track_id,
                reason="duplicate_anonymous_in_tick",
            )
            return None
        updated_anonymous_ids.add(match.matched_id)
        seen_count = int(profile["seen_count"]) + 1
        familiar_score = max(float(profile["familiar_score"]), match.match_score)
        observed_duration_ms = _updated_observed_duration_ms(
            profile,
            current_frame_timestamp_ms=plan.cached.frame.timestamp_ms,
            query_interval_ms=self.query_interval_ms,
        )
        self.store.update_anonymous_profile(
            anonymous_id=match.matched_id,
            seen_count=seen_count,
            last_seen_at_ms=plan.cached.frame.timestamp_ms,
            familiar_score=familiar_score,
            observed_duration_ms=observed_duration_ms,
        )
        profile = {
            **profile,
            "seen_count": seen_count,
            "last_seen_at_ms": plan.cached.frame.timestamp_ms,
            "familiar_score": familiar_score,
            "observed_duration_ms": observed_duration_ms,
        }
        if (
            seen_count < self.familiar_seen_count
            or observed_duration_ms < self.familiar_observed_duration_ms
            or familiar_score < self.familiar_threshold
        ):
            self._identity_overlay.put_unknown(
                connection_id=plan.cached.connection_id,
                camera=plan.cached.frame.camera,
                track_id=target.track_id,
                reason="not_familiar",
            )
            return None
        self._identity_overlay.put_familiar_unknown(
            connection_id=plan.cached.connection_id,
            camera=plan.cached.frame.camera,
            track_id=target.track_id,
            anonymous_profile=profile,
            match=match,
        )
        with self._query_state_lock:
            if not self._gate.allow(
                _event_gate_scope(plan.cached),
                "familiar_unknown_present",
                match.matched_id,
                now_ms=plan.cached.frame.timestamp_ms,
            ):
                return None
            event_id = self._next_event_id(plan.cached.frame.camera)
        source = SourceFrameRef(
            camera=plan.cached.frame.camera,
            frame_id=plan.cached.frame.frame_id,
            frame_timestamp_ms=plan.cached.frame.timestamp_ms,
            source_target_mode=target.source_target_mode,
            track_id=target.track_id,
        )
        self._record_match(match, event_id=event_id, source=source)
        return build_familiar_unknown_event(
            event_id=event_id,
            match=match,
            source=source,
            anonymous_profile=profile,
        )

    def _query_scene(self, plan: _QueryPlan) -> dict[str, Any] | None:
        target = ResolvedTarget(
            source_target_mode="scene",
            target_type="scene",
            bbox_xyxy=(
                0.0,
                0.0,
                float(plan.cached.frame.width),
                float(plan.cached.frame.height),
            ),
            track_id=None,
            quality="usable",
        )
        try:
            embedding = self.embedding_backend.embed_scene(
                _target_bytes(plan.cached.frame.jpeg_bytes, target)
            )
        except EmbeddingUnavailable:
            return None
        match = self._retriever.query_scene(
            embedding,
            threshold=self.scene_threshold,
            margin=0.0,
        )
        if match is None:
            return None
        scene = self.store.get_scene_memory(match.matched_id)
        if scene is None:
            return None
        with self._query_state_lock:
            if not self._gate.allow(
                _event_gate_scope(plan.cached),
                "scene_activated",
                match.matched_id,
                now_ms=plan.cached.frame.timestamp_ms,
            ):
                return None
            event_id = self._next_event_id(plan.cached.frame.camera)
        source = SourceFrameRef(
            camera=plan.cached.frame.camera,
            frame_id=plan.cached.frame.frame_id,
            frame_timestamp_ms=plan.cached.frame.timestamp_ms,
            source_target_mode="scene",
            track_id=None,
        )
        self._record_match(match, event_id=event_id, source=source)
        return build_scene_event(
            event_id=event_id,
            match=match,
            source=source,
            scene_memory=scene,
        )

    def _record_match(
        self,
        match: MemoryMatch,
        *,
        event_id: str,
        source: SourceFrameRef,
    ) -> None:
        self.store.add_memory_match_record(
            event_id=event_id,
            matched_type=match.matched_type,
            matched_id=match.matched_id,
            embedding_id=match.embedding_id,
            match_score=match.match_score,
            top2_margin=match.top2_margin,
            source_target_mode=source.source_target_mode,
            camera=source.camera,
            frame_id=source.frame_id,
            frame_timestamp_ms=source.frame_timestamp_ms,
            now_ms=self._clock_ms(),
            memory_match_id=match.memory_match_id,
        )

    def _next_event_id(self, camera: str) -> str:
        next_value = self._event_counters.get(camera, 0) + 1
        self._event_counters[camera] = next_value
        return f"{camera}:mem_evt_{next_value:06d}"

    def _enqueue(self, stream_key: _StreamKey, event: dict[str, Any]) -> None:
        queue = self._completed_by_stream.setdefault(
            stream_key,
            deque(maxlen=self._queue_size),
        )
        queue.append(event)

    def _store_count_snapshot(self) -> dict[str, int]:
        return self.store.memory_table_counts()

    def _store_delta(self, before: dict[str, int]) -> dict[str, dict[str, int]]:
        after = self._store_count_snapshot()
        return {
            "before": before,
            "after": after,
            "delta": {
                table: after.get(table, 0) - before.get(table, 0)
                for table in sorted(set(before) | set(after))
            },
        }

    def _teach_match_evidence(
        self,
        match: MemoryMatch,
        cached: CachedFrame,
        target: ResolvedTarget,
        *,
        crop_hash: str | None = None,
        crop_path_or_artifact_ref: str | None = None,
        embedding: EmbeddingResult | None = None,
        person_embedding_candidate: _PersonEmbeddingInputCandidate | None = None,
        interaction_snapshot: RequestInteractionSnapshot | None = None,
    ) -> dict[str, Any]:
        evidence = self._target_evidence(
            cached,
            target,
            crop_hash=crop_hash,
            crop_path_or_artifact_ref=crop_path_or_artifact_ref,
            embedding=embedding,
            person_embedding_candidate=person_embedding_candidate,
            interaction_snapshot=interaction_snapshot,
        )
        evidence.update(
            {
                "memory_match_id": match.memory_match_id,
                "matched_type": match.matched_type,
                "matched_id": match.matched_id,
                "embedding_id": match.embedding_id,
                "match_type": match.match_type,
                "match_score": match.match_score,
                "top2_margin": match.top2_margin,
                "embedding_model": match.embedding_model,
                "embedding_version": match.embedding_version,
            }
        )
        return evidence

    def _teach_person_conflict_error(
        self,
        *,
        matched_person_id: str,
        evidence: dict[str, Any],
        store_before: dict[str, int],
        external_user_ref: str | None = None,
        external_user_person_id: str | None = None,
    ) -> MemoryServiceError:
        details: dict[str, Any] = {
            "error_code": "person_teach_conflict",
            "outcome": "conflict",
            "matched_person_id": matched_person_id,
            "evidence": evidence,
            "store_delta": self._store_delta(store_before),
        }
        if not external_user_ref:
            return MemoryServiceError(
                "person_teach_conflict",
                "teach_person matched an existing person with a different display_name",
                status_code=409,
                details=details,
            )
        details["external_user_ref"] = external_user_ref
        if external_user_person_id is not None:
            details["external_user_person_id"] = external_user_person_id
        return MemoryServiceError(
            "person_teach_conflict",
            "external_user_ref is already linked to a different person",
            status_code=409,
            details=details,
        )

    def _teach_anonymous_external_ref_conflict_error(
        self,
        *,
        matched_anonymous_id: str,
        evidence: dict[str, Any],
        store_before: dict[str, int],
        external_user_ref: str,
        external_user_person_id: str,
    ) -> MemoryServiceError:
        return MemoryServiceError(
            "person_teach_conflict",
            "external_user_ref is already linked to a different person",
            status_code=409,
            details={
                "error_code": "person_teach_conflict",
                "outcome": "conflict",
                "matched_anonymous_id": matched_anonymous_id,
                "external_user_ref": external_user_ref,
                "external_user_person_id": external_user_person_id,
                "evidence": evidence,
                "store_delta": self._store_delta(store_before),
            },
        )

    def _teach_created_external_ref_conflict_error(
        self,
        *,
        evidence: dict[str, Any],
        store_before: dict[str, int],
        external_user_ref: str,
        external_user_person_id: str,
    ) -> MemoryServiceError:
        return MemoryServiceError(
            "person_teach_conflict",
            "external_user_ref is already linked to a different person",
            status_code=409,
            details={
                "error_code": "person_teach_conflict",
                "outcome": "conflict",
                "external_user_ref": external_user_ref,
                "external_user_person_id": external_user_person_id,
                "evidence": evidence,
                "store_delta": self._store_delta(store_before),
            },
        )

    def _external_ref_decision(
        self,
        profile: dict[str, Any],
        matched_person_id: str,
    ) -> _ExternalRefDecision:
        external_user_ref = _optional_text(profile.get("external_user_ref"))
        if not external_user_ref:
            return _ExternalRefDecision(
                external_user_ref="",
                linked_person_id=None,
                conflict=False,
                same_person=False,
                should_link=False,
            )
        linked = self.store.get_person_by_external_user(external_user_ref)
        linked_person_id = linked["person_id"] if linked is not None else None
        same_person = linked_person_id == matched_person_id
        return _ExternalRefDecision(
            external_user_ref=external_user_ref,
            linked_person_id=linked_person_id,
            conflict=linked_person_id is not None and not same_person,
            same_person=same_person,
            should_link=linked_person_id is None,
        )

    def _interaction_window_evidence_for_request(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        stream_ref = _optional_text(request.get("stream_ref"))
        if not stream_ref:
            return {}
        try:
            _snapshot, _ambiguity_type, evidence = self._request_interaction_snapshot(
                stream_ref,
                _required_text(request, "camera"),
            )
        except MemoryServiceError:
            return {}
        return evidence

    def _newest_snapshot_frame(
        self,
        frames: tuple[CachedFrame, ...],
    ) -> CachedFrame | None:
        for cached in reversed(frames):
            if cached.memory_snapshot is not None:
                return cached
        return None

    def _request_evidence(
        self,
        cached: CachedFrame,
        *,
        interaction_snapshot: RequestInteractionSnapshot | None = None,
        stability_window: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if interaction_snapshot is not None:
            return {
                "request_snapshot_ref": interaction_snapshot.request_snapshot_ref,
                "source_frame_ref": interaction_snapshot.source_frame_ref,
                "frame_id": interaction_snapshot.selected.frame.frame_id,
                "frame_timestamp_ms": interaction_snapshot.frame_timestamp_ms,
                "observed_at_ms": interaction_snapshot.observed_at_ms,
                "frame_cache_ttl_ms": interaction_snapshot.frame_cache_ttl_ms,
                "stability_window": interaction_snapshot.stability_window,
            }

        evidence: dict[str, Any] = {}
        if cached.memory_snapshot is not None:
            evidence["request_snapshot_ref"] = cached.memory_snapshot.snapshot_ref
            source_frame_ref = cached.memory_snapshot.source_frame_ref
        else:
            source_frame_ref = _source_frame_ref(cached)
        evidence.update(
            {
                "source_frame_ref": source_frame_ref,
                "frame_id": cached.frame.frame_id,
                "frame_timestamp_ms": cached.frame.timestamp_ms,
                "observed_at_ms": cached.observed_at_ms,
                "frame_cache_ttl_ms": self._cache.max_age_ms,
            }
        )
        if stability_window is not None:
            evidence["stability_window"] = stability_window
        return evidence

    def _target_evidence(
        self,
        cached: CachedFrame,
        target: ResolvedTarget,
        *,
        crop_hash: str | None = None,
        crop_path_or_artifact_ref: str | None = None,
        embedding: EmbeddingResult | None = None,
        person_embedding_candidate: _PersonEmbeddingInputCandidate | None = None,
        interaction_snapshot: RequestInteractionSnapshot | None = None,
    ) -> dict[str, Any]:
        evidence = self._request_evidence(
            cached,
            interaction_snapshot=interaction_snapshot,
        )
        evidence["resolution_reason"] = target.source_target_mode
        source_track_ref = _source_track_ref(cached, target)
        if source_track_ref is not None:
            evidence["source_track_ref"] = source_track_ref
        evidence["resolver_target_ref"] = _resolver_target_ref(cached, target)
        introducer_ref: str | None = None
        if target.source_target_mode == "pose_pointing_to_person":
            introducer_ref = (
                f"{cached.frame.camera}:track:"
                f"{interaction_snapshot.active_target_track_id}"
                if interaction_snapshot is not None
                else self._pose_pointing_introducer_ref(cached)
            )
            if introducer_ref is not None:
                evidence["introducer_ref"] = introducer_ref
        if target.evidence:
            evidence.update(target.evidence)
        if target.source_target_mode == "pose_pointing_to_person":
            evidence = _with_pose_visual_service_evidence(
                evidence,
                introducer_ref=introducer_ref,
                target_ref=evidence["resolver_target_ref"],
                pose_stability_window=evidence.get("pose_stability_window"),
            )
        if crop_hash is not None:
            evidence["crop_hash"] = crop_hash
        if crop_path_or_artifact_ref is not None:
            evidence["crop_path_or_artifact_ref"] = crop_path_or_artifact_ref
        if (
            crop_hash is not None
            and target.target_type in {"person", "region"}
            and person_embedding_candidate is not None
        ):
            evidence["person_visual_evidence"] = _person_visual_evidence(
                source_frame_ref=str(evidence["source_frame_ref"]),
                target=target,
                candidate=person_embedding_candidate,
                crop_hash=crop_hash,
                crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                embedding=embedding,
            )
        return evidence

    def _pose_pointing_introducer_ref(self, cached: CachedFrame) -> str | None:
        introducer, _ambiguity_type = self._active_interaction_target_from_frame(cached)
        if introducer is None or introducer.track_id is None:
            return None
        return _resolver_target_ref(cached, introducer)

    def _preview_evidence(
        self,
        cached: CachedFrame,
        preview: Any,
    ) -> dict[str, Any] | None:
        evidence = self._request_evidence(cached)
        if preview.status == "resolved" and preview.candidates:
            candidate = preview.candidates[0]
            evidence["resolution_reason"] = candidate.reason
            if candidate.track_id is not None:
                evidence["source_track_ref"] = _candidate_source_track_ref(
                    cached,
                    candidate,
                )
            evidence["resolver_target_ref"] = _candidate_resolver_target_ref(
                cached,
                candidate,
            )
        if getattr(preview, "evidence", None):
            evidence.update(preview.evidence)
        return evidence


def _identify_current_response(
    *,
    status: str,
    reason: str | None = None,
    people: list[dict[str, Any]] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "ok": True,
        "status": status,
        "people": list(people or []),
        "evidence": dict(evidence or {}),
    }
    if reason is not None:
        response["reason"] = reason
    return response


def _consume_identify_future_exception(
    future: asyncio.Future[_PersonIdentifyResult],
) -> None:
    try:
        future.result()
    except Exception:
        _LOGGER.debug("timed-out identify-current worker failed", exc_info=True)


def _identify_current_evidence(cached: CachedFrame) -> dict[str, Any]:
    evidence: dict[str, Any] = {"source_frame_ref": _source_frame_ref(cached)}
    snapshot = cached.memory_snapshot
    if snapshot is not None:
        evidence["source_frame_ref"] = snapshot.source_frame_ref
        evidence["request_snapshot_ref"] = snapshot.snapshot_ref
    return evidence


def _person_visual_evidence(
    *,
    source_frame_ref: str,
    target: ResolvedTarget,
    candidate: _PersonEmbeddingInputCandidate,
    crop_hash: str,
    crop_path_or_artifact_ref: str | None,
    embedding: EmbeddingResult | None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "source_frame_ref": source_frame_ref,
        "source_bbox_xyxy": [float(value) for value in target.bbox_xyxy],
        "source_bbox_coordinate_space": "source_frame",
        "crop_box_xyxy": [float(value) for value in candidate.crop_box_xyxy],
        "crop_box_coordinate_space": candidate.crop_box_coordinate_space,
        "embedding_crop_hash": crop_hash,
    }
    if crop_path_or_artifact_ref is not None:
        evidence["embedding_crop_path"] = crop_path_or_artifact_ref
    face_detection = _embedding_face_detection(embedding)
    if face_detection is not None:
        evidence["face_detection"] = face_detection
    return evidence


def _embedding_face_detection(
    embedding: EmbeddingResult | None,
) -> dict[str, Any] | None:
    if embedding is None or not isinstance(embedding.metadata, dict):
        return None
    raw = embedding.metadata.get("face_detection")
    if not isinstance(raw, dict):
        return None

    coordinate_space = raw.get("coordinate_space")
    bbox = _finite_float_list(raw.get("face_bbox_xyxy"), length=4)
    score = _finite_float(raw.get("score"))
    if not isinstance(coordinate_space, str) or not coordinate_space or bbox is None:
        return None
    result: dict[str, Any] = {
        "coordinate_space": coordinate_space,
        "face_bbox_xyxy": bbox,
    }
    landmarks = _finite_point_list(raw.get("landmarks_5"), length=5)
    if landmarks is not None:
        result["landmarks_5"] = landmarks
    if score is not None:
        result["score"] = score
    source = raw.get("source")
    if isinstance(source, str) and source:
        result["source"] = source
    return result


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite_float_list(value: Any, *, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    result: list[float] = []
    for item in value:
        number = _finite_float(item)
        if number is None:
            return None
        result.append(number)
    return result


def _finite_point_list(value: Any, *, length: int) -> list[list[float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    result: list[list[float]] = []
    for point in value:
        xy = _finite_float_list(point, length=2)
        if xy is None:
            return None
        result.append(xy)
    return result


def _target_request(request: dict[str, Any]) -> TargetRequest:
    target = _required_mapping(request, "target")
    mode = _required_text(target, "mode")
    if mode == "track_id":
        track_id = target.get("track_id")
        if not isinstance(track_id, int):
            raise MemoryServiceError("invalid_target_request", "track_id is required")
        return TargetRequest(mode=mode, track_id=track_id)
    if mode == "bbox":
        raw_bbox = target.get("bbox_xyxy")
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise MemoryServiceError("invalid_target_request", "bbox_xyxy is required")
        return TargetRequest(
            mode=mode,
            bbox_xyxy=tuple(float(value) for value in raw_bbox),  # type: ignore[arg-type]
        )
    if mode == "point_uv":
        raw_point = target.get("point_uv")
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            raise MemoryServiceError("invalid_target_request", "point_uv is required")
        return TargetRequest(
            mode=mode,
            point_uv=(float(raw_point[0]), float(raw_point[1])),
        )
    return TargetRequest(mode=mode)


def _public_person_target(request: dict[str, Any]) -> dict[str, Any] | None:
    target = _required_mapping(request, "target")
    if target.get("kind") != "person":
        return None
    return target


def _ambiguous_target_response(
    ambiguity_type: str,
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = {
        "ok": True,
        "status": "ambiguous",
        "retryable": True,
        "ask_user_hint": True,
        "ambiguity_type": ambiguity_type,
        "candidates": [],
    }
    if evidence is not None:
        response["evidence"] = evidence
    return response


def _target_ambiguous_error(
    ambiguity_type: str,
    *,
    evidence: dict[str, Any] | None = None,
) -> MemoryServiceError:
    return MemoryServiceError(
        "target_ambiguous",
        "target requires an active interaction target",
        status_code=409,
        details=_ambiguous_target_response(
            ambiguity_type,
            evidence=evidence,
        ),
    )


def _attach_store_delta_to_error(
    error: MemoryServiceError,
    store_delta: dict[str, dict[str, int]],
) -> None:
    if "store_delta" not in error.details:
        error.details["store_delta"] = store_delta


def _recognition_track_eligible(track: TrackSnapshot) -> bool:
    return _recognition_track_skip_reason(track) is None


def _recognition_track_skip_reason(track: TrackSnapshot) -> str | None:
    if track.class_name != "person":
        return "not_person"
    if track.lost_ms != 0:
        return "lost"
    if track.hits <= 0:
        return "no_hits"
    return None


def _visible_person_track(
    tracks: list[TrackSnapshot],
    track_id: int,
) -> TrackSnapshot | None:
    for track in tracks:
        if (
            track.track_id == track_id
            and track.lost_ms == 0
            and track.class_name == "person"
            and track.hits > 0
        ):
            return track
    return None


def _cached_visible_person_track(cached: CachedFrame, track_id: int) -> bool:
    return track_id in set(_visible_person_track_ids(cached.visual_state))


def _source_frame_ref(cached: CachedFrame) -> str:
    frame = cached.frame
    return f"{frame.camera}:{frame.frame_id}:{frame.timestamp_ms}"


def _source_track_ref(cached: CachedFrame, target: ResolvedTarget) -> str | None:
    if target.track_id is None:
        return None
    return f"{cached.frame.camera}:track:{target.track_id}"


def _resolver_target_ref(cached: CachedFrame, target: ResolvedTarget) -> str:
    if target.source_target_mode == "scene":
        return "scene"
    source_track_ref = _source_track_ref(cached, target)
    if source_track_ref is not None:
        return source_track_ref
    bbox = ",".join(_short_float(value) for value in target.bbox_xyxy)
    return f"{target.source_target_mode}:{target.target_type}:{bbox}"


def _candidate_source_track_ref(cached: CachedFrame, candidate: Any) -> str | None:
    if candidate.track_id is None:
        return None
    return f"{cached.frame.camera}:track:{candidate.track_id}"


def _candidate_resolver_target_ref(cached: CachedFrame, candidate: Any) -> str:
    if candidate.target_type == "scene":
        return "scene"
    source_track_ref = _candidate_source_track_ref(cached, candidate)
    if source_track_ref is not None:
        return source_track_ref
    bbox = ",".join(_short_float(value) for value in candidate.bbox_xyxy)
    return f"{candidate.reason}:{candidate.target_type}:{bbox}"


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _readable_artifact_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _short_float(value: float) -> str:
    text = f"{float(value):.3f}"
    return text.rstrip("0").rstrip(".")


def _tracks_from_visual_state(
    visual_state: dict[str, Any],
    *,
    frame: FrameMessage,
) -> list[TrackSnapshot]:
    tracks: list[TrackSnapshot] = []
    for raw in visual_state.get("tracks") or []:
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        track_id = raw.get("track_id")
        if not isinstance(track_id, int):
            continue
        age_ms = int(raw.get("age_ms") or 0)
        head = raw.get("head_uv") or [0.0, 0.0]
        velocity = raw.get("velocity_uv_s") or [0.0, 0.0]
        if not isinstance(head, (list, tuple)) or len(head) != 2:
            head = [0.0, 0.0]
        if not isinstance(velocity, (list, tuple)) or len(velocity) != 2:
            velocity = [0.0, 0.0]
        tracks.append(
            TrackSnapshot(
                track_id=track_id,
                first_seen_ms=frame.timestamp_ms - max(0, age_ms),
                last_seen_ms=frame.timestamp_ms - int(raw.get("lost_ms") or 0),
                frame_timestamp_ms=frame.timestamp_ms,
                bbox_xyxy=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
                confidence=float(raw.get("confidence") or 0.0),
                pose_confidence=float(raw.get("pose_confidence") or 0.0),
                head_uv=(float(head[0]), float(head[1])),
                velocity_uv_s=(float(velocity[0]), float(velocity[1])),
                lost_ms=int(raw.get("lost_ms") or 0),
                hits=1,
                misses=0,
                class_name=str(raw.get("class") or "person"),
            )
        )
    return tracks


def _attention_from_visual_state(visual_state: dict[str, Any]) -> AttentionResult | None:
    raw = visual_state.get("attention")
    if not isinstance(raw, dict):
        return None
    track_id = raw.get("target_track_id")
    target_uv = raw.get("target_uv")
    if not isinstance(track_id, int) or not isinstance(target_uv, (list, tuple)):
        return None
    if len(target_uv) != 2:
        return None
    return AttentionResult(
        target_track_id=track_id,
        target_uv=(float(target_uv[0]), float(target_uv[1])),
        reason=str(raw.get("reason") or "memory_query"),
        confidence=float(raw.get("confidence") or 0.0),
        largest_person_stable=True,
    )


def _target_bytes(jpeg_bytes: bytes, target: ResolvedTarget) -> bytes:
    if target.target_type == "scene":
        return jpeg_bytes
    if target.target_type == "person":
        return _person_embedding_bytes(jpeg_bytes, target)
    if target.target_type == "region":
        return _crop_target_jpeg(jpeg_bytes, target)
    raise MemoryServiceError(
        "invalid_target_type",
        f"unsupported memory target type {target.target_type}",
    )


def _person_embedding_bytes(jpeg_bytes: bytes, target: ResolvedTarget) -> bytes:
    if target.target_type not in {"person", "region"}:
        raise MemoryServiceError(
            "invalid_target_type",
            "person embedding target must resolve to a person or region",
        )
    return _crop_target_jpeg(
        jpeg_bytes,
        target,
        margin_ratio=_PERSON_EMBEDDING_CROP_MARGIN_RATIO,
    )


def _person_embedding_input_candidates(
    jpeg_bytes: bytes,
    target: ResolvedTarget,
    *,
    tracks: Iterable[TrackSnapshot] | None = None,
) -> Iterator[bytes]:
    for candidate in _person_embedding_input_candidate_records(
        jpeg_bytes,
        target,
        tracks=tracks,
    ):
        yield candidate.payload


def _person_embedding_input_candidate_records(
    jpeg_bytes: bytes,
    target: ResolvedTarget,
    *,
    tracks: Iterable[TrackSnapshot] | None = None,
) -> Iterator[_PersonEmbeddingInputCandidate]:
    yield _PersonEmbeddingInputCandidate(
        payload=_person_embedding_bytes(jpeg_bytes, target),
        crop_box_xyxy=_target_crop_box_for_jpeg(
            jpeg_bytes,
            target,
            margin_ratio=_PERSON_EMBEDDING_CROP_MARGIN_RATIO,
        ),
        crop_box_coordinate_space="source_frame",
    )
    if target.target_type != "person":
        return

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise MemoryServiceError(
            "image_crop_unavailable",
            "Pillow is required to crop memory target images",
            status_code=503,
        ) from exc

    try:
        with Image.open(BytesIO(jpeg_bytes)) as image:
            source = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MemoryServiceError(
            "invalid_frame_image",
            "cached frame JPEG could not be decoded",
        ) from exc

    left, top, right, bottom = _target_crop_box(
        target,
        image_width=source.width,
        image_height=source.height,
        margin_ratio=_PERSON_EMBEDDING_CONTEXT_MARGIN_RATIO,
    )
    masked = Image.new("RGB", source.size, (0, 0, 0))
    masked.paste(source.crop((left, top, right, bottom)), (left, top))
    if tracks is not None:
        for track in tracks:
            if (
                track.lost_ms != 0
                or track.class_name != "person"
                or track.track_id == target.track_id
            ):
                continue
            other = ResolvedTarget(
                source_target_mode="track_id",
                target_type="person",
                bbox_xyxy=track.bbox_xyxy,
                track_id=track.track_id,
                quality="usable",
            )
            try:
                other_left, other_top, other_right, other_bottom = _target_crop_box(
                    other,
                    image_width=source.width,
                    image_height=source.height,
                    margin_ratio=0.0,
                )
            except MemoryServiceError:
                continue
            masked.paste(
                (0, 0, 0),
                (other_left, other_top, other_right, other_bottom),
            )

    buffer = BytesIO()
    masked.save(buffer, format="JPEG", quality=95)
    yield _PersonEmbeddingInputCandidate(
        payload=buffer.getvalue(),
        crop_box_xyxy=(0.0, 0.0, float(source.width), float(source.height)),
        crop_box_coordinate_space="source_frame",
    )


def _target_crop_box_for_jpeg(
    jpeg_bytes: bytes,
    target: ResolvedTarget,
    *,
    margin_ratio: float,
) -> tuple[float, float, float, float]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise MemoryServiceError(
            "image_crop_unavailable",
            "Pillow is required to inspect memory target images",
            status_code=503,
        ) from exc

    try:
        with Image.open(BytesIO(jpeg_bytes)) as image:
            image_width, image_height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MemoryServiceError(
            "invalid_frame_image",
            "cached frame JPEG could not be decoded",
        ) from exc

    return tuple(
        float(value)
        for value in _target_crop_box(
            target,
            image_width=image_width,
            image_height=image_height,
            margin_ratio=margin_ratio,
        )
    )


def _crop_target_jpeg(
    jpeg_bytes: bytes,
    target: ResolvedTarget,
    *,
    margin_ratio: float = 0.0,
) -> bytes:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise MemoryServiceError(
            "image_crop_unavailable",
            "Pillow is required to crop memory target images",
            status_code=503,
        ) from exc

    try:
        with Image.open(BytesIO(jpeg_bytes)) as image:
            source = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MemoryServiceError(
            "invalid_frame_image",
            "cached frame JPEG could not be decoded",
        ) from exc

    left, top, right, bottom = _target_crop_box(
        target,
        image_width=source.width,
        image_height=source.height,
        margin_ratio=margin_ratio,
    )
    crop = source.crop((left, top, right, bottom))
    if crop.width <= 0 or crop.height <= 0:
        raise MemoryServiceError("invalid_target_bbox", "target crop is empty")

    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _target_crop_box(
    target: ResolvedTarget,
    *,
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    try:
        bbox = tuple(float(value) for value in target.bbox_xyxy)
    except (TypeError, ValueError) as exc:
        raise MemoryServiceError("invalid_target_bbox", "target bbox is invalid") from exc
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise MemoryServiceError("invalid_target_bbox", "target bbox is invalid")

    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        raise MemoryServiceError("invalid_target_bbox", "target bbox is empty")
    if margin_ratio > 0.0:
        margin_x = (x2 - x1) * margin_ratio
        margin_y = (y2 - y1) * margin_ratio
        x1 -= margin_x
        y1 -= margin_y
        x2 += margin_x
        y2 += margin_y

    left = math.floor(x1)
    top = math.floor(y1)
    right = math.ceil(x2)
    bottom = math.ceil(y2)
    if (
        margin_ratio == 0.0
        and (left < 0 or top < 0 or right > image_width or bottom > image_height)
    ):
        raise MemoryServiceError("invalid_target_bbox", "target bbox is outside image")

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)
    if left >= right or top >= bottom:
        raise MemoryServiceError("invalid_target_bbox", "target crop is empty")
    return left, top, right, bottom


def _candidate_to_dict(candidate: Any) -> dict[str, Any]:
    result = {
        "target_type": candidate.target_type,
        "track_id": candidate.track_id,
        "bbox_xyxy": [float(value) for value in candidate.bbox_xyxy],
        "confidence": float(candidate.confidence),
        "reason": candidate.reason,
    }
    evidence = getattr(candidate, "evidence", None)
    if evidence:
        result["evidence"] = evidence
    return result


def _with_pose_visual_service_evidence(
    evidence: dict[str, Any],
    *,
    introducer_ref: str | None,
    target_ref: str | None,
    pose_stability_window: dict[str, Any] | None,
    clear_target: bool = False,
) -> dict[str, Any]:
    visual = evidence.get("pose_visual_evidence")
    if not isinstance(visual, dict):
        return evidence

    updated = dict(visual)
    if introducer_ref is not None:
        updated["introducer_ref"] = introducer_ref
    if clear_target:
        for key in ("target_ref", "target_track_id", "target_bbox_xyxy"):
            updated.pop(key, None)
    elif target_ref is not None and (
        "target_track_id" in updated or "target_bbox_xyxy" in updated
    ):
        updated["target_ref"] = target_ref
    if pose_stability_window is not None:
        updated["pose_stability_window"] = pose_stability_window

    result = dict(evidence)
    result["pose_visual_evidence"] = updated
    return result


def _pose_preview_arm_side(evidence: dict[str, Any] | None) -> str | None:
    if not isinstance(evidence, dict):
        return None
    scoring = evidence.get("pose_pointing_scoring")
    if not isinstance(scoring, dict):
        return None
    arm_side = scoring.get("arm_side")
    if not isinstance(arm_side, str) or not arm_side:
        return None
    return arm_side


def _resolved_target_to_candidate_dict(
    target: ResolvedTarget,
    *,
    reason: str,
) -> dict[str, Any]:
    result = {
        "target_type": target.target_type,
        "track_id": target.track_id,
        "bbox_xyxy": [float(value) for value in target.bbox_xyxy],
        "confidence": 1.0 if target.quality == "usable" else 0.0,
        "reason": reason,
    }
    if target.evidence:
        result["evidence"] = target.evidence
    return result


def _updated_observed_duration_ms(
    profile: dict[str, Any],
    *,
    current_frame_timestamp_ms: int,
    query_interval_ms: int,
) -> int:
    previous_last_seen_at_ms = int(profile["last_seen_at_ms"])
    previous_observed_duration_ms = int(profile.get("observed_duration_ms", 0))
    delta_ms = max(0, int(current_frame_timestamp_ms) - previous_last_seen_at_ms)
    increment_ms = min(delta_ms, int(query_interval_ms))
    return previous_observed_duration_ms + increment_ms


def _stream_key(connection_id: str, camera: str) -> _StreamKey:
    return (connection_id, camera)


def _event_gate_scope(cached: CachedFrame) -> str:
    return f"{cached.connection_id}:{cached.frame.camera}"


def _required_mapping(request: dict[str, Any], key: str) -> dict[str, Any]:
    value = request.get(key)
    if not isinstance(value, dict):
        raise MemoryServiceError("invalid_memory_request", f"{key} must be an object")
    return value


def _required_text(request: dict[str, Any], key: str) -> str:
    value = _optional_text(request.get(key))
    if not value:
        raise MemoryServiceError("invalid_memory_request", f"{key} is required")
    return value


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _public_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _system_time_ms() -> int:
    return int(time.time() * 1000)
