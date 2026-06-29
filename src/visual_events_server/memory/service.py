from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from visual_events_server.attention import AttentionResult
from visual_events_server.protocol import FrameMessage
from visual_events_server.tracking import TrackSnapshot

from .embedding import EmbeddingUnavailable, MemoryEmbeddingBackend
from .events import (
    MemoryEventGate,
    MemoryMatch,
    SourceFrameRef,
    build_familiar_unknown_event,
    build_known_person_event,
    build_scene_event,
)
from .frame_cache import CachedFrame, FrameCache, FrameCacheError, MemoryFrameSnapshot
from .retriever import MemoryRetriever
from .store import MemoryStore
from .target_resolver import (
    ResolvedTarget,
    TargetRequest,
    TargetResolveError,
    TargetResolver,
)


_LOGGER = logging.getLogger(__name__)

_PERSON_EMBEDDING_CROP_MARGIN_RATIO = 0.10


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
    tracks: list[TrackSnapshot]
    attention: AttentionResult | None


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
        familiar_threshold: float,
        scene_threshold: float,
        event_cooldown_ms: int,
        clock_ms: Callable[[], int] | None = None,
        target_resolver: TargetResolver | None = None,
    ) -> None:
        if query_interval_ms <= 0:
            raise ValueError("query_interval_ms must be positive")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self.store = store
        self.embedding_backend = embedding_backend
        self.query_interval_ms = int(query_interval_ms)
        self.known_person_threshold = float(known_person_threshold)
        self.known_person_margin = float(known_person_margin)
        self.anonymous_threshold = float(anonymous_threshold)
        self.anonymous_margin = float(anonymous_margin)
        self.familiar_seen_count = int(familiar_seen_count)
        self.familiar_threshold = float(familiar_threshold)
        self.scene_threshold = float(scene_threshold)
        self._clock_ms = clock_ms or _system_time_ms
        self._cache = FrameCache(
            max_age_ms=int(frame_cache_seconds) * 1000,
            clock_ms=self._clock_ms,
        )
        self._resolver = target_resolver or TargetResolver()
        self._retriever = MemoryRetriever(store)
        self._gate = MemoryEventGate(cooldown_ms=event_cooldown_ms)
        self._completed_by_camera: dict[str, deque[dict[str, Any]]] = {}
        self._queue_size = int(queue_size)
        self._last_query_frame_timestamp_ms: dict[str, int] = {}
        self._event_counters: dict[str, int] = {}
        self._pending_queries_by_camera: dict[
            str,
            asyncio.Future[list[tuple[str, dict[str, Any]]]],
        ] = {}
        self._query_state_lock = threading.Lock()

    async def observe_visual_state(
        self,
        *,
        connection_id: str,
        frame: FrameMessage,
        visual_state: dict[str, Any],
        memory_snapshot: MemoryFrameSnapshot | None = None,
    ) -> None:
        self._cache.update(
            connection_id=connection_id,
            frame=frame,
            visual_state=visual_state,
            memory_snapshot=memory_snapshot,
        )
        if not self._query_due(frame):
            return
        pending = self._pending_queries_by_camera.get(frame.camera)
        if pending is not None:
            if pending.done():
                self._collect_completed_query(frame.camera, pending)
            else:
                return
        self._last_query_frame_timestamp_ms[frame.camera] = frame.timestamp_ms
        plan = _QueryPlan(
            cached=self._cache.get_fresh(frame.camera),
            tracks=_tracks_from_visual_state(visual_state, frame=frame),
            attention=_attention_from_visual_state(visual_state),
        )
        loop = asyncio.get_running_loop()
        self._pending_queries_by_camera[frame.camera] = loop.run_in_executor(
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
        pending = self._pending_queries_by_camera.get(camera)
        if pending is not None and pending.done():
            self._collect_completed_query(camera, pending)
        queue = self._completed_by_camera.get(camera)
        if not queue:
            return []
        events = list(queue)
        queue.clear()
        return events

    async def teach_person(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            cached = self._fresh_cached_frame(request)
        except MemoryServiceError as exc:
            if _public_person_target(request) is not None and exc.code == "frame_cache_expired":
                raise _target_ambiguous_error("stale_interaction") from exc
            raise
        target = self._resolve_person_teach_target(cached, request)
        if target.target_type not in {"person", "region"}:
            raise MemoryServiceError(
                "invalid_target_type",
                "teach_person target must resolve to a person or region",
            )
        profile = _required_mapping(request, "profile")
        display_name = _required_text(profile, "display_name")
        description = _optional_text(profile.get("description"))
        tags = tuple(str(tag) for tag in profile.get("tags", []) if str(tag))
        embedding_bytes = _person_embedding_bytes(cached.frame.jpeg_bytes, target)
        try:
            embedding = self.embedding_backend.embed_person(embedding_bytes)
        except EmbeddingUnavailable as exc:
            raise MemoryServiceError(exc.code, exc.message, status_code=503) from exc

        now_ms = self._clock_ms()
        person_id = _public_id("person")
        created = self.store.create_person_with_embedding(
            person_id=person_id,
            display_name=display_name,
            description=description,
            tags=tags,
            embedding=embedding,
            source_target_type=target.source_target_mode,
            source_track_ref=_source_track_ref(cached, target),
            source_frame_ref=_source_frame_ref(cached),
            crop_hash=_sha256_hex(embedding_bytes),
            crop_path_or_artifact_ref=None,
            resolver_target_ref=_resolver_target_ref(cached, target),
            resolution_reason=target.source_target_mode,
            now_ms=now_ms,
        )
        return {
            "ok": True,
            "person_id": created["person_id"],
            "embedding_id": created["embedding_id"],
            "embedding_count": 1,
            "target_quality": target.quality,
        }

    async def teach_scene(self, request: dict[str, Any]) -> dict[str, Any]:
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
        try:
            embedding = self.embedding_backend.embed_scene(embedding_bytes)
        except EmbeddingUnavailable as exc:
            raise MemoryServiceError(exc.code, exc.message, status_code=503) from exc

        now_ms = self._clock_ms()
        scene_id = _public_id("scene")
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
            crop_hash=_sha256_hex(embedding_bytes),
            crop_path_or_artifact_ref=None,
            resolver_target_ref=_resolver_target_ref(cached, target),
            resolution_reason=target.source_target_mode,
            now_ms=now_ms,
        )
        return {
            "ok": True,
            "scene_id": created["scene_id"],
            "embedding_id": created["embedding_id"],
            "embedding_count": 1,
            "target_quality": target.quality,
        }

    async def resolve_target(self, request: dict[str, Any]) -> dict[str, Any]:
        public_person_target = _public_person_target(request)
        try:
            cached = self._fresh_cached_frame(request)
        except MemoryServiceError as exc:
            if public_person_target is not None and exc.code == "frame_cache_expired":
                return _ambiguous_target_response("stale_interaction")
            raise
        if public_person_target is not None:
            return self._preview_public_person_target(cached, public_person_target)
        preview = self._preview_cached_target(cached, _target_request(request))
        return {
            "ok": True,
            "status": preview.status,
            "candidates": [_candidate_to_dict(candidate) for candidate in preview.candidates],
        }

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
            if self.store.get_person_profile(person_id) is None:
                raise MemoryServiceError(
                    "person_not_found",
                    "person profile not found",
                    status_code=404,
                )
        else:
            if not isinstance(profile, dict):
                raise MemoryServiceError(
                    "invalid_memory_request",
                    "profile is required when person_id is absent",
                )
            person_id = _public_id("person")
            self.store.upsert_person_profile(
                person_id=person_id,
                display_name=_required_text(profile, "display_name"),
                description=_optional_text(profile.get("description")),
                tags=tuple(str(tag) for tag in profile.get("tags", []) if str(tag)),
                now_ms=now_ms,
            )

        copied_ids = self.store.copy_anonymous_embeddings_to_person(
            anonymous_id=anonymous_id,
            person_id=person_id,
            now_ms=now_ms,
        )
        self.store.mark_anonymous_profile_merged(
            anonymous_id=anonymous_id,
            person_id=person_id,
            now_ms=now_ms,
        )
        merge_id = self.store.add_profile_merge_history(
            anonymous_id=anonymous_id,
            person_id=person_id,
            merge_reason=merge_reason,
            now_ms=now_ms,
        )
        return {
            "ok": True,
            "anonymous_id": anonymous_id,
            "person_id": person_id,
            "copied_embedding_count": len(copied_ids),
            "merge_id": merge_id,
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
        try:
            return self._cache.get_fresh(camera)
        except FrameCacheError as exc:
            raise MemoryServiceError(exc.code, exc.message, status_code=409) from exc

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
    ) -> ResolvedTarget:
        public_target = _public_person_target(request)
        if public_target is None:
            return self._resolve_cached_target(cached, _target_request(request))

        intent = _required_text(public_target, "intent")
        if intent == "self_introduction":
            target, ambiguity_type = self._active_interaction_target(cached)
            if target is not None:
                return target
            raise _target_ambiguous_error(ambiguity_type)
        if intent == "third_person_introduction":
            raise _target_ambiguous_error("introducer_unclear")
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
            resolved, ambiguity_type = self._active_interaction_target(cached)
            if resolved is None:
                return _ambiguous_target_response(ambiguity_type)
            return {
                "ok": True,
                "status": "resolved",
                "candidates": [
                    _resolved_target_to_candidate_dict(
                        resolved,
                        reason="active_interaction_target",
                    )
                ],
            }
        if intent == "third_person_introduction":
            return _ambiguous_target_response("introducer_unclear")
        return {
            **_ambiguous_target_response("target_unclear"),
            "error_code": "unsupported_person_intent",
        }

    def _active_interaction_target(
        self,
        cached: CachedFrame,
    ) -> tuple[ResolvedTarget | None, str]:
        snapshot = cached.memory_snapshot
        if snapshot is None:
            return None, "no_active_interaction_target"
        if self._clock_ms() - snapshot.observed_at_ms > self._cache.max_age_ms:
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

    def _query_due(self, frame: FrameMessage) -> bool:
        last = self._last_query_frame_timestamp_ms.get(frame.camera)
        return last is None or frame.timestamp_ms - last >= self.query_interval_ms

    def _collect_completed_query(
        self,
        camera: str,
        future: asyncio.Future[list[tuple[str, dict[str, Any]]]],
    ) -> None:
        try:
            completed = future.result()
        except Exception:
            _LOGGER.exception(
                "memory query failed for camera %s",
                camera,
            )
            completed = []
        for event_camera, event in completed:
            self._enqueue(event_camera, event)
        if self._pending_queries_by_camera.get(camera) is future:
            self._pending_queries_by_camera.pop(camera, None)

    def _run_query(self, plan: _QueryPlan) -> list[tuple[str, dict[str, Any]]]:
        completed: list[tuple[str, dict[str, Any]]] = []
        person_target = self._query_person_target(plan)
        if person_target is not None:
            person_event = self._query_person(plan, person_target)
            if person_event is not None:
                completed.append((plan.cached.frame.camera, person_event))
        scene_event = self._query_scene(plan)
        if scene_event is not None:
            completed.append((plan.cached.frame.camera, scene_event))
        return completed

    def _query_person_target(
        self,
        plan: _QueryPlan,
    ) -> ResolvedTarget | None:
        if plan.attention is None:
            return None
        try:
            return self._resolver.resolve(
                TargetRequest(mode="attention_target"),
                image_width=plan.cached.frame.width,
                image_height=plan.cached.frame.height,
                tracks=plan.tracks,
                attention=plan.attention,
            )
        except TargetResolveError:
            return None

    def _query_person(
        self,
        plan: _QueryPlan,
        target: ResolvedTarget,
    ) -> dict[str, Any] | None:
        try:
            embedding = self.embedding_backend.embed_person(
                _person_embedding_bytes(plan.cached.frame.jpeg_bytes, target)
            )
        except EmbeddingUnavailable:
            return None
        match = self._retriever.query_person(
            embedding,
            threshold=self.known_person_threshold,
            margin=self.known_person_margin,
        )
        if match is None:
            return self._query_anonymous_person(plan, target, embedding)
        profile = self.store.get_person_profile(match.matched_id)
        if profile is None:
            return None
        with self._query_state_lock:
            if not self._gate.allow(
                plan.cached.frame.camera,
                "known_person_present",
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
        return build_known_person_event(
            event_id=event_id,
            match=match,
            source=source,
            person_profile=profile,
            conversation_summaries=tuple(
                self.store.get_conversation_summaries(match.matched_id, limit=3)
            ),
        )

    def _query_anonymous_person(
        self,
        plan: _QueryPlan,
        target: ResolvedTarget,
        embedding,
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
            )
            self.store.add_anonymous_embedding(
                anonymous_id=anonymous_id,
                result=embedding,
                source_target_type=target.source_target_mode,
                now_ms=now_ms,
            )
            return None

        profile = self.store.get_active_anonymous_profile(match.matched_id)
        if profile is None:
            return None
        seen_count = int(profile["seen_count"]) + 1
        familiar_score = max(float(profile["familiar_score"]), match.match_score)
        self.store.update_anonymous_profile(
            anonymous_id=match.matched_id,
            seen_count=seen_count,
            last_seen_at_ms=plan.cached.frame.timestamp_ms,
            familiar_score=familiar_score,
        )
        profile = {
            **profile,
            "seen_count": seen_count,
            "last_seen_at_ms": plan.cached.frame.timestamp_ms,
            "familiar_score": familiar_score,
        }
        if seen_count < self.familiar_seen_count or familiar_score < self.familiar_threshold:
            return None
        with self._query_state_lock:
            if not self._gate.allow(
                plan.cached.frame.camera,
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
                plan.cached.frame.camera,
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

    def _enqueue(self, camera: str, event: dict[str, Any]) -> None:
        queue = self._completed_by_camera.setdefault(
            camera,
            deque(maxlen=self._queue_size),
        )
        queue.append(event)


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


def _ambiguous_target_response(ambiguity_type: str) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "ambiguous",
        "retryable": True,
        "ask_user_hint": True,
        "ambiguity_type": ambiguity_type,
        "candidates": [],
    }


def _target_ambiguous_error(ambiguity_type: str) -> MemoryServiceError:
    return MemoryServiceError(
        "target_ambiguous",
        "target requires an active interaction target",
        status_code=409,
        details=_ambiguous_target_response(ambiguity_type),
    )


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


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    return {
        "target_type": candidate.target_type,
        "track_id": candidate.track_id,
        "bbox_xyxy": [float(value) for value in candidate.bbox_xyxy],
        "confidence": float(candidate.confidence),
        "reason": candidate.reason,
    }


def _resolved_target_to_candidate_dict(
    target: ResolvedTarget,
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "target_type": target.target_type,
        "track_id": target.track_id,
        "bbox_xyxy": [float(value) for value in target.bbox_xyxy],
        "confidence": 1.0 if target.quality == "usable" else 0.0,
        "reason": reason,
    }


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
