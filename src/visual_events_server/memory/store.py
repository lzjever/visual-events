from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embedding import EmbeddingResult, normalize_vector

try:
    import sqlite_vec
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests.
    sqlite_vec = None  # type: ignore[assignment]


class MemoryStoreError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})


@dataclass(frozen=True)
class VectorCandidate:
    matched_type: str
    matched_id: str
    embedding_id: str
    match_type: str
    embedding_model: str
    embedding_version: str
    distance: float


_MEMORY_COUNT_TABLES = (
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
)


class MemoryStore:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        person_dim: int,
        scene_dim: int,
    ) -> None:
        self.connection = connection
        self.person_dim = person_dim
        self.scene_dim = scene_dim
        self._lock = threading.RLock()

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        person_dim: int,
        scene_dim: int,
    ) -> MemoryStore:
        if sqlite_vec is None:
            raise MemoryStoreError(
                "sqlite_vec_unavailable",
                "sqlite-vec is required for memory vector search",
            )
        if person_dim <= 0 or scene_dim <= 0:
            raise ValueError("person_dim and scene_dim must be positive")
        db_path = Path(path)
        if db_path.parent and str(db_path.parent) != ".":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.enable_load_extension(True)
        try:
            sqlite_vec.load(connection)
        finally:
            connection.enable_load_extension(False)
        store = cls(connection, person_dim=person_dim, scene_dim=scene_dim)
        store._initialize_schema()
        return store

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def memory_table_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._lock:
            for table in _MEMORY_COUNT_TABLES:
                row = self.connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table}",
                ).fetchone()
                counts[table] = int(row["count"])
        return counts

    def upsert_person_profile(
        self,
        *,
        person_id: str,
        display_name: str,
        description: str,
        tags: tuple[str, ...],
        now_ms: int,
    ) -> None:
        with self._lock:
            with self.connection:
                self._upsert_person_profile(
                    person_id=person_id,
                    display_name=display_name,
                    description=description,
                    tags=tags,
                    now_ms=now_ms,
                )

    def get_person_profile(self, person_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT person_id, display_name, description, tags_json
                FROM person_profiles
                WHERE person_id = ?
                """,
                (person_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "person_id": row["person_id"],
            "display_name": row["display_name"],
            "description": row["description"],
            "tags": json.loads(row["tags_json"]),
        }

    def create_person_with_embedding(
        self,
        *,
        person_id: str,
        display_name: str,
        description: str,
        tags: tuple[str, ...],
        embedding: EmbeddingResult,
        source_target_type: str,
        source_track_ref: str | None,
        source_frame_ref: str,
        crop_hash: str,
        crop_path_or_artifact_ref: str | None,
        resolver_target_ref: str,
        resolution_reason: str,
        now_ms: int,
        external_user_ref: str | None = None,
    ) -> dict[str, str]:
        vector = self._checked_vector(embedding.vector, expected_dim=self.person_dim)
        embedding_id = _new_id("emb_person")
        vector_blob = _serialize_vector(vector)
        with self._lock:
            with self.connection:
                self._upsert_person_profile(
                    person_id=person_id,
                    display_name=display_name,
                    description=description,
                    tags=tags,
                    now_ms=now_ms,
                )
                if external_user_ref:
                    self._link_external_user_if_available(
                        person_id=person_id,
                        external_user_ref=external_user_ref,
                        now_ms=now_ms,
                    )
                self._insert_person_embedding_rows(
                    embedding_id=embedding_id,
                    person_id=person_id,
                    result=embedding,
                    source_target_type=source_target_type,
                    vector_blob=vector_blob,
                    now_ms=now_ms,
                )
                self._insert_embedding_provenance(
                    embedding_id=embedding_id,
                    owner_type="person",
                    owner_id=person_id,
                    source_track_ref=source_track_ref,
                    source_frame_ref=source_frame_ref,
                    crop_hash=crop_hash,
                    crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                    resolver_target_ref=resolver_target_ref,
                    resolution_reason=resolution_reason,
                    embedding_type=embedding.embedding_type,
                    embedding_model=embedding.embedding_model,
                    embedding_version=embedding.embedding_version,
                    embedding_dim=self.person_dim,
                    now_ms=now_ms,
                )
        return {"person_id": person_id, "embedding_id": embedding_id}

    def promote_anonymous_to_person(
        self,
        *,
        anonymous_id: str,
        person_id: str,
        display_name: str,
        description: str,
        tags: tuple[str, ...],
        now_ms: int,
        merge_reason: str,
        external_user_ref: str | None = None,
        fresh_embedding: EmbeddingResult | None = None,
        source_target_type: str | None = None,
        source_track_ref: str | None = None,
        source_frame_ref: str | None = None,
        crop_hash: str | None = None,
        crop_path_or_artifact_ref: str | None = None,
        resolver_target_ref: str | None = None,
        resolution_reason: str | None = None,
    ) -> dict[str, Any]:
        fresh_embedding_id: str | None = None
        fresh_vector_blob: bytes | None = None
        if fresh_embedding is not None:
            vector = self._checked_vector(
                fresh_embedding.vector,
                expected_dim=self.person_dim,
            )
            fresh_vector_blob = _serialize_vector(vector)
            fresh_embedding_id = _new_id("emb_person")
            if not all(
                value is not None
                for value in (
                    source_target_type,
                    source_frame_ref,
                    crop_hash,
                    resolver_target_ref,
                    resolution_reason,
                )
            ):
                raise MemoryStoreError(
                    "invalid_embedding_provenance",
                    "fresh person embedding provenance is incomplete",
                )
        merge_id = _new_id("merge")
        with self._lock:
            with self.connection:
                active_anonymous = self.connection.execute(
                    """
                    SELECT anonymous_id
                    FROM anonymous_profiles
                    WHERE anonymous_id = ? AND status = 'active'
                    """,
                    (anonymous_id,),
                ).fetchone()
                if active_anonymous is None:
                    raise MemoryStoreError(
                        "anonymous_not_found",
                        "active anonymous profile not found",
                    )
                self._upsert_person_profile(
                    person_id=person_id,
                    display_name=display_name,
                    description=description,
                    tags=tags,
                    now_ms=now_ms,
                )
                if external_user_ref:
                    self._link_external_user_if_available(
                        person_id=person_id,
                        external_user_ref=external_user_ref,
                        now_ms=now_ms,
                    )
                if fresh_embedding is not None:
                    assert fresh_embedding_id is not None
                    assert fresh_vector_blob is not None
                    assert source_target_type is not None
                    assert source_frame_ref is not None
                    assert crop_hash is not None
                    assert resolver_target_ref is not None
                    assert resolution_reason is not None
                    self._insert_person_embedding_rows(
                        embedding_id=fresh_embedding_id,
                        person_id=person_id,
                        result=fresh_embedding,
                        source_target_type=source_target_type,
                        vector_blob=fresh_vector_blob,
                        now_ms=now_ms,
                    )
                    self._insert_embedding_provenance(
                        embedding_id=fresh_embedding_id,
                        owner_type="person",
                        owner_id=person_id,
                        source_track_ref=source_track_ref,
                        source_frame_ref=source_frame_ref,
                        crop_hash=crop_hash,
                        crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                        resolver_target_ref=resolver_target_ref,
                        resolution_reason=resolution_reason,
                        embedding_type=fresh_embedding.embedding_type,
                        embedding_model=fresh_embedding.embedding_model,
                        embedding_version=fresh_embedding.embedding_version,
                        embedding_dim=self.person_dim,
                        now_ms=now_ms,
                    )
                copied_embedding_ids = self._copy_anonymous_embeddings_to_person(
                    anonymous_id=anonymous_id,
                    person_id=person_id,
                    now_ms=now_ms,
                )
                self.connection.execute(
                    """
                    UPDATE anonymous_profiles
                    SET status = 'merged',
                        merged_person_id = ?,
                        updated_at_ms = ?
                    WHERE anonymous_id = ? AND status = 'active'
                    """,
                    (person_id, now_ms, anonymous_id),
                )
                self._insert_profile_merge_history(
                    merge_id=merge_id,
                    anonymous_id=anonymous_id,
                    person_id=person_id,
                    merge_reason=merge_reason,
                    now_ms=now_ms,
                )
        return {
            "person_id": person_id,
            "embedding_id": fresh_embedding_id,
            "copied_embedding_ids": copied_embedding_ids,
            "merge_id": merge_id,
        }

    def create_scene_memory(
        self,
        *,
        scene_id: str,
        title: str,
        description: str,
        activation_hint: str,
        target_type: str,
        now_ms: int,
        region_id: str | None = None,
    ) -> None:
        with self._lock:
            with self.connection:
                self._upsert_scene_memory(
                    scene_id=scene_id,
                    title=title,
                    description=description,
                    activation_hint=activation_hint,
                    target_type=target_type,
                    region_id=region_id,
                    now_ms=now_ms,
                )

    def get_scene_memory(self, scene_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT scene_id, title, description, activation_hint, target_type, region_id
                FROM scene_memories
                WHERE scene_id = ?
                """,
                (scene_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "scene_id": row["scene_id"],
            "title": row["title"],
            "description": row["description"],
            "activation_hint": row["activation_hint"],
            "target_type": row["target_type"],
            "region_id": row["region_id"],
        }

    def create_scene_with_embedding(
        self,
        *,
        scene_id: str,
        title: str,
        description: str,
        activation_hint: str,
        target_type: str,
        region_id: str | None,
        embedding: EmbeddingResult,
        source_target_type: str,
        source_track_ref: str | None,
        source_frame_ref: str,
        crop_hash: str,
        crop_path_or_artifact_ref: str | None,
        resolver_target_ref: str,
        resolution_reason: str,
        now_ms: int,
    ) -> dict[str, str]:
        vector = self._checked_vector(embedding.vector, expected_dim=self.scene_dim)
        embedding_id = _new_id("emb_scene")
        vector_blob = _serialize_vector(vector)
        with self._lock:
            with self.connection:
                self._upsert_scene_memory(
                    scene_id=scene_id,
                    title=title,
                    description=description,
                    activation_hint=activation_hint,
                    target_type=target_type,
                    region_id=region_id,
                    now_ms=now_ms,
                )
                self._insert_scene_embedding_rows(
                    embedding_id=embedding_id,
                    scene_id=scene_id,
                    result=embedding,
                    source_target_type=source_target_type,
                    vector_blob=vector_blob,
                    now_ms=now_ms,
                )
                self._insert_embedding_provenance(
                    embedding_id=embedding_id,
                    owner_type="scene",
                    owner_id=scene_id,
                    source_track_ref=source_track_ref,
                    source_frame_ref=source_frame_ref,
                    crop_hash=crop_hash,
                    crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                    resolver_target_ref=resolver_target_ref,
                    resolution_reason=resolution_reason,
                    embedding_type=embedding.embedding_type,
                    embedding_model=embedding.embedding_model,
                    embedding_version=embedding.embedding_version,
                    embedding_dim=self.scene_dim,
                    now_ms=now_ms,
                )
        return {"scene_id": scene_id, "embedding_id": embedding_id}

    def create_anonymous_profile(
        self,
        *,
        anonymous_id: str,
        seen_count: int,
        first_seen_at_ms: int,
        last_seen_at_ms: int,
        familiar_score: float,
        observed_duration_ms: int = 0,
        status: str = "active",
        merged_person_id: str | None = None,
    ) -> None:
        with self._lock:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO anonymous_profiles (
                      anonymous_id, seen_count, first_seen_at_ms, last_seen_at_ms,
                      familiar_score, observed_duration_ms, status, merged_person_id,
                      created_at_ms, updated_at_ms
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        anonymous_id,
                        int(seen_count),
                        int(first_seen_at_ms),
                        int(last_seen_at_ms),
                        float(familiar_score),
                        int(observed_duration_ms),
                        status,
                        merged_person_id,
                        int(first_seen_at_ms),
                        int(last_seen_at_ms),
                    ),
                )

    def update_anonymous_profile(
        self,
        *,
        anonymous_id: str,
        seen_count: int | None = None,
        last_seen_at_ms: int | None = None,
        familiar_score: float | None = None,
        observed_duration_ms: int | None = None,
        status: str | None = None,
        merged_person_id: str | None = None,
    ) -> None:
        assignments: list[str] = []
        values: list[Any] = []
        if seen_count is not None:
            assignments.append("seen_count = ?")
            values.append(int(seen_count))
        if last_seen_at_ms is not None:
            assignments.append("last_seen_at_ms = ?")
            values.append(int(last_seen_at_ms))
            assignments.append("updated_at_ms = ?")
            values.append(int(last_seen_at_ms))
        if familiar_score is not None:
            assignments.append("familiar_score = ?")
            values.append(float(familiar_score))
        if observed_duration_ms is not None:
            assignments.append("observed_duration_ms = ?")
            values.append(int(observed_duration_ms))
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if merged_person_id is not None:
            assignments.append("merged_person_id = ?")
            values.append(merged_person_id)
        if not assignments:
            return
        values.append(anonymous_id)
        with self._lock:
            with self.connection:
                self.connection.execute(
                    f"""
                    UPDATE anonymous_profiles
                    SET {", ".join(assignments)}
                    WHERE anonymous_id = ?
                    """,
                    tuple(values),
                )

    def get_active_anonymous_profile(self, anonymous_id: str) -> dict[str, Any] | None:
        return self._get_anonymous_profile(anonymous_id, active_only=True)

    def _get_anonymous_profile(
        self,
        anonymous_id: str,
        *,
        active_only: bool,
    ) -> dict[str, Any] | None:
        where = "anonymous_id = ?"
        if active_only:
            where += " AND status = 'active'"
        with self._lock:
            row = self.connection.execute(
                f"""
                SELECT
                  anonymous_id, seen_count, first_seen_at_ms, last_seen_at_ms,
                  familiar_score, observed_duration_ms, status, merged_person_id
                FROM anonymous_profiles
                WHERE {where}
                """,
                (anonymous_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "anonymous_id": row["anonymous_id"],
            "seen_count": int(row["seen_count"]),
            "first_seen_at_ms": int(row["first_seen_at_ms"]),
            "last_seen_at_ms": int(row["last_seen_at_ms"]),
            "familiar_score": float(row["familiar_score"]),
            "observed_duration_ms": int(row["observed_duration_ms"]),
            "status": row["status"],
            "merged_person_id": row["merged_person_id"],
        }

    def add_conversation_summary(
        self,
        *,
        person_id: str,
        summary: str,
        source: str,
        source_conversation_id: str | None,
        now_ms: int,
    ) -> str:
        summary_id = _new_id("summary")
        short_summary = _short_text(summary)
        with self._lock:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO conversation_summaries (
                      summary_id, person_id, summary, source, source_conversation_id,
                      created_at_ms
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        summary_id,
                        person_id,
                        short_summary,
                        source,
                        source_conversation_id,
                        now_ms,
                    ),
                )
        return summary_id

    def get_conversation_summaries(
        self,
        person_id: str,
        *,
        limit: int,
    ) -> list[str]:
        if limit <= 0:
            return []
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT summary
                FROM conversation_summaries
                WHERE person_id = ?
                ORDER BY created_at_ms DESC, summary_id DESC
                LIMIT ?
                """,
                (person_id, limit),
            ).fetchall()
        return [row["summary"] for row in rows]

    def link_external_user(
        self,
        *,
        person_id: str,
        external_user_ref: str,
        now_ms: int,
    ) -> None:
        with self._lock:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO external_user_links (
                      external_user_ref, person_id, created_at_ms
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(external_user_ref) DO UPDATE SET
                      person_id = excluded.person_id,
                      created_at_ms = excluded.created_at_ms
                    """,
                    (external_user_ref, person_id, now_ms),
                )

    def get_person_by_external_user(
        self,
        external_user_ref: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT p.person_id, p.display_name, p.description, p.tags_json
                FROM external_user_links AS l
                JOIN person_profiles AS p ON p.person_id = l.person_id
                WHERE l.external_user_ref = ?
                """,
                (external_user_ref,),
            ).fetchone()
        if row is None:
            return None
        return {
            "person_id": row["person_id"],
            "display_name": row["display_name"],
            "description": row["description"],
            "tags": json.loads(row["tags_json"]),
        }

    def add_memory_match_record(
        self,
        *,
        event_id: str,
        matched_type: str,
        matched_id: str,
        embedding_id: str,
        match_score: float,
        top2_margin: float,
        source_target_mode: str,
        camera: str,
        frame_id: int,
        frame_timestamp_ms: int,
        now_ms: int,
        memory_match_id: str | None = None,
    ) -> str:
        match_id = memory_match_id or _new_id("match")
        with self._lock:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO memory_match_records (
                      memory_match_id, event_id, matched_type, matched_id, embedding_id,
                      match_score, top2_margin, source_target_mode, camera, frame_id,
                      frame_timestamp_ms, created_at_ms
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match_id,
                        event_id,
                        matched_type,
                        matched_id,
                        embedding_id,
                        float(match_score),
                        float(top2_margin),
                        source_target_mode,
                        camera,
                        int(frame_id),
                        int(frame_timestamp_ms),
                        now_ms,
                    ),
                )
        return match_id

    def get_memory_match_record(
        self,
        memory_match_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT
                  memory_match_id, event_id, matched_type, matched_id, embedding_id,
                  match_score, top2_margin, source_target_mode, camera, frame_id,
                  frame_timestamp_ms
                FROM memory_match_records
                WHERE memory_match_id = ?
                """,
                (memory_match_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "memory_match_id": row["memory_match_id"],
            "event_id": row["event_id"],
            "matched_type": row["matched_type"],
            "matched_id": row["matched_id"],
            "embedding_id": row["embedding_id"],
            "match_score": float(row["match_score"]),
            "top2_margin": float(row["top2_margin"]),
            "source_target_mode": row["source_target_mode"],
            "camera": row["camera"],
            "frame_id": int(row["frame_id"]),
            "frame_timestamp_ms": int(row["frame_timestamp_ms"]),
        }

    def add_person_embedding(
        self,
        *,
        person_id: str,
        result: EmbeddingResult,
        source_target_type: str,
        now_ms: int,
    ) -> str:
        vector = self._checked_vector(result.vector, expected_dim=self.person_dim)
        embedding_id = _new_id("emb_person")
        vector_blob = _serialize_vector(vector)
        with self._lock:
            with self.connection:
                self._insert_person_embedding_rows(
                    embedding_id=embedding_id,
                    person_id=person_id,
                    result=result,
                    source_target_type=source_target_type,
                    vector_blob=vector_blob,
                    now_ms=now_ms,
                )
        return embedding_id

    def add_scene_embedding(
        self,
        *,
        scene_id: str,
        result: EmbeddingResult,
        source_target_type: str,
        now_ms: int,
    ) -> str:
        vector = self._checked_vector(result.vector, expected_dim=self.scene_dim)
        embedding_id = _new_id("emb_scene")
        vector_blob = _serialize_vector(vector)
        with self._lock:
            with self.connection:
                self._insert_scene_embedding_rows(
                    embedding_id=embedding_id,
                    scene_id=scene_id,
                    result=result,
                    source_target_type=source_target_type,
                    vector_blob=vector_blob,
                    now_ms=now_ms,
                )
        return embedding_id

    def get_embedding_provenance(self, embedding_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT
                  embedding_id, owner_type, owner_id, source_track_ref, source_frame_ref,
                  crop_hash, crop_path_or_artifact_ref, resolver_target_ref,
                  resolution_reason, embedding_type, embedding_model, embedding_version,
                  embedding_dim, created_at_ms
                FROM embedding_provenance
                WHERE embedding_id = ?
                """,
                (embedding_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "embedding_id": row["embedding_id"],
            "owner_type": row["owner_type"],
            "owner_id": row["owner_id"],
            "source_track_ref": row["source_track_ref"],
            "source_frame_ref": row["source_frame_ref"],
            "crop_hash": row["crop_hash"],
            "crop_path_or_artifact_ref": row["crop_path_or_artifact_ref"],
            "resolver_target_ref": row["resolver_target_ref"],
            "resolution_reason": row["resolution_reason"],
            "embedding_type": row["embedding_type"],
            "embedding_model": row["embedding_model"],
            "embedding_version": row["embedding_version"],
            "embedding_dim": int(row["embedding_dim"]),
            "created_at_ms": int(row["created_at_ms"]),
        }

    def add_anonymous_embedding(
        self,
        *,
        anonymous_id: str,
        result: EmbeddingResult,
        source_target_type: str,
        source_track_ref: str | None = None,
        source_frame_ref: str | None = None,
        crop_hash: str | None = None,
        crop_path_or_artifact_ref: str | None = None,
        resolver_target_ref: str | None = None,
        resolution_reason: str | None = None,
        now_ms: int,
    ) -> str:
        vector = self._checked_vector(result.vector, expected_dim=self.person_dim)
        embedding_id = _new_id("emb_anon")
        vector_blob = _serialize_vector(vector)
        has_provenance = any(
            value is not None
            for value in (
                source_frame_ref,
                crop_hash,
                resolver_target_ref,
                resolution_reason,
            )
        )
        if has_provenance and not all(
            value is not None
            for value in (
                source_frame_ref,
                crop_hash,
                resolver_target_ref,
                resolution_reason,
            )
        ):
            raise MemoryStoreError(
                "invalid_embedding_provenance",
                "anonymous embedding provenance is incomplete",
            )
        with self._lock:
            with self.connection:
                cursor = self.connection.execute(
                    """
                    INSERT INTO anonymous_embeddings (
                      embedding_id, anonymous_id, embedding_type, embedding_model,
                      embedding_version, embedding_dim, source_target_type,
                      vector_blob, quality, created_at_ms
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        embedding_id,
                        anonymous_id,
                        result.embedding_type,
                        result.embedding_model,
                        result.embedding_version,
                        self.person_dim,
                        source_target_type,
                        vector_blob,
                        result.quality,
                        now_ms,
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO anonymous_embedding_vectors(
                      rowid, embedding, embedding_id, anonymous_id, embedding_type,
                      embedding_model, embedding_version, embedding_dim, source_target_type
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cursor.lastrowid,
                        vector_blob,
                        embedding_id,
                        anonymous_id,
                        result.embedding_type,
                        result.embedding_model,
                        result.embedding_version,
                        self.person_dim,
                        source_target_type,
                    ),
                )
                if has_provenance:
                    self._insert_embedding_provenance(
                        embedding_id=embedding_id,
                        owner_type="anonymous",
                        owner_id=anonymous_id,
                        source_track_ref=source_track_ref,
                        source_frame_ref=str(source_frame_ref),
                        crop_hash=str(crop_hash),
                        crop_path_or_artifact_ref=crop_path_or_artifact_ref,
                        resolver_target_ref=str(resolver_target_ref),
                        resolution_reason=str(resolution_reason),
                        embedding_type=result.embedding_type,
                        embedding_model=result.embedding_model,
                        embedding_version=result.embedding_version,
                        embedding_dim=self.person_dim,
                        now_ms=now_ms,
                    )
        return embedding_id

    def list_anonymous_embeddings(self, anonymous_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT
                  embedding_id, anonymous_id, embedding_type, embedding_model,
                  embedding_version, embedding_dim, source_target_type, quality, created_at_ms
                FROM anonymous_embeddings
                WHERE anonymous_id = ?
                ORDER BY created_at_ms, vector_rowid
                """,
                (anonymous_id,),
            ).fetchall()
        return [
            {
                "embedding_id": row["embedding_id"],
                "anonymous_id": row["anonymous_id"],
                "embedding_type": row["embedding_type"],
                "embedding_model": row["embedding_model"],
                "embedding_version": row["embedding_version"],
                "embedding_dim": int(row["embedding_dim"]),
                "source_target_type": row["source_target_type"],
                "quality": float(row["quality"]),
                "created_at_ms": int(row["created_at_ms"]),
            }
            for row in rows
        ]

    def copy_anonymous_embeddings_to_person(
        self,
        *,
        anonymous_id: str,
        person_id: str,
        now_ms: int,
    ) -> list[str]:
        with self._lock:
            with self.connection:
                return self._copy_anonymous_embeddings_to_person(
                    anonymous_id=anonymous_id,
                    person_id=person_id,
                    now_ms=now_ms,
                )

    def mark_anonymous_profile_merged(
        self,
        *,
        anonymous_id: str,
        person_id: str,
        now_ms: int,
    ) -> None:
        with self._lock:
            with self.connection:
                self.connection.execute(
                    """
                    UPDATE anonymous_profiles
                    SET status = 'merged',
                        merged_person_id = ?,
                        updated_at_ms = ?
                    WHERE anonymous_id = ?
                    """,
                    (person_id, now_ms, anonymous_id),
                )

    def add_profile_merge_history(
        self,
        *,
        anonymous_id: str,
        person_id: str,
        merge_reason: str,
        now_ms: int,
    ) -> str:
        merge_id = _new_id("merge")
        with self._lock:
            with self.connection:
                self._insert_profile_merge_history(
                    merge_id=merge_id,
                    anonymous_id=anonymous_id,
                    person_id=person_id,
                    merge_reason=merge_reason,
                    now_ms=now_ms,
                )
        return merge_id

    def get_profile_merge_history(self, merge_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT merge_id, anonymous_id, person_id, merge_reason, created_at_ms
                FROM profile_merge_history
                WHERE merge_id = ?
                """,
                (merge_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "merge_id": row["merge_id"],
            "anonymous_id": row["anonymous_id"],
            "person_id": row["person_id"],
            "merge_reason": row["merge_reason"],
            "created_at_ms": int(row["created_at_ms"]),
        }

    def add_negative_identity_match(
        self,
        *,
        memory_match_id: str,
        wrong_person_id: str,
        embedding_id: str,
        now_ms: int,
    ) -> None:
        with self._lock:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO negative_identity_matches (
                      memory_match_id, wrong_person_id, embedding_id, created_at_ms
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(wrong_person_id, embedding_id) DO UPDATE SET
                      memory_match_id = excluded.memory_match_id,
                      created_at_ms = excluded.created_at_ms
                    """,
                    (memory_match_id, wrong_person_id, embedding_id, now_ms),
                )

    def get_negative_identity_match(
        self,
        *,
        wrong_person_id: str,
        embedding_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT memory_match_id, wrong_person_id, embedding_id, created_at_ms
                FROM negative_identity_matches
                WHERE wrong_person_id = ? AND embedding_id = ?
                """,
                (wrong_person_id, embedding_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "memory_match_id": row["memory_match_id"],
            "wrong_person_id": row["wrong_person_id"],
            "embedding_id": row["embedding_id"],
            "created_at_ms": int(row["created_at_ms"]),
        }

    def is_negative_identity_candidate(
        self,
        *,
        wrong_person_id: str,
        embedding_id: str,
    ) -> bool:
        return self.get_negative_identity_match(
            wrong_person_id=wrong_person_id,
            embedding_id=embedding_id,
        ) is not None

    def search_person_embeddings(
        self,
        result: EmbeddingResult,
        *,
        limit: int,
    ) -> list[VectorCandidate]:
        vector_blob = _serialize_vector(
            self._checked_vector(result.vector, expected_dim=self.person_dim)
        )
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT
                  person_id AS matched_id,
                  embedding_id,
                  embedding_type,
                  embedding_model,
                  embedding_version,
                  distance
                FROM person_embedding_vectors AS v
                WHERE embedding MATCH ?
                  AND k = ?
                  AND embedding_type = ?
                  AND embedding_model = ?
                  AND embedding_version = ?
                  AND embedding_dim = ?
                ORDER BY distance
                """,
                (
                    vector_blob,
                    limit,
                    result.embedding_type,
                    result.embedding_model,
                    result.embedding_version,
                    self.person_dim,
                ),
            ).fetchall()
        return [
            VectorCandidate(
                matched_type="person",
                matched_id=row["matched_id"],
                embedding_id=row["embedding_id"],
                match_type=row["embedding_type"],
                embedding_model=row["embedding_model"],
                embedding_version=row["embedding_version"],
                distance=float(row["distance"]),
            )
            for row in rows
        ]

    def search_scene_embeddings(
        self,
        result: EmbeddingResult,
        *,
        limit: int,
    ) -> list[VectorCandidate]:
        vector_blob = _serialize_vector(
            self._checked_vector(result.vector, expected_dim=self.scene_dim)
        )
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT
                  scene_id AS matched_id,
                  embedding_id,
                  embedding_type,
                  embedding_model,
                  embedding_version,
                  distance
                FROM scene_embedding_vectors AS v
                WHERE embedding MATCH ?
                  AND k = ?
                  AND embedding_type = ?
                  AND embedding_model = ?
                  AND embedding_version = ?
                  AND embedding_dim = ?
                ORDER BY distance
                """,
                (
                    vector_blob,
                    limit,
                    result.embedding_type,
                    result.embedding_model,
                    result.embedding_version,
                    self.scene_dim,
                ),
            ).fetchall()
        return [
            VectorCandidate(
                matched_type="scene",
                matched_id=row["matched_id"],
                embedding_id=row["embedding_id"],
                match_type=row["embedding_type"],
                embedding_model=row["embedding_model"],
                embedding_version=row["embedding_version"],
                distance=float(row["distance"]),
            )
            for row in rows
        ]

    def search_anonymous_embeddings(
        self,
        result: EmbeddingResult,
        *,
        limit: int,
        active_only: bool = True,
    ) -> list[VectorCandidate]:
        vector_blob = _serialize_vector(
            self._checked_vector(result.vector, expected_dim=self.person_dim)
        )
        active_clause = "AND p.status = 'active'" if active_only else ""
        with self._lock:
            rows = self.connection.execute(
                f"""
                SELECT
                  v.anonymous_id AS matched_id,
                  v.embedding_id,
                  v.embedding_type,
                  v.embedding_model,
                  v.embedding_version,
                  v.distance
                FROM anonymous_embedding_vectors AS v
                JOIN anonymous_profiles AS p ON p.anonymous_id = v.anonymous_id
                WHERE v.embedding MATCH ?
                  AND k = ?
                  AND v.embedding_type = ?
                  AND v.embedding_model = ?
                  AND v.embedding_version = ?
                  AND v.embedding_dim = ?
                  {active_clause}
                ORDER BY v.distance
                """,
                (
                    vector_blob,
                    limit,
                    result.embedding_type,
                    result.embedding_model,
                    result.embedding_version,
                    self.person_dim,
                ),
            ).fetchall()
        return [
            VectorCandidate(
                matched_type="anonymous_person",
                matched_id=row["matched_id"],
                embedding_id=row["embedding_id"],
                match_type=row["embedding_type"],
                embedding_model=row["embedding_model"],
                embedding_version=row["embedding_version"],
                distance=float(row["distance"]),
            )
            for row in rows
        ]

    def _upsert_person_profile(
        self,
        *,
        person_id: str,
        display_name: str,
        description: str,
        tags: tuple[str, ...],
        now_ms: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO person_profiles (
              person_id, display_name, description, tags_json, created_at_ms, updated_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
              display_name = excluded.display_name,
              description = excluded.description,
              tags_json = excluded.tags_json,
              updated_at_ms = excluded.updated_at_ms
            """,
            (
                person_id,
                display_name,
                description,
                json.dumps(list(tags), ensure_ascii=False, separators=(",", ":")),
                now_ms,
                now_ms,
            ),
        )

    def _upsert_scene_memory(
        self,
        *,
        scene_id: str,
        title: str,
        description: str,
        activation_hint: str,
        target_type: str,
        region_id: str | None,
        now_ms: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO scene_memories (
              scene_id, title, description, activation_hint, target_type, region_id,
              created_at_ms, updated_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scene_id) DO UPDATE SET
              title = excluded.title,
              description = excluded.description,
              activation_hint = excluded.activation_hint,
              target_type = excluded.target_type,
              region_id = excluded.region_id,
              updated_at_ms = excluded.updated_at_ms
            """,
            (
                scene_id,
                title,
                description,
                activation_hint,
                target_type,
                region_id,
                now_ms,
                now_ms,
            ),
        )

    def _link_external_user_if_available(
        self,
        *,
        person_id: str,
        external_user_ref: str,
        now_ms: int,
    ) -> None:
        linked = self.connection.execute(
            """
            SELECT person_id
            FROM external_user_links
            WHERE external_user_ref = ?
            """,
            (external_user_ref,),
        ).fetchone()
        if linked is not None and linked["person_id"] != person_id:
            raise MemoryStoreError(
                "external_user_ref_conflict",
                "external_user_ref is already linked to a different person",
                details={
                    "external_user_ref": external_user_ref,
                    "external_user_person_id": linked["person_id"],
                },
            )
        if linked is None:
            self.connection.execute(
                """
                INSERT INTO external_user_links (
                  external_user_ref, person_id, created_at_ms
                )
                VALUES (?, ?, ?)
                """,
                (external_user_ref, person_id, now_ms),
            )
            return
        self.connection.execute(
            """
            UPDATE external_user_links
            SET created_at_ms = ?
            WHERE external_user_ref = ? AND person_id = ?
            """,
            (now_ms, external_user_ref, person_id),
        )

    def _insert_person_embedding_rows(
        self,
        *,
        embedding_id: str,
        person_id: str,
        result: EmbeddingResult,
        source_target_type: str,
        vector_blob: bytes,
        now_ms: int,
    ) -> None:
        cursor = self.connection.execute(
            """
            INSERT INTO person_embeddings (
              embedding_id, person_id, embedding_type, embedding_model,
              embedding_version, embedding_dim, source_target_type,
              vector_blob, quality, created_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embedding_id,
                person_id,
                result.embedding_type,
                result.embedding_model,
                result.embedding_version,
                self.person_dim,
                source_target_type,
                vector_blob,
                result.quality,
                now_ms,
            ),
        )
        self.connection.execute(
            """
            INSERT INTO person_embedding_vectors(
              rowid, embedding, embedding_id, person_id, embedding_type,
              embedding_model, embedding_version, embedding_dim, source_target_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cursor.lastrowid,
                vector_blob,
                embedding_id,
                person_id,
                result.embedding_type,
                result.embedding_model,
                result.embedding_version,
                self.person_dim,
                source_target_type,
            ),
        )

    def _insert_scene_embedding_rows(
        self,
        *,
        embedding_id: str,
        scene_id: str,
        result: EmbeddingResult,
        source_target_type: str,
        vector_blob: bytes,
        now_ms: int,
    ) -> None:
        cursor = self.connection.execute(
            """
            INSERT INTO scene_embeddings (
              embedding_id, scene_id, embedding_type, embedding_model,
              embedding_version, embedding_dim, source_target_type,
              vector_blob, quality, created_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embedding_id,
                scene_id,
                result.embedding_type,
                result.embedding_model,
                result.embedding_version,
                self.scene_dim,
                source_target_type,
                vector_blob,
                result.quality,
                now_ms,
            ),
        )
        self.connection.execute(
            """
            INSERT INTO scene_embedding_vectors(
              rowid, embedding, embedding_id, scene_id, embedding_type,
              embedding_model, embedding_version, embedding_dim, source_target_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cursor.lastrowid,
                vector_blob,
                embedding_id,
                scene_id,
                result.embedding_type,
                result.embedding_model,
                result.embedding_version,
                self.scene_dim,
                source_target_type,
            ),
        )

    def _copy_anonymous_embeddings_to_person(
        self,
        *,
        anonymous_id: str,
        person_id: str,
        now_ms: int,
    ) -> list[str]:
        rows = self.connection.execute(
            """
            SELECT
              embedding_id, embedding_type, embedding_model, embedding_version, embedding_dim,
              source_target_type, vector_blob, quality
            FROM anonymous_embeddings
            WHERE anonymous_id = ?
            ORDER BY created_at_ms, vector_rowid
            """,
            (anonymous_id,),
        ).fetchall()
        copied_ids: list[str] = []
        for row in rows:
            if int(row["embedding_dim"]) != self.person_dim:
                raise MemoryStoreError(
                    "memory_db_dimension_mismatch",
                    f"anonymous embedding dim is {row['embedding_dim']}, expected {self.person_dim}",
                )
            embedding_id = _new_id("emb_person")
            cursor = self.connection.execute(
                """
                INSERT INTO person_embeddings (
                  embedding_id, person_id, embedding_type, embedding_model,
                  embedding_version, embedding_dim, source_target_type,
                  vector_blob, quality, created_at_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    embedding_id,
                    person_id,
                    row["embedding_type"],
                    row["embedding_model"],
                    row["embedding_version"],
                    self.person_dim,
                    row["source_target_type"],
                    row["vector_blob"],
                    row["quality"],
                    now_ms,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO person_embedding_vectors(
                  rowid, embedding, embedding_id, person_id, embedding_type,
                  embedding_model, embedding_version, embedding_dim, source_target_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cursor.lastrowid,
                    row["vector_blob"],
                    embedding_id,
                    person_id,
                    row["embedding_type"],
                    row["embedding_model"],
                    row["embedding_version"],
                    self.person_dim,
                    row["source_target_type"],
                ),
            )
            provenance = self.connection.execute(
                """
                SELECT
                  source_track_ref, source_frame_ref, crop_hash,
                  crop_path_or_artifact_ref, resolver_target_ref,
                  resolution_reason, embedding_type, embedding_model,
                  embedding_version, embedding_dim
                FROM embedding_provenance
                WHERE embedding_id = ?
                """,
                (row["embedding_id"],),
            ).fetchone()
            if provenance is not None:
                self._insert_embedding_provenance(
                    embedding_id=embedding_id,
                    owner_type="person",
                    owner_id=person_id,
                    source_track_ref=provenance["source_track_ref"],
                    source_frame_ref=provenance["source_frame_ref"],
                    crop_hash=provenance["crop_hash"],
                    crop_path_or_artifact_ref=provenance[
                        "crop_path_or_artifact_ref"
                    ],
                    resolver_target_ref=provenance["resolver_target_ref"],
                    resolution_reason=provenance["resolution_reason"],
                    embedding_type=provenance["embedding_type"],
                    embedding_model=provenance["embedding_model"],
                    embedding_version=provenance["embedding_version"],
                    embedding_dim=int(provenance["embedding_dim"]),
                    now_ms=now_ms,
                )
            copied_ids.append(embedding_id)
        return copied_ids

    def _insert_profile_merge_history(
        self,
        *,
        merge_id: str,
        anonymous_id: str,
        person_id: str,
        merge_reason: str,
        now_ms: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO profile_merge_history (
              merge_id, anonymous_id, person_id, merge_reason, created_at_ms
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (merge_id, anonymous_id, person_id, merge_reason, now_ms),
        )

    def _insert_embedding_provenance(
        self,
        *,
        embedding_id: str,
        owner_type: str,
        owner_id: str,
        source_track_ref: str | None,
        source_frame_ref: str,
        crop_hash: str,
        crop_path_or_artifact_ref: str | None,
        resolver_target_ref: str,
        resolution_reason: str,
        embedding_type: str,
        embedding_model: str,
        embedding_version: str,
        embedding_dim: int,
        now_ms: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO embedding_provenance (
              embedding_id, owner_type, owner_id, source_track_ref, source_frame_ref,
              crop_hash, crop_path_or_artifact_ref, resolver_target_ref,
              resolution_reason, embedding_type, embedding_model, embedding_version,
              embedding_dim, created_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embedding_id,
                owner_type,
                owner_id,
                source_track_ref,
                source_frame_ref,
                crop_hash,
                crop_path_or_artifact_ref,
                resolver_target_ref,
                resolution_reason,
                embedding_type,
                embedding_model,
                embedding_version,
                embedding_dim,
                now_ms,
            ),
        )

    def _initialize_schema(self) -> None:
        self.connection.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS memory_meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS person_profiles (
              person_id TEXT PRIMARY KEY,
              display_name TEXT NOT NULL,
              description TEXT NOT NULL,
              tags_json TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS person_embeddings (
              vector_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
              embedding_id TEXT NOT NULL UNIQUE,
              person_id TEXT NOT NULL REFERENCES person_profiles(person_id) ON DELETE CASCADE,
              embedding_type TEXT NOT NULL,
              embedding_model TEXT NOT NULL,
              embedding_version TEXT NOT NULL,
              embedding_dim INTEGER NOT NULL,
              source_target_type TEXT NOT NULL,
              vector_blob BLOB NOT NULL,
              quality REAL NOT NULL,
              created_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scene_memories (
              scene_id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              description TEXT NOT NULL,
              activation_hint TEXT NOT NULL,
              target_type TEXT NOT NULL,
              region_id TEXT,
              created_at_ms INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS anonymous_profiles (
              anonymous_id TEXT PRIMARY KEY,
              seen_count INTEGER NOT NULL,
              first_seen_at_ms INTEGER NOT NULL,
              last_seen_at_ms INTEGER NOT NULL,
              familiar_score REAL NOT NULL,
              observed_duration_ms INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL,
              merged_person_id TEXT,
              created_at_ms INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scene_embeddings (
              vector_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
              embedding_id TEXT NOT NULL UNIQUE,
              scene_id TEXT NOT NULL REFERENCES scene_memories(scene_id) ON DELETE CASCADE,
              embedding_type TEXT NOT NULL,
              embedding_model TEXT NOT NULL,
              embedding_version TEXT NOT NULL,
              embedding_dim INTEGER NOT NULL,
              source_target_type TEXT NOT NULL,
              vector_blob BLOB NOT NULL,
              quality REAL NOT NULL,
              created_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS anonymous_embeddings (
              vector_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
              embedding_id TEXT NOT NULL UNIQUE,
              anonymous_id TEXT NOT NULL REFERENCES anonymous_profiles(anonymous_id) ON DELETE CASCADE,
              embedding_type TEXT NOT NULL,
              embedding_model TEXT NOT NULL,
              embedding_version TEXT NOT NULL,
              embedding_dim INTEGER NOT NULL,
              source_target_type TEXT NOT NULL,
              vector_blob BLOB NOT NULL,
              quality REAL NOT NULL,
              created_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embedding_provenance (
              embedding_id TEXT PRIMARY KEY,
              owner_type TEXT NOT NULL,
              owner_id TEXT NOT NULL,
              source_track_ref TEXT,
              source_frame_ref TEXT NOT NULL,
              crop_hash TEXT NOT NULL,
              crop_path_or_artifact_ref TEXT,
              resolver_target_ref TEXT NOT NULL,
              resolution_reason TEXT NOT NULL,
              embedding_type TEXT NOT NULL,
              embedding_model TEXT NOT NULL,
              embedding_version TEXT NOT NULL,
              embedding_dim INTEGER NOT NULL,
              created_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_summaries (
              summary_id TEXT PRIMARY KEY,
              person_id TEXT NOT NULL REFERENCES person_profiles(person_id) ON DELETE CASCADE,
              summary TEXT NOT NULL,
              source TEXT NOT NULL,
              source_conversation_id TEXT,
              created_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS external_user_links (
              external_user_ref TEXT PRIMARY KEY,
              person_id TEXT NOT NULL REFERENCES person_profiles(person_id) ON DELETE CASCADE,
              created_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_match_records (
              memory_match_id TEXT PRIMARY KEY,
              event_id TEXT NOT NULL,
              matched_type TEXT NOT NULL,
              matched_id TEXT NOT NULL,
              embedding_id TEXT NOT NULL,
              match_score REAL NOT NULL,
              top2_margin REAL NOT NULL,
              source_target_mode TEXT NOT NULL,
              camera TEXT NOT NULL,
              frame_id INTEGER NOT NULL,
              frame_timestamp_ms INTEGER NOT NULL,
              created_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS profile_merge_history (
              merge_id TEXT PRIMARY KEY,
              anonymous_id TEXT NOT NULL,
              person_id TEXT NOT NULL REFERENCES person_profiles(person_id) ON DELETE CASCADE,
              merge_reason TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS negative_identity_matches (
              memory_match_id TEXT NOT NULL,
              wrong_person_id TEXT NOT NULL REFERENCES person_profiles(person_id) ON DELETE CASCADE,
              embedding_id TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL,
              UNIQUE(wrong_person_id, embedding_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS person_embedding_vectors USING vec0(
              embedding float[{self.person_dim}],
              embedding_id TEXT,
              person_id TEXT,
              embedding_type TEXT,
              embedding_model TEXT,
              embedding_version TEXT,
              embedding_dim INTEGER,
              source_target_type TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS anonymous_embedding_vectors USING vec0(
              embedding float[{self.person_dim}],
              embedding_id TEXT,
              anonymous_id TEXT,
              embedding_type TEXT,
              embedding_model TEXT,
              embedding_version TEXT,
              embedding_dim INTEGER,
              source_target_type TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS scene_embedding_vectors USING vec0(
              embedding float[{self.scene_dim}],
              embedding_id TEXT,
              scene_id TEXT,
              embedding_type TEXT,
              embedding_model TEXT,
              embedding_version TEXT,
              embedding_dim INTEGER,
              source_target_type TEXT
            );
            """
        )
        self._ensure_column("scene_memories", "region_id", "TEXT")
        self._ensure_column(
            "anonymous_profiles",
            "observed_duration_ms",
            "INTEGER NOT NULL DEFAULT 0",
        )
        self._assert_or_set_meta("person_dim", str(self.person_dim))
        self._assert_or_set_meta("scene_dim", str(self.scene_dim))
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        if any(row["name"] == column for row in rows):
            return
        self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _assert_or_set_meta(self, key: str, value: str) -> None:
        row = self.connection.execute(
            "SELECT value FROM memory_meta WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO memory_meta(key, value) VALUES (?, ?)",
                (key, value),
            )
            return
        if row["value"] != value:
            raise MemoryStoreError(
                "memory_db_dimension_mismatch",
                f"{key} is {row['value']}, expected {value}",
            )

    def _checked_vector(
        self,
        vector: tuple[float, ...],
        *,
        expected_dim: int,
    ) -> tuple[float, ...]:
        if len(vector) != expected_dim:
            raise ValueError(f"embedding vector expected {expected_dim} dimensions")
        return normalize_vector(vector)


def _serialize_vector(vector: tuple[float, ...]) -> bytes:
    if sqlite_vec is None:
        raise MemoryStoreError(
            "sqlite_vec_unavailable",
            "sqlite-vec is required for memory vector search",
        )
    return sqlite_vec.serialize_float32(vector)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _short_text(value: str, *, max_chars: int = 240) -> str:
    text = " ".join(str(value).split())
    return text[:max_chars]
