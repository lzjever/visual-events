from __future__ import annotations

import math
import uuid
from collections.abc import Callable

from .embedding import EmbeddingResult
from .events import MemoryMatch
from .store import MemoryStore, VectorCandidate


_MAX_VECTOR_CANDIDATES = 64


class MemoryRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def query_person(
        self,
        result: EmbeddingResult,
        *,
        threshold: float,
        margin: float,
        top_k: int = 2,
    ) -> MemoryMatch | None:
        window_limit, fetch_limit = _candidate_limits(top_k)
        raw_candidates = self.store.search_person_embeddings(
            result,
            limit=fetch_limit,
        )
        return self._query_candidates(
            raw_candidates[:window_limit],
            threshold=threshold,
            margin=margin,
            window_truncated=_window_truncated(
                raw_candidates,
                window_limit=window_limit,
                fetch_limit=fetch_limit,
            ),
            candidate_filter=lambda candidate: not self.store.is_negative_identity_candidate(
                wrong_person_id=candidate.matched_id,
                embedding_id=candidate.embedding_id,
            ),
        )

    def query_scene(
        self,
        result: EmbeddingResult,
        *,
        threshold: float,
        margin: float,
        top_k: int = 2,
    ) -> MemoryMatch | None:
        window_limit, fetch_limit = _candidate_limits(top_k)
        raw_candidates = self.store.search_scene_embeddings(
            result,
            limit=fetch_limit,
        )
        return self._query_candidates(
            raw_candidates[:window_limit],
            threshold=threshold,
            margin=margin,
            window_truncated=_window_truncated(
                raw_candidates,
                window_limit=window_limit,
                fetch_limit=fetch_limit,
            ),
        )

    def query_anonymous_person(
        self,
        result: EmbeddingResult,
        *,
        threshold: float,
        margin: float,
        top_k: int = 2,
    ) -> MemoryMatch | None:
        window_limit, fetch_limit = _candidate_limits(top_k)
        active_candidates = self.store.search_anonymous_embeddings(
            result,
            limit=fetch_limit,
        )
        match = self._query_candidates(
            active_candidates[:window_limit],
            threshold=threshold,
            margin=margin,
            window_truncated=_window_truncated(
                active_candidates,
                window_limit=window_limit,
                fetch_limit=fetch_limit,
            ),
        )
        if match is None:
            return None
        raw_candidates = self.store.search_anonymous_embeddings(
            result,
            limit=fetch_limit,
            active_only=False,
        )
        if self._inactive_anonymous_beats_match(raw_candidates, match):
            return None
        return match

    def _query_candidates(
        self,
        candidates: list[VectorCandidate],
        *,
        threshold: float,
        margin: float,
        window_truncated: bool,
        candidate_filter: Callable[[VectorCandidate], bool] | None = None,
    ) -> MemoryMatch | None:
        if candidate_filter is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate_filter(candidate)
            ]
        if not candidates:
            return None
        scored = [
            (candidate, _score_from_l2_distance(candidate.distance))
            for candidate in candidates
        ]
        best, best_score = scored[0]
        second_score = 0.0
        for candidate, score in scored[1:]:
            if (
                candidate.matched_type,
                candidate.matched_id,
            ) != (best.matched_type, best.matched_id):
                second_score = score
                break
        else:
            if window_truncated:
                return None
        top2_margin = best_score - second_score
        if best_score < threshold or top2_margin < margin:
            return None
        return MemoryMatch(
            memory_match_id=f"match_{uuid.uuid4().hex[:16]}",
            matched_type=best.matched_type,
            matched_id=best.matched_id,
            embedding_id=best.embedding_id,
            match_type=best.match_type,
            match_score=best_score,
            top2_margin=top2_margin,
            embedding_model=best.embedding_model,
            embedding_version=best.embedding_version,
        )

    def _inactive_anonymous_beats_match(
        self,
        candidates: list[VectorCandidate],
        match: MemoryMatch,
    ) -> bool:
        for candidate in candidates:
            if self.store.get_active_anonymous_profile(candidate.matched_id) is not None:
                continue
            if _score_from_l2_distance(candidate.distance) > match.match_score:
                return True
        return False


def _score_from_l2_distance(distance: float) -> float:
    if not math.isfinite(distance):
        return 0.0
    return max(0.0, min(1.0, 1.0 - ((float(distance) * float(distance)) / 2.0)))


def _candidate_limits(top_k: int) -> tuple[int, int]:
    requested = max(2, int(top_k))
    window_limit = min(_MAX_VECTOR_CANDIDATES, max(2, requested * 4))
    fetch_limit = min(_MAX_VECTOR_CANDIDATES, window_limit + 1)
    return window_limit, fetch_limit


def _window_truncated(
    candidates: list[VectorCandidate],
    *,
    window_limit: int,
    fetch_limit: int,
) -> bool:
    if len(candidates) > window_limit:
        return True
    return fetch_limit == window_limit and len(candidates) == window_limit
