from __future__ import annotations

import math
import uuid
from collections.abc import Callable

from .embedding import EmbeddingResult
from .events import MemoryMatch
from .store import MemoryStore, VectorCandidate


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
        candidate_limit = max(8, top_k * 4, 2)
        return self._query_candidates(
            self.store.search_person_embeddings(result, limit=candidate_limit),
            threshold=threshold,
            margin=margin,
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
        return self._query_candidates(
            self.store.search_scene_embeddings(result, limit=max(2, top_k)),
            threshold=threshold,
            margin=margin,
        )

    def query_anonymous_person(
        self,
        result: EmbeddingResult,
        *,
        threshold: float,
        margin: float,
        top_k: int = 2,
    ) -> MemoryMatch | None:
        return self._query_candidates(
            self.store.search_anonymous_embeddings(result, limit=max(2, top_k)),
            threshold=threshold,
            margin=margin,
        )

    def _query_candidates(
        self,
        candidates: list[VectorCandidate],
        *,
        threshold: float,
        margin: float,
        candidate_filter: Callable[[VectorCandidate], bool] | None = None,
    ) -> MemoryMatch | None:
        if candidate_filter is not None:
            candidates = [candidate for candidate in candidates if candidate_filter(candidate)]
        if not candidates:
            return None
        scored = [(candidate, _score_from_l2_distance(candidate.distance)) for candidate in candidates]
        best, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
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


def _score_from_l2_distance(distance: float) -> float:
    if not math.isfinite(distance):
        return 0.0
    return max(0.0, min(1.0, 1.0 - ((float(distance) * float(distance)) / 2.0)))
