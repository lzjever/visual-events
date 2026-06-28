from __future__ import annotations

from .embedding import (
    DisabledEmbeddingBackend,
    EmbeddingResult,
    EmbeddingUnavailable,
    FakeEmbeddingBackend,
    MemoryEmbeddingBackend,
)
from .events import (
    MemoryEventGate,
    MemoryMatch,
    SourceFrameRef,
    build_familiar_unknown_event,
    build_known_person_event,
    build_scene_event,
)
from .retriever import MemoryRetriever
from .service import AppMemoryService, MemoryServiceError
from .store import MemoryStore, MemoryStoreError
from .target_resolver import (
    ResolvedTarget,
    TargetCandidate,
    TargetPreview,
    TargetRequest,
    TargetResolveError,
    TargetResolver,
)

__all__ = [
    "DisabledEmbeddingBackend",
    "EmbeddingResult",
    "EmbeddingUnavailable",
    "FakeEmbeddingBackend",
    "MemoryEmbeddingBackend",
    "MemoryEventGate",
    "MemoryMatch",
    "MemoryRetriever",
    "AppMemoryService",
    "MemoryServiceError",
    "MemoryStore",
    "MemoryStoreError",
    "ResolvedTarget",
    "SourceFrameRef",
    "TargetCandidate",
    "TargetPreview",
    "TargetRequest",
    "TargetResolveError",
    "TargetResolver",
    "build_familiar_unknown_event",
    "build_known_person_event",
    "build_scene_event",
]
