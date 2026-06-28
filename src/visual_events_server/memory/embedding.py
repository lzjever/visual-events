from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingResult:
    vector: tuple[float, ...]
    embedding_type: str
    embedding_model: str
    embedding_version: str
    quality: float


class EmbeddingUnavailable(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class MemoryEmbeddingBackend(Protocol):
    def embed_person(self, image_crop: bytes) -> EmbeddingResult:
        ...

    def embed_scene(self, image_or_crop: bytes) -> EmbeddingResult:
        ...


class DisabledEmbeddingBackend:
    def embed_person(self, image_crop: bytes) -> EmbeddingResult:
        raise EmbeddingUnavailable("embedding_disabled", "memory embedding is disabled")

    def embed_scene(self, image_or_crop: bytes) -> EmbeddingResult:
        raise EmbeddingUnavailable("embedding_disabled", "memory embedding is disabled")


@dataclass(frozen=True)
class FakeEmbeddingBackend:
    person_dim: int = 32
    scene_dim: int = 32
    person_model: str = "fake-face"
    scene_model: str = "fake-scene"
    model_version: str = "test-v1"

    def __post_init__(self) -> None:
        if self.person_dim <= 0:
            raise ValueError("person_dim must be positive")
        if self.scene_dim <= 0:
            raise ValueError("scene_dim must be positive")

    def embed_person(self, image_crop: bytes) -> EmbeddingResult:
        return EmbeddingResult(
            vector=_deterministic_unit_vector(
                b"person" + image_crop,
                dim=self.person_dim,
            ),
            embedding_type="face",
            embedding_model=self.person_model,
            embedding_version=self.model_version,
            quality=1.0,
        )

    def embed_scene(self, image_or_crop: bytes) -> EmbeddingResult:
        return EmbeddingResult(
            vector=_deterministic_unit_vector(
                b"scene" + image_or_crop,
                dim=self.scene_dim,
            ),
            embedding_type="scene",
            embedding_model=self.scene_model,
            embedding_version=self.model_version,
            quality=1.0,
        )


def normalize_vector(vector: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if not vector:
        raise ValueError("embedding vector must not be empty")
    length = math.sqrt(sum(float(value) * float(value) for value in vector))
    if length <= 0.0 or not math.isfinite(length):
        raise ValueError("embedding vector norm must be finite and positive")
    normalized = tuple(float(value) / length for value in vector)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("embedding vector values must be finite")
    return normalized


def _deterministic_unit_vector(seed: bytes, *, dim: int) -> tuple[float, ...]:
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for offset in range(0, len(digest), 4):
            if len(values) >= dim:
                break
            raw = struct.unpack(">I", digest[offset : offset + 4])[0]
            values.append((raw / 0xFFFFFFFF) * 2.0 - 1.0)
        counter += 1
    return normalize_vector(values)
