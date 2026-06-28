from __future__ import annotations

import math

import pytest

from visual_events_server.memory.embedding import (
    DisabledEmbeddingBackend,
    EmbeddingUnavailable,
    FakeEmbeddingBackend,
)


def test_disabled_embedding_backend_fails_fast() -> None:
    backend = DisabledEmbeddingBackend()

    with pytest.raises(EmbeddingUnavailable) as exc:
        backend.embed_person(b"image-bytes")

    assert exc.value.code == "embedding_disabled"


def test_fake_embedding_backend_is_deterministic_and_normalized() -> None:
    backend = FakeEmbeddingBackend(
        person_dim=8,
        scene_dim=6,
        person_model="fake-face",
        scene_model="fake-scene",
        model_version="test-v1",
    )

    first = backend.embed_person(b"same-person")
    second = backend.embed_person(b"same-person")
    scene = backend.embed_scene(b"same-person")

    assert first.vector == second.vector
    assert first.embedding_type == "face"
    assert first.embedding_model == "fake-face"
    assert first.embedding_version == "test-v1"
    assert len(first.vector) == 8
    assert math.isclose(
        math.sqrt(sum(value * value for value in first.vector)),
        1.0,
        rel_tol=1e-6,
    )
    assert scene.embedding_type == "scene"
    assert scene.embedding_model == "fake-scene"
    assert len(scene.vector) == 6
