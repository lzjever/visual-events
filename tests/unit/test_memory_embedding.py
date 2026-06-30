from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visual_events_server.memory.embedding import (
    DisabledEmbeddingBackend,
    EmbeddingUnavailable,
    FakeEmbeddingBackend,
    LocalEmbeddingBackend,
    _load_face_bundle,
    _load_scene_bundle,
    _OnnxLocalEmbeddingLoader,
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


class StubLocalLoader:
    def __init__(self) -> None:
        self.person_calls: list[bytes] = []
        self.scene_calls: list[bytes] = []

    def embed_person(self, image_crop: bytes) -> list[float]:
        self.person_calls.append(image_crop)
        return [3.0, 4.0, 0.0]

    def embed_scene(self, image_or_crop: bytes) -> list[float]:
        self.scene_calls.append(image_or_crop)
        return [0.0, 6.0, 8.0, 0.0]


class MetadataLocalLoader(StubLocalLoader):
    def embed_person(self, image_crop: bytes):
        self.person_calls.append(image_crop)
        return {
            "vector": [3.0, 4.0, 0.0],
            "quality": 0.87,
            "metadata": {
                "face_detection": {
                    "coordinate_space": "crop",
                    "face_bbox_xyxy": [11.0, 12.0, 51.0, 62.0],
                    "landmarks_5": [
                        [20.0, 30.0],
                        [40.0, 30.0],
                        [30.0, 42.0],
                        [22.0, 54.0],
                        [38.0, 54.0],
                    ],
                    "score": 0.87,
                    "source": "local_embedding_scrfd",
                }
            },
        }


class NoFaceLocalLoader:
    def embed_person(self, image_crop: bytes) -> list[float]:
        raise EmbeddingUnavailable("no_usable_face", "no usable face detected")

    def embed_scene(self, image_or_crop: bytes) -> list[float]:
        return [1.0, 0.0]


def test_local_embedding_backend_normalizes_stub_loader_outputs(tmp_path: Path) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        model_name="local-face",
        version="face-v1",
        dim=3,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        model_name="local-scene",
        version="scene-v2",
        dim=4,
        files={"model": "scene.onnx"},
    )
    loader = StubLocalLoader()
    backend = LocalEmbeddingBackend(
        person_model_path=person_bundle,
        scene_model_path=scene_bundle,
        loader=loader,
    )

    person = backend.embed_person(b"person-jpeg")
    scene = backend.embed_scene(b"scene-jpeg")

    assert backend.person_dim == 3
    assert backend.scene_dim == 4
    assert person.embedding_type == "face"
    assert person.embedding_model == "local-face"
    assert person.embedding_version == "face-v1"
    assert person.vector == (0.6, 0.8, 0.0)
    assert scene.embedding_type == "scene"
    assert scene.embedding_model == "local-scene"
    assert scene.embedding_version == "scene-v2"
    assert math.isclose(
        math.sqrt(sum(value * value for value in scene.vector)),
        1.0,
        rel_tol=1e-6,
    )
    assert loader.person_calls == [b"person-jpeg"]
    assert loader.scene_calls == [b"scene-jpeg"]


def test_local_embedding_backend_preserves_person_face_metadata(
    tmp_path: Path,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        model_name="local-face",
        version="face-v1",
        dim=3,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        model_name="local-scene",
        version="scene-v2",
        dim=4,
        files={"model": "scene.onnx"},
    )
    loader = MetadataLocalLoader()
    backend = LocalEmbeddingBackend(
        person_model_path=person_bundle,
        scene_model_path=scene_bundle,
        loader=loader,
    )

    person = backend.embed_person(b"person-jpeg")

    assert person.vector == (0.6, 0.8, 0.0)
    assert person.quality == pytest.approx(0.87)
    assert person.metadata == {
        "face_detection": {
            "coordinate_space": "crop",
            "face_bbox_xyxy": [11.0, 12.0, 51.0, 62.0],
            "landmarks_5": [
                [20.0, 30.0],
                [40.0, 30.0],
                [30.0, 42.0],
                [22.0, 54.0],
                [38.0, 54.0],
            ],
            "score": 0.87,
            "source": "local_embedding_scrfd",
        }
    }


def test_local_embedding_backend_propagates_no_usable_face(tmp_path: Path) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=2,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=2,
        files={"model": "scene.onnx"},
    )
    backend = LocalEmbeddingBackend(
        person_model_path=person_bundle,
        scene_model_path=scene_bundle,
        loader=NoFaceLocalLoader(),
    )

    with pytest.raises(EmbeddingUnavailable) as exc:
        backend.embed_person(b"person-jpeg")

    assert exc.value.code == "no_usable_face"


def test_local_embedding_backend_fails_fast_on_metadata_errors(
    tmp_path: Path,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=0,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=2,
        files={"model": "scene.onnx"},
    )

    with pytest.raises(ValueError, match="person model metadata dim must be positive"):
        LocalEmbeddingBackend(
            person_model_path=person_bundle,
            scene_model_path=scene_bundle,
            loader=StubLocalLoader(),
        )


def test_local_embedding_backend_fails_fast_on_missing_metadata_field(
    tmp_path: Path,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=2,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=2,
        files={"model": "scene.onnx"},
    )
    manifest_path = person_bundle / "manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace('"version":"v1",', ""),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="person model metadata missing required field: version",
    ):
        LocalEmbeddingBackend(
            person_model_path=person_bundle,
            scene_model_path=scene_bundle,
            loader=StubLocalLoader(),
        )


def test_local_embedding_backend_requires_face_mainline_fields(
    tmp_path: Path,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=2,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=2,
        files={"model": "scene.onnx"},
    )
    manifest_path = person_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["input_size"]["recognizer"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"person model metadata missing required field: input_size\.recognizer",
    ):
        LocalEmbeddingBackend(
            person_model_path=person_bundle,
            scene_model_path=scene_bundle,
            loader=StubLocalLoader(),
        )


def test_local_embedding_backend_requires_scene_mainline_fields(
    tmp_path: Path,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=2,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=2,
        files={"model": "scene.onnx"},
    )
    manifest_path = scene_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["output_name"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="scene model metadata missing required field: output_name",
    ):
        LocalEmbeddingBackend(
            person_model_path=person_bundle,
            scene_model_path=scene_bundle,
            loader=StubLocalLoader(),
        )


@pytest.mark.parametrize(
    ("resize_mode", "match"),
    [
        (None, r"scene model metadata missing required field: preprocess\.resize_mode"),
        (
            "stretch",
            "scene model metadata preprocess.resize_mode must be "
            "'resize_shorter_center_crop'",
        ),
    ],
)
def test_local_embedding_backend_requires_supported_scene_resize_mode(
    tmp_path: Path,
    resize_mode: str | None,
    match: str,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=2,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=2,
        files={"model": "scene.onnx"},
    )
    manifest_path = scene_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if resize_mode is None:
        del manifest["preprocess"]["resize_mode"]
    else:
        manifest["preprocess"]["resize_mode"] = resize_mode
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        LocalEmbeddingBackend(
            person_model_path=person_bundle,
            scene_model_path=scene_bundle,
            loader=StubLocalLoader(),
        )


@pytest.mark.parametrize("input_size", [224, [1, 3, 224, 224], [224], [224, 0]])
def test_local_embedding_backend_requires_explicit_width_height_input_size(
    tmp_path: Path,
    input_size,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=2,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=2,
        files={"model": "scene.onnx"},
    )
    manifest_path = scene_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_size"] = input_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"scene model metadata input_size must be \[width, height\] "
        "with two positive integers",
    ):
        LocalEmbeddingBackend(
            person_model_path=person_bundle,
            scene_model_path=scene_bundle,
            loader=StubLocalLoader(),
        )


def test_local_embedding_backend_fails_fast_on_declared_missing_file(
    tmp_path: Path,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=2,
        files={"detector": "detector.onnx", "recognizer": "missing.onnx"},
    )
    (person_bundle / "missing.onnx").unlink()
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=2,
        files={"model": "scene.onnx"},
    )

    with pytest.raises(
        ValueError,
        match="person model metadata declares missing file: missing.onnx",
    ):
        LocalEmbeddingBackend(
            person_model_path=person_bundle,
            scene_model_path=scene_bundle,
            loader=StubLocalLoader(),
        )


def test_local_embedding_backend_fails_fast_on_missing_or_unknown_paths(
    tmp_path: Path,
) -> None:
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=2,
        files={"model": "scene.onnx"},
    )

    with pytest.raises(ValueError, match="person_model_path is required"):
        LocalEmbeddingBackend(
            person_model_path=None,
            scene_model_path=scene_bundle,
            loader=StubLocalLoader(),
        )

    with pytest.raises(ValueError, match="person_model_path does not exist"):
        LocalEmbeddingBackend(
            person_model_path=tmp_path / "missing-person-bundle",
            scene_model_path=scene_bundle,
            loader=StubLocalLoader(),
        )


def test_onnx_face_loader_letterboxes_scrfd_and_aligns_arcface_input(
    tmp_path: Path,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=3,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=4,
        files={"model": "scene.onnx"},
    )
    detector = FakeOnnxSession(
        input_name="scrfd_input",
        output_names=(
            "score_8",
            "score_16",
            "score_32",
            "bbox_8",
            "bbox_16",
            "bbox_32",
            "kps_8",
            "kps_16",
            "kps_32",
        ),
        runner=lambda output_names, feeds: _scrfd_9_outputs(),
    )
    recognizer = FakeOnnxSession(
        input_name="arcface_input",
        output_names=("face_embedding",),
        runner=lambda output_names, feeds: [np.array([[1.0, 2.0, 2.0]], dtype=np.float32)],
    )
    scene = FakeOnnxSession(
        input_name="mobileclip_image",
        output_names=("mobileclip_embedding",),
        runner=lambda output_names, feeds: [np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)],
    )
    loader = _fake_onnx_loader(
        person_bundle,
        scene_bundle,
        {
            "detector.onnx": detector,
            "recognizer.onnx": recognizer,
            "scene.onnx": scene,
        },
    )
    image_bytes = _rgb_png_bytes(128, 64)
    image = loader._decode_rgb(image_bytes)

    detections = loader._detect_faces(image)
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.95)
    assert detections[0].bbox == pytest.approx((32.0, 0.0, 96.0, 64.0))
    np.testing.assert_allclose(
        detections[0].landmarks,
        (
            (48.0, 16.0),
            (80.0, 16.0),
            (64.0, 32.0),
            (48.0, 48.0),
            (80.0, 48.0),
        ),
    )
    detector_input = detector.calls[0]["feeds"]["scrfd_input"]
    assert detector_input.shape == (1, 3, 64, 64)
    np.testing.assert_allclose(detector_input[:, :, :16, :], -1.0)
    np.testing.assert_allclose(detector_input[:, :, 48:, :], -1.0)
    assert not np.allclose(detector_input[:, :, 16:48, :], -1.0)

    embedding = loader.embed_person(image_bytes)

    assert embedding.vector == [1.0, 2.0, 2.0]
    assert embedding.quality == pytest.approx(0.95)
    face_detection = embedding.metadata["face_detection"]
    assert face_detection["coordinate_space"] == "crop"
    assert face_detection["face_bbox_xyxy"] == pytest.approx([32.0, 0.0, 96.0, 64.0])
    np.testing.assert_allclose(
        face_detection["landmarks_5"],
        [
            [48.0, 16.0],
            [80.0, 16.0],
            [64.0, 32.0],
            [48.0, 48.0],
            [80.0, 48.0],
        ],
    )
    assert face_detection["score"] == pytest.approx(0.95)
    assert face_detection["source"] == "local_embedding_scrfd"
    assert len(recognizer.calls) == 1
    aligned = recognizer.calls[0]["feeds"]["arcface_input"]
    assert aligned.shape == (1, 3, 112, 112)
    direct = _rgb_nchw(image.resize((112, 112)))
    assert not np.allclose(aligned, direct)


def test_onnx_face_loader_rejects_scrfd_detector_without_kps_contract(
    tmp_path: Path,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=3,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=4,
        files={"model": "scene.onnx"},
    )
    detector = FakeOnnxSession(
        input_name="scrfd_input",
        output_names=("score_8", "score_16", "score_32", "bbox_8", "bbox_16", "bbox_32"),
        runner=lambda output_names, feeds: _scrfd_6_outputs(),
    )
    recognizer = FakeOnnxSession(
        input_name="arcface_input",
        output_names=("face_embedding",),
        runner=lambda output_names, feeds: [np.array([[1.0, 2.0, 2.0]], dtype=np.float32)],
    )
    scene = FakeOnnxSession(
        input_name="mobileclip_image",
        output_names=("mobileclip_embedding",),
        runner=lambda output_names, feeds: [np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)],
    )
    with pytest.raises(RuntimeError, match="SCRFD detector must expose 9 outputs"):
        _fake_onnx_loader(
            person_bundle,
            scene_bundle,
            {
                "detector.onnx": detector,
                "recognizer.onnx": recognizer,
                "scene.onnx": scene,
            },
        )
    assert recognizer.calls == []


def test_onnx_scene_loader_uses_declared_names_preprocess_and_center_crop(
    tmp_path: Path,
) -> None:
    person_bundle = _write_bundle(
        tmp_path,
        "person",
        dim=3,
        files={"detector": "detector.onnx", "recognizer": "recognizer.onnx"},
    )
    scene_bundle = _write_bundle(
        tmp_path,
        "scene",
        dim=4,
        files={"model": "scene.onnx"},
    )
    detector = FakeOnnxSession(
        input_name="scrfd_input",
        output_names=(
            "score_8",
            "score_16",
            "score_32",
            "bbox_8",
            "bbox_16",
            "bbox_32",
            "kps_8",
            "kps_16",
            "kps_32",
        ),
        runner=lambda output_names, feeds: _scrfd_9_outputs(),
    )
    recognizer = FakeOnnxSession(
        input_name="arcface_input",
        output_names=("face_embedding",),
        runner=lambda output_names, feeds: [np.array([[1.0, 0.0, 0.0]], dtype=np.float32)],
    )
    scene = FakeOnnxSession(
        input_name="mobileclip_image",
        output_names=("unused_first", "mobileclip_embedding"),
        runner=lambda output_names, feeds: [
            np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
        ],
    )
    loader = _fake_onnx_loader(
        person_bundle,
        scene_bundle,
        {
            "detector.onnx": detector,
            "recognizer.onnx": recognizer,
            "scene.onnx": scene,
        },
    )
    image_bytes = _rgb_png_bytes(
        4,
        2,
        pixels=(
            (0, 0, 0),
            (10, 20, 30),
            (40, 50, 60),
            (70, 80, 90),
            (100, 110, 120),
            (130, 140, 150),
            (160, 170, 180),
            (190, 200, 210),
        ),
    )

    embedding = loader.embed_scene(image_bytes)

    assert embedding.vector == [3.0, 4.0, 0.0, 0.0]
    assert len(scene.calls) == 1
    assert scene.calls[0]["output_names"] == ["mobileclip_embedding"]
    assert set(scene.calls[0]["feeds"]) == {"mobileclip_image"}
    actual = scene.calls[0]["feeds"]["mobileclip_image"]
    expected = _scene_expected_tensor(
        pixels=(
            (10, 20, 30),
            (40, 50, 60),
            (130, 140, 150),
            (160, 170, 180),
        ),
        mean=(0.1, 0.2, 0.3),
        std=(0.2, 0.4, 0.5),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def _write_bundle(
    tmp_path: Path,
    name: str,
    *,
    dim: int,
    files: dict[str, str],
    model_name: str | None = None,
    version: str = "v1",
) -> Path:
    bundle_path = tmp_path / name
    bundle_path.mkdir()
    for relative_path in files.values():
        (bundle_path / relative_path).write_bytes(b"dummy onnx")
    manifest = {
        "model_name": model_name or f"{name}-model",
        "version": version,
        "dim": dim,
        "runtime": "onnxruntime",
        "files": files,
    }
    if set(files) == {"detector", "recognizer"}:
        manifest["input_size"] = {
            "detector": [64, 64],
            "recognizer": [112, 112],
        }
    else:
        manifest.update(
            {
                "input_size": [2, 2],
                "input_name": "mobileclip_image",
                "output_name": "mobileclip_embedding",
                "preprocess": {
                    "resize_mode": "resize_shorter_center_crop",
                    "mean": [0.1, 0.2, 0.3],
                    "std": [0.2, 0.4, 0.5],
                },
            }
        )
    (bundle_path / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")),
        encoding="utf-8",
    )
    return bundle_path


class FakeOnnxNode:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeOnnxSession:
    def __init__(self, *, input_name: str, output_names: tuple[str, ...], runner) -> None:
        self.input_name = input_name
        self.output_names = output_names
        self.runner = runner
        self.calls: list[dict] = []

    def get_inputs(self) -> list[FakeOnnxNode]:
        return [FakeOnnxNode(self.input_name)]

    def get_outputs(self) -> list[FakeOnnxNode]:
        return [FakeOnnxNode(name) for name in self.output_names]

    def run(self, output_names, feeds):
        self.calls.append({"output_names": output_names, "feeds": feeds})
        return self.runner(output_names, feeds)


def _fake_onnx_loader(
    person_bundle: Path,
    scene_bundle: Path,
    sessions_by_file: dict[str, FakeOnnxSession],
) -> _OnnxLocalEmbeddingLoader:
    return _OnnxLocalEmbeddingLoader(
        _load_face_bundle(
            person_bundle,
            config_key="person_model_path",
            label="person model",
        ),
        _load_scene_bundle(
            scene_bundle,
            config_key="scene_model_path",
            label="scene model",
        ),
        session_factory=lambda model_path: sessions_by_file[Path(model_path).name],
    )


def _scrfd_9_outputs() -> list[np.ndarray]:
    scores, boxes, kps = _scrfd_base_outputs(include_kps=True)
    return scores + boxes + kps


def _scrfd_6_outputs() -> list[np.ndarray]:
    scores, boxes, _kps = _scrfd_base_outputs(include_kps=False)
    return scores + boxes


def _scrfd_base_outputs(*, include_kps: bool) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    counts = (128, 32, 8)
    scores = [np.zeros((1, count, 1), dtype=np.float32) for count in counts]
    boxes = [np.zeros((1, count, 4), dtype=np.float32) for count in counts]
    kps = [np.zeros((1, count, 10), dtype=np.float32) for count in counts]
    anchor_index = (4 * 8 + 4) * 2
    scores[0][0, anchor_index, 0] = 0.95
    boxes[0][0, anchor_index] = [2.0, 2.0, 2.0, 2.0]
    if include_kps:
        kps[0][0, anchor_index] = [
            -1.0,
            -1.0,
            1.0,
            -1.0,
            0.0,
            0.0,
            -1.0,
            1.0,
            1.0,
            1.0,
        ]

    suppressed_index = anchor_index + 1
    scores[0][0, suppressed_index, 0] = 0.90
    boxes[0][0, suppressed_index] = [2.0, 2.0, 2.0, 2.0]
    if include_kps:
        kps[0][0, suppressed_index] = kps[0][0, anchor_index]
    return scores, boxes, kps


def _rgb_png_bytes(
    width: int,
    height: int,
    *,
    pixels: tuple[tuple[int, int, int], ...] | None = None,
) -> bytes:
    image = Image.new("RGB", (width, height))
    if pixels is None:
        image.putdata(
            [
                ((x * 3 + y * 5) % 256, (x * 7) % 256, (y * 11) % 256)
                for y in range(height)
                for x in range(width)
            ]
        )
    else:
        image.putdata(pixels)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _rgb_nchw(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    array = (array / 255.0 - 0.5) / 0.5
    return array.transpose(2, 0, 1)[np.newaxis, :, :, :]


def _scene_expected_tensor(
    *,
    pixels: tuple[tuple[int, int, int], ...],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    array = np.asarray(pixels, dtype=np.float32).reshape(2, 2, 3) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std,
        dtype=np.float32,
    )
    return array.transpose(2, 0, 1)[np.newaxis, :, :, :]
