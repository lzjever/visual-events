from __future__ import annotations

import hashlib
import io
import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


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


class _LocalEmbeddingLoader(Protocol):
    def embed_person(self, image_crop: bytes) -> Any:
        ...

    def embed_scene(self, image_or_crop: bytes) -> Any:
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


class LocalEmbeddingBackend:
    def __init__(
        self,
        *,
        person_model_path: str | Path | None,
        scene_model_path: str | Path | None,
        loader: _LocalEmbeddingLoader | None = None,
    ) -> None:
        self._person_bundle = _load_face_bundle(
            person_model_path,
            config_key="person_model_path",
            label="person model",
        )
        self._scene_bundle = _load_scene_bundle(
            scene_model_path,
            config_key="scene_model_path",
            label="scene model",
        )
        self.person_dim = self._person_bundle.dim
        self.scene_dim = self._scene_bundle.dim
        self.person_model = self._person_bundle.model_name
        self.scene_model = self._scene_bundle.model_name
        self.person_version = self._person_bundle.version
        self.scene_version = self._scene_bundle.version
        self._loader = loader or _OnnxLocalEmbeddingLoader(
            self._person_bundle,
            self._scene_bundle,
        )

    def embed_person(self, image_crop: bytes) -> EmbeddingResult:
        try:
            output = self._loader.embed_person(image_crop)
        except EmbeddingUnavailable:
            raise
        except Exception as exc:
            raise EmbeddingUnavailable(
                "embedding_runtime_error",
                f"person embedding inference failed: {exc}",
            ) from exc
        vector, quality = _coerce_loader_output(
            output,
            expected_dim=self.person_dim,
            label="person",
        )
        return EmbeddingResult(
            vector=vector,
            embedding_type="face",
            embedding_model=self.person_model,
            embedding_version=self.person_version,
            quality=quality,
        )

    def embed_scene(self, image_or_crop: bytes) -> EmbeddingResult:
        try:
            output = self._loader.embed_scene(image_or_crop)
        except EmbeddingUnavailable:
            raise
        except Exception as exc:
            raise EmbeddingUnavailable(
                "embedding_runtime_error",
                f"scene embedding inference failed: {exc}",
            ) from exc
        vector, quality = _coerce_loader_output(
            output,
            expected_dim=self.scene_dim,
            label="scene",
        )
        return EmbeddingResult(
            vector=vector,
            embedding_type="scene",
            embedding_model=self.scene_model,
            embedding_version=self.scene_version,
            quality=quality,
        )


@dataclass(frozen=True)
class _FaceBundle:
    root: Path
    manifest_path: Path
    model_name: str
    version: str
    dim: int
    runtime: str
    detector_path: Path
    recognizer_path: Path
    detector_input_size: tuple[int, int]
    recognizer_input_size: tuple[int, int]


@dataclass(frozen=True)
class _ScenePreprocess:
    resize_mode: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class _SceneBundle:
    root: Path
    manifest_path: Path
    model_name: str
    version: str
    dim: int
    runtime: str
    model_path: Path
    input_size: tuple[int, int]
    input_name: str
    output_name: str
    preprocess: _ScenePreprocess


@dataclass(frozen=True)
class _DetectedFace:
    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...] | None
    score: float


@dataclass(frozen=True)
class _LetterboxTransform:
    scale: float
    pad_x: float
    pad_y: float


@dataclass(frozen=True)
class _LoaderEmbedding:
    vector: list[float]
    quality: float = 1.0


class _OnnxLocalEmbeddingLoader:
    _SCRFD_STRIDES = (8, 16, 32)
    _SCRFD_ANCHORS_PER_POINT = 2
    _FACE_SCORE_THRESHOLD = 0.5
    _FACE_NMS_THRESHOLD = 0.4
    _ARCFACE_TEMPLATE_112 = (
        (38.2946, 51.6963),
        (73.5318, 51.5014),
        (56.0252, 71.7366),
        (41.5493, 92.3655),
        (70.7299, 92.2041),
    )

    def __init__(
        self,
        person_bundle: _FaceBundle,
        scene_bundle: _SceneBundle,
        *,
        session_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on optional extra.
            raise RuntimeError(
                "memory local embedding backend requires the optional "
                "'memory-local' dependencies: pillow and numpy"
            ) from exc

        self._np = np
        self._image = Image
        self._person_bundle = person_bundle
        self._scene_bundle = scene_bundle
        self._session_factory = session_factory or self._default_session_factory()
        self._face_detector = self._create_session(person_bundle.detector_path)
        self._face_recognizer = self._create_session(person_bundle.recognizer_path)
        self._scene_encoder = self._create_session(scene_bundle.model_path)
        self._validate_face_session_contract()
        self._validate_scene_session_contract()

    def embed_person(self, image_crop: bytes) -> _LoaderEmbedding:
        image = self._decode_rgb(image_crop)
        detections = self._detect_faces(image)
        if not detections:
            raise EmbeddingUnavailable(
                "no_usable_face",
                "no usable face detected in person crop",
            )
        face = detections[0]
        if face.landmarks is None or len(face.landmarks) != 5:
            raise EmbeddingUnavailable(
                "no_usable_face",
                "no usable face detected with five landmarks in person crop",
            )
        aligned = self._align_arcface(image, face.landmarks)
        recognizer_input = self._to_arcface_nchw(aligned)
        recognizer_outputs = self._run_single_input_session(
            self._face_recognizer,
            recognizer_input,
        )
        return _LoaderEmbedding(
            vector=self._single_embedding_output(
                recognizer_outputs,
                label="ArcFace recognizer",
            ),
            quality=face.score,
        )

    def embed_scene(self, image_or_crop: bytes) -> _LoaderEmbedding:
        image = self._decode_rgb(image_or_crop)
        scene_input = self._to_mobileclip_nchw(image)
        scene_outputs = self._scene_encoder.run(
            [self._scene_bundle.output_name],
            {self._scene_bundle.input_name: scene_input},
        )
        return _LoaderEmbedding(
            vector=self._single_embedding_output(
                list(scene_outputs),
                label="MobileCLIP scene encoder",
            )
        )

    def _detect_faces(self, image: Any) -> list[_DetectedFace]:
        detector_input, transform = self._to_scrfd_nchw(image)
        detector_outputs = self._run_single_input_session(
            self._face_detector,
            detector_input,
        )
        decoded = self._decode_scrfd_outputs(detector_outputs)
        scaled = self._scale_faces_to_image(decoded, image.size, transform)
        return self._nms_faces(scaled)

    def _default_session_factory(self) -> Callable[[Path], Any]:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional extra.
            raise RuntimeError(
                "memory local embedding backend requires the optional "
                "'memory-local' dependency: onnxruntime"
            ) from exc

        def create(model_path: Path) -> Any:
            return ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

        return create

    def _create_session(self, model_path: Path) -> Any:
        try:
            return self._session_factory(model_path)
        except Exception as exc:
            raise RuntimeError(f"failed to load ONNX model file {model_path}: {exc}") from exc

    def _decode_rgb(self, image_bytes: bytes) -> Any:
        try:
            with self._image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB")
        except Exception as exc:
            raise EmbeddingUnavailable(
                "invalid_image",
                "image bytes must be decodable",
            ) from exc

    def _to_scrfd_nchw(self, image: Any) -> tuple[Any, _LetterboxTransform]:
        letterboxed, transform = self._letterbox_image(
            image,
            self._person_bundle.detector_input_size,
        )
        return self._unit_nchw_from_rgb(letterboxed), transform

    def _to_arcface_nchw(self, image: Any) -> Any:
        return self._to_unit_nchw(image, self._person_bundle.recognizer_input_size)

    def _to_unit_nchw(self, image: Any, size: tuple[int, int]) -> Any:
        resized = image.resize(size, resample=self._resample_bilinear())
        return self._unit_nchw_from_rgb(resized)

    def _unit_nchw_from_rgb(self, image: Any) -> Any:
        array = self._np.asarray(image, dtype=self._np.float32)
        array = (array / 255.0 - 0.5) / 0.5
        return array.transpose(2, 0, 1)[self._np.newaxis, :, :, :]

    def _to_mobileclip_nchw(self, image: Any) -> Any:
        resized = self._resize_shorter_center_crop(image, self._scene_bundle.input_size)
        array = self._np.asarray(resized, dtype=self._np.float32) / 255.0
        mean = self._np.asarray(self._scene_bundle.preprocess.mean, dtype=self._np.float32)
        std = self._np.asarray(self._scene_bundle.preprocess.std, dtype=self._np.float32)
        array = (array - mean) / std
        return array.transpose(2, 0, 1)[self._np.newaxis, :, :, :]

    def _letterbox_image(
        self,
        image: Any,
        size: tuple[int, int],
    ) -> tuple[Any, _LetterboxTransform]:
        target_width, target_height = size
        source_width, source_height = image.size
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, min(target_width, int(round(source_width * scale))))
        resized_height = max(1, min(target_height, int(round(source_height * scale))))
        pad_x = (target_width - resized_width) // 2
        pad_y = (target_height - resized_height) // 2
        resized = image.resize(
            (resized_width, resized_height),
            resample=self._resample_bilinear(),
        )
        letterboxed = self._image.new("RGB", size, (0, 0, 0))
        letterboxed.paste(resized, (pad_x, pad_y))
        return letterboxed, _LetterboxTransform(
            scale=scale,
            pad_x=float(pad_x),
            pad_y=float(pad_y),
        )

    def _resize_shorter_center_crop(self, image: Any, size: tuple[int, int]) -> Any:
        target_width, target_height = size
        source_width, source_height = image.size
        scale = max(target_width / source_width, target_height / source_height)
        resized_width = max(target_width, int(round(source_width * scale)))
        resized_height = max(target_height, int(round(source_height * scale)))
        resized = image.resize(
            (resized_width, resized_height),
            resample=self._resample_bilinear(),
        )
        left = (resized_width - target_width) // 2
        top = (resized_height - target_height) // 2
        return resized.crop((left, top, left + target_width, top + target_height))

    def _run_single_input_session(self, session: Any, input_tensor: Any) -> list[Any]:
        input_name = session.get_inputs()[0].name
        return list(session.run(None, {input_name: input_tensor}))

    def _decode_scrfd_outputs(self, outputs: list[Any]) -> list[_DetectedFace]:
        if len(outputs) not in (6, 9):
            raise RuntimeError(
                "SCRFD detector must produce 6 or 9 outputs "
                "(scores, bbox_preds, optional kps_preds for strides 8/16/32)"
            )
        use_kps = len(outputs) == 9
        score_outputs = outputs[:3]
        bbox_outputs = outputs[3:6]
        kps_outputs = outputs[6:9] if use_kps else (None, None, None)
        detections: list[_DetectedFace] = []
        for index, stride in enumerate(self._SCRFD_STRIDES):
            scores = self._np.asarray(score_outputs[index], dtype=self._np.float32)
            bbox_preds = self._np.asarray(bbox_outputs[index], dtype=self._np.float32)
            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            if scores.shape[0] != bbox_preds.shape[0]:
                raise RuntimeError(
                    f"SCRFD stride {stride} score and bbox output sizes differ"
                )
            centers = self._scrfd_anchor_centers(stride, bbox_preds.shape[0])
            bboxes = self._distance_to_bbox(centers, bbox_preds * float(stride))
            landmarks_by_anchor = None
            if use_kps:
                kps_preds = self._np.asarray(
                    kps_outputs[index],
                    dtype=self._np.float32,
                ).reshape(-1, 5, 2)
                if kps_preds.shape[0] != bbox_preds.shape[0]:
                    raise RuntimeError(
                        f"SCRFD stride {stride} score and kps output sizes differ"
                    )
                landmarks_by_anchor = centers[:, self._np.newaxis, :] + (
                    kps_preds * float(stride)
                )
            for anchor_index in self._np.where(scores >= self._FACE_SCORE_THRESHOLD)[0]:
                score = float(scores[anchor_index])
                if not math.isfinite(score):
                    continue
                landmarks = None
                if landmarks_by_anchor is not None:
                    landmarks = tuple(
                        (float(point[0]), float(point[1]))
                        for point in landmarks_by_anchor[anchor_index]
                    )
                box = bboxes[anchor_index]
                detections.append(
                    _DetectedFace(
                        bbox=(
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        ),
                        landmarks=landmarks,
                        score=score,
                    )
                )
        return detections

    def _scrfd_anchor_centers(self, stride: int, expected_count: int) -> Any:
        detector_width, detector_height = self._person_bundle.detector_input_size
        feat_width = detector_width // stride
        feat_height = detector_height // stride
        centers = self._np.stack(
            self._np.mgrid[:feat_height, :feat_width][::-1],
            axis=-1,
        ).astype(self._np.float32)
        centers = (centers * float(stride)).reshape(-1, 2)
        centers = self._np.repeat(
            centers,
            self._SCRFD_ANCHORS_PER_POINT,
            axis=0,
        )
        if centers.shape[0] != expected_count:
            raise RuntimeError(
                f"SCRFD stride {stride} expected {centers.shape[0]} anchors, "
                f"got {expected_count}"
            )
        return centers

    def _distance_to_bbox(self, centers: Any, distances: Any) -> Any:
        return self._np.stack(
            (
                centers[:, 0] - distances[:, 0],
                centers[:, 1] - distances[:, 1],
                centers[:, 0] + distances[:, 2],
                centers[:, 1] + distances[:, 3],
            ),
            axis=1,
        )

    def _scale_faces_to_image(
        self,
        faces: list[_DetectedFace],
        image_size: tuple[int, int],
        transform: _LetterboxTransform,
    ) -> list[_DetectedFace]:
        image_width, image_height = image_size
        scaled: list[_DetectedFace] = []

        def project(point: tuple[float, float]) -> tuple[float, float]:
            x, y = point
            return (
                (x - transform.pad_x) / transform.scale,
                (y - transform.pad_y) / transform.scale,
            )

        for face in faces:
            raw_x1, raw_y1 = project((face.bbox[0], face.bbox[1]))
            raw_x2, raw_y2 = project((face.bbox[2], face.bbox[3]))
            x1 = max(0.0, min(float(image_width), raw_x1))
            y1 = max(0.0, min(float(image_height), raw_y1))
            x2 = max(0.0, min(float(image_width), raw_x2))
            y2 = max(0.0, min(float(image_height), raw_y2))
            if x2 <= x1 or y2 <= y1:
                continue
            landmarks = None
            if face.landmarks is not None:
                landmarks = tuple(project((x, y)) for x, y in face.landmarks)
            scaled.append(
                _DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    landmarks=landmarks,
                    score=face.score,
                )
            )
        return scaled

    def _nms_faces(self, faces: list[_DetectedFace]) -> list[_DetectedFace]:
        remaining = sorted(faces, key=lambda face: face.score, reverse=True)
        kept: list[_DetectedFace] = []
        while remaining:
            current = remaining.pop(0)
            kept.append(current)
            remaining = [
                face
                for face in remaining
                if _bbox_iou(current.bbox, face.bbox) <= self._FACE_NMS_THRESHOLD
            ]
        return kept

    def _align_arcface(
        self,
        image: Any,
        landmarks: tuple[tuple[float, float], ...],
    ) -> Any:
        width, height = self._person_bundle.recognizer_input_size
        src = self._np.asarray(landmarks, dtype=self._np.float64)
        dst = self._np.asarray(self._ARCFACE_TEMPLATE_112, dtype=self._np.float64)
        dst[:, 0] *= width / 112.0
        dst[:, 1] *= height / 112.0
        transform = self._similarity_from_points(src, dst)
        inverse = self._np.linalg.inv(transform)
        coeffs = (
            float(inverse[0, 0]),
            float(inverse[0, 1]),
            float(inverse[0, 2]),
            float(inverse[1, 0]),
            float(inverse[1, 1]),
            float(inverse[1, 2]),
        )
        return image.transform(
            (width, height),
            self._image.Transform.AFFINE,
            coeffs,
            resample=self._resample_bilinear(),
        )

    def _similarity_from_points(self, src: Any, dst: Any) -> Any:
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        src_variance = float((src_centered * src_centered).sum() / src.shape[0])
        if src_variance <= 0.0 or not math.isfinite(src_variance):
            raise EmbeddingUnavailable(
                "no_usable_face",
                "face landmarks could not be aligned",
            )
        covariance = (dst_centered.T @ src_centered) / src.shape[0]
        left, singular_values, right_t = self._np.linalg.svd(covariance)
        direction = self._np.ones(2, dtype=self._np.float64)
        if self._np.linalg.det(left) * self._np.linalg.det(right_t) < 0.0:
            direction[-1] = -1.0
        rotation = left @ self._np.diag(direction) @ right_t
        scale = float((singular_values * direction).sum() / src_variance)
        translation = dst_mean - scale * (rotation @ src_mean)
        transform = self._np.eye(3, dtype=self._np.float64)
        transform[:2, :2] = scale * rotation
        transform[:2, 2] = translation
        return transform

    def _single_embedding_output(self, outputs: list[Any], *, label: str) -> list[float]:
        if len(outputs) != 1:
            raise RuntimeError(f"{label} must produce exactly one embedding output")
        array = self._np.asarray(outputs[0], dtype=self._np.float32)
        if array.size == 0:
            raise RuntimeError(f"{label} produced an empty embedding output")
        return [float(value) for value in array.reshape(-1).tolist()]

    def _validate_face_session_contract(self) -> None:
        output_names = [item.name for item in self._face_detector.get_outputs()]
        if len(output_names) != 9:
            raise RuntimeError(
                "SCRFD detector must expose 9 outputs including kps for strides "
                "8/16/32"
            )

    def _validate_scene_session_contract(self) -> None:
        input_names = [item.name for item in self._scene_encoder.get_inputs()]
        if self._scene_bundle.input_name not in input_names:
            raise RuntimeError(
                "MobileCLIP scene encoder input "
                f"{self._scene_bundle.input_name!r} is not present in ONNX model"
            )
        output_names = [item.name for item in self._scene_encoder.get_outputs()]
        if self._scene_bundle.output_name not in output_names:
            raise RuntimeError(
                "MobileCLIP scene encoder output "
                f"{self._scene_bundle.output_name!r} is not present in ONNX model"
            )

    def _resample_bilinear(self) -> Any:
        return getattr(self._image, "Resampling", self._image).BILINEAR


def _load_face_bundle(
    bundle_path: str | Path | None,
    *,
    config_key: str,
    label: str,
) -> _FaceBundle:
    data, manifest_path, root = _load_bundle_manifest(
        bundle_path,
        config_key=config_key,
        label=label,
    )
    base = _parse_bundle_base(data, label=label)
    files = _metadata_mapping(data["files"], field="files", label=label)
    input_size = _metadata_mapping(data["input_size"], field="input_size", label=label)
    detector_path = _required_bundle_file(
        files,
        key="detector",
        root=root,
        label=label,
    )
    recognizer_path = _required_bundle_file(
        files,
        key="recognizer",
        root=root,
        label=label,
    )
    detector_input_size = _required_input_size(
        input_size,
        key="detector",
        label=label,
    )
    recognizer_input_size = _required_input_size(
        input_size,
        key="recognizer",
        label=label,
    )
    if recognizer_input_size != (112, 112):
        raise ValueError(
            f"{label} metadata input_size.recognizer must be [112, 112] "
            "for ArcFace alignment"
        )
    return _FaceBundle(
        root=root,
        manifest_path=manifest_path,
        model_name=base["model_name"],
        version=base["version"],
        dim=base["dim"],
        runtime=base["runtime"],
        detector_path=detector_path,
        recognizer_path=recognizer_path,
        detector_input_size=detector_input_size,
        recognizer_input_size=recognizer_input_size,
    )


def _load_scene_bundle(
    bundle_path: str | Path | None,
    *,
    config_key: str,
    label: str,
) -> _SceneBundle:
    data, manifest_path, root = _load_bundle_manifest(
        bundle_path,
        config_key=config_key,
        label=label,
    )
    required_fields = ("input_name", "output_name", "preprocess")
    missing = [field for field in required_fields if field not in data]
    if missing:
        _raise_missing_fields(label, missing)
    base = _parse_bundle_base(data, label=label)
    files = _metadata_mapping(data["files"], field="files", label=label)
    preprocess = _metadata_mapping(
        data["preprocess"],
        field="preprocess",
        label=label,
    )
    if "mean" not in preprocess:
        raise ValueError(f"{label} metadata missing required field: preprocess.mean")
    if "std" not in preprocess:
        raise ValueError(f"{label} metadata missing required field: preprocess.std")
    if "resize_mode" not in preprocess:
        raise ValueError(
            f"{label} metadata missing required field: preprocess.resize_mode"
        )
    resize_mode = _metadata_text(
        preprocess["resize_mode"],
        field="preprocess.resize_mode",
        label=label,
    )
    if resize_mode != "resize_shorter_center_crop":
        raise ValueError(
            f"{label} metadata preprocess.resize_mode must be "
            "'resize_shorter_center_crop'"
        )
    return _SceneBundle(
        root=root,
        manifest_path=manifest_path,
        model_name=base["model_name"],
        version=base["version"],
        dim=base["dim"],
        runtime=base["runtime"],
        model_path=_required_bundle_file(
            files,
            key="model",
            root=root,
            label=label,
        ),
        input_size=_parse_input_size(data["input_size"], label=label),
        input_name=_metadata_text(
            data["input_name"],
            field="input_name",
            label=label,
        ),
        output_name=_metadata_text(
            data["output_name"],
            field="output_name",
            label=label,
        ),
        preprocess=_ScenePreprocess(
            resize_mode=resize_mode,
            mean=_metadata_float_triplet(
                preprocess["mean"],
                field="preprocess.mean",
                label=label,
            ),
            std=_metadata_float_triplet(
                preprocess["std"],
                field="preprocess.std",
                label=label,
                positive=True,
            ),
        ),
    )


def _load_bundle_manifest(
    bundle_path: str | Path | None,
    *,
    config_key: str,
    label: str,
) -> tuple[dict[str, Any], Path, Path]:
    if bundle_path is None:
        raise ValueError(f"{config_key} is required for memory local embedding backend")
    path = Path(bundle_path)
    if not path.exists():
        raise ValueError(f"{config_key} does not exist: {path}")
    manifest_path, root = _resolve_manifest_path(path, config_key=config_key)
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} metadata must be valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} metadata must be a JSON object")
    return data, manifest_path, root


def _parse_bundle_base(data: dict[str, Any], *, label: str) -> dict[str, Any]:
    required_fields = ("model_name", "version", "dim", "runtime", "input_size", "files")
    missing = [field for field in required_fields if field not in data]
    if missing:
        _raise_missing_fields(label, missing)

    model_name = _metadata_text(data["model_name"], field="model_name", label=label)
    version = _metadata_text(data["version"], field="version", label=label)
    runtime = _metadata_text(data["runtime"], field="runtime", label=label)
    if runtime.lower() != "onnxruntime":
        raise ValueError(f"{label} metadata runtime must be 'onnxruntime'")
    dim = _metadata_positive_int(data["dim"], field="dim", label=label)
    return {
        "model_name": model_name,
        "version": version,
        "runtime": runtime,
        "dim": dim,
    }


def _raise_missing_fields(label: str, missing: list[str]) -> None:
    suffix = "field" if len(missing) == 1 else "fields"
    raise ValueError(
        f"{label} metadata missing required {suffix}: {', '.join(missing)}"
    )


def _resolve_manifest_path(path: Path, *, config_key: str) -> tuple[Path, Path]:
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            return manifest_path, path
        raise ValueError(
            f"{config_key} metadata file not found: expected manifest.json"
        )
    if path.is_file() and path.suffix.lower() == ".json":
        return path, path.parent
    raise ValueError(f"{config_key} must be a bundle directory or metadata JSON file")


def _metadata_text(value: Any, *, field: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} metadata {field} must be a non-empty string")
    return value.strip()


def _metadata_positive_int(value: Any, *, field: str, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} metadata {field} must be positive")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} metadata {field} must be positive") from exc
    if number <= 0:
        raise ValueError(f"{label} metadata {field} must be positive")
    return number


def _resolve_bundle_file(root: Path, file_text: str) -> Path:
    path = Path(file_text)
    if path.is_absolute():
        return path
    return root / path


def _metadata_mapping(value: Any, *, field: str, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} metadata {field} must be an object")
    return value


def _required_bundle_file(
    files: dict[str, Any],
    *,
    key: str,
    root: Path,
    label: str,
) -> Path:
    if key not in files:
        raise ValueError(f"{label} metadata missing required field: files.{key}")
    file_text = _metadata_text(files[key], field=f"files.{key}", label=label)
    file_path = _resolve_bundle_file(root, file_text)
    if not file_path.exists():
        raise ValueError(f"{label} metadata declares missing file: {file_text}")
    return file_path


def _required_input_size(
    input_size: dict[str, Any],
    *,
    key: str,
    label: str,
) -> tuple[int, int]:
    if key not in input_size:
        raise ValueError(f"{label} metadata missing required field: input_size.{key}")
    return _parse_input_size(input_size[key], label=label)


def _parse_input_size(value: Any, *, label: str) -> tuple[int, int]:
    message = (
        f"{label} metadata input_size must be [width, height] with two positive "
        "integers"
    )
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ValueError(message)
    if len(value) != 2:
        raise ValueError(message)
    width, height = value
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(message)
    return width, height


def _metadata_float_triplet(
    value: Any,
    *,
    field: str,
    label: str,
    positive: bool = False,
) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ValueError(f"{label} metadata {field} must contain three numbers")
    try:
        numbers = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} metadata {field} must contain three numbers"
        ) from exc
    if len(numbers) != 3 or not all(math.isfinite(item) for item in numbers):
        raise ValueError(f"{label} metadata {field} must contain three numbers")
    if positive and any(item <= 0.0 for item in numbers):
        raise ValueError(f"{label} metadata {field} values must be positive")
    return numbers


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _coerce_loader_output(
    output: Any,
    *,
    expected_dim: int,
    label: str,
) -> tuple[tuple[float, ...], float]:
    vector_like = getattr(output, "vector", output)
    try:
        vector = _flatten_numeric_vector(vector_like)
    except (TypeError, ValueError) as exc:
        raise EmbeddingUnavailable(
            "embedding_runtime_error",
            f"{label} embedding inference produced a non-numeric vector",
        ) from exc
    if len(vector) != expected_dim:
        raise EmbeddingUnavailable(
            "embedding_runtime_error",
            f"{label} embedding inference produced {len(vector)} dimensions; "
            f"expected {expected_dim}",
        )
    try:
        normalized = normalize_vector(vector)
    except ValueError as exc:
        raise EmbeddingUnavailable(
            "embedding_runtime_error",
            f"{label} embedding inference produced an invalid vector: {exc}",
        ) from exc

    quality = float(getattr(output, "quality", 1.0))
    if not math.isfinite(quality) or quality < 0.0:
        raise EmbeddingUnavailable(
            "embedding_runtime_error",
            f"{label} embedding inference produced invalid quality",
        )
    return normalized, min(quality, 1.0)


def _flatten_numeric_vector(value: Any) -> list[float]:
    flattened: list[float] = []

    def append(item: Any) -> None:
        if hasattr(item, "tolist") and not isinstance(item, (bytes, bytearray, str)):
            append(item.tolist())
            return
        if isinstance(item, Sequence) and not isinstance(
            item,
            (bytes, bytearray, str),
        ):
            for child in item:
                append(child)
            return
        flattened.append(float(item))

    append(value)
    return flattened


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
