from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

from visual_events_server.protocol import FrameMessage

from .base import (
    BBoxXYXY,
    COCO17_KEYPOINT_NAMES,
    PersonPoseDetection,
    PoseDetections,
    PoseKeypoint,
    bbox_area,
    clip_bbox,
)


class InferenceConfigError(RuntimeError):
    pass


class UltralyticsPoseBackend:
    def __init__(
        self,
        *,
        model_path: Path,
        device: str | None,
        imgsz: int,
        conf: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise InferenceConfigError(
                f"Ultralytics model file not found: {self.model_path}"
            )
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self._model: Any | None = None
        self._clock = clock or time.perf_counter
        self._last_phase_latencies_ms: dict[str, float] = {}

    async def infer(self, frame: FrameMessage) -> PoseDetections:
        self._last_phase_latencies_ms = {}
        decode_start = self._clock()
        image, image_width, image_height = _decode_jpeg(frame.jpeg_bytes)
        decode_ms = _elapsed_ms(decode_start, self._clock)

        infer_start = self._clock()
        results = self._load_model().predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        infer_ms = _elapsed_ms(infer_start, self._clock)

        postprocess_start = self._clock()
        if not results:
            detections = PoseDetections(persons=[])
        else:
            detections = result_to_pose_detections(
                results[0],
                image_width=image_width,
                image_height=image_height,
                conf_threshold=self.conf,
            )
        postprocess_ms = _elapsed_ms(postprocess_start, self._clock)
        self._last_phase_latencies_ms = {
            "decode": decode_ms,
            "infer": infer_ms,
            "postprocess": postprocess_ms,
        }
        return detections

    def consume_phase_metrics(self) -> Mapping[str, float]:
        phase_latencies = self._last_phase_latencies_ms
        self._last_phase_latencies_ms = {}
        return phase_latencies

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise InferenceConfigError(
                    "Ultralytics backend requires installing the inference extra"
                ) from exc
            self._model = YOLO(str(self.model_path))
        return self._model


def result_to_pose_detections(
    result: Any,
    *,
    image_width: int,
    image_height: int,
    conf_threshold: float,
) -> PoseDetections:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return PoseDetections(persons=[])

    xyxy_values = _to_list(getattr(boxes, "xyxy", []))
    confidences = _to_list(getattr(boxes, "conf", []))
    classes = _to_list(getattr(boxes, "cls", []))
    keypoints = getattr(result, "keypoints", None)
    keypoint_xy = _to_list(getattr(keypoints, "xy", [])) if keypoints is not None else []
    keypoint_conf = (
        _to_list(getattr(keypoints, "conf", [])) if keypoints is not None else []
    )

    persons: list[PersonPoseDetection] = []
    for index, raw_bbox in enumerate(xyxy_values):
        class_id = _value_at(classes, index, default=-1)
        if int(float(class_id)) != 0:
            continue

        confidence = float(_value_at(confidences, index, default=0.0))
        if confidence < conf_threshold:
            continue

        bbox = _as_bbox(raw_bbox)
        if bbox is None:
            continue
        clipped_bbox = clip_bbox(bbox, image_width, image_height)
        area = bbox_area(clipped_bbox)
        if area <= 0:
            continue

        persons.append(
            PersonPoseDetection(
                bbox_xyxy=clipped_bbox,
                bbox_area=area,
                confidence=confidence,
                keypoints=_build_keypoints(
                    _value_at(keypoint_xy, index, default=[]),
                    _value_at(keypoint_conf, index, default=[]),
                ),
            )
        )

    return PoseDetections(persons=persons)


def _decode_jpeg(jpeg_bytes: bytes) -> tuple[Any, int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise InferenceConfigError(
            "Ultralytics backend requires Pillow from the inference extra"
        ) from exc

    with Image.open(BytesIO(jpeg_bytes)) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
    return rgb_image, width, height


def _elapsed_ms(start: float, clock: Callable[[], float]) -> float:
    return max(0.0, (clock() - start) * 1000.0)


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    for method in ("detach", "cpu", "numpy"):
        method_value = getattr(value, method, None)
        if callable(method_value):
            value = method_value()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _as_bbox(raw_bbox: Any) -> BBoxXYXY | None:
    values = list(raw_bbox) if isinstance(raw_bbox, (list, tuple)) else []
    if len(values) < 4:
        return None
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _build_keypoints(raw_xy: Any, raw_conf: Any) -> list[PoseKeypoint]:
    xy_values = list(raw_xy) if isinstance(raw_xy, (list, tuple)) else []
    conf_values = list(raw_conf) if isinstance(raw_conf, (list, tuple)) else []
    keypoints: list[PoseKeypoint] = []
    for index, name in enumerate(COCO17_KEYPOINT_NAMES):
        if index >= len(xy_values):
            break
        point = xy_values[index]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        confidence = (
            float(conf_values[index]) if index < len(conf_values) else None
        )
        keypoints.append(
            PoseKeypoint(
                name=name,
                x=float(point[0]),
                y=float(point[1]),
                confidence=confidence,
            )
        )
    return keypoints


def _value_at(values: list[Any], index: int, *, default: Any) -> Any:
    if index >= len(values):
        return default
    return values[index]
