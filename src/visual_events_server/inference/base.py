from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from visual_events_server.protocol import FrameMessage

BBoxXYXY = tuple[float, float, float, float]

COCO17_KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass(frozen=True)
class PoseKeypoint:
    name: str
    x: float
    y: float
    confidence: float | None


@dataclass(frozen=True)
class PersonPoseDetection:
    bbox_xyxy: BBoxXYXY
    bbox_area: float
    confidence: float
    keypoints: list[PoseKeypoint]


@dataclass(frozen=True)
class PoseDetections:
    persons: list[PersonPoseDetection]


class InferBackend(Protocol):
    async def infer(self, frame: FrameMessage) -> PoseDetections:
        ...


def clip_bbox(bbox: BBoxXYXY, image_width: int, image_height: int) -> BBoxXYXY:
    x1, y1, x2, y2 = bbox
    width = float(image_width)
    height = float(image_height)
    return (
        min(max(float(x1), 0.0), width),
        min(max(float(y1), 0.0), height),
        min(max(float(x2), 0.0), width),
        min(max(float(y2), 0.0), height),
    )


def bbox_area(bbox: BBoxXYXY) -> float:
    x1, y1, x2, y2 = bbox
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    return width * height
