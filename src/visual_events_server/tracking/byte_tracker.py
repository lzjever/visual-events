from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from visual_events_server.inference.base import (
    BBoxXYXY,
    PersonPoseDetection,
    PoseDetections,
    PoseKeypoint,
    bbox_area,
)
from visual_events_server.protocol import FrameMessage

_FACE_KEYPOINT_NAMES = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}
_MIN_VELOCITY_DT_MS = 100


@dataclass(frozen=True)
class TrackingConfig:
    high_conf: float = 0.6
    low_conf: float = 0.1
    new_track_conf: float = 0.7
    match_iou: float = 0.3
    lost_ttl_ms: int = 1500
    history_ms: int = 3000
    velocity_window_ms: int = 1000

    def __post_init__(self) -> None:
        for field_name in ("high_conf", "low_conf", "new_track_conf", "match_iou"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if self.low_conf > self.high_conf:
            raise ValueError("low_conf must be <= high_conf")
        if self.lost_ttl_ms < 0:
            raise ValueError("lost_ttl_ms must be non-negative")
        if self.history_ms <= 0:
            raise ValueError("history_ms must be positive")
        if self.velocity_window_ms <= 0:
            raise ValueError("velocity_window_ms must be positive")


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    first_seen_ms: int
    last_seen_ms: int
    frame_timestamp_ms: int
    bbox_xyxy: BBoxXYXY
    confidence: float
    pose_confidence: float
    head_uv: tuple[float, float]
    velocity_uv_s: tuple[float, float]
    lost_ms: int
    hits: int
    misses: int
    class_name: str = "person"

    @property
    def age_ms(self) -> int:
        return max(0, self.frame_timestamp_ms - self.first_seen_ms)

    def to_protocol(self, *, image_width: int, image_height: int) -> dict[str, object]:
        x1, y1, x2, y2 = self.bbox_xyxy
        image_area = float(image_width) * float(image_height)
        area_ratio = bbox_area(self.bbox_xyxy) / image_area if image_area > 0 else 0.0
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_area_ratio": area_ratio,
            "center_uv": [_center_x(self.bbox_xyxy), _center_y(self.bbox_xyxy)],
            "head_uv": [float(self.head_uv[0]), float(self.head_uv[1])],
            "velocity_uv_s": [
                float(self.velocity_uv_s[0]),
                float(self.velocity_uv_s[1]),
            ],
            "age_ms": self.age_ms,
            "lost_ms": max(0, int(self.lost_ms)),
            "confidence": float(self.confidence),
            "pose_confidence": float(self.pose_confidence),
        }


@dataclass(frozen=True)
class _DetectionCandidate:
    index: int
    detection: PersonPoseDetection
    pose_confidence: float
    head_uv: tuple[float, float]

    @property
    def bbox_xyxy(self) -> BBoxXYXY:
        return self.detection.bbox_xyxy

    @property
    def confidence(self) -> float:
        return float(self.detection.confidence)


@dataclass
class _Observation:
    timestamp_ms: int
    center_uv: tuple[float, float]


@dataclass
class _TrackState:
    track_id: int
    first_seen_ms: int
    last_seen_ms: int
    bbox_xyxy: BBoxXYXY
    confidence: float
    pose_confidence: float
    head_uv: tuple[float, float]
    velocity_uv_s: tuple[float, float] = (0.0, 0.0)
    history: list[_Observation] = field(default_factory=list)
    lost_ms: int = 0
    hits: int = 1
    misses: int = 0
    class_name: str = "person"

    @classmethod
    def create(
        cls,
        *,
        track_id: int,
        timestamp_ms: int,
        candidate: _DetectionCandidate,
    ) -> _TrackState:
        center = _bbox_center(candidate.bbox_xyxy)
        return cls(
            track_id=track_id,
            first_seen_ms=timestamp_ms,
            last_seen_ms=timestamp_ms,
            bbox_xyxy=candidate.bbox_xyxy,
            confidence=candidate.confidence,
            pose_confidence=candidate.pose_confidence,
            head_uv=candidate.head_uv,
            history=[_Observation(timestamp_ms=timestamp_ms, center_uv=center)],
        )

    def update(
        self,
        *,
        timestamp_ms: int,
        candidate: _DetectionCandidate,
        velocity_window_ms: int,
    ) -> None:
        center = _bbox_center(candidate.bbox_xyxy)
        self.velocity_uv_s = self._compute_velocity(
            timestamp_ms,
            center,
            velocity_window_ms=velocity_window_ms,
        )
        self.last_seen_ms = timestamp_ms
        self.bbox_xyxy = candidate.bbox_xyxy
        self.confidence = candidate.confidence
        self.pose_confidence = candidate.pose_confidence
        self.head_uv = candidate.head_uv
        self.lost_ms = 0
        self.hits += 1
        self.misses = 0
        self.history.append(_Observation(timestamp_ms=timestamp_ms, center_uv=center))

    def mark_missed(self, *, timestamp_ms: int) -> None:
        self.lost_ms = max(0, timestamp_ms - self.last_seen_ms)
        self.misses += 1

    def prune_history(self, *, timestamp_ms: int, history_ms: int) -> None:
        oldest_ms = timestamp_ms - history_ms
        self.history = [
            observation
            for observation in self.history
            if observation.timestamp_ms >= oldest_ms
        ]

    def snapshot(self, *, frame_timestamp_ms: int) -> TrackSnapshot:
        return TrackSnapshot(
            track_id=self.track_id,
            first_seen_ms=self.first_seen_ms,
            last_seen_ms=self.last_seen_ms,
            frame_timestamp_ms=frame_timestamp_ms,
            bbox_xyxy=self.bbox_xyxy,
            confidence=self.confidence,
            pose_confidence=self.pose_confidence,
            head_uv=self.head_uv,
            velocity_uv_s=self.velocity_uv_s,
            lost_ms=self.lost_ms,
            hits=self.hits,
            misses=self.misses,
            class_name=self.class_name,
        )

    def _compute_velocity(
        self,
        timestamp_ms: int,
        center_uv: tuple[float, float],
        *,
        velocity_window_ms: int,
    ) -> tuple[float, float]:
        recent = [
            observation
            for observation in self.history
            if 0 < timestamp_ms - observation.timestamp_ms <= velocity_window_ms
        ]
        if not recent:
            return self.velocity_uv_s
        reference = recent[0]
        dt_ms = timestamp_ms - reference.timestamp_ms
        if dt_ms < _MIN_VELOCITY_DT_MS:
            return self.velocity_uv_s
        dt_s = dt_ms / 1000.0
        return (
            (center_uv[0] - reference.center_uv[0]) / dt_s,
            (center_uv[1] - reference.center_uv[1]) / dt_s,
        )


class ByteTrackStyleTracker:
    def __init__(self, *, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._tracks: list[_TrackState] = []
        self._next_track_id = 1
        self._last_timestamp_ms: int | None = None
        self._last_frame_id: int | None = None

    def reset(self) -> None:
        self._tracks = []
        self._next_track_id = 1
        self._last_timestamp_ms = None
        self._last_frame_id = None

    def update(
        self,
        frame: FrameMessage,
        detections: PoseDetections,
    ) -> list[TrackSnapshot]:
        if self._should_reset(frame):
            self.reset()

        timestamp_ms = frame.timestamp_ms
        self._drop_expired(timestamp_ms)
        candidates = [
            _candidate_from_detection(index, detection)
            for index, detection in enumerate(detections.persons)
            if detection.confidence >= self.config.low_conf
        ]

        matched_track_ids: set[int] = set()
        matched_candidate_indices: set[int] = set()
        high_indices = [
            candidate.index
            for candidate in candidates
            if candidate.confidence >= self.config.high_conf
        ]
        self._match_candidates(
            candidates,
            candidate_indices=high_indices,
            matched_track_ids=matched_track_ids,
            matched_candidate_indices=matched_candidate_indices,
            timestamp_ms=timestamp_ms,
        )

        low_indices = [
            candidate.index
            for candidate in candidates
            if self.config.low_conf <= candidate.confidence < self.config.high_conf
        ]
        self._match_candidates(
            candidates,
            candidate_indices=low_indices,
            matched_track_ids=matched_track_ids,
            matched_candidate_indices=matched_candidate_indices,
            timestamp_ms=timestamp_ms,
        )

        for track in self._tracks:
            if track.track_id not in matched_track_ids:
                track.mark_missed(timestamp_ms=timestamp_ms)

        self._drop_expired(timestamp_ms)
        for candidate in candidates:
            if candidate.index in matched_candidate_indices:
                continue
            if candidate.confidence < self.config.new_track_conf:
                continue
            self._tracks.append(
                _TrackState.create(
                    track_id=self._next_track_id,
                    timestamp_ms=timestamp_ms,
                    candidate=candidate,
                )
            )
            self._next_track_id += 1

        for track in self._tracks:
            track.prune_history(
                timestamp_ms=timestamp_ms,
                history_ms=self.config.history_ms,
            )

        self._last_timestamp_ms = timestamp_ms
        self._last_frame_id = frame.frame_id
        return [
            track.snapshot(frame_timestamp_ms=timestamp_ms)
            for track in sorted(self._tracks, key=lambda item: item.track_id)
        ]

    def _match_candidates(
        self,
        candidates: list[_DetectionCandidate],
        *,
        candidate_indices: Iterable[int],
        matched_track_ids: set[int],
        matched_candidate_indices: set[int],
        timestamp_ms: int,
    ) -> None:
        candidate_index_set = set(candidate_indices)
        if not candidate_index_set:
            return
        candidate_by_index = {candidate.index: candidate for candidate in candidates}
        pairs: list[tuple[float, int, int]] = []
        for track in self._tracks:
            if track.track_id in matched_track_ids:
                continue
            for candidate_index in candidate_index_set:
                if candidate_index in matched_candidate_indices:
                    continue
                candidate = candidate_by_index[candidate_index]
                iou = bbox_iou(track.bbox_xyxy, candidate.bbox_xyxy)
                if iou >= self.config.match_iou:
                    pairs.append((iou, track.track_id, candidate_index))

        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        tracks_by_id = {track.track_id: track for track in self._tracks}
        for _iou, track_id, candidate_index in pairs:
            if track_id in matched_track_ids:
                continue
            if candidate_index in matched_candidate_indices:
                continue
            track = tracks_by_id[track_id]
            track.update(
                timestamp_ms=timestamp_ms,
                candidate=candidate_by_index[candidate_index],
                velocity_window_ms=self.config.velocity_window_ms,
            )
            matched_track_ids.add(track_id)
            matched_candidate_indices.add(candidate_index)

    def _drop_expired(self, timestamp_ms: int) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if max(0, timestamp_ms - track.last_seen_ms) <= self.config.lost_ttl_ms
        ]

    def _should_reset(self, frame: FrameMessage) -> bool:
        if self._last_timestamp_ms is not None and frame.timestamp_ms < self._last_timestamp_ms:
            return True
        return self._last_frame_id is not None and frame.frame_id < self._last_frame_id


def bbox_iou(first: BBoxXYXY, second: BBoxXYXY) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection <= 0:
        return 0.0
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _candidate_from_detection(
    index: int,
    detection: PersonPoseDetection,
) -> _DetectionCandidate:
    return _DetectionCandidate(
        index=index,
        detection=detection,
        pose_confidence=_pose_confidence(detection.keypoints),
        head_uv=_head_uv(detection),
    )


def _pose_confidence(keypoints: list[PoseKeypoint]) -> float:
    values = [
        float(keypoint.confidence)
        for keypoint in keypoints
        if keypoint.confidence is not None
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _head_uv(detection: PersonPoseDetection) -> tuple[float, float]:
    face_points = [
        (keypoint.x, keypoint.y)
        for keypoint in detection.keypoints
        if keypoint.name in _FACE_KEYPOINT_NAMES and _valid_keypoint(keypoint)
    ]
    if face_points:
        return (
            sum(point[0] for point in face_points) / len(face_points),
            sum(point[1] for point in face_points) / len(face_points),
        )

    x1, y1, x2, y2 = detection.bbox_xyxy
    return ((x1 + x2) / 2.0, y1 + ((y2 - y1) * 0.28))


def _valid_keypoint(keypoint: PoseKeypoint) -> bool:
    if not math.isfinite(keypoint.x) or not math.isfinite(keypoint.y):
        return False
    return keypoint.confidence is None or keypoint.confidence > 0.0


def _bbox_center(bbox: BBoxXYXY) -> tuple[float, float]:
    return (_center_x(bbox), _center_y(bbox))


def _center_x(bbox: BBoxXYXY) -> float:
    x1, _y1, x2, _y2 = bbox
    return (x1 + x2) / 2.0


def _center_y(bbox: BBoxXYXY) -> float:
    _x1, y1, _x2, y2 = bbox
    return (y1 + y2) / 2.0
