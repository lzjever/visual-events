from __future__ import annotations

import math
from dataclasses import dataclass

from visual_events_server.inference.base import bbox_area
from visual_events_server.protocol import FrameMessage
from visual_events_server.tracking import TrackSnapshot


@dataclass(frozen=True)
class AttentionConfig:
    stable_min_hits: int = 2
    stable_min_age_ms: int = 300
    switch_area_ratio: float = 1.25
    switch_confirm_ms: int = 500
    lost_hold_ms: int = 600

    def __post_init__(self) -> None:
        if self.stable_min_hits <= 0:
            raise ValueError("stable_min_hits must be positive")
        if self.stable_min_age_ms < 0:
            raise ValueError("stable_min_age_ms must be non-negative")
        if self.switch_area_ratio < 1.0:
            raise ValueError("switch_area_ratio must be >= 1.0")
        if self.switch_confirm_ms < 0:
            raise ValueError("switch_confirm_ms must be non-negative")
        if self.lost_hold_ms < 0:
            raise ValueError("lost_hold_ms must be non-negative")


@dataclass(frozen=True)
class AttentionResult:
    target_track_id: int
    target_uv: tuple[float, float]
    reason: str
    confidence: float
    largest_person_stable: bool

    def to_protocol(self) -> dict[str, object]:
        return {
            "target_track_id": self.target_track_id,
            "target_uv": [self.target_uv[0], self.target_uv[1]],
            "reason": self.reason,
            "confidence": self.confidence,
        }


class AttentionSelector:
    def __init__(self, *, config: AttentionConfig | None = None) -> None:
        self.config = config or AttentionConfig()
        self._target_track_id: int | None = None
        self._challenger_track_id: int | None = None
        self._challenger_since_ms: int | None = None
        self._last_timestamp_ms: int | None = None
        self._last_frame_id: int | None = None

    def reset(self) -> None:
        self._target_track_id = None
        self._challenger_track_id = None
        self._challenger_since_ms = None
        self._last_timestamp_ms = None
        self._last_frame_id = None

    def update(
        self,
        frame: FrameMessage,
        tracks: list[TrackSnapshot],
    ) -> AttentionResult | None:
        if self._should_reset(frame):
            self.reset()

        result = self._update_after_reset_check(frame, tracks)
        self._last_timestamp_ms = frame.timestamp_ms
        self._last_frame_id = frame.frame_id
        return result

    def _update_after_reset_check(
        self,
        frame: FrameMessage,
        tracks: list[TrackSnapshot],
    ) -> AttentionResult | None:
        stable_visible = [
            track for track in tracks if self._is_stable_visible_person(track)
        ]
        stable_by_id = {track.track_id: track for track in stable_visible}
        tracks_by_id = {track.track_id: track for track in tracks}

        current = (
            stable_by_id.get(self._target_track_id)
            if self._target_track_id is not None
            else None
        )
        if current is not None:
            switched = self._maybe_switch_current_target(
                frame,
                current=current,
                stable_visible=stable_visible,
            )
            if switched is not None:
                return self._result_for(
                    frame,
                    switched,
                    reason="largest_stable_person",
                    largest_person_stable=True,
                )
            return self._result_for(
                frame,
                current,
                reason="largest_stable_person",
                largest_person_stable=True,
            )

        held_lost = (
            tracks_by_id.get(self._target_track_id)
            if self._target_track_id is not None
            else None
        )
        if (
            held_lost is not None
            and held_lost.class_name == "person"
            and 0 < held_lost.lost_ms <= self.config.lost_hold_ms
        ):
            self._clear_challenger()
            return self._result_for(
                frame,
                held_lost,
                reason="held_lost_target",
                largest_person_stable=False,
            )

        self._target_track_id = None
        self._clear_challenger()
        selected = self._best_track(stable_visible, frame=frame)
        if selected is None:
            return None
        self._target_track_id = selected.track_id
        return self._result_for(
            frame,
            selected,
            reason="largest_stable_person",
            largest_person_stable=True,
        )

    def _maybe_switch_current_target(
        self,
        frame: FrameMessage,
        *,
        current: TrackSnapshot,
        stable_visible: list[TrackSnapshot],
    ) -> TrackSnapshot | None:
        current_area = bbox_area(current.bbox_xyxy)
        minimum_area = current_area * self.config.switch_area_ratio
        challengers = [
            track
            for track in stable_visible
            if track.track_id != current.track_id
            and bbox_area(track.bbox_xyxy) >= minimum_area
        ]
        challenger = self._best_track(challengers, frame=frame)
        if challenger is None:
            self._clear_challenger()
            return None

        if challenger.track_id != self._challenger_track_id:
            self._challenger_track_id = challenger.track_id
            self._challenger_since_ms = frame.timestamp_ms

        since_ms = self._challenger_since_ms
        if since_ms is None:
            return None
        if frame.timestamp_ms - since_ms < self.config.switch_confirm_ms:
            return None

        self._target_track_id = challenger.track_id
        self._clear_challenger()
        return challenger

    def _result_for(
        self,
        frame: FrameMessage,
        track: TrackSnapshot,
        *,
        reason: str,
        largest_person_stable: bool,
    ) -> AttentionResult:
        return AttentionResult(
            target_track_id=track.track_id,
            target_uv=_target_uv(
                track,
                image_width=frame.width,
                image_height=frame.height,
            ),
            reason=reason,
            confidence=_clamp(float(track.confidence), 0.0, 1.0),
            largest_person_stable=largest_person_stable,
        )

    def _is_stable_visible_person(self, track: TrackSnapshot) -> bool:
        return (
            track.class_name == "person"
            and track.lost_ms == 0
            and track.hits >= self.config.stable_min_hits
            and track.age_ms >= self.config.stable_min_age_ms
        )

    def _best_track(
        self,
        tracks: list[TrackSnapshot],
        *,
        frame: FrameMessage,
    ) -> TrackSnapshot | None:
        if not tracks:
            return None
        image_area = float(frame.width) * float(frame.height)
        return max(
            tracks,
            key=lambda track: (
                _selection_score(track, image_area=image_area),
                bbox_area(track.bbox_xyxy),
                float(track.confidence),
                -track.track_id,
            ),
        )

    def _clear_challenger(self) -> None:
        self._challenger_track_id = None
        self._challenger_since_ms = None

    def _should_reset(self, frame: FrameMessage) -> bool:
        if (
            self._last_timestamp_ms is not None
            and frame.timestamp_ms < self._last_timestamp_ms
        ):
            return True
        return self._last_frame_id is not None and frame.frame_id < self._last_frame_id


def _selection_score(track: TrackSnapshot, *, image_area: float) -> float:
    area_ratio = bbox_area(track.bbox_xyxy) / image_area if image_area > 0.0 else 0.0
    return area_ratio * max(0.0, float(track.confidence))


def _target_uv(
    track: TrackSnapshot,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    x, y = track.head_uv
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        x, y = _bbox_center(track)
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        x = float(image_width) / 2.0
        y = float(image_height) / 2.0
    return (
        _clamp(float(x), 0.0, float(image_width)),
        _clamp(float(y), 0.0, float(image_height)),
    )


def _bbox_center(track: TrackSnapshot) -> tuple[float, float]:
    x1, y1, x2, y2 = track.bbox_xyxy
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
