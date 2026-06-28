from __future__ import annotations

import math
from dataclasses import dataclass

from visual_events_server.attention import AttentionResult
from visual_events_server.inference.base import BBoxXYXY, PoseKeypoint, bbox_area
from visual_events_server.protocol import FrameMessage
from visual_events_server.tracking import TrackSnapshot


EVENT_ORDER = (
    "person_appeared",
    "person_left",
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
    "person_waving",
    "attention_target_changed",
)

MOTION_SENSITIVE_EVENTS = {
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
}

_EVENT_TEXT = {
    "person_appeared": "有人进入画面",
    "person_left": "有人离开画面",
    "person_passing_by": "有人从机器人旁经过",
    "person_approaching_robot": "有人正在靠近机器人",
    "person_stopped_near_robot": "有人停在机器人附近",
    "person_waving": "有人在挥手",
    "attention_target_changed": "注意目标已切换",
}

_WAVE_WRIST_ABOVE_SHOULDER_MIN_PX = 4.0
_WAVE_WRIST_ABOVE_SHOULDER_MIN_BBOX_RATIO = 0.03


@dataclass(frozen=True)
class EventConfig:
    history_ms: int = 3000
    cooldown_ms: int = 5000
    stable_min_hits: int = 2
    stable_min_age_ms: int = 300
    left_lost_ms: int = 1500
    stop_duration_ms: int = 1500
    near_area_ratio: float = 0.075
    near_height_ratio: float = 0.55
    stop_max_speed_px_s: float = 35.0
    approach_duration_ms: int = 500
    approach_area_growth_ratio: float = 1.35
    approach_min_area_delta: float = 0.015
    approach_min_current_area: float = 0.035
    passing_duration_ms: int = 1000
    passing_min_dx_ratio: float = 0.45
    passing_min_abs_vx_px_s: float = 80.0
    keypoint_min_conf: float = 0.3
    wave_window_ms: int = 1800
    wave_min_x_span_px: float = 35.0
    wave_min_x_span_bbox_ratio: float = 0.12
    reacquire_alias_window_ms: int = 5000
    reacquire_center_distance_ratio: float = 0.08

    def __post_init__(self) -> None:
        integer_fields = (
            "history_ms",
            "cooldown_ms",
            "stable_min_hits",
            "stable_min_age_ms",
            "left_lost_ms",
            "stop_duration_ms",
            "approach_duration_ms",
            "passing_duration_ms",
            "wave_window_ms",
            "reacquire_alias_window_ms",
        )
        for field_name in integer_fields:
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.history_ms <= 0:
            raise ValueError("history_ms must be positive")
        if self.stable_min_hits <= 0:
            raise ValueError("stable_min_hits must be positive")
        ratio_fields = (
            "near_area_ratio",
            "near_height_ratio",
            "approach_area_growth_ratio",
            "approach_min_area_delta",
            "approach_min_current_area",
            "passing_min_dx_ratio",
            "keypoint_min_conf",
            "wave_min_x_span_bbox_ratio",
            "reacquire_center_distance_ratio",
        )
        for field_name in ratio_fields:
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.approach_area_growth_ratio < 1.0:
            raise ValueError("approach_area_growth_ratio must be >= 1.0")
        if self.stop_max_speed_px_s < 0.0:
            raise ValueError("stop_max_speed_px_s must be non-negative")
        if self.passing_min_abs_vx_px_s < 0.0:
            raise ValueError("passing_min_abs_vx_px_s must be non-negative")
        if self.wave_min_x_span_px < 0.0:
            raise ValueError("wave_min_x_span_px must be non-negative")


@dataclass(frozen=True)
class _Observation:
    timestamp_ms: int
    bbox_xyxy: BBoxXYXY
    confidence: float
    velocity_uv_s: tuple[float, float]
    keypoints: tuple[PoseKeypoint, ...]


@dataclass(frozen=True)
class _EventProposal:
    event: str
    track_id: int
    confidence: float
    duration_ms: int
    rising_edge: bool = True


class EventEngine:
    def __init__(self, *, config: EventConfig | None = None) -> None:
        self.config = config or EventConfig()
        self._next_event_number = 1
        self._reset_state()

    def reset(self) -> None:
        self._reset_state()

    def update(
        self,
        frame: FrameMessage,
        tracks: list[TrackSnapshot],
        attention: AttentionResult | None,
    ) -> list[dict[str, object]]:
        if self._should_reset(frame):
            self._reset_state()

        now_ms = int(frame.timestamp_ms)
        visible_tracks = [
            track
            for track in tracks
            if track.class_name == "person" and int(track.lost_ms) == 0
        ]
        tracks_by_id = {track.track_id: track for track in tracks}
        self._observe_tracks(
            frame,
            visible_tracks=visible_tracks,
        )

        salient = self._salient_track(frame, visible_tracks, attention)
        if salient is not None:
            self._salient_person_slots.add(
                self._person_slot_for_track_id(salient.track_id)
            )

        proposals: list[_EventProposal] = []
        appeared = self._person_appeared_proposal(salient)
        if appeared is not None:
            proposals.append(appeared)
        proposals.extend(self._person_left_proposals(now_ms, tracks_by_id))

        for track in visible_tracks:
            passing = self._person_passing_by_proposal(frame, track)
            if passing is not None:
                proposals.append(passing)
        for track in visible_tracks:
            approaching = self._person_approaching_robot_proposal(frame, track)
            if approaching is not None:
                proposals.append(approaching)
        for track in visible_tracks:
            stopped = self._person_stopped_near_robot_proposal(frame, track)
            if stopped is not None:
                proposals.append(stopped)
        for track in visible_tracks:
            waving = self._person_waving_proposal(frame, track)
            if waving is not None:
                proposals.append(waving)

        attention_changed = self._attention_target_changed_proposal(attention)
        if attention_changed is not None:
            proposals.append(attention_changed)

        proposals.sort(key=lambda proposal: EVENT_ORDER.index(proposal.event))
        events = [
            event
            for proposal in proposals
            if (event := self._emit_if_allowed(frame, proposal)) is not None
        ]

        self._last_timestamp_ms = frame.timestamp_ms
        self._last_frame_id = frame.frame_id
        return events

    def _reset_state(self) -> None:
        self._last_timestamp_ms: int | None = None
        self._last_frame_id: int | None = None
        self._histories: dict[int, list[_Observation]] = {}
        self._motion_histories: dict[int, list[_Observation]] = {}
        self._next_person_slot = 1
        self._track_person_slots: dict[int, int] = {}
        self._track_alias_assigned_ms: dict[int, int] = {}
        self._slot_last_seen_ms: dict[int, int] = {}
        self._slot_last_bbox: dict[int, BBoxXYXY] = {}
        self._slot_last_confidence: dict[int, float] = {}
        self._slot_last_track_id: dict[int, int] = {}
        self._appeared_person_slots: set[int] = set()
        self._salient_person_slots: set[int] = set()
        self._left_emitted_person_slots: set[int] = set()
        self._active_conditions: set[tuple[int, str]] = set()
        self._last_event_by_type_ms: dict[str, int] = {}
        self._last_person_event_ms: dict[tuple[int, str], int] = {}
        self._last_attention_track_id: int | None = None

    def _should_reset(self, frame: FrameMessage) -> bool:
        if (
            self._last_timestamp_ms is not None
            and frame.timestamp_ms < self._last_timestamp_ms
        ):
            return True
        return self._last_frame_id is not None and frame.frame_id < self._last_frame_id

    def _observe_tracks(
        self,
        frame: FrameMessage,
        *,
        visible_tracks: list[TrackSnapshot],
    ) -> None:
        now_ms = int(frame.timestamp_ms)
        oldest_ms = now_ms - self.config.history_ms
        self._assign_person_slots(frame, visible_tracks)

        for track in visible_tracks:
            slot = self._person_slot_for_track_id(track.track_id)
            self._slot_last_seen_ms[slot] = int(track.last_seen_ms)
            self._slot_last_bbox[slot] = track.bbox_xyxy
            self._slot_last_confidence[slot] = float(track.confidence)
            self._slot_last_track_id[slot] = track.track_id

            observation = _observation_from_track(track, timestamp_ms=now_ms)
            history = self._histories.setdefault(track.track_id, [])
            history.append(observation)
            self._histories[track.track_id] = [
                item for item in history if item.timestamp_ms >= oldest_ms
            ]

        if frame.head_motion_state != "stationary":
            self._motion_histories.clear()
            for event in MOTION_SENSITIVE_EVENTS:
                self._clear_active_for_event(event)
            return

        for track in visible_tracks:
            observation = _observation_from_track(track, timestamp_ms=now_ms)
            history = self._motion_histories.setdefault(track.track_id, [])
            history.append(observation)
            self._motion_histories[track.track_id] = [
                item for item in history if item.timestamp_ms >= oldest_ms
            ]

    def _salient_track(
        self,
        frame: FrameMessage,
        visible_tracks: list[TrackSnapshot],
        attention: AttentionResult | None,
    ) -> TrackSnapshot | None:
        stable_visible = [
            track for track in visible_tracks if self._is_stable_visible(track)
        ]
        if not stable_visible:
            return None

        if attention is not None:
            stable_by_id = {track.track_id: track for track in stable_visible}
            target = stable_by_id.get(attention.target_track_id)
            return target

        image_area = float(frame.width) * float(frame.height)
        return max(
            stable_visible,
            key=lambda track: (
                _area_ratio(track.bbox_xyxy, image_area=image_area)
                * max(0.0, float(track.confidence)),
                bbox_area(track.bbox_xyxy),
                float(track.confidence),
                -track.track_id,
            ),
        )

    def _is_stable_visible(self, track: TrackSnapshot) -> bool:
        return (
            track.class_name == "person"
            and int(track.lost_ms) == 0
            and int(track.hits) >= self.config.stable_min_hits
            and int(track.age_ms) >= self.config.stable_min_age_ms
        )

    def _person_appeared_proposal(
        self,
        salient: TrackSnapshot | None,
    ) -> _EventProposal | None:
        if salient is None:
            return None
        slot = self._person_slot_for_track_id(salient.track_id)
        if slot in self._appeared_person_slots:
            return None
        return _EventProposal(
            event="person_appeared",
            track_id=salient.track_id,
            confidence=float(salient.confidence),
            duration_ms=int(salient.age_ms),
            rising_edge=False,
        )

    def _person_left_proposals(
        self,
        now_ms: int,
        tracks_by_id: dict[int, TrackSnapshot],
    ) -> list[_EventProposal]:
        proposals: list[_EventProposal] = []
        known_slots = sorted(self._appeared_person_slots | self._salient_person_slots)
        visible_slots = {
            self._person_slot_for_track_id(track.track_id)
            for track in tracks_by_id.values()
            if track.class_name == "person" and int(track.lost_ms) == 0
        }
        for slot in known_slots:
            if slot in self._left_emitted_person_slots:
                continue
            if slot in visible_slots:
                continue

            last_seen_ms = self._slot_last_seen_ms.get(slot)
            if last_seen_ms is None:
                continue

            slot_tracks = [
                track
                for track in tracks_by_id.values()
                if self._track_person_slots.get(track.track_id) == slot
            ]
            for track in slot_tracks:
                last_seen_ms = max(last_seen_ms, int(track.last_seen_ms))
            lost_duration_ms = max(0, now_ms - last_seen_ms)
            track_lost_ms = max(
                (int(track.lost_ms) for track in slot_tracks),
                default=0,
            )
            duration_ms = max(lost_duration_ms, track_lost_ms)
            if duration_ms < self.config.left_lost_ms:
                continue
            track_id = self._slot_last_track_id.get(slot)
            if track_id is None:
                continue
            proposals.append(
                _EventProposal(
                    event="person_left",
                    track_id=track_id,
                    confidence=self._slot_last_confidence.get(slot, 0.8),
                    duration_ms=duration_ms,
                    rising_edge=False,
                )
            )
        return proposals

    def _person_passing_by_proposal(
        self,
        frame: FrameMessage,
        track: TrackSnapshot,
    ) -> _EventProposal | None:
        event = "person_passing_by"
        history = self._motion_histories.get(track.track_id, [])
        if len(history) < 2:
            self._clear_active(track.track_id, event)
            return None

        current = history[-1]
        reference = self._oldest_reference(history, self.config.passing_duration_ms)
        if reference is None:
            self._clear_active(track.track_id, event)
            return None

        duration_ms = current.timestamp_ms - reference.timestamp_ms
        dx = _center_x(current.bbox_xyxy) - _center_x(reference.bbox_xyxy)
        dx_ratio = abs(dx) / float(frame.width) if frame.width > 0 else 0.0
        avg_vx = dx / (duration_ms / 1000.0) if duration_ms > 0 else 0.0
        crossed_side_bands = self._crossed_side_bands(reference, current, frame)
        swept_ltr_fallback = (
            dx > 0.0
            and crossed_side_bands
            and self._bbox_swept_dx_ratio(reference, current, frame)
            >= self.config.passing_min_dx_ratio
        )
        has_lateral_evidence = (
            dx_ratio >= self.config.passing_min_dx_ratio
            or swept_ltr_fallback
        )
        if (
            not has_lateral_evidence
            or abs(avg_vx) < self.config.passing_min_abs_vx_px_s
            or self._entered_stopped_near_state(history, frame)
            or not crossed_side_bands
        ):
            self._clear_active(track.track_id, event)
            return None

        if self._is_active(track.track_id, event):
            return None
        return _EventProposal(
            event=event,
            track_id=track.track_id,
            confidence=current.confidence,
            duration_ms=duration_ms,
        )

    def _person_approaching_robot_proposal(
        self,
        frame: FrameMessage,
        track: TrackSnapshot,
    ) -> _EventProposal | None:
        event = "person_approaching_robot"
        history = self._motion_histories.get(track.track_id, [])
        if len(history) < 2:
            self._clear_active(track.track_id, event)
            return None

        current = history[-1]
        reference = self._oldest_reference(history, self.config.approach_duration_ms)
        if reference is None:
            self._clear_active(track.track_id, event)
            return None

        image_area = float(frame.width) * float(frame.height)
        start_area = _area_ratio(reference.bbox_xyxy, image_area=image_area)
        current_area = _area_ratio(current.bbox_xyxy, image_area=image_area)
        if start_area <= 0.0:
            self._clear_active(track.track_id, event)
            return None
        duration_ms = current.timestamp_ms - reference.timestamp_ms
        growth = current_area / start_area
        area_delta = current_area - start_area
        if (
            current_area < self.config.approach_min_current_area
            or growth < self.config.approach_area_growth_ratio
            or area_delta < self.config.approach_min_area_delta
            or not self._area_trend_is_steady(history, frame, since_ms=reference.timestamp_ms)
        ):
            self._clear_active(track.track_id, event)
            return None

        if self._is_active(track.track_id, event):
            return None
        return _EventProposal(
            event=event,
            track_id=track.track_id,
            confidence=current.confidence,
            duration_ms=duration_ms,
        )

    def _person_stopped_near_robot_proposal(
        self,
        frame: FrameMessage,
        track: TrackSnapshot,
    ) -> _EventProposal | None:
        event = "person_stopped_near_robot"
        history = self._motion_histories.get(track.track_id, [])
        if not history or not self._near_and_slow(history[-1], frame):
            self._clear_active(track.track_id, event)
            return None

        streak: list[_Observation] = []
        for observation in reversed(history):
            if not self._near_and_slow(observation, frame):
                break
            streak.append(observation)
        if not streak:
            self._clear_active(track.track_id, event)
            return None

        start = streak[-1]
        current = history[-1]
        duration_ms = current.timestamp_ms - start.timestamp_ms
        if duration_ms < self.config.stop_duration_ms:
            self._clear_active(track.track_id, event)
            return None
        if self._is_active(track.track_id, event):
            return None
        return _EventProposal(
            event=event,
            track_id=track.track_id,
            confidence=current.confidence,
            duration_ms=duration_ms,
        )

    def _person_waving_proposal(
        self,
        frame: FrameMessage,
        track: TrackSnapshot,
    ) -> _EventProposal | None:
        event = "person_waving"
        history = [
            observation
            for observation in self._histories.get(track.track_id, [])
            if frame.timestamp_ms - observation.timestamp_ms <= self.config.wave_window_ms
        ]
        result = self._wave_result(history)
        if result is None:
            self._clear_active(track.track_id, event)
            return None
        duration_ms, confidence = result
        if self._is_active(track.track_id, event):
            return None
        return _EventProposal(
            event=event,
            track_id=track.track_id,
            confidence=min(float(track.confidence), confidence),
            duration_ms=duration_ms,
        )

    def _attention_target_changed_proposal(
        self,
        attention: AttentionResult | None,
    ) -> _EventProposal | None:
        if attention is None:
            self._last_attention_track_id = None
            return None
        current_id = int(attention.target_track_id)
        previous_id = self._last_attention_track_id
        self._last_attention_track_id = current_id
        if previous_id is None or current_id == previous_id:
            return None
        if attention.reason == "held_lost_target":
            return None
        return _EventProposal(
            event="attention_target_changed",
            track_id=current_id,
            confidence=float(attention.confidence),
            duration_ms=0,
            rising_edge=False,
        )

    def _emit_if_allowed(
        self,
        frame: FrameMessage,
        proposal: _EventProposal,
    ) -> dict[str, object] | None:
        now_ms = int(frame.timestamp_ms)
        last_type_ms = self._last_event_by_type_ms.get(proposal.event)
        if last_type_ms is not None and now_ms - last_type_ms < self.config.cooldown_ms:
            return None

        person_event_key = self._person_event_key(proposal)
        last_track_event_ms = self._last_person_event_ms.get(person_event_key)
        alias_assigned_ms = self._track_alias_assigned_ms.get(proposal.track_id)
        if (
            proposal.event != "attention_target_changed"
            and alias_assigned_ms is not None
            and now_ms - alias_assigned_ms <= self.config.reacquire_alias_window_ms
            and last_track_event_ms is not None
        ):
            return None
        if (
            last_track_event_ms is not None
            and now_ms - last_track_event_ms < self.config.cooldown_ms
        ):
            return None

        event = {
            "type": "semantic_event",
            "event_id": f"{frame.camera}:evt_{self._next_event_number:06d}",
            "event": proposal.event,
            "camera": frame.camera,
            "track_id": proposal.track_id,
            "confidence": _clamp(float(proposal.confidence), 0.0, 1.0),
            "duration_ms": max(0, int(proposal.duration_ms)),
            "text": _EVENT_TEXT[proposal.event],
        }
        self._next_event_number += 1
        self._last_event_by_type_ms[proposal.event] = now_ms
        self._last_person_event_ms[person_event_key] = now_ms

        slot = person_event_key[0]
        if proposal.event == "person_appeared":
            self._appeared_person_slots.add(slot)
            self._salient_person_slots.add(slot)
        elif proposal.event == "person_left":
            self._left_emitted_person_slots.add(slot)

        if proposal.rising_edge:
            self._active_conditions.add(person_event_key)
        return event

    def _oldest_reference(
        self,
        history: list[_Observation],
        minimum_duration_ms: int,
    ) -> _Observation | None:
        current = history[-1]
        candidates = [
            observation
            for observation in history
            if current.timestamp_ms - observation.timestamp_ms >= minimum_duration_ms
        ]
        return candidates[0] if candidates else None

    def _entered_stopped_near_state(
        self,
        history: list[_Observation],
        frame: FrameMessage,
    ) -> bool:
        streak: list[_Observation] = []
        for observation in reversed(history):
            if not self._near_and_slow(observation, frame):
                break
            streak.append(observation)
        return bool(streak) and (
            history[-1].timestamp_ms - streak[-1].timestamp_ms
            >= self.config.stop_duration_ms
        )

    def _crossed_side_bands(
        self,
        reference: _Observation,
        current: _Observation,
        frame: FrameMessage,
    ) -> bool:
        width = float(frame.width)
        if width <= 0.0:
            return False
        side_band = width * 0.38
        left_band = side_band
        right_band = width - side_band
        start_x = _center_x(reference.bbox_xyxy)
        end_x = _center_x(current.bbox_xyxy)
        if end_x > start_x:
            return start_x <= left_band and end_x >= right_band
        return start_x >= right_band and end_x <= left_band

    def _bbox_swept_dx_ratio(
        self,
        reference: _Observation,
        current: _Observation,
        frame: FrameMessage,
    ) -> float:
        width = float(frame.width)
        if width <= 0.0:
            return 0.0
        ref_x1, _, ref_x2, _ = reference.bbox_xyxy
        cur_x1, _, cur_x2, _ = current.bbox_xyxy
        if _center_x(current.bbox_xyxy) > _center_x(reference.bbox_xyxy):
            swept_dx = float(cur_x2) - float(ref_x1)
        else:
            swept_dx = float(ref_x2) - float(cur_x1)
        return max(0.0, swept_dx) / width

    def _area_trend_is_steady(
        self,
        history: list[_Observation],
        frame: FrameMessage,
        *,
        since_ms: int,
    ) -> bool:
        image_area = float(frame.width) * float(frame.height)
        observations = [
            observation for observation in history if observation.timestamp_ms >= since_ms
        ]
        ratios = [
            _area_ratio(observation.bbox_xyxy, image_area=image_area)
            for observation in observations
        ]
        if len(ratios) < 2:
            return False
        drop_epsilon = 0.002
        drops = [
            ratios[index - 1] - ratios[index]
            for index in range(1, len(ratios))
            if ratios[index] < ratios[index - 1] - drop_epsilon
        ]
        if not drops and ratios[-1] >= max(ratios[:-1]):
            return True

        net_gain = ratios[-1] - ratios[0]
        if net_gain <= 0.0:
            return False
        jitter_min_current_area = max(
            self.config.approach_min_current_area,
            self.config.near_area_ratio * 0.72,
        )
        if ratios[-1] < jitter_min_current_area:
            return False
        peak_gap = max(ratios) - ratios[-1]
        peak_gap_limit = max(0.003, min(0.006, net_gain * 0.15))
        total_drop_limit = max(0.006, min(0.012, net_gain * 0.35))
        single_drop_limit = max(0.004, min(0.008, net_gain * 0.25))
        return (
            len(drops) <= 2
            and peak_gap <= peak_gap_limit
            and sum(drops) <= total_drop_limit
            and max(drops, default=0.0) <= single_drop_limit
        )

    def _near_and_slow(self, observation: _Observation, frame: FrameMessage) -> bool:
        vx, vy = observation.velocity_uv_s
        speed = math.hypot(float(vx), float(vy))
        return self._is_near(observation, frame) and speed <= self.config.stop_max_speed_px_s

    def _is_near(self, observation: _Observation, frame: FrameMessage) -> bool:
        image_area = float(frame.width) * float(frame.height)
        height = float(frame.height)
        _, y1, _, y2 = observation.bbox_xyxy
        height_ratio = (float(y2) - float(y1)) / height if height > 0.0 else 0.0
        return (
            _area_ratio(observation.bbox_xyxy, image_area=image_area)
            >= self.config.near_area_ratio
            or height_ratio >= self.config.near_height_ratio
        )

    def _wave_result(
        self,
        history: list[_Observation],
    ) -> tuple[int, float] | None:
        for side in ("left", "right"):
            samples: list[tuple[int, float, float]] = []
            for observation in history:
                shoulder = _keypoint_by_name(observation.keypoints, f"{side}_shoulder")
                wrist = _keypoint_by_name(observation.keypoints, f"{side}_wrist")
                if shoulder is None or wrist is None:
                    continue
                bbox_height = _bbox_height(observation.bbox_xyxy)
                if not self._is_wave_pose_sample(
                    shoulder=shoulder,
                    wrist=wrist,
                    bbox_height=bbox_height,
                ):
                    continue
                samples.append(
                    (
                        observation.timestamp_ms,
                        float(wrist.x),
                        min(float(shoulder.confidence or 0.0), float(wrist.confidence or 0.0)),
                    )
                )

            if len(samples) < 3:
                continue
            xs = [sample[1] for sample in samples]
            span = max(xs) - min(xs)
            bbox_width = _bbox_width(history[-1].bbox_xyxy)
            required_span = max(
                self.config.wave_min_x_span_px,
                bbox_width * self.config.wave_min_x_span_bbox_ratio,
            )
            if span < required_span or not _has_direction_reversal(xs):
                continue
            return samples[-1][0] - samples[0][0], sum(sample[2] for sample in samples) / len(samples)
        return None

    def _is_wave_pose_sample(
        self,
        *,
        shoulder: PoseKeypoint,
        wrist: PoseKeypoint,
        bbox_height: float,
    ) -> bool:
        if not self._valid_keypoint(shoulder) or not self._valid_keypoint(wrist):
            return False
        min_vertical_clearance = max(
            _WAVE_WRIST_ABOVE_SHOULDER_MIN_PX,
            float(bbox_height) * _WAVE_WRIST_ABOVE_SHOULDER_MIN_BBOX_RATIO,
        )
        return float(wrist.y) <= float(shoulder.y) - min_vertical_clearance

    def _valid_keypoint(self, keypoint: PoseKeypoint) -> bool:
        confidence = keypoint.confidence
        return (
            math.isfinite(float(keypoint.x))
            and math.isfinite(float(keypoint.y))
            and confidence is not None
            and float(confidence) >= self.config.keypoint_min_conf
        )

    def _is_active(self, track_id: int, event: str) -> bool:
        return (
            self._person_slot_for_track_id(track_id),
            event,
        ) in self._active_conditions

    def _clear_active(self, track_id: int, event: str) -> None:
        self._active_conditions.discard(
            (self._person_slot_for_track_id(track_id), event)
        )

    def _clear_active_for_event(self, event: str) -> None:
        self._active_conditions = {
            item for item in self._active_conditions if item[1] != event
        }

    def _assign_person_slots(
        self,
        frame: FrameMessage,
        visible_tracks: list[TrackSnapshot],
    ) -> None:
        now_ms = int(frame.timestamp_ms)
        assigned_visible_slots = {
            self._track_person_slots[track.track_id]
            for track in visible_tracks
            if track.track_id in self._track_person_slots
        }
        for track in visible_tracks:
            if track.track_id in self._track_person_slots:
                continue
            alias_slot = self._alias_slot_for_track(
                frame,
                track,
                excluded_slots=assigned_visible_slots,
            )
            slot = alias_slot if alias_slot is not None else self._create_person_slot()
            self._track_person_slots[track.track_id] = slot
            if alias_slot is not None:
                self._track_alias_assigned_ms[track.track_id] = now_ms
            assigned_visible_slots.add(slot)

    def _alias_slot_for_track(
        self,
        frame: FrameMessage,
        track: TrackSnapshot,
        *,
        excluded_slots: set[int],
    ) -> int | None:
        now_ms = int(frame.timestamp_ms)
        threshold_px = (
            math.hypot(float(frame.width), float(frame.height))
            * self.config.reacquire_center_distance_ratio
        )
        candidates: list[tuple[float, int, int]] = []
        for slot, last_seen_ms in self._slot_last_seen_ms.items():
            if slot in excluded_slots:
                continue
            elapsed_ms = now_ms - last_seen_ms
            if elapsed_ms < 0 or elapsed_ms > self.config.reacquire_alias_window_ms:
                continue
            previous_bbox = self._slot_last_bbox.get(slot)
            if previous_bbox is None:
                continue
            distance_px = _center_distance(track.bbox_xyxy, previous_bbox)
            if distance_px <= threshold_px:
                candidates.append((distance_px, elapsed_ms, slot))
        if not candidates:
            return None
        return min(candidates)[2]

    def _create_person_slot(self) -> int:
        slot = self._next_person_slot
        self._next_person_slot += 1
        return slot

    def _person_slot_for_track_id(self, track_id: int) -> int:
        slot = self._track_person_slots.get(track_id)
        if slot is None:
            slot = self._create_person_slot()
            self._track_person_slots[track_id] = slot
        return slot

    def _person_event_key(self, proposal: _EventProposal) -> tuple[int, str]:
        if proposal.event == "attention_target_changed":
            return (proposal.track_id, proposal.event)
        return (self._person_slot_for_track_id(proposal.track_id), proposal.event)


def _observation_from_track(
    track: TrackSnapshot,
    *,
    timestamp_ms: int,
) -> _Observation:
    return _Observation(
        timestamp_ms=timestamp_ms,
        bbox_xyxy=track.bbox_xyxy,
        confidence=float(track.confidence),
        velocity_uv_s=track.velocity_uv_s,
        keypoints=tuple(track.keypoints),
    )


def _area_ratio(bbox: BBoxXYXY, *, image_area: float) -> float:
    return bbox_area(bbox) / image_area if image_area > 0.0 else 0.0


def _center_x(bbox: BBoxXYXY) -> float:
    x1, _, x2, _ = bbox
    return (float(x1) + float(x2)) / 2.0


def _center_y(bbox: BBoxXYXY) -> float:
    _, y1, _, y2 = bbox
    return (float(y1) + float(y2)) / 2.0


def _center_distance(left: BBoxXYXY, right: BBoxXYXY) -> float:
    return math.hypot(
        _center_x(left) - _center_x(right),
        _center_y(left) - _center_y(right),
    )


def _bbox_width(bbox: BBoxXYXY) -> float:
    x1, _, x2, _ = bbox
    return max(0.0, float(x2) - float(x1))


def _bbox_height(bbox: BBoxXYXY) -> float:
    _, y1, _, y2 = bbox
    return max(0.0, float(y2) - float(y1))


def _keypoint_by_name(
    keypoints: tuple[PoseKeypoint, ...],
    name: str,
) -> PoseKeypoint | None:
    for keypoint in keypoints:
        if keypoint.name == name:
            return keypoint
    return None


def _has_direction_reversal(values: list[float]) -> bool:
    signs: list[int] = []
    for index in range(1, len(values)):
        delta = values[index] - values[index - 1]
        if abs(delta) < 1e-6:
            continue
        signs.append(1 if delta > 0 else -1)
    return any(signs[index] != signs[index - 1] for index in range(1, len(signs)))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
