from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Any, Literal

from visual_events_server.attention import AttentionResult
from visual_events_server.inference.base import BBoxXYXY, PoseKeypoint, bbox_area, clip_bbox
from visual_events_server.tracking import TrackSnapshot


@dataclass(frozen=True)
class TargetRequest:
    mode: str
    track_id: int | None = None
    bbox_xyxy: BBoxXYXY | None = None
    point_uv: tuple[float, float] | None = None


@dataclass(frozen=True)
class ResolvedTarget:
    source_target_mode: str
    target_type: str
    bbox_xyxy: BBoxXYXY
    track_id: int | None
    quality: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetCandidate:
    target_type: str
    track_id: int | None
    bbox_xyxy: BBoxXYXY
    confidence: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetPreview:
    status: Literal["resolved", "ambiguous", "not_found"]
    candidates: list[TargetCandidate]
    ambiguity_type: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PointingArm:
    side: str
    origin: tuple[float, float]
    direction: tuple[float, float]
    shoulder: tuple[float, float]
    elbow: tuple[float, float]
    wrist: tuple[float, float]
    keypoint_confidences: dict[str, float]


@dataclass(frozen=True)
class _PoseCandidateScore:
    track: TrackSnapshot
    score: float
    perpendicular_distance: float
    arm: _PointingArm
    ray_intersects_bbox: bool


class TargetResolveError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class TargetResolver:
    min_box_size_px: float = 12.0
    ambiguous_score_margin: float = 0.08
    attention_prior_weight: float = 0.04

    def preview(
        self,
        request: TargetRequest,
        *,
        image_width: int,
        image_height: int,
        tracks: list[TrackSnapshot],
        attention: AttentionResult | None,
    ) -> TargetPreview:
        if image_width <= 0 or image_height <= 0:
            raise TargetResolveError("invalid_frame_size", "image dimensions must be positive")

        if request.mode == "scene":
            return TargetPreview(
                status="resolved",
                candidates=[
                    TargetCandidate(
                        target_type="scene",
                        track_id=None,
                        bbox_xyxy=(0.0, 0.0, float(image_width), float(image_height)),
                        confidence=1.0,
                        reason="full_scene",
                    )
                ],
            )
        if request.mode == "attention_target":
            if attention is None:
                return TargetPreview(status="not_found", candidates=[])
            track = _visible_track_by_id(tracks, attention.target_track_id)
            if track is None:
                return TargetPreview(status="not_found", candidates=[])
            return TargetPreview(
                status="resolved",
                candidates=[
                    self._candidate_from_track(
                        track,
                        confidence=_clamp_confidence(attention.confidence),
                        reason="attention_target",
                        image_width=image_width,
                        image_height=image_height,
                    )
                ],
            )
        if request.mode == "track_id":
            if request.track_id is None:
                raise TargetResolveError("invalid_target_request", "track_id is required")
            track = _visible_track_by_id(tracks, request.track_id)
            if track is None:
                return TargetPreview(status="not_found", candidates=[])
            return TargetPreview(
                status="resolved",
                candidates=[
                    self._candidate_from_track(
                        track,
                        confidence=_clamp_confidence(track.confidence),
                        reason="track_id",
                        image_width=image_width,
                        image_height=image_height,
                    )
                ],
            )
        if request.mode == "bbox":
            if request.bbox_xyxy is None:
                raise TargetResolveError("invalid_target_request", "bbox_xyxy is required")
            return TargetPreview(
                status="resolved",
                candidates=[
                    TargetCandidate(
                        target_type="region",
                        track_id=None,
                        bbox_xyxy=_checked_bbox(
                            request.bbox_xyxy,
                            image_width=image_width,
                            image_height=image_height,
                            min_box_size_px=self.min_box_size_px,
                        ),
                        confidence=1.0,
                        reason="bbox",
                    )
                ],
            )
        if request.mode == "point_uv":
            if request.point_uv is None:
                raise TargetResolveError("invalid_target_request", "point_uv is required")
            return self._preview_point(
                request.point_uv,
                image_width=image_width,
                image_height=image_height,
                tracks=tracks,
                attention=attention,
            )
        raise TargetResolveError(
            "unsupported_target_mode",
            f"unsupported target mode {request.mode}",
        )

    def resolve_candidates(
        self,
        request: TargetRequest,
        *,
        image_width: int,
        image_height: int,
        tracks: list[TrackSnapshot],
        attention: AttentionResult | None,
    ) -> TargetPreview:
        return self.preview(
            request,
            image_width=image_width,
            image_height=image_height,
            tracks=tracks,
            attention=attention,
        )

    def preview_pose_pointing_person(
        self,
        *,
        introducer_track_id: int,
        image_width: int,
        image_height: int,
        tracks: list[TrackSnapshot],
    ) -> TargetPreview:
        if image_width <= 0 or image_height <= 0:
            raise TargetResolveError("invalid_frame_size", "image dimensions must be positive")

        visible_tracks = self._usable_person_tracks(
            tracks,
            image_width=image_width,
            image_height=image_height,
        )
        introducer = next(
            (track for track in visible_tracks if track.track_id == introducer_track_id),
            None,
        )
        if introducer is None:
            return TargetPreview(
                status="ambiguous",
                candidates=[],
                ambiguity_type="no_active_interaction_target",
                evidence={
                    "pose_visual_evidence": {
                        "coordinate_space": "source_frame",
                        "introducer_track_id": introducer_track_id,
                        "ambiguity_type": "no_active_interaction_target",
                        "candidate_scores": [],
                    }
                },
            )

        arms = _pointing_arms(introducer)
        if not arms:
            return TargetPreview(
                status="ambiguous",
                candidates=[],
                ambiguity_type="pose_unclear",
                evidence={
                    "pose_visual_evidence": _pose_visual_evidence(
                        introducer=introducer,
                        selected_track=None,
                        arm=None,
                        score_entries=[],
                        image_width=image_width,
                        image_height=image_height,
                        ambiguity_type="pose_unclear",
                    )
                },
            )

        candidate_tracks = [
            track for track in visible_tracks if track.track_id != introducer_track_id
        ]
        multiple_ray_hits = any(
            _arm_has_multiple_ray_hits(arm, candidate_tracks) for arm in arms
        )
        multiple_ray_hit_checks = (
            {"multiple_ray_hits": True} if multiple_ray_hits else None
        )
        score_entries = _pose_candidate_scores(
            candidate_tracks,
            arms=arms,
            image_width=image_width,
            image_height=image_height,
        )
        positive_entries = [entry for entry in score_entries if entry.score > 0.0]
        if not positive_entries:
            arm = arms[0]
            checks = {"candidate_found": False}
            if multiple_ray_hit_checks is not None:
                checks.update(multiple_ray_hit_checks)
            scoring = _pose_pointing_scoring_evidence(
                arm=arm,
                score_entries=score_entries,
                score_margin=None,
                ambiguous_score_margin=self.ambiguous_score_margin,
                margin_ok=False,
                checks=checks,
            )
            return TargetPreview(
                status="not_found",
                candidates=[],
                ambiguity_type="target_unclear",
                evidence=_pose_pointing_evidence(
                    scoring=scoring,
                    introducer=introducer,
                    selected_track=None,
                    arm=arm,
                    score_entries=score_entries,
                    image_width=image_width,
                    image_height=image_height,
                    ambiguity_type="target_unclear",
                ),
            )

        selected_arm = positive_entries[0].arm
        score_margin = _pose_score_margin(positive_entries)
        margin_ok = True
        if len(positive_entries) > 1:
            if (
                score_margin is not None
                and score_margin <= self.ambiguous_score_margin
            ):
                margin_ok = False
                scoring = _pose_pointing_scoring_evidence(
                    arm=selected_arm,
                    score_entries=score_entries,
                    score_margin=score_margin,
                    ambiguous_score_margin=self.ambiguous_score_margin,
                    margin_ok=margin_ok,
                    checks=multiple_ray_hit_checks,
                )
                evidence = _pose_pointing_evidence(
                    scoring=scoring,
                    introducer=introducer,
                    selected_track=None,
                    arm=selected_arm,
                    score_entries=score_entries,
                    image_width=image_width,
                    image_height=image_height,
                    ambiguity_type="multiple_candidates",
                )
                candidates = [
                    self._candidate_from_track(
                        entry.track,
                        confidence=entry.score,
                        reason="pose_pointing_to_person",
                        image_width=image_width,
                        image_height=image_height,
                        evidence=evidence,
                    )
                    for entry in positive_entries
                ]
                return TargetPreview(
                    status="ambiguous",
                    candidates=candidates[:3],
                    ambiguity_type="multiple_candidates",
                    evidence=evidence,
                )

        scoring = _pose_pointing_scoring_evidence(
            arm=selected_arm,
            score_entries=score_entries,
            score_margin=score_margin,
            ambiguous_score_margin=self.ambiguous_score_margin,
            margin_ok=margin_ok,
            checks=multiple_ray_hit_checks,
        )
        selected = positive_entries[0]
        evidence = _pose_pointing_evidence(
            scoring=scoring,
            introducer=introducer,
            selected_track=selected.track,
            arm=selected_arm,
            score_entries=score_entries,
            image_width=image_width,
            image_height=image_height,
            ambiguity_type=None,
        )
        return TargetPreview(
            status="resolved",
            candidates=[
                self._candidate_from_track(
                    selected.track,
                    confidence=selected.score,
                    reason="pose_pointing_to_person",
                    image_width=image_width,
                    image_height=image_height,
                    evidence=evidence,
                )
            ],
            evidence=evidence,
        )

    def resolve(
        self,
        request: TargetRequest,
        *,
        image_width: int,
        image_height: int,
        tracks: list[TrackSnapshot],
        attention: AttentionResult | None,
    ) -> ResolvedTarget:
        if image_width <= 0 or image_height <= 0:
            raise TargetResolveError("invalid_frame_size", "image dimensions must be positive")

        if request.mode == "scene":
            return ResolvedTarget(
                source_target_mode="scene",
                target_type="scene",
                bbox_xyxy=(0.0, 0.0, float(image_width), float(image_height)),
                track_id=None,
                quality="usable",
            )
        if request.mode == "attention_target":
            if attention is None:
                raise TargetResolveError("attention_target_missing", "attention target is missing")
            track = _visible_track_by_id(tracks, attention.target_track_id)
            if track is None:
                raise TargetResolveError("target_not_visible", "attention target is not visible")
            return self._target_from_track(
                track,
                source_target_mode="attention_target",
                image_width=image_width,
                image_height=image_height,
            )
        if request.mode == "track_id":
            if request.track_id is None:
                raise TargetResolveError("invalid_target_request", "track_id is required")
            track = _visible_track_by_id(tracks, request.track_id)
            if track is None:
                raise TargetResolveError("target_not_visible", "track is not visible")
            return self._target_from_track(
                track,
                source_target_mode="track_id",
                image_width=image_width,
                image_height=image_height,
            )
        if request.mode == "bbox":
            if request.bbox_xyxy is None:
                raise TargetResolveError("invalid_target_request", "bbox_xyxy is required")
            return self._target_from_bbox(
                request.bbox_xyxy,
                source_target_mode="bbox",
                image_width=image_width,
                image_height=image_height,
            )
        if request.mode == "point_uv":
            if request.point_uv is None:
                raise TargetResolveError("invalid_target_request", "point_uv is required")
            preview = self._preview_point(
                request.point_uv,
                image_width=image_width,
                image_height=image_height,
                tracks=tracks,
                attention=attention,
            )
            if preview.status == "ambiguous":
                raise TargetResolveError("target_ambiguous", "target point is ambiguous")
            return self._target_from_point(
                request.point_uv,
                image_width=image_width,
                image_height=image_height,
                tracks=tracks,
            )
        raise TargetResolveError(
            "unsupported_target_mode",
            f"unsupported target mode {request.mode}",
        )

    def _target_from_track(
        self,
        track: TrackSnapshot,
        *,
        source_target_mode: str,
        image_width: int,
        image_height: int,
    ) -> ResolvedTarget:
        bbox = _checked_bbox(
            track.bbox_xyxy,
            image_width=image_width,
            image_height=image_height,
            min_box_size_px=self.min_box_size_px,
        )
        return ResolvedTarget(
            source_target_mode=source_target_mode,
            target_type="person",
            bbox_xyxy=bbox,
            track_id=track.track_id,
            quality="usable",
        )

    def _target_from_bbox(
        self,
        bbox: BBoxXYXY,
        *,
        source_target_mode: str,
        image_width: int,
        image_height: int,
    ) -> ResolvedTarget:
        return ResolvedTarget(
            source_target_mode=source_target_mode,
            target_type="region",
            bbox_xyxy=_checked_bbox(
                bbox,
                image_width=image_width,
                image_height=image_height,
                min_box_size_px=self.min_box_size_px,
            ),
            track_id=None,
            quality="usable",
        )

    def _candidate_from_track(
        self,
        track: TrackSnapshot,
        *,
        confidence: float,
        reason: str,
        image_width: int,
        image_height: int,
        evidence: dict[str, Any] | None = None,
    ) -> TargetCandidate:
        return TargetCandidate(
            target_type="person",
            track_id=track.track_id,
            bbox_xyxy=_checked_bbox(
                track.bbox_xyxy,
                image_width=image_width,
                image_height=image_height,
                min_box_size_px=self.min_box_size_px,
            ),
            confidence=_clamp_confidence(confidence),
            reason=reason,
            evidence=evidence or {},
        )

    def _preview_point(
        self,
        point_uv: tuple[float, float],
        *,
        image_width: int,
        image_height: int,
        tracks: list[TrackSnapshot],
        attention: AttentionResult | None,
    ) -> TargetPreview:
        x, y = point_uv
        if not (0.0 <= float(x) <= float(image_width)) or not (
            0.0 <= float(y) <= float(image_height)
        ):
            raise TargetResolveError("point_out_of_bounds", "point_uv is outside image bounds")

        visible_tracks = self._usable_person_tracks(
            tracks,
            image_width=image_width,
            image_height=image_height,
        )
        if not visible_tracks:
            return TargetPreview(
                status="not_found",
                candidates=[
                    TargetCandidate(
                        target_type="region",
                        track_id=None,
                        bbox_xyxy=self._point_region_bbox(
                            point_uv,
                            image_width=image_width,
                            image_height=image_height,
                        ),
                        confidence=0.2,
                        reason="point_region_fallback",
                    )
                ],
            )

        containing_tracks = [
            track for track in visible_tracks if _point_in_bbox(point_uv, track.bbox_xyxy)
        ]
        if containing_tracks:
            candidates = [
                self._candidate_from_track(
                    track,
                    confidence=self._point_inside_confidence(
                        track,
                        image_width=image_width,
                        image_height=image_height,
                        attention=attention,
                    ),
                    reason="point_inside_bbox",
                    image_width=image_width,
                    image_height=image_height,
                )
                for track in containing_tracks
            ]
            candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
            return TargetPreview(
                status="resolved" if len(candidates) == 1 else "ambiguous",
                candidates=candidates,
            )

        scored: list[tuple[float, TargetCandidate]] = []
        distance_scale = max(float(min(image_width, image_height)) * 0.45, self.min_box_size_px)
        for track in visible_tracks:
            distance = _point_to_bbox_distance(point_uv, track.bbox_xyxy)
            proximity = max(0.0, 1.0 - (distance / distance_scale))
            confidence = 0.25 + (0.55 * proximity)
            confidence += self._attention_bonus(track, attention)
            scored.append(
                (
                    distance,
                    self._candidate_from_track(
                        track,
                        confidence=confidence,
                        reason="nearest_bbox_to_point",
                        image_width=image_width,
                        image_height=image_height,
                    ),
                )
            )
        scored.sort(key=lambda item: (-item[1].confidence, item[0], item[1].track_id or -1))
        candidates = [candidate for _, candidate in scored]
        if not candidates or candidates[0].confidence < 0.35:
            return TargetPreview(status="not_found", candidates=candidates[:3])
        if len(candidates) > 1:
            score_margin = candidates[0].confidence - candidates[1].confidence
            if score_margin <= self.ambiguous_score_margin:
                return TargetPreview(status="ambiguous", candidates=candidates[:3])
        return TargetPreview(status="resolved", candidates=[candidates[0]])

    def _usable_person_tracks(
        self,
        tracks: list[TrackSnapshot],
        *,
        image_width: int,
        image_height: int,
    ) -> list[TrackSnapshot]:
        usable: list[TrackSnapshot] = []
        for track in tracks:
            if track.lost_ms != 0 or track.class_name != "person":
                continue
            try:
                _checked_bbox(
                    track.bbox_xyxy,
                    image_width=image_width,
                    image_height=image_height,
                    min_box_size_px=self.min_box_size_px,
                )
            except TargetResolveError:
                continue
            usable.append(track)
        return usable

    def _point_inside_confidence(
        self,
        track: TrackSnapshot,
        *,
        image_width: int,
        image_height: int,
        attention: AttentionResult | None,
    ) -> float:
        image_area = max(float(image_width) * float(image_height), 1.0)
        area_ratio = bbox_area(track.bbox_xyxy) / image_area
        smaller_box_bonus = max(0.0, min(0.16, (1.0 - area_ratio) * 0.16))
        return 0.78 + smaller_box_bonus + self._attention_bonus(track, attention)

    def _attention_bonus(
        self,
        track: TrackSnapshot,
        attention: AttentionResult | None,
    ) -> float:
        if attention is None or attention.target_track_id != track.track_id:
            return 0.0
        return self.attention_prior_weight * _clamp_confidence(attention.confidence)

    def _point_region_bbox(
        self,
        point_uv: tuple[float, float],
        *,
        image_width: int,
        image_height: int,
    ) -> BBoxXYXY:
        x, y = point_uv
        half_size = max(float(min(image_width, image_height)) * 0.08, self.min_box_size_px)
        return _checked_bbox(
            (x - half_size, y - half_size, x + half_size, y + half_size),
            image_width=image_width,
            image_height=image_height,
            min_box_size_px=self.min_box_size_px,
        )

    def _target_from_point(
        self,
        point_uv: tuple[float, float],
        *,
        image_width: int,
        image_height: int,
        tracks: list[TrackSnapshot],
    ) -> ResolvedTarget:
        x, y = point_uv
        if not (0.0 <= float(x) <= float(image_width)) or not (
            0.0 <= float(y) <= float(image_height)
        ):
            raise TargetResolveError("point_out_of_bounds", "point_uv is outside image bounds")
        containing_tracks = [
            track
            for track in tracks
            if track.lost_ms == 0 and _point_in_bbox(point_uv, track.bbox_xyxy)
        ]
        if containing_tracks:
            selected = min(containing_tracks, key=lambda track: bbox_area(track.bbox_xyxy))
            return self._target_from_track(
                selected,
                source_target_mode="point_uv",
                image_width=image_width,
                image_height=image_height,
            )

        return self._target_from_bbox(
            self._point_region_bbox(
                point_uv,
                image_width=image_width,
                image_height=image_height,
            ),
            source_target_mode="point_uv",
            image_width=image_width,
            image_height=image_height,
        )


def _visible_track_by_id(
    tracks: list[TrackSnapshot],
    track_id: int,
) -> TrackSnapshot | None:
    for track in tracks:
        if track.track_id == track_id and track.lost_ms == 0:
            return track
    return None


def _checked_bbox(
    bbox: BBoxXYXY,
    *,
    image_width: int,
    image_height: int,
    min_box_size_px: float,
) -> BBoxXYXY:
    clipped = clip_bbox(bbox, image_width, image_height)
    x1, y1, x2, y2 = clipped
    if x2 <= x1 or y2 <= y1:
        raise TargetResolveError("target_empty", "target bbox is empty")
    if x2 - x1 < min_box_size_px or y2 - y1 < min_box_size_px:
        raise TargetResolveError("target_too_small", "target bbox is too small")
    return clipped


def _point_in_bbox(point_uv: tuple[float, float], bbox: BBoxXYXY) -> bool:
    x, y = point_uv
    x1, y1, x2, y2 = bbox
    return float(x1) <= float(x) <= float(x2) and float(y1) <= float(y) <= float(y2)


def _point_to_bbox_distance(point_uv: tuple[float, float], bbox: BBoxXYXY) -> float:
    x, y = point_uv
    x1, y1, x2, y2 = bbox
    dx = max(float(x1) - float(x), 0.0, float(x) - float(x2))
    dy = max(float(y1) - float(y), 0.0, float(y) - float(y2))
    return hypot(dx, dy)


def _pointing_arms(
    track: TrackSnapshot,
) -> list[_PointingArm]:
    arms: list[_PointingArm] = []
    min_length = max(
        24.0,
        min(_bbox_width(track.bbox_xyxy), _bbox_height(track.bbox_xyxy)) * 0.25,
    )
    for side in ("left", "right"):
        shoulder = _keypoint_by_name(track.keypoints, f"{side}_shoulder")
        elbow = _keypoint_by_name(track.keypoints, f"{side}_elbow")
        wrist = _keypoint_by_name(track.keypoints, f"{side}_wrist")
        if not (
            _valid_pose_keypoint(shoulder)
            and _valid_pose_keypoint(elbow)
            and _valid_pose_keypoint(wrist)
        ):
            continue
        assert shoulder is not None
        assert elbow is not None
        assert wrist is not None
        direction = (
            float(wrist.x) - float(shoulder.x),
            float(wrist.y) - float(shoulder.y),
        )
        if hypot(direction[0], direction[1]) < min_length:
            continue
        assert shoulder.confidence is not None
        assert elbow.confidence is not None
        assert wrist.confidence is not None
        arms.append(
            _PointingArm(
                side=side,
                origin=(float(wrist.x), float(wrist.y)),
                direction=direction,
                shoulder=(float(shoulder.x), float(shoulder.y)),
                elbow=(float(elbow.x), float(elbow.y)),
                wrist=(float(wrist.x), float(wrist.y)),
                keypoint_confidences={
                    shoulder.name: float(shoulder.confidence),
                    elbow.name: float(elbow.confidence),
                    wrist.name: float(wrist.confidence),
                },
            )
        )
    return arms


def _pose_candidate_scores(
    candidate_tracks: list[TrackSnapshot],
    *,
    arms: list[_PointingArm],
    image_width: int,
    image_height: int,
) -> list[_PoseCandidateScore]:
    score_entries: list[_PoseCandidateScore] = []
    for candidate_track in candidate_tracks:
        best_entry: _PoseCandidateScore | None = None
        for arm in arms:
            score, perp = _pose_pointing_score(
                origin=arm.origin,
                direction=arm.direction,
                candidate_bbox=candidate_track.bbox_xyxy,
                image_width=image_width,
                image_height=image_height,
            )
            entry = _PoseCandidateScore(
                track=candidate_track,
                score=score,
                perpendicular_distance=perp,
                arm=arm,
                ray_intersects_bbox=_ray_intersects_bbox(
                    arm.origin,
                    arm.direction,
                    candidate_track.bbox_xyxy,
                ),
            )
            if best_entry is None or entry.score > best_entry.score:
                best_entry = entry
        if best_entry is not None:
            score_entries.append(best_entry)
    score_entries.sort(
        key=lambda entry: (
            -entry.score,
            entry.perpendicular_distance,
            entry.track.track_id or -1,
        )
    )
    return score_entries


def _pose_score_margin(score_entries: list[_PoseCandidateScore]) -> float | None:
    if len(score_entries) < 2:
        return None
    return float(score_entries[0].score) - float(score_entries[1].score)


def _arm_has_multiple_ray_hits(
    arm: _PointingArm,
    candidate_tracks: list[TrackSnapshot],
) -> bool:
    hit_count = 0
    for track in candidate_tracks:
        if _ray_intersects_bbox(arm.origin, arm.direction, track.bbox_xyxy):
            hit_count += 1
            if hit_count > 1:
                return True
    return False


def _pose_pointing_evidence(
    *,
    scoring: dict[str, Any],
    introducer: TrackSnapshot,
    selected_track: TrackSnapshot | None,
    arm: _PointingArm,
    score_entries: list[_PoseCandidateScore],
    image_width: int,
    image_height: int,
    ambiguity_type: str | None,
) -> dict[str, Any]:
    return {
        "pose_pointing_scoring": scoring,
        "pose_visual_evidence": _pose_visual_evidence(
            introducer=introducer,
            selected_track=selected_track,
            arm=arm,
            score_entries=score_entries,
            image_width=image_width,
            image_height=image_height,
            ambiguity_type=ambiguity_type,
        ),
    }


def _pose_visual_evidence(
    *,
    introducer: TrackSnapshot,
    selected_track: TrackSnapshot | None,
    arm: _PointingArm | None,
    score_entries: list[_PoseCandidateScore],
    image_width: int,
    image_height: int,
    ambiguity_type: str | None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "coordinate_space": "source_frame",
        "introducer_track_id": introducer.track_id,
        "introducer_bbox_xyxy": _bbox_xyxy_list(introducer.bbox_xyxy),
        "candidate_scores": [
            _pose_candidate_score_evidence(entry) for entry in score_entries
        ],
    }
    if ambiguity_type is not None:
        evidence["ambiguity_type"] = ambiguity_type
    if selected_track is not None:
        evidence["target_track_id"] = selected_track.track_id
        evidence["target_bbox_xyxy"] = _bbox_xyxy_list(selected_track.bbox_xyxy)
    if arm is not None:
        evidence.update(
            {
                "arm_side": arm.side,
                "shoulder_xy": _xy_list(arm.shoulder),
                "elbow_xy": _xy_list(arm.elbow),
                "wrist_xy": _xy_list(arm.wrist),
                "ray_start_xy": _xy_list(arm.origin),
                "ray_end_xy": _xy_list(
                    _ray_end_xy(
                        arm.origin,
                        arm.direction,
                        image_width=image_width,
                        image_height=image_height,
                    )
                ),
            }
        )
    return evidence


def _pose_candidate_score_evidence(
    entry: _PoseCandidateScore,
) -> dict[str, Any]:
    return {
        "track_id": entry.track.track_id,
        "bbox_xyxy": _bbox_xyxy_list(entry.track.bbox_xyxy),
        "score": float(entry.score),
        "arm_side": entry.arm.side,
        "perpendicular_distance": float(entry.perpendicular_distance),
        "ray_intersects_bbox": bool(entry.ray_intersects_bbox),
    }


def _pose_pointing_scoring_evidence(
    *,
    arm: _PointingArm,
    score_entries: list[_PoseCandidateScore],
    score_margin: float | None,
    ambiguous_score_margin: float,
    margin_ok: bool,
    checks: dict[str, bool] | None = None,
) -> dict[str, Any]:
    all_checks = {"keypoints_ok": True, "margin_ok": margin_ok}
    if checks is not None:
        all_checks.update(checks)
    return {
        "arm_side": arm.side,
        "keypoint_confidences": arm.keypoint_confidences,
        "arm_vector": [float(arm.direction[0]), float(arm.direction[1])],
        "candidate_scores": [
            _pose_candidate_score_evidence(entry) for entry in score_entries
        ],
        "score_margin": None if score_margin is None else float(score_margin),
        "ambiguous_score_margin": float(ambiguous_score_margin),
        "checks": all_checks,
    }


def _pose_pointing_score(
    *,
    origin: tuple[float, float],
    direction: tuple[float, float],
    candidate_bbox: BBoxXYXY,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    dir_len = hypot(direction[0], direction[1])
    if dir_len <= 0.0:
        return 0.0, 0.0
    ray_hit_t = _ray_bbox_intersection_t(origin, direction, candidate_bbox)
    ray_intersects_bbox = ray_hit_t is not None
    target = _person_pointing_target(candidate_bbox)
    to_target = (target[0] - origin[0], target[1] - origin[1])
    forward = (to_target[0] * direction[0] + to_target[1] * direction[1]) / dir_len
    if forward <= 0.0 and not ray_intersects_bbox:
        return 0.0, 0.0
    perp = abs(direction[0] * to_target[1] - direction[1] * to_target[0]) / dir_len
    candidate_size = min(_bbox_width(candidate_bbox), _bbox_height(candidate_bbox))
    tolerance = max(60.0, candidate_size * 0.45)
    if ray_intersects_bbox:
        hit_distance = max(0.0, float(ray_hit_t) * dir_len)
        ray_scale = max(float(max(image_width, image_height)) * 0.45, 1.0)
        center_alignment = max(0.0, 1.0 - min(perp / tolerance, 1.0))
        proximity = max(0.0, 1.0 - min(hit_distance / ray_scale, 1.0))
        return _clamp_confidence(
            0.52 + (0.08 * center_alignment) + (0.40 * proximity)
        ), perp
    if perp > tolerance:
        return 0.0, perp
    forward_scale = max(float(max(image_width, image_height)) * 0.45, 1.0)
    alignment = 1.0 - (perp / tolerance)
    forward_bonus = min(1.0, forward / forward_scale)
    return _clamp_confidence(0.35 + (0.55 * alignment) + (0.10 * forward_bonus)), perp


def _ray_bbox_intersection_t(
    origin: tuple[float, float],
    direction: tuple[float, float],
    bbox: BBoxXYXY,
) -> float | None:
    x1, y1, x2, y2 = bbox
    ox, oy = origin
    dx, dy = direction
    t_min = 0.0
    t_max = float("inf")
    for start, delta, lower, upper in (
        (ox, dx, float(x1), float(x2)),
        (oy, dy, float(y1), float(y2)),
    ):
        if abs(delta) < 1e-9:
            if start < lower or start > upper:
                return None
            continue
        t1 = (lower - start) / delta
        t2 = (upper - start) / delta
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return None
    if t_max < 0.0:
        return None
    return max(t_min, 0.0)


def _ray_intersects_bbox(
    origin: tuple[float, float],
    direction: tuple[float, float],
    bbox: BBoxXYXY,
) -> bool:
    return _ray_bbox_intersection_t(origin, direction, bbox) is not None


def _ray_end_xy(
    origin: tuple[float, float],
    direction: tuple[float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    ox, oy = origin
    dx, dy = direction
    candidates: list[float] = []
    if dx > 0.0:
        candidates.append((float(image_width) - ox) / dx)
    elif dx < 0.0:
        candidates.append((0.0 - ox) / dx)
    if dy > 0.0:
        candidates.append((float(image_height) - oy) / dy)
    elif dy < 0.0:
        candidates.append((0.0 - oy) / dy)
    positive = [t for t in candidates if isfinite(t) and t >= 0.0]
    if not positive:
        return origin
    t = min(positive)
    return (
        max(0.0, min(float(image_width), ox + (dx * t))),
        max(0.0, min(float(image_height), oy + (dy * t))),
    )


def _person_pointing_target(bbox: BBoxXYXY) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, float(y1) + (_bbox_height(bbox) * 0.45))


def _keypoint_by_name(
    keypoints: tuple[PoseKeypoint, ...],
    name: str,
) -> PoseKeypoint | None:
    for keypoint in keypoints:
        if keypoint.name == name:
            return keypoint
    return None


def _valid_pose_keypoint(keypoint: PoseKeypoint | None) -> bool:
    if keypoint is None:
        return False
    confidence = keypoint.confidence
    return (
        isfinite(float(keypoint.x))
        and isfinite(float(keypoint.y))
        and confidence is not None
        and isfinite(float(confidence))
        and float(confidence) >= 0.3
    )


def _bbox_width(bbox: BBoxXYXY) -> float:
    x1, _, x2, _ = bbox
    return max(0.0, float(x2) - float(x1))


def _bbox_height(bbox: BBoxXYXY) -> float:
    _, y1, _, y2 = bbox
    return max(0.0, float(y2) - float(y1))


def _bbox_xyxy_list(bbox: BBoxXYXY) -> list[float]:
    return [float(value) for value in bbox]


def _xy_list(point: tuple[float, float]) -> list[float]:
    return [float(point[0]), float(point[1])]


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
