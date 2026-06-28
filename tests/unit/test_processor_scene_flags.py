import asyncio

from visual_events_server.events import EventEngineResult
from visual_events_server.inference.base import PoseDetections
from visual_events_server.processor import BackendVisualStreamSession, build_visual_state
from visual_events_server.protocol import FrameMessage
from visual_events_server.tracking import TrackSnapshot


def frame(*, width: int = 640, height: int = 480) -> FrameMessage:
    return FrameMessage(
        camera="front",
        frame_id=42,
        timestamp_ms=1710000000000,
        width=width,
        height=height,
        jpeg_bytes=b"",
    )


def person_track(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    lost_ms: int = 0,
) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=1,
        first_seen_ms=1710000000000,
        last_seen_ms=1710000000000 - lost_ms,
        frame_timestamp_ms=1710000000000,
        bbox_xyxy=bbox_xyxy,
        confidence=0.9,
        pose_confidence=0.8,
        head_uv=(0.5, 0.25),
        velocity_uv_s=(0.0, 0.0),
        lost_ms=lost_ms,
        hits=1,
        misses=1 if lost_ms > 0 else 0,
        class_name="person",
    )


def test_someone_near_center_true_for_visible_person_centered_bbox():
    visual_state = build_visual_state(
        frame(),
        [person_track((300.0, 220.0, 340.0, 260.0))],
    )

    assert visual_state["scene_flags"]["someone_near_center"] is True


def test_someone_near_center_false_for_visible_person_away_from_center():
    visual_state = build_visual_state(
        frame(),
        [person_track((10.0, 10.0, 50.0, 50.0))],
    )

    assert visual_state["scene_flags"]["someone_near_center"] is False


def test_someone_near_center_false_for_lost_person_centered_bbox():
    visual_state = build_visual_state(
        frame(),
        [person_track((300.0, 220.0, 340.0, 260.0), lost_ms=100)],
    )

    assert visual_state["scene_flags"]["someone_near_center"] is False


def test_default_scene_context_is_present_and_stable():
    visual_state = build_visual_state(frame(), [])

    assert visual_state["scene_context"] == {
        "engagement_state": "no_target",
        "attention_available": False,
        "target_track_id": None,
        "no_engage_reasons": ["no_visible_person"],
        "target_reacquired": None,
    }


def test_custom_scene_context_is_passed_through_unchanged():
    scene_context = {
        "engagement_state": "tracking_target",
        "attention_available": True,
        "target_track_id": 7,
        "no_engage_reasons": [],
        "target_reacquired": False,
        "source": {"engine": "fixture"},
    }

    visual_state = build_visual_state(frame(), [], scene_context=scene_context)

    assert visual_state["scene_context"] is scene_context


def test_scene_context_does_not_change_existing_scene_flags_behavior():
    visual_state = build_visual_state(
        frame(),
        [person_track((300.0, 220.0, 340.0, 260.0))],
        scene_context={
            "engagement_state": "no_target",
            "attention_available": False,
            "target_track_id": None,
            "no_engage_reasons": ["custom_reason"],
            "target_reacquired": None,
        },
    )

    assert visual_state["scene_flags"] == {
        "has_person": True,
        "person_count": 1,
        "largest_person_stable": False,
        "someone_near_center": True,
    }


class EmptyBackend:
    async def infer(self, frame: FrameMessage) -> PoseDetections:
        return PoseDetections(persons=[])


class StaticTracker:
    def __init__(self, tracks: list[TrackSnapshot]) -> None:
        self.tracks = tracks

    def update(
        self,
        frame: FrameMessage,
        detections: PoseDetections,
    ) -> list[TrackSnapshot]:
        return self.tracks

    def reset(self) -> None:
        pass


class NoAttentionSelector:
    def update(
        self,
        frame: FrameMessage,
        tracks: list[TrackSnapshot],
    ) -> None:
        return None

    def reset(self) -> None:
        pass


class StaticEventEngine:
    def __init__(self, result: EventEngineResult) -> None:
        self.result = result

    def update(
        self,
        frame: FrameMessage,
        tracks: list[TrackSnapshot],
        attention: None,
    ) -> EventEngineResult:
        return self.result

    def reset(self) -> None:
        pass


async def test_process_frame_passes_event_engine_scene_context_through():
    scene_context = {
        "engagement_state": "available",
        "attention_available": True,
        "target_track_id": 7,
        "no_engage_reasons": [],
        "target_reacquired": None,
    }
    semantic_events = [
        {
            "type": "semantic_event",
            "event_id": "front:evt_000001",
            "event": "person_appeared",
        }
    ]
    session = BackendVisualStreamSession(
        EmptyBackend(),
        backend_lock=asyncio.Lock(),
        tracker=StaticTracker([person_track((300.0, 220.0, 340.0, 260.0))]),
        attention_selector=NoAttentionSelector(),
        event_engine=StaticEventEngine(
            EventEngineResult(
                semantic_events=semantic_events,
                scene_context=scene_context,
            )
        ),
    )

    visual_state = await session.process_frame(frame())

    assert visual_state["scene_context"] is scene_context
    assert visual_state["semantic_events"] is semantic_events
