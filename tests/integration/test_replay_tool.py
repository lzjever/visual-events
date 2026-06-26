import asyncio
import json
from pathlib import Path

import pytest

from tools.replay_val_data import (
    ReplayStats,
    _stats_passed,
    async_main,
    discover_scene_dirs,
    iter_scene_frames,
    parse_args,
    replay_scene,
)
from visual_events_server.protocol import decode_frame_message


JPEG_BYTES = b"\xff\xd8\xff\xe0minimal-jpeg\xff\xd9"


def write_jpeg(path: Path) -> None:
    path.write_bytes(JPEG_BYTES)


def test_iter_scene_frames_sorts_by_filename_timestamp_and_builds_headers(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000200000000.jpeg")
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    write_jpeg(scene / "img_1710000000100000000.jpeg")

    frames = list(
        iter_scene_frames(
            scene,
            camera="front",
            fps=10,
            head_motion="unknown",
        )
    )

    assert [frame.path.name for frame in frames] == [
        "img_1710000000000000000.jpeg",
        "img_1710000000100000000.jpeg",
        "img_1710000000200000000.jpeg",
    ]
    assert [frame.header["frame_id"] for frame in frames] == [0, 1, 2]
    assert [frame.header["timestamp_ms"] for frame in frames] == [
        1710000000000,
        1710000000100,
        1710000000200,
    ]
    assert frames[0].header["head_motion"] == {"state": "unknown"}
    assert frames[0].header["width"] == 1280
    assert frames[0].header["height"] == 720


def test_discover_scene_dirs_accepts_single_scene_or_val_data_root(tmp_path):
    single_scene = tmp_path / "pic_hello"
    single_scene.mkdir()
    write_jpeg(single_scene / "img_1.jpeg")

    all_scenes = tmp_path / "val-data"
    nested_scene = all_scenes / "pic_leave"
    nested_scene.mkdir(parents=True)
    write_jpeg(nested_scene / "img_2.jpeg")

    assert discover_scene_dirs(single_scene) == [single_scene]
    assert discover_scene_dirs(all_scenes) == [nested_scene]


class FakeWebSocket:
    def __init__(self):
        self.sent_payloads = []
        self.awaiting_recv = False

    async def send(self, payload):
        assert self.awaiting_recv is False
        self.sent_payloads.append(payload)
        self.awaiting_recv = True

    async def recv(self):
        assert self.awaiting_recv is True
        self.awaiting_recv = False
        frame = decode_frame_message(self.sent_payloads[-1])
        return json.dumps(
            {
                "type": "visual_state",
                "schema_version": 1,
                "camera": frame.camera,
                "frame_id": frame.frame_id,
                "frame_timestamp_ms": frame.timestamp_ms,
                "server_timestamp_ms": frame.timestamp_ms + 5,
                "image_size": [frame.width, frame.height],
                "tracks": [],
                "attention": None,
                "scene_flags": {
                    "has_person": False,
                    "person_count": 0,
                    "largest_person_stable": False,
                    "someone_near_center": False,
                },
                "semantic_events": [],
            }
        )


class FakeConnect:
    def __init__(self):
        self.websocket = FakeWebSocket()
        self.urls = []

    def __call__(self, url, *, max_size=None):
        self.urls.append((url, max_size))
        return self

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb):
        return False


def track_payload(
    track_id: int,
    *,
    age_ms: int,
    bbox_area_ratio: float = 0.1,
    bbox_xyxy: list[float] | None = None,
    lost_ms: int = 0,
) -> dict:
    bbox = bbox_xyxy or [10.0, 20.0, 110.0, 220.0]
    center_uv = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    return {
        "track_id": track_id,
        "class": "person",
        "bbox_xyxy": bbox,
        "bbox_area_ratio": bbox_area_ratio,
        "center_uv": center_uv,
        "head_uv": [center_uv[0], bbox[1] + ((bbox[3] - bbox[1]) * 0.28)],
        "velocity_uv_s": [0.0, 0.0],
        "age_ms": age_ms,
        "lost_ms": lost_ms,
        "confidence": 0.9,
        "pose_confidence": 0.0,
    }


def event_payload(
    event: str = "person_waving",
    *,
    event_id: str = "front:evt_000001",
    camera: str = "front",
    track_id: int = 1,
    confidence: float = 0.9,
    duration_ms: int = 1200,
    text: str = "有人在挥手",
) -> dict:
    return {
        "type": "semantic_event",
        "event_id": event_id,
        "event": event,
        "camera": camera,
        "track_id": track_id,
        "confidence": confidence,
        "duration_ms": duration_ms,
        "text": text,
    }


def attention_payload(
    track_id: int,
    *,
    target_uv: list[float] | None = None,
    reason: str = "largest_stable_person",
    confidence: float = 0.9,
) -> dict:
    return {
        "target_track_id": track_id,
        "target_uv": target_uv or [60.0, 76.0],
        "reason": reason,
        "confidence": confidence,
    }


class SequenceResponseWebSocket(FakeWebSocket):
    def __init__(self, response_tracks, response_attentions=None, response_events=None):
        super().__init__()
        self.response_tracks = list(response_tracks)
        self.response_attentions = (
            [None for _ in response_tracks]
            if response_attentions is None
            else list(response_attentions)
        )
        self.response_events = (
            [[] for _ in response_tracks] if response_events is None else list(response_events)
        )

    async def recv(self):
        assert self.awaiting_recv is True
        self.awaiting_recv = False
        frame = decode_frame_message(self.sent_payloads[-1])
        tracks = self.response_tracks.pop(0)
        attention = self.response_attentions.pop(0)
        events = self.response_events.pop(0)
        return json.dumps(
            {
                "type": "visual_state",
                "schema_version": 1,
                "camera": frame.camera,
                "frame_id": frame.frame_id,
                "frame_timestamp_ms": frame.timestamp_ms,
                "server_timestamp_ms": frame.timestamp_ms + 5,
                "image_size": [frame.width, frame.height],
                "tracks": tracks,
                "attention": attention,
                "scene_flags": {
                    "has_person": bool(tracks),
                    "person_count": len(tracks),
                    "largest_person_stable": attention is not None,
                    "someone_near_center": False,
                },
                "semantic_events": events,
            }
        )


class SequenceResponseConnect(FakeConnect):
    def __init__(self, response_tracks, response_attentions=None, response_events=None):
        self.websocket = SequenceResponseWebSocket(
            response_tracks,
            response_attentions,
            response_events,
        )
        self.urls = []


class HangingWebSocket(FakeWebSocket):
    async def recv(self):
        await asyncio.sleep(1)


class HangingConnect(FakeConnect):
    def __init__(self):
        self.websocket = HangingWebSocket()
        self.urls = []


@pytest.mark.asyncio
async def test_replay_scene_sends_one_frame_at_a_time_and_saves_jsonl(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    write_jpeg(scene / "img_1710000000100000000.jpeg")
    save_jsonl = tmp_path / "visual_state.jsonl"
    connector = FakeConnect()

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        save_jsonl=save_jsonl,
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert isinstance(stats, ReplayStats)
    assert stats.frames_sent == 2
    assert stats.frames_ok == 2
    assert stats.errors == 0
    assert stats.ok_rate == 1.0
    assert stats.frames_with_person == 0
    assert stats.person_frame_rate == 0.0
    assert stats.frame_id_mismatch == 0
    assert connector.urls == [("ws://127.0.0.1:8765/v1/stream", None)]
    assert len(connector.websocket.sent_payloads) == 2
    assert [
        decode_frame_message(payload).frame_id
        for payload in connector.websocket.sent_payloads
    ] == [0, 1]

    lines = [json.loads(line) for line in save_jsonl.read_text().splitlines()]
    assert [line["frame_id"] for line in lines] == [0, 1]
    assert [line["response"]["type"] for line in lines] == ["visual_state", "visual_state"]
    assert all(line["latency_ms"] >= 0 for line in lines)


@pytest.mark.asyncio
async def test_replay_scene_empty_tracks_do_not_pass_s3_tracking_gate(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    connector = FakeConnect()

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.frames_ok == 1
    assert stats.track_frames == 0
    assert stats.visible_counts_by_id == {}
    assert _stats_passed(stats) is False


@pytest.mark.asyncio
async def test_replay_scene_separates_largest_bbox_diagnostic_from_tracking_switches(
    tmp_path,
):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    write_jpeg(scene / "img_1710000000100000000.jpeg")
    write_jpeg(scene / "img_1710000000200000000.jpeg")
    left = [10.0, 20.0, 110.0, 220.0]
    right = [300.0, 20.0, 400.0, 220.0]
    connector = SequenceResponseConnect(
        [
            [
                track_payload(1, age_ms=0, bbox_area_ratio=0.30, bbox_xyxy=left),
                track_payload(2, age_ms=0, bbox_area_ratio=0.20, bbox_xyxy=right),
            ],
            [
                track_payload(1, age_ms=100, bbox_area_ratio=0.20, bbox_xyxy=left),
                track_payload(2, age_ms=100, bbox_area_ratio=0.30, bbox_xyxy=right),
            ],
            [
                track_payload(1, age_ms=200, bbox_area_ratio=0.31, bbox_xyxy=left),
                track_payload(2, age_ms=200, bbox_area_ratio=0.21, bbox_xyxy=right),
            ],
        ]
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.track_frames == 3
    assert stats.largest_bbox_track_switches == 2
    assert stats.largest_bbox_track_id == 1
    assert stats.largest_bbox_track_coverage == pytest.approx(2 / 3)
    assert stats.largest_bbox_track_max_gap_ms == 200
    assert stats.single_visible_id_switches == 0
    assert stats.adjacent_track_matches == 4
    assert stats.association_id_switches == 0
    assert stats.duplicate_track_id_frames == 0
    assert stats.visible_counts_by_id == {"1": 3, "2": 3}
    assert stats.track_schema_errors == 0
    assert stats.age_monotonic_violations == 0


@pytest.mark.asyncio
async def test_replay_scene_counts_association_id_switches_for_high_iou_tracks(
    tmp_path,
):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    write_jpeg(scene / "img_1710000000100000000.jpeg")
    connector = SequenceResponseConnect(
        [
            [
                track_payload(
                    1,
                    age_ms=0,
                    bbox_area_ratio=0.10,
                    bbox_xyxy=[10.0, 20.0, 110.0, 220.0],
                )
            ],
            [
                track_payload(
                    2,
                    age_ms=0,
                    bbox_area_ratio=0.10,
                    bbox_xyxy=[12.0, 20.0, 112.0, 220.0],
                )
            ],
        ]
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.adjacent_track_matches == 1
    assert stats.association_id_switches == 1
    assert stats.single_visible_id_switches == 1
    assert stats.visible_counts_by_id == {"1": 1, "2": 1}


@pytest.mark.asyncio
async def test_replay_scene_counts_duplicate_visible_track_id_frames(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    connector = SequenceResponseConnect(
        [
            [
                track_payload(
                    1,
                    age_ms=0,
                    bbox_area_ratio=0.10,
                    bbox_xyxy=[10.0, 20.0, 110.0, 220.0],
                ),
                track_payload(
                    1,
                    age_ms=0,
                    bbox_area_ratio=0.08,
                    bbox_xyxy=[300.0, 20.0, 400.0, 220.0],
                ),
            ]
        ]
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.duplicate_track_id_frames == 1
    assert stats.visible_counts_by_id == {"1": 1}
    assert stats.track_schema_errors == 0


@pytest.mark.asyncio
async def test_replay_scene_counts_duplicate_track_ids_across_visible_and_lost_tracks(
    tmp_path,
):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    connector = SequenceResponseConnect(
        [
            [
                track_payload(
                    1,
                    age_ms=100,
                    bbox_area_ratio=0.10,
                    bbox_xyxy=[10.0, 20.0, 110.0, 220.0],
                    lost_ms=0,
                ),
                track_payload(
                    1,
                    age_ms=100,
                    bbox_area_ratio=0.08,
                    bbox_xyxy=[300.0, 20.0, 400.0, 220.0],
                    lost_ms=100,
                ),
            ]
        ]
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.duplicate_track_id_frames == 1
    assert stats.visible_counts_by_id == {"1": 1}
    assert _stats_passed(stats) is False


@pytest.mark.asyncio
async def test_replay_scene_counts_track_schema_and_age_violations(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    write_jpeg(scene / "img_1710000000100000000.jpeg")
    write_jpeg(scene / "img_1710000000200000000.jpeg")
    invalid_track = {"track_id": 3, "class": "person"}
    connector = SequenceResponseConnect(
        [
            [track_payload(1, age_ms=100)],
            [track_payload(1, age_ms=50)],
            [invalid_track],
        ]
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.track_frames == 2
    assert stats.track_schema_errors == 1
    assert stats.age_monotonic_violations == 1


@pytest.mark.asyncio
async def test_replay_scene_summarizes_attention_targets_switches_and_lost_hold(
    tmp_path,
):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    for index in range(4):
        write_jpeg(scene / f"img_1710000000{index}00000000.jpeg")
    left = [10.0, 20.0, 110.0, 220.0]
    right = [300.0, 20.0, 440.0, 240.0]
    connector = SequenceResponseConnect(
        [
            [track_payload(1, age_ms=400, bbox_area_ratio=0.10, bbox_xyxy=left)],
            [
                track_payload(1, age_ms=500, bbox_area_ratio=0.10, bbox_xyxy=left),
                track_payload(2, age_ms=500, bbox_area_ratio=0.22, bbox_xyxy=right),
            ],
            [
                track_payload(
                    1,
                    age_ms=600,
                    bbox_area_ratio=0.10,
                    bbox_xyxy=left,
                    lost_ms=300,
                )
            ],
            [track_payload(2, age_ms=700, bbox_area_ratio=0.22, bbox_xyxy=right)],
        ],
        [
            attention_payload(1),
            attention_payload(1),
            attention_payload(1, reason="held_lost_target"),
            attention_payload(2, target_uv=[370.0, 81.6]),
        ],
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.attention_frames == 4
    assert stats.attention_null_frames == 0
    assert stats.attention_coverage == 1.0
    assert stats.attention_target_switches == 1
    assert stats.attention_target_counts_by_id == {"1": 3, "2": 1}
    assert stats.attention_schema_errors == 0
    assert stats.attention_invalid_uv_frames == 0
    assert stats.attention_target_missing_track_frames == 0
    assert stats.attention_target_lost_frames == 1
    assert stats.attention_max_lost_hold_ms == 300
    assert stats.attention_largest_bbox_disagreement_frames == 1
    assert _stats_passed(stats, gate="attention") is True


@pytest.mark.asyncio
async def test_replay_scene_counts_attention_schema_uv_and_missing_target_errors(
    tmp_path,
):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    for index in range(3):
        write_jpeg(scene / f"img_1710000000{index}00000000.jpeg")
    connector = SequenceResponseConnect(
        [
            [track_payload(1, age_ms=400)],
            [track_payload(1, age_ms=500)],
            [track_payload(1, age_ms=600)],
        ],
        [
            {"target_track_id": 1, "reason": "largest_stable_person", "confidence": 0.9},
            attention_payload(1, target_uv=[2000.0, 76.0]),
            attention_payload(2),
        ],
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.attention_frames == 2
    assert stats.attention_schema_errors == 1
    assert stats.attention_invalid_uv_frames == 1
    assert stats.attention_target_missing_track_frames == 1
    assert _stats_passed(stats, gate="attention") is False


@pytest.mark.asyncio
async def test_replay_scene_empty_attention_fails_attention_gate_only(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    connector = SequenceResponseConnect([[track_payload(1, age_ms=400)]])

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.attention_frames == 0
    assert stats.attention_null_frames == 1
    assert stats.attention_coverage == 0.0
    assert _stats_passed(stats, gate="tracking") is True
    assert _stats_passed(stats, gate="attention") is False
    assert _stats_passed(stats, gate="none") is True


def test_stable_attention_scene_low_coverage_fails_attention_gate():
    stats = ReplayStats(
        scene="pci_stand",
        frames_sent=10,
        frames_ok=10,
        errors=0,
        elapsed_s=1.0,
        attention_frames=1,
        attention_null_frames=9,
        attention_target_counts_by_id={"1": 1},
    )

    assert stats.attention_coverage == 0.1
    assert _stats_passed(stats, gate="attention") is False


def test_stable_attention_scene_excessive_switches_fail_attention_gate():
    stats = ReplayStats(
        scene="pic_walk_in_stop",
        frames_sent=10,
        frames_ok=10,
        errors=0,
        elapsed_s=1.0,
        attention_frames=10,
        attention_target_switches=3,
        attention_target_counts_by_id={"1": 4, "2": 3, "3": 3},
    )

    assert stats.attention_coverage == 1.0
    assert _stats_passed(stats, gate="attention") is False


def test_non_stable_attention_scene_uses_generic_evidence_gate():
    stats = ReplayStats(
        scene="pic_hello",
        frames_sent=10,
        frames_ok=10,
        errors=0,
        elapsed_s=1.0,
        attention_frames=1,
        attention_null_frames=9,
        attention_target_counts_by_id={"1": 1},
    )

    assert stats.attention_coverage == 0.1
    assert _stats_passed(stats, gate="attention") is True


@pytest.mark.asyncio
async def test_replay_scene_summarizes_semantic_events_and_passes_events_gate(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    write_jpeg(scene / "img_1710000000100000000.jpeg")
    connector = SequenceResponseConnect(
        [[track_payload(1, age_ms=400)], [track_payload(1, age_ms=500)]],
        response_events=[
            [],
            [
                event_payload(
                    "person_waving",
                    event_id="front:evt_000001",
                    track_id=1,
                )
            ],
        ],
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.semantic_event_frames == 1
    assert stats.semantic_event_count == 1
    assert stats.semantic_event_counts_by_type == {"person_waving": 1}
    assert stats.semantic_event_first_frame_by_type == {"person_waving": 1}
    assert stats.semantic_event_expected_missing == 0
    assert _stats_passed(stats, gate="events") is True


@pytest.mark.asyncio
async def test_replay_scene_counts_semantic_event_validation_errors(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    connector = SequenceResponseConnect(
        [[track_payload(1, age_ms=400)]],
        response_events=[
            [
                {"type": "semantic_event", "event_id": "front:evt_000001"},
                event_payload("robot_dancing", event_id="front:evt_000002"),
                event_payload("attention_target_changed", event_id="bad-id"),
                event_payload("person_waving", event_id="front:evt_000003"),
                event_payload("person_appeared", event_id="front:evt_000003"),
                event_payload("person_waving", event_id="front:evt_000004"),
                event_payload(
                    "person_left",
                    event_id="front:evt_000005",
                    confidence=1.2,
                    track_id=99,
                ),
                event_payload(
                    "person_approaching_robot",
                    event_id="front:evt_000006",
                    duration_ms=-1,
                    track_id=99,
                ),
                event_payload(
                    "person_stopped_near_robot",
                    event_id="front:evt_000007",
                    text="",
                ),
            ]
        ],
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.semantic_event_schema_errors == 1
    assert stats.semantic_event_unknown_type_count == 1
    assert stats.semantic_event_id_format_errors == 1
    assert stats.semantic_event_duplicate_id_count == 1
    assert stats.semantic_event_duplicate_track_event_count == 1
    assert stats.semantic_event_type_cooldown_errors == 1
    assert stats.semantic_event_confidence_errors == 1
    assert stats.semantic_event_duration_errors == 1
    assert stats.semantic_event_empty_text_count == 1
    assert stats.semantic_event_track_missing_frames == 1
    assert _stats_passed(stats, gate="events") is False


@pytest.mark.asyncio
async def test_events_gate_fails_same_event_type_across_tracks_within_cooldown(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    write_jpeg(scene / "img_1710000000100000000.jpeg")
    connector = SequenceResponseConnect(
        [
            [
                track_payload(1, age_ms=400),
                track_payload(2, age_ms=400, bbox_xyxy=[300.0, 20.0, 400.0, 220.0]),
            ],
            [
                track_payload(1, age_ms=500),
                track_payload(2, age_ms=500, bbox_xyxy=[300.0, 20.0, 400.0, 220.0]),
            ],
        ],
        response_events=[
            [
                event_payload(
                    "person_waving",
                    event_id="front:evt_000001",
                    track_id=1,
                )
            ],
            [
                event_payload(
                    "person_waving",
                    event_id="front:evt_000002",
                    track_id=2,
                )
            ],
        ],
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.semantic_event_count == 2
    assert stats.semantic_event_duplicate_track_event_count == 0
    assert stats.semantic_event_type_cooldown_errors == 1
    assert _stats_passed(stats, gate="events") is False


@pytest.mark.asyncio
async def test_events_gate_uses_configured_semantic_event_cooldown(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    write_jpeg(scene / "img_1710000000100000000.jpeg")
    connector = SequenceResponseConnect(
        [
            [
                track_payload(1, age_ms=400),
                track_payload(2, age_ms=400, bbox_xyxy=[300.0, 20.0, 400.0, 220.0]),
            ],
            [
                track_payload(1, age_ms=500),
                track_payload(2, age_ms=500, bbox_xyxy=[300.0, 20.0, 400.0, 220.0]),
            ],
        ],
        response_events=[
            [
                event_payload(
                    "person_waving",
                    event_id="front:evt_000001",
                    track_id=1,
                )
            ],
            [
                event_payload(
                    "person_waving",
                    event_id="front:evt_000002",
                    track_id=2,
                )
            ],
        ],
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
        semantic_event_cooldown_ms=50,
    )

    assert stats.semantic_event_cooldown_ms == 50
    assert stats.semantic_event_type_cooldown_errors == 0
    assert _stats_passed(stats, gate="events") is True


@pytest.mark.asyncio
async def test_events_gate_checks_expected_missing_and_unexpected_by_scene(tmp_path):
    scene = tmp_path / "pic_1_l_to_r"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    connector = SequenceResponseConnect(
        [[track_payload(1, age_ms=400)]],
        response_events=[
            [
                event_payload(
                    "person_stopped_near_robot",
                    event_id="front:evt_000001",
                    track_id=1,
                )
            ]
        ],
    )

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert stats.semantic_event_expected_missing == 1
    assert stats.semantic_event_unexpected_by_scene == 1
    assert _stats_passed(stats, gate="events") is False


@pytest.mark.asyncio
async def test_events_gate_head_motion_unknown_skips_stationary_motion_expectations(
    tmp_path,
):
    scene = tmp_path / "pic_walk_in_stop"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    no_event_connector = SequenceResponseConnect([[track_payload(1, age_ms=400)]])
    motion_event_connector = SequenceResponseConnect(
        [[track_payload(1, age_ms=400)]],
        response_events=[
            [
                event_payload(
                    "person_approaching_robot",
                    event_id="front:evt_000001",
                    track_id=1,
                )
            ]
        ],
    )

    no_event_stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="unknown",
        connector=no_event_connector,
        realtime=False,
        response_timeout_ms=250,
    )
    motion_event_stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="moving",
        connector=motion_event_connector,
        realtime=False,
        response_timeout_ms=250,
    )

    assert no_event_stats.semantic_event_expected_missing == 0
    assert no_event_stats.semantic_event_motion_sensitive_count == 0
    assert _stats_passed(no_event_stats, gate="events") is True
    assert motion_event_stats.semantic_event_motion_sensitive_count == 1
    assert _stats_passed(motion_event_stats, gate="events") is False


def test_parse_args_accepts_events_gate_and_all_includes_events():
    args = parse_args(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            "/tmp/val-data",
            "--gate",
            "events",
            "--semantic-event-cooldown-ms",
            "50",
        ]
    )
    stats = ReplayStats(
        scene="pic_hello",
        frames_sent=2,
        frames_ok=2,
        errors=0,
        elapsed_s=0.1,
        track_frames=2,
        visible_counts_by_id={"1": 2},
        attention_frames=1,
        attention_target_counts_by_id={"1": 1},
        semantic_event_expected_missing=1,
    )

    assert args.gate == "events"
    assert args.semantic_event_cooldown_ms == 50
    assert _stats_passed(stats, gate="tracking") is True
    assert _stats_passed(stats, gate="attention") is True
    assert _stats_passed(stats, gate="events") is False
    assert _stats_passed(stats, gate="all") is False


@pytest.mark.asyncio
async def test_replay_scene_response_timeout_records_error_and_returns(tmp_path):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    connector = HangingConnect()

    stats = await replay_scene(
        server="ws://127.0.0.1:8765/v1/stream",
        scene_dir=scene,
        camera="front",
        fps=10,
        head_motion="stationary",
        connector=connector,
        realtime=False,
        response_timeout_ms=1,
    )

    assert stats.frames_sent == 1
    assert stats.frames_ok == 0
    assert stats.errors == 1
    assert stats.ok_rate == 0.0


@pytest.mark.asyncio
async def test_async_main_writes_summary_json(tmp_path, monkeypatch):
    scene = tmp_path / "pic_hello"
    scene.mkdir()
    write_jpeg(scene / "img_1710000000000000000.jpeg")
    summary_json = tmp_path / "summary.json"

    async def fake_replay_data_dir(**kwargs):
        return [
            ReplayStats(
                scene="pic_hello",
                frames_sent=2,
                frames_ok=2,
                errors=0,
                elapsed_s=0.25,
                frames_with_person=1,
                frame_id_mismatch=1,
            )
        ]

    monkeypatch.setattr("tools.replay_val_data.replay_data_dir", fake_replay_data_dir)

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(scene),
            "--summary-json",
            str(summary_json),
        ]
    )

    assert exit_code == 1
    assert json.loads(summary_json.read_text()) == [
        {
            "scene": "pic_hello",
            "frames_sent": 2,
            "frames_ok": 2,
            "errors": 0,
            "ok_rate": 1.0,
            "frames_with_person": 1,
            "person_frame_rate": 0.5,
            "frame_id_mismatch": 1,
            "track_frames": 0,
            "largest_bbox_track_switches": 0,
            "largest_bbox_track_id": None,
            "largest_bbox_track_coverage": 0.0,
            "largest_bbox_track_max_gap_ms": 0,
            "duplicate_track_id_frames": 0,
            "single_visible_id_switches": 0,
            "adjacent_track_matches": 0,
            "association_id_switches": 0,
            "visible_counts_by_id": {},
            "track_schema_errors": 0,
            "age_monotonic_violations": 0,
            "attention_frames": 0,
            "attention_null_frames": 0,
            "attention_coverage": 0.0,
            "attention_target_switches": 0,
            "attention_target_counts_by_id": {},
            "attention_schema_errors": 0,
            "attention_invalid_uv_frames": 0,
            "attention_target_missing_track_frames": 0,
            "attention_target_lost_frames": 0,
            "attention_max_lost_hold_ms": 0,
            "attention_largest_bbox_disagreement_frames": 0,
            "semantic_event_frames": 0,
            "semantic_event_count": 0,
            "semantic_event_counts_by_type": {},
            "semantic_event_first_frame_by_type": {},
            "semantic_event_schema_errors": 0,
            "semantic_event_unknown_type_count": 0,
            "semantic_event_id_format_errors": 0,
            "semantic_event_duplicate_id_count": 0,
            "semantic_event_duplicate_track_event_count": 0,
            "semantic_event_cooldown_ms": 5000,
            "semantic_event_type_cooldown_errors": 0,
            "semantic_event_confidence_errors": 0,
            "semantic_event_duration_errors": 0,
            "semantic_event_empty_text_count": 0,
            "semantic_event_track_missing_frames": 0,
            "semantic_event_motion_sensitive_count": 0,
            "semantic_event_expected_missing": 0,
            "semantic_event_unexpected_by_scene": 0,
            "tracking_pass": False,
            "attention_pass": False,
            "events_pass": False,
            "passed": False,
            "elapsed_s": 0.25,
        }
    ]
