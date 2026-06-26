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


class SequenceResponseWebSocket(FakeWebSocket):
    def __init__(self, response_tracks):
        super().__init__()
        self.response_tracks = list(response_tracks)

    async def recv(self):
        assert self.awaiting_recv is True
        self.awaiting_recv = False
        frame = decode_frame_message(self.sent_payloads[-1])
        tracks = self.response_tracks.pop(0)
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
                "attention": None,
                "scene_flags": {
                    "has_person": bool(tracks),
                    "person_count": len(tracks),
                    "largest_person_stable": False,
                    "someone_near_center": False,
                },
                "semantic_events": [],
            }
        )


class SequenceResponseConnect(FakeConnect):
    def __init__(self, response_tracks):
        self.websocket = SequenceResponseWebSocket(response_tracks)
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
            "elapsed_s": 0.25,
        }
    ]
