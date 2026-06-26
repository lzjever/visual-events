import json
from pathlib import Path

import pytest

from tools.replay_val_data import (
    ReplayStats,
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
    )

    assert isinstance(stats, ReplayStats)
    assert stats.frames_sent == 2
    assert stats.frames_ok == 2
    assert stats.errors == 0
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
