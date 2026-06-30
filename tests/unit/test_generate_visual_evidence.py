from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import generate_visual_evidence as module

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeImage:
    def __init__(self, source: Path) -> None:
        self.source = source


def _make_scene(data_dir: Path, scene: str, frame_count: int) -> None:
    scene_dir = data_dir / scene
    scene_dir.mkdir(parents=True)
    for frame_id in range(frame_count):
        (scene_dir / f"frame_{frame_id:03d}.jpg").write_bytes(b"jpeg")


def _state(
    *,
    frame_id: int,
    tracks: list[dict[str, Any]] | None = None,
    attention: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    person_count: int = 0,
    frame_timestamp_ms: int | None = None,
    server_timestamp_ms: int | None = None,
) -> dict[str, Any]:
    state = {
        "type": "visual_state",
        "camera": "front",
        "frame_id": frame_id,
        "scene_flags": {
            "has_person": person_count > 0,
            "person_count": person_count,
        },
        "tracks": tracks or [],
        "attention": attention,
        "semantic_events": events or [],
    }
    if frame_timestamp_ms is not None:
        state["frame_timestamp_ms"] = frame_timestamp_ms
    if server_timestamp_ms is not None:
        state["server_timestamp_ms"] = server_timestamp_ms
    return state


def _event(event_type: str, track_id: int = 7) -> dict[str, Any]:
    return {
        "type": "semantic_event",
        "event": event_type,
        "event_id": f"front:evt_{track_id:06d}",
        "track_id": track_id,
        "confidence": 0.9,
        "duration_ms": 100,
        "text": event_type,
        "evidence": {"runtime_person_slot": 1},
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _wrapped(scene: str, frame_id: int, response: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": scene,
        "frame_id": frame_id,
        "latency_ms": 12.5 + frame_id,
        "response": response,
    }


def _patch_image_io(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {"decode": [], "draw": [], "write": []}

    def fake_decode(path: Path) -> FakeImage:
        calls["decode"].append(path)
        return FakeImage(path)

    def fake_draw(
        image: FakeImage,
        response: dict[str, Any],
        *,
        scene: str,
        frame_id: int,
    ) -> dict[str, Any]:
        calls["draw"].append((image.source, response, scene, frame_id))
        return {"source": image.source, "scene": scene, "frame_id": frame_id}

    def fake_write(path: Path, image: Any) -> None:
        calls["write"].append((path, image))
        path.write_bytes(b"annotated")

    monkeypatch.setattr(module, "_decode_image", fake_decode)
    monkeypatch.setattr(module, "draw_visual_state", fake_draw)
    monkeypatch.setattr(module, "_write_jpeg", fake_write)
    return calls


def test_generate_visual_evidence_direct_script_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "generate_visual_evidence.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Generate artifact-first visual evidence pages" in result.stdout
    assert result.stderr == ""


def test_wrapped_jsonl_generates_root_scene_outputs_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    out = tmp_path / "evidence"
    _make_scene(data_dir, "lobby", 3)
    _make_scene(data_dir, "hall", 1)
    records = [
        _wrapped(
            "lobby",
            0,
            _state(
                frame_id=0,
                tracks=[
                    {
                        "track_id": 7,
                        "bbox_xyxy": [1, 2, 3, 4],
                        "lost_ms": 0,
                    }
                ],
                attention={"target_track_id": 7, "target_uv": [2, 3]},
                events=[],
                person_count=1,
                frame_timestamp_ms=1000,
                server_timestamp_ms=1033,
            ),
        ),
        _wrapped(
            "lobby",
            1,
            _state(
                frame_id=1,
                tracks=[
                    {
                        "track_id": 8,
                        "bbox_xyxy": [1, 2, 5, 6],
                        "lost_ms": 0,
                    }
                ],
                attention={"target_track_id": 8, "target_uv": [3, 4]},
                events=[_event("person_waving", track_id=8)],
                person_count=2,
            ),
        ),
        _wrapped(
            "lobby",
            2,
            {
                "type": "error",
                "code": "response_timeout",
            },
        ),
        _wrapped("hall", 0, _state(frame_id=0, attention=None, person_count=0)),
    ]
    input_jsonl = tmp_path / "artifact" / "visual_state.jsonl"
    _write_jsonl(input_jsonl, records)
    calls = _patch_image_io(monkeypatch)

    summary = module.generate_visual_evidence(
        records=module.read_wrapped_visual_state_jsonl(input_jsonl),
        source_images=module.map_source_images(
            data_dir,
            records,
            camera="front",
            fps=10.0,
            head_motion="stationary",
        ),
        out=out,
        input_jsonl=input_jsonl,
    )

    assert (out / "index.html").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "visual_state.jsonl").is_file()
    assert (out / "scenes" / "lobby" / "index.html").is_file()
    assert (out / "scenes" / "lobby" / "summary.json").is_file()
    assert (out / "scenes" / "lobby" / "visual_state.jsonl").is_file()
    assert (out / "scenes" / "lobby" / "frames" / "000000.jpg").read_bytes() == b"annotated"
    assert (out / "scenes" / "lobby" / "states" / "000001.json").is_file()
    assert len(calls["draw"]) == 4
    assert {call[2:] for call in calls["draw"]} == {
        ("hall", 0),
        ("lobby", 0),
        ("lobby", 1),
        ("lobby", 2),
    }

    assert summary["frames_total"] == 4
    assert summary["frames_ok"] == 3
    assert summary["errors"] == 1
    assert summary["person"]["frames_with_person"] == 2
    assert summary["person"]["max_person_count"] == 2
    assert summary["tracking"]["unique_track_count"] == 2
    assert summary["tracking"]["track_ids"] == [7, 8]
    assert summary["tracking"]["max_tracks_per_frame"] == 1
    assert summary["attention"]["available_frames"] == 2
    assert summary["attention"]["null_frames"] == 1
    assert summary["attention"]["available_ratio"] == pytest.approx(2 / 3)
    assert summary["attention"]["target_switches"] == 1
    assert summary["semantic_events"]["total"] == 1
    assert summary["semantic_events"]["counts_by_type"] == {"person_waving": 1}
    assert summary["semantic_events"]["first_frame_by_type"] == {
        "person_waving": {"scene": "lobby", "frame_id": 1}
    }
    assert summary["keyframes"] == {
        "hall": {
            "first_frame": 0,
            "first_person_frame": None,
            "first_attention_frame": None,
            "first_event_frame_by_type": {},
            "last_frame": 0,
        },
        "lobby": {
            "first_frame": 0,
            "first_person_frame": 0,
            "first_attention_frame": 0,
            "first_event_frame_by_type": {"person_waving": 1},
            "last_frame": 2,
        },
    }
    assert set(summary["scenes"]) == {"hall", "lobby"}
    assert summary["scenes"]["lobby"]["keyframes"] == summary["keyframes"]["lobby"]
    assert summary["scenes"]["lobby"]["semantic_events"]["first_frame_by_type"] == {
        "person_waving": 1
    }

    summary_json = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary_json == summary
    assert "passed" not in json.dumps(summary_json)
    assert "quality_gate" not in json.dumps(summary_json)
    assert "oracle" not in json.dumps(summary_json)

    root_html = (out / "index.html").read_text(encoding="utf-8")
    assert "<table" in root_html
    assert "<th>Scene</th>" in root_html
    assert "<th>Frames</th>" in root_html
    assert "<th>OK / Errors</th>" in root_html
    assert "<th>Person Frames / Max</th>" in root_html
    assert "<th>Tracks</th>" in root_html
    assert "<th>Attention</th>" in root_html
    assert "<th>Events</th>" in root_html
    assert "<th>Keyframes</th>" in root_html
    assert "2 / 1" in root_html
    assert "2 / 2" in root_html
    assert "ids=[7,8]" in root_html
    assert "available=2 null=0 switches=1" in root_html
    assert "person_waving=1" in root_html
    assert "Keyframes" in root_html
    assert 'href="scenes/lobby/index.html#frame-0">lobby first_frame</a>' in root_html
    assert (
        'href="scenes/lobby/index.html#frame-0">lobby first_person_frame</a>'
        in root_html
    )
    assert (
        'href="scenes/lobby/index.html#frame-0">lobby first_attention_frame</a>'
        in root_html
    )
    assert (
        'href="scenes/lobby/index.html#frame-1">lobby person_waving first_event</a>'
        in root_html
    )
    assert 'href="scenes/lobby/index.html#frame-2">lobby last_frame</a>' in root_html
    assert "Semantic Event Timeline" in root_html
    assert 'href="scenes/lobby/index.html#frame-1"' in root_html
    assert "person_waving track=8" in root_html

    scene_html = (out / "scenes" / "hall" / "index.html").read_text(encoding="utf-8")
    assert 'id="frame-0"' in scene_html
    assert "events=none" in scene_html
    assert "path=" in scene_html
    assert "latency_ms=12.5" in scene_html
    assert "track_ids=[]" in scene_html
    assert "timestamp=-" in scene_html
    assert "server_timestamp_ms=-" in scene_html
    assert "<details><summary>visual_state</summary>" in scene_html

    lobby_html = (out / "scenes" / "lobby" / "index.html").read_text(encoding="utf-8")
    assert "track_ids=[7]" in lobby_html
    assert "frame_timestamp_ms=1000" in lobby_html
    assert "server_timestamp_ms=1033" in lobby_html
    assert "latency_ms=12.5" in lobby_html


def test_attention_summary_uses_ok_frames_and_available_attention_only() -> None:
    records = [
        _wrapped(
            "lobby",
            0,
            _state(
                frame_id=0,
                attention={"target_track_id": 7},
                person_count=1,
            ),
        ),
        _wrapped(
            "lobby",
            1,
            _state(frame_id=1, attention={"debug": "dict-without-target"}),
        ),
        _wrapped(
            "lobby",
            2,
            _state(frame_id=2, attention={"target_uv": [10, 12]}),
        ),
        _wrapped("lobby", 3, _state(frame_id=3, attention=None)),
        _wrapped("lobby", 4, {"type": "error", "code": "response_timeout"}),
        _wrapped(
            "lobby",
            5,
            _state(frame_id=5, attention={"target_track_id": 9}),
        ),
    ]

    summary = module.summarize_records(records)

    assert summary["frames_total"] == 6
    assert summary["frames_ok"] == 5
    assert summary["errors"] == 1
    assert summary["attention"] == {
        "available_frames": 3,
        "available_ratio": pytest.approx(3 / 5),
        "null_frames": 2,
        "target_switches": 1,
    }
    assert summary["keyframes"]["lobby"]["first_attention_frame"] == 0


def test_visual_evidence_caption_shows_overlay_and_event_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    out = tmp_path / "evidence"
    _make_scene(data_dir, "lobby", 1)
    identity = {
        "status": "known_person",
        "source": "cache",
        "person": {
            "person_id": "person_identity",
            "display_name": "张三",
            "embedding": [1.0, 0.0],
        },
    }
    state = _state(
        frame_id=0,
        tracks=[{"track_id": 7, "bbox_xyxy": [1, 2, 3, 4], "lost_ms": 0}],
        attention={"target_track_id": 7, "target_uv": [2, 3]},
        events=[
            {
                **_event("person_waving", track_id=7),
                "identity_context": identity,
            }
        ],
        person_count=1,
    )
    state["identity_context"] = {
        "overlay_status": "ready",
        "tracks": [{"track_id": 7, "identity": identity}],
    }
    input_jsonl = tmp_path / "artifact" / "visual_state.jsonl"
    _write_jsonl(input_jsonl, [_wrapped("lobby", 0, state)])
    _patch_image_io(monkeypatch)

    module.generate_visual_evidence(
        records=module.read_wrapped_visual_state_jsonl(input_jsonl),
        source_images=module.map_source_images(
            data_dir,
            [_wrapped("lobby", 0, state)],
            camera="front",
            fps=10.0,
            head_motion="stationary",
        ),
        out=out,
        input_jsonl=input_jsonl,
    )

    html = (out / "scenes" / "lobby" / "index.html").read_text(encoding="utf-8")
    assert "overlay=ready identity=张三" in html
    assert "person_waving track=7 identity=张三" in html
    raw_state = json.loads((out / "scenes" / "lobby" / "states" / "000000.json").read_text())
    assert raw_state["identity_context"]["tracks"][0]["identity"]["person"][
        "embedding"
    ] == [1.0, 0.0]


def test_raw_visual_state_jsonl_is_rejected(tmp_path: Path) -> None:
    raw_jsonl = tmp_path / "visual_state.jsonl"
    _write_jsonl(raw_jsonl, [_state(frame_id=0)])

    with pytest.raises(SystemExit) as exc:
        module.read_wrapped_visual_state_jsonl(raw_jsonl)

    assert "raw visual_state-only JSONL is not accepted" in str(exc.value)
    assert "tools.replay_val_data" in str(exc.value)


def test_non_numeric_latency_ms_is_rejected(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "visual_state.jsonl"
    record = _wrapped("lobby", 0, _state(frame_id=0))
    record["latency_ms"] = "12.5"
    _write_jsonl(jsonl_path, [record])

    with pytest.raises(SystemExit) as exc:
        module.read_wrapped_visual_state_jsonl(jsonl_path)

    assert "wrapped record latency_ms must be a number" in str(exc.value)


def test_scene_and_frame_mapping_failures_are_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _make_scene(data_dir, "lobby", 1)

    with pytest.raises(SystemExit) as missing_scene:
        module.map_source_images(
            data_dir,
            [_wrapped("missing", 0, _state(frame_id=0))],
            camera="front",
            fps=10.0,
            head_motion="stationary",
        )
    assert "unknown scene 'missing'" in str(missing_scene.value)

    with pytest.raises(SystemExit) as bad_frame:
        module.map_source_images(
            data_dir,
            [_wrapped("lobby", 3, _state(frame_id=3))],
            camera="front",
            fps=10.0,
            head_motion="stationary",
        )
    assert "outside scene 'lobby' range" in str(bad_frame.value)


async def test_out_inside_data_dir_is_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(SystemExit) as exc:
        await module.generate_visual_evidence_from_args(
            argparse.Namespace(
                data_dir=data_dir,
                out=data_dir / "evidence",
                visual_state_jsonl=tmp_path / "visual_state.jsonl",
                replay_artifact=None,
                run_replay=False,
                server=None,
                camera="front",
                fps=10.0,
                head_motion="stationary",
                response_timeout_ms=None,
                no_realtime=False,
            )
        )

    assert "--out must not be inside --data-dir" in str(exc.value)


async def test_online_mode_calls_replay_data_dir_without_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    out = tmp_path / "out"
    _make_scene(data_dir, "lobby", 1)
    _patch_image_io(monkeypatch)
    calls: list[dict[str, Any]] = []

    async def fake_replay_data_dir(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        _write_jsonl(
            kwargs["save_jsonl"],
            [_wrapped("lobby", 0, _state(frame_id=0, person_count=1))],
        )
        return []

    monkeypatch.setattr(module, "replay_data_dir", fake_replay_data_dir)

    summary = await module.generate_visual_evidence_from_args(
        argparse.Namespace(
            data_dir=data_dir,
            out=out,
            visual_state_jsonl=None,
            replay_artifact=None,
            run_replay=True,
            server="ws://127.0.0.1:8765/v1/stream",
            camera="front",
            fps=10.0,
            head_motion="stationary",
            response_timeout_ms=50,
            no_realtime=True,
        )
    )

    assert calls == [
        {
            "server": "ws://127.0.0.1:8765/v1/stream",
            "data_dir": data_dir,
            "camera": "front",
            "fps": 10.0,
            "head_motion": "stationary",
            "save_jsonl": out / "visual_state.jsonl",
            "realtime": False,
            "response_timeout_ms": 50,
            "continue_on_timeout": True,
        }
    ]
    assert summary["frames_total"] == 1
    assert (out / "index.html").is_file()
    assert not hasattr(module, "websockets")
