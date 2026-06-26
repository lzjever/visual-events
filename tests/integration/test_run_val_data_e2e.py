import json
from pathlib import Path

import pytest

from tools.replay_val_data import ReplayStats
from tools.run_val_data_e2e import REQUIRED_SCENE_NAMES, async_main


JPEG_BYTES = b"\xff\xd8\xff\xe0minimal-jpeg\xff\xd9"


def write_jpeg(path: Path) -> None:
    path.write_bytes(JPEG_BYTES)


def make_val_data_root(tmp_path: Path, name: str = "val-data") -> Path:
    root = tmp_path / name
    for scene in REQUIRED_SCENE_NAMES:
        scene_dir = root / scene
        scene_dir.mkdir(parents=True)
        write_jpeg(scene_dir / "img_1710000000000000000.jpeg")
    return root


def passing_stats(scene: str, head_motion: str) -> ReplayStats:
    return ReplayStats(
        scene=scene,
        frames_sent=3,
        frames_ok=3,
        errors=0,
        elapsed_s=0.3,
        head_motion=head_motion,
        track_frames=3,
        visible_counts_by_id={"1": 3},
        attention_frames=3,
        attention_target_counts_by_id={"1": 3},
    )


def write_fake_jsonl(save_jsonl: Path, latencies: list[float] | None = None) -> None:
    save_jsonl.parent.mkdir(parents=True, exist_ok=True)
    samples = latencies or [10.0, 20.0, 30.0]
    save_jsonl.write_text(
        "\n".join(
            json.dumps(
                {
                    "frame_id": index,
                    "latency_ms": latency,
                    "response": {"type": "visual_state"},
                },
                separators=(",", ":"),
            )
            for index, latency in enumerate(samples)
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_preflight_requires_full_val_data_root(tmp_path, monkeypatch):
    calls = []

    async def fake_replay_scene(**kwargs):
        calls.append(kwargs)
        write_fake_jsonl(kwargs["save_jsonl"])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)

    missing_root = tmp_path / "missing-val-data"
    assert (
        await async_main(
            [
                "--server",
                "ws://127.0.0.1:8765/v1/stream",
                "--data-dir",
                str(missing_root),
            ]
        )
        == 1
    )

    empty_scene_root = make_val_data_root(tmp_path, "empty-scene-val-data")
    for jpeg in (empty_scene_root / "pic_hello").glob("*.jpeg"):
        jpeg.unlink()
    assert (
        await async_main(
            [
                "--server",
                "ws://127.0.0.1:8765/v1/stream",
                "--data-dir",
                str(empty_scene_root),
            ]
        )
        == 1
    )

    missing_scene_root = make_val_data_root(tmp_path, "missing-scene-val-data")
    for jpeg in (missing_scene_root / "pic_leave").glob("*.jpeg"):
        jpeg.unlink()
    (missing_scene_root / "pic_leave").rmdir()
    assert (
        await async_main(
            [
                "--server",
                "ws://127.0.0.1:8765/v1/stream",
                "--data-dir",
                str(missing_scene_root),
            ]
        )
        == 1
    )

    single_scene = tmp_path / "single-scene" / "pic_hello"
    single_scene.mkdir(parents=True)
    write_jpeg(single_scene / "img_1710000000000000000.jpeg")
    assert (
        await async_main(
            [
                "--server",
                "ws://127.0.0.1:8765/v1/stream",
                "--data-dir",
                str(single_scene),
            ]
        )
        == 1
    )

    assert calls == []


@pytest.mark.asyncio
async def test_preflight_rejects_unsafe_out_paths_without_replay(tmp_path, monkeypatch):
    data_dir = make_val_data_root(tmp_path)
    calls = []

    async def fake_replay_scene(**kwargs):
        calls.append(kwargs)
        write_fake_jsonl(kwargs["save_jsonl"])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)

    for unsafe_out in (tmp_path / "e2e", data_dir / "artifacts" / "e2e"):
        assert (
            await async_main(
                [
                    "--server",
                    "ws://127.0.0.1:8765/v1/stream",
                    "--data-dir",
                    str(data_dir),
                    "--out",
                    str(unsafe_out),
                ]
            )
            == 1
        )

    assert calls == []


@pytest.mark.asyncio
async def test_runner_replays_stationary_and_unknown_rounds_and_writes_artifacts(
    tmp_path,
    monkeypatch,
):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"
    calls = []

    async def fake_replay_scene(**kwargs):
        calls.append(kwargs)
        write_fake_jsonl(kwargs["save_jsonl"])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--camera",
            "rear",
            "--fps",
            "10",
            "--response-timeout-ms",
            "250",
            "--semantic-event-cooldown-ms",
            "50",
            "--no-realtime",
        ]
    )

    assert exit_code == 0
    assert len(calls) == len(REQUIRED_SCENE_NAMES) * 2
    assert [call["head_motion"] for call in calls] == (
        ["stationary"] * len(REQUIRED_SCENE_NAMES)
        + ["unknown"] * len(REQUIRED_SCENE_NAMES)
    )

    stationary_calls = calls[: len(REQUIRED_SCENE_NAMES)]
    unknown_calls = calls[len(REQUIRED_SCENE_NAMES) :]
    assert [Path(call["scene_dir"]).name for call in stationary_calls] == list(
        REQUIRED_SCENE_NAMES
    )
    assert [Path(call["scene_dir"]).name for call in unknown_calls] == list(
        REQUIRED_SCENE_NAMES
    )
    assert all(call["server"] == "ws://127.0.0.1:8765/v1/stream" for call in calls)
    assert all(call["camera"] == "rear" for call in calls)
    assert all(call["fps"] == 10.0 for call in calls)
    assert all(call["response_timeout_ms"] == 250 for call in calls)
    assert all(call["semantic_event_cooldown_ms"] == 50 for call in calls)
    assert all(call["realtime"] is False for call in calls)
    assert [
        call["save_jsonl"].relative_to(out)
        for call in stationary_calls[:2] + unknown_calls[:2]
    ] == [
        Path("pci_stand/visual_state.jsonl"),
        Path("pic_1_l_to_r/visual_state.jsonl"),
        Path("pci_stand__head_unknown/visual_state.jsonl"),
        Path("pic_1_l_to_r__head_unknown/visual_state.jsonl"),
    ]

    for scene in REQUIRED_SCENE_NAMES:
        for case_dir, gate, head_motion in (
            (out / scene, "all", "stationary"),
            (out / f"{scene}__head_unknown", "events", "unknown"),
        ):
            assert (case_dir / "visual_state.jsonl").is_file()
            assert (case_dir / "summary.md").is_file()
            summary = json.loads((case_dir / "summary.json").read_text())
            assert summary["scene"] == scene
            assert summary["head_motion"] == head_motion
            assert summary["gate"] == gate
            assert summary["passed"] is True

    report = json.loads((out / "report.json").read_text())
    assert report["overall_pass"] is True
    assert len(report["cases"]) == len(REQUIRED_SCENE_NAMES) * 2
    assert report["thresholds"]["hz_min"] == 9.0
    assert report["cases"][0]["artifacts"]["visual_state_jsonl"].endswith(
        "pci_stand/visual_state.jsonl"
    )

    perf = json.loads((tmp_path / "artifacts" / "perf" / "server_perf.json").read_text())
    assert perf["total_latency_ms"] == {
        "available": True,
        "p50": 20.0,
        "p95": 30.0,
        "p99": 30.0,
    }
    assert perf["hz"] == 10.0
    assert perf["error_rate"] == 0.0
    assert perf["frames"] == {
        "sent": len(REQUIRED_SCENE_NAMES) * 2 * 3,
        "ok": len(REQUIRED_SCENE_NAMES) * 2 * 3,
        "errors": 0,
        "latency_samples": len(REQUIRED_SCENE_NAMES) * 2 * 3,
    }
    assert perf["server_phase_latency_ms"]["infer"] == {"available": False}
    assert perf["vram"] == {"available": False}
    assert perf["memory"] == {"available": False}


@pytest.mark.asyncio
async def test_any_replay_gate_failure_returns_nonzero_and_reports_reason(
    tmp_path,
    monkeypatch,
):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"

    async def fake_replay_scene(**kwargs):
        scene = Path(kwargs["scene_dir"]).name
        write_fake_jsonl(kwargs["save_jsonl"])
        stats = passing_stats(scene, kwargs["head_motion"])
        if kwargs["head_motion"] == "stationary" and scene == "pic_hello":
            stats = ReplayStats(
                scene=scene,
                frames_sent=3,
                frames_ok=3,
                errors=0,
                elapsed_s=0.3,
                head_motion="stationary",
            )
        return stats

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--no-realtime",
        ]
    )

    assert exit_code == 1
    report = json.loads((out / "report.json").read_text())
    failed = [case for case in report["cases"] if not case["passed"]]
    assert [case["case"] for case in failed] == ["pic_hello"]
    assert failed[0]["gate"] == "all"
    assert "tracking gate failed" in failed[0]["failure_reasons"]


@pytest.mark.asyncio
async def test_replay_exception_writes_failure_report_and_perf(tmp_path, monkeypatch):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"
    calls = []

    async def fake_replay_scene(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("connection refused")
        write_fake_jsonl(kwargs["save_jsonl"])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--no-realtime",
        ]
    )

    assert exit_code == 1
    report = json.loads((out / "report.json").read_text())
    perf = json.loads((tmp_path / "artifacts" / "perf" / "server_perf.json").read_text())
    assert report["overall_pass"] is False
    assert report["cases"][0]["case"] == "pci_stand"
    assert len(report["cases"]) == 1
    assert "e2e exception: RuntimeError: connection refused" in report[
        "failure_reasons"
    ]
    assert perf["passed"] is False
    assert "e2e exception: RuntimeError: connection refused" in perf["failure_reasons"]
    assert perf["frames"] == {
        "sent": 0,
        "ok": 0,
        "errors": 0,
        "latency_samples": 0,
    }
    assert perf["total_latency_ms"]["available"] is False
    assert perf["hz"] == 0.0
    assert perf["server_phase_latency_ms"]["decode"] == {"available": False}
    assert perf["vram"] == {"available": False}
    assert perf["memory"] == {"available": False}


@pytest.mark.asyncio
async def test_perf_threshold_failure_returns_nonzero(tmp_path, monkeypatch):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"

    async def fake_replay_scene(**kwargs):
        write_fake_jsonl(kwargs["save_jsonl"], latencies=[10.0, 20.0, 210.0])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--no-realtime",
        ]
    )

    assert exit_code == 1
    report = json.loads((out / "report.json").read_text())
    perf = json.loads((tmp_path / "artifacts" / "perf" / "server_perf.json").read_text())
    assert report["overall_pass"] is False
    assert perf["passed"] is False
    assert perf["total_latency_ms"]["p95"] == 210.0
    assert "total_latency_p95_ms" in perf["failure_reasons"]


@pytest.mark.asyncio
async def test_unknown_motion_sensitive_failure_blocks_e2e(tmp_path, monkeypatch):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"

    async def fake_replay_scene(**kwargs):
        scene = Path(kwargs["scene_dir"]).name
        write_fake_jsonl(kwargs["save_jsonl"])
        stats = passing_stats(scene, kwargs["head_motion"])
        if kwargs["head_motion"] == "unknown" and scene == "pic_walk_in_stop":
            stats = ReplayStats(
                scene=scene,
                frames_sent=3,
                frames_ok=3,
                errors=0,
                elapsed_s=0.3,
                head_motion="unknown",
                semantic_event_motion_sensitive_count=1,
            )
        return stats

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--no-realtime",
        ]
    )

    assert exit_code == 1
    report = json.loads((out / "report.json").read_text())
    failed = [case for case in report["cases"] if not case["passed"]]
    assert [case["case"] for case in failed] == ["pic_walk_in_stop__head_unknown"]
    assert "motion-sensitive events emitted for unknown head motion" in failed[0][
        "failure_reasons"
    ]
