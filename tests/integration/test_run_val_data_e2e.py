import json
from pathlib import Path

import pytest

from tools.replay_val_data import ReplayStats
from tools.run_val_data_e2e import (
    MOVING_SUPPRESSION_SCENE_NAMES,
    REQUIRED_SCENE_NAMES,
    async_main,
)


JPEG_BYTES = b"\xff\xd8\xff\xe0minimal-jpeg\xff\xd9"
EXPECTED_MOVING_SUPPRESSION_SCENE_NAMES = (
    "pci_stand",
    "pic_1_l_to_r",
    "pic_1_r_to_l",
    "pic_persone_walk_in",
    "pic_walk_in_stop",
)
STATIONARY_UNKNOWN_CASE_COUNT = len(REQUIRED_SCENE_NAMES) * 2
FULL_MATRIX_CASE_COUNT = STATIONARY_UNKNOWN_CASE_COUNT + len(
    EXPECTED_MOVING_SUPPRESSION_SCENE_NAMES
)


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
    monkeypatch.setattr("tools.run_val_data_e2e.read_process_rss_mb", lambda pid: None)

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
async def test_runner_replays_stationary_unknown_and_moving_rounds_and_writes_artifacts(
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
    assert MOVING_SUPPRESSION_SCENE_NAMES == EXPECTED_MOVING_SUPPRESSION_SCENE_NAMES
    assert len(calls) == FULL_MATRIX_CASE_COUNT
    assert [call["head_motion"] for call in calls] == (
        ["stationary"] * len(REQUIRED_SCENE_NAMES)
        + ["unknown"] * len(REQUIRED_SCENE_NAMES)
        + ["moving"] * len(EXPECTED_MOVING_SUPPRESSION_SCENE_NAMES)
    )

    stationary_calls = calls[: len(REQUIRED_SCENE_NAMES)]
    unknown_calls = calls[
        len(REQUIRED_SCENE_NAMES) : STATIONARY_UNKNOWN_CASE_COUNT
    ]
    moving_calls = calls[STATIONARY_UNKNOWN_CASE_COUNT:]
    assert [Path(call["scene_dir"]).name for call in stationary_calls] == list(
        REQUIRED_SCENE_NAMES
    )
    assert [Path(call["scene_dir"]).name for call in unknown_calls] == list(
        REQUIRED_SCENE_NAMES
    )
    assert [Path(call["scene_dir"]).name for call in moving_calls] == list(
        EXPECTED_MOVING_SUPPRESSION_SCENE_NAMES
    )
    assert all(call["server"] == "ws://127.0.0.1:8765/v1/stream" for call in calls)
    assert all(call["camera"] == "rear" for call in calls)
    assert all(call["fps"] == 10.0 for call in calls)
    assert all(call["response_timeout_ms"] == 250 for call in calls)
    assert all(call["semantic_event_cooldown_ms"] == 50 for call in calls)
    assert all(call["realtime"] is False for call in calls)
    assert [
        call["save_jsonl"].relative_to(out)
        for call in stationary_calls[:2] + unknown_calls[:2] + moving_calls[:2]
    ] == [
        Path("pci_stand/visual_state.jsonl"),
        Path("pic_1_l_to_r/visual_state.jsonl"),
        Path("pci_stand__head_unknown/visual_state.jsonl"),
        Path("pic_1_l_to_r__head_unknown/visual_state.jsonl"),
        Path("pci_stand__head_moving/visual_state.jsonl"),
        Path("pic_1_l_to_r__head_moving/visual_state.jsonl"),
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

    for scene in EXPECTED_MOVING_SUPPRESSION_SCENE_NAMES:
        case_dir = out / f"{scene}__head_moving"
        assert (case_dir / "visual_state.jsonl").is_file()
        assert (case_dir / "summary.md").is_file()
        summary = json.loads((case_dir / "summary.json").read_text())
        assert summary["scene"] == scene
        assert summary["head_motion"] == "moving"
        assert summary["gate"] == "events"
        assert summary["passed"] is True

    report = json.loads((out / "report.json").read_text())
    assert report["overall_pass"] is True
    assert len(report["cases"]) == FULL_MATRIX_CASE_COUNT
    assert report["thresholds"]["hz_min"] == 9.0
    assert report["cases"][0]["artifacts"]["visual_state_jsonl"].endswith(
        "pci_stand/visual_state.jsonl"
    )
    assert report["cases"][STATIONARY_UNKNOWN_CASE_COUNT]["case"] == (
        "pci_stand__head_moving"
    )
    assert report["cases"][STATIONARY_UNKNOWN_CASE_COUNT]["gate"] == "events"
    assert report["cases"][STATIONARY_UNKNOWN_CASE_COUNT]["head_motion"] == "moving"

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
        "sent": FULL_MATRIX_CASE_COUNT * 3,
        "ok": FULL_MATRIX_CASE_COUNT * 3,
        "errors": 0,
        "latency_samples": FULL_MATRIX_CASE_COUNT * 3,
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


@pytest.mark.parametrize("head_motion", ["unknown", "moving"])
@pytest.mark.asyncio
async def test_non_stationary_motion_sensitive_failure_blocks_e2e(
    tmp_path,
    monkeypatch,
    head_motion,
):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"

    async def fake_replay_scene(**kwargs):
        scene = Path(kwargs["scene_dir"]).name
        write_fake_jsonl(kwargs["save_jsonl"])
        stats = passing_stats(scene, kwargs["head_motion"])
        if kwargs["head_motion"] == head_motion and scene == "pic_walk_in_stop":
            stats = ReplayStats(
                scene=scene,
                frames_sent=3,
                frames_ok=3,
                errors=0,
                elapsed_s=0.3,
                head_motion=head_motion,
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
    assert [case["case"] for case in failed] == [f"pic_walk_in_stop__head_{head_motion}"]
    assert f"motion-sensitive events emitted for {head_motion} head motion" in failed[0][
        "failure_reasons"
    ]


@pytest.mark.asyncio
async def test_soak_preflight_requires_timeout_and_server_pid_without_replay(
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

    missing_timeout = [
        "--server",
        "ws://127.0.0.1:8765/v1/stream",
        "--data-dir",
        str(data_dir),
        "--out",
        str(out),
        "--soak-seconds",
        "5",
        "--server-pid",
        "1234",
    ]
    assert await async_main(missing_timeout) == 1

    missing_pid = [
        "--server",
        "ws://127.0.0.1:8765/v1/stream",
        "--data-dir",
        str(data_dir),
        "--out",
        str(out),
        "--soak-seconds",
        "5",
        "--response-timeout-ms",
        "250",
    ]
    assert await async_main(missing_pid) == 1

    no_realtime = [
        "--server",
        "ws://127.0.0.1:8765/v1/stream",
        "--data-dir",
        str(data_dir),
        "--out",
        str(out),
        "--soak-seconds",
        "0.001",
        "--response-timeout-ms",
        "250",
        "--server-pid",
        "1234",
        "--no-realtime",
    ]
    assert await async_main(no_realtime) == 1

    assert calls == []


@pytest.mark.asyncio
async def test_soak_runs_full_val_data_loops_until_wall_clock_target(
    tmp_path,
    monkeypatch,
):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"
    calls = []
    perf_clock = iter([100.0, 103.0, 106.0])
    rss_samples = [100.0, 101.0, 102.0, 103.0]

    async def fake_replay_scene(**kwargs):
        calls.append(kwargs)
        write_fake_jsonl(kwargs["save_jsonl"])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)
    monkeypatch.setattr(
        "tools.run_val_data_e2e._perf_counter",
        lambda: next(perf_clock),
    )
    monkeypatch.setattr(
        "tools.run_val_data_e2e.read_process_rss_mb",
        lambda pid: rss_samples.pop(0),
    )

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--response-timeout-ms",
            "250",
            "--soak-seconds",
            "5",
            "--server-pid",
            "1234",
            "--soak-memory-growth-max-mb",
            "10",
            "--soak-sample-interval-s",
            "1",
        ]
    )

    assert exit_code == 0
    assert len(calls) == FULL_MATRIX_CASE_COUNT + STATIONARY_UNKNOWN_CASE_COUNT * 2
    warm_up_calls = calls[:FULL_MATRIX_CASE_COUNT]
    soak_loop_calls = calls[FULL_MATRIX_CASE_COUNT:]
    assert [call["head_motion"] for call in warm_up_calls] == (
        ["stationary"] * len(REQUIRED_SCENE_NAMES)
        + ["unknown"] * len(REQUIRED_SCENE_NAMES)
        + ["moving"] * len(EXPECTED_MOVING_SUPPRESSION_SCENE_NAMES)
    )
    assert all(call["head_motion"] != "moving" for call in soak_loop_calls)

    first_soak_loop = soak_loop_calls[:STATIONARY_UNKNOWN_CASE_COUNT]
    second_soak_loop = soak_loop_calls[STATIONARY_UNKNOWN_CASE_COUNT:]
    expected_soak_loop_motions = (
        ["stationary"] * len(REQUIRED_SCENE_NAMES)
        + ["unknown"] * len(REQUIRED_SCENE_NAMES)
    )
    assert [call["head_motion"] for call in first_soak_loop] == expected_soak_loop_motions
    assert [call["head_motion"] for call in second_soak_loop] == expected_soak_loop_motions
    assert first_soak_loop[0]["save_jsonl"].relative_to(out) == Path(
        "soak/loop_0001/pci_stand/visual_state.jsonl"
    )
    assert first_soak_loop[-1]["save_jsonl"].relative_to(out) == Path(
        "soak/loop_0001/pic_walk_in_stop__head_unknown/visual_state.jsonl"
    )
    assert second_soak_loop[0]["save_jsonl"].relative_to(out) == Path(
        "soak/loop_0002/pci_stand/visual_state.jsonl"
    )
    assert all(call["response_timeout_ms"] == 250 for call in calls)

    assert (out / "pci_stand" / "visual_state.jsonl").is_file()
    assert (out / "pci_stand__head_moving" / "visual_state.jsonl").is_file()
    assert (out / "soak" / "loop_0001" / "pci_stand" / "visual_state.jsonl").is_file()
    assert (out / "soak" / "loop_0002" / "pci_stand" / "visual_state.jsonl").is_file()
    assert not (out / "soak" / "loop_0001" / "pci_stand__head_moving").exists()
    assert not (out / "soak" / "loop_0002" / "pci_stand__head_moving").exists()

    report = json.loads((out / "report.json").read_text())
    perf = json.loads((tmp_path / "artifacts" / "perf" / "server_perf.json").read_text())
    assert report["overall_pass"] is True
    soak_cases_completed = STATIONARY_UNKNOWN_CASE_COUNT * 2
    soak_frames = soak_cases_completed * 3
    soak_without_hz = {
        key: value for key, value in report["soak"].items() if key != "hz"
    }
    assert soak_without_hz == {
        "enabled": True,
        "passed": True,
        "failure_reasons": [],
        "target_seconds": 5.0,
        "elapsed_s": 6.0,
        "loops_completed": 2,
        "cases_completed": soak_cases_completed,
        "frames": {
            "sent": soak_frames,
            "ok": soak_frames,
            "errors": 0,
            "latency_samples": soak_frames,
        },
        "error_rate": 0.0,
        "total_latency_ms": {
            "available": True,
            "p50": 20.0,
            "p95": 30.0,
            "p99": 30.0,
        },
        "server_pid": 1234,
        "rss_mb": {
            "available": True,
            "start": 100.0,
            "end": 103.0,
            "growth": 3.0,
            "max_growth": 10.0,
            "samples": 4,
        },
    }
    assert report["soak"]["hz"] == pytest.approx(10.0)
    assert perf["passed"] is True
    assert perf["soak"] == report["soak"]


@pytest.mark.asyncio
async def test_soak_rss_baseline_is_sampled_after_initial_full_matrix(
    tmp_path,
    monkeypatch,
):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"
    calls = []
    rss_call_replay_counts = []
    perf_clock = iter([100.0, 106.0])

    async def fake_replay_scene(**kwargs):
        calls.append(kwargs)
        write_fake_jsonl(kwargs["save_jsonl"])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    def fake_read_process_rss_mb(pid):
        rss_call_replay_counts.append(len(calls))
        return [512.0, 514.0, 515.0][len(rss_call_replay_counts) - 1]

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)
    monkeypatch.setattr(
        "tools.run_val_data_e2e._perf_counter",
        lambda: next(perf_clock),
    )
    monkeypatch.setattr(
        "tools.run_val_data_e2e.read_process_rss_mb",
        fake_read_process_rss_mb,
    )

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--response-timeout-ms",
            "250",
            "--soak-seconds",
            "5",
            "--server-pid",
            "1234",
            "--soak-memory-growth-max-mb",
            "10",
            "--soak-sample-interval-s",
            "1",
        ]
    )

    assert exit_code == 0
    assert rss_call_replay_counts[0] == FULL_MATRIX_CASE_COUNT
    report = json.loads((out / "report.json").read_text())
    assert report["soak"]["rss_mb"]["start"] == 512.0
    assert report["soak"]["rss_mb"]["growth"] == 3.0


@pytest.mark.asyncio
async def test_soak_memory_growth_failure_returns_nonzero(tmp_path, monkeypatch):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"
    perf_clock = iter([200.0, 206.0])
    rss_samples = [100.0, 120.5, 121.0]

    async def fake_replay_scene(**kwargs):
        write_fake_jsonl(kwargs["save_jsonl"])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)
    monkeypatch.setattr(
        "tools.run_val_data_e2e._perf_counter",
        lambda: next(perf_clock),
    )
    monkeypatch.setattr(
        "tools.run_val_data_e2e.read_process_rss_mb",
        lambda pid: rss_samples.pop(0),
    )

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--response-timeout-ms",
            "250",
            "--soak-seconds",
            "5",
            "--server-pid",
            "1234",
            "--soak-memory-growth-max-mb",
            "10",
            "--soak-sample-interval-s",
            "1",
        ]
    )

    assert exit_code == 1
    report = json.loads((out / "report.json").read_text())
    perf = json.loads((tmp_path / "artifacts" / "perf" / "server_perf.json").read_text())
    assert report["overall_pass"] is False
    assert perf["passed"] is False
    assert perf["soak"]["passed"] is False
    assert perf["soak"]["rss_mb"]["growth"] == 21.0
    assert "soak_memory_growth_mb" in perf["soak"]["failure_reasons"]
    assert "perf: soak_memory_growth_mb" in report["failure_reasons"]


@pytest.mark.asyncio
async def test_soak_latency_threshold_failure_returns_nonzero(tmp_path, monkeypatch):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"
    perf_clock = iter([500.0, 506.0])
    rss_samples = [100.0, 100.0, 100.0]
    calls = []

    async def fake_replay_scene(**kwargs):
        calls.append(kwargs)
        if "soak" in kwargs["save_jsonl"].parts:
            write_fake_jsonl(kwargs["save_jsonl"], latencies=[10.0, 20.0, 210.0])
        else:
            write_fake_jsonl(kwargs["save_jsonl"])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)
    monkeypatch.setattr(
        "tools.run_val_data_e2e._perf_counter",
        lambda: next(perf_clock),
    )
    monkeypatch.setattr(
        "tools.run_val_data_e2e.read_process_rss_mb",
        lambda pid: rss_samples.pop(0),
    )

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--response-timeout-ms",
            "250",
            "--soak-seconds",
            "5",
            "--server-pid",
            "1234",
            "--soak-memory-growth-max-mb",
            "10",
            "--soak-sample-interval-s",
            "1",
        ]
    )

    assert exit_code == 1
    report = json.loads((out / "report.json").read_text())
    perf = json.loads((tmp_path / "artifacts" / "perf" / "server_perf.json").read_text())
    soak_cases_completed = STATIONARY_UNKNOWN_CASE_COUNT
    soak_frames = soak_cases_completed * 3

    assert report["overall_pass"] is False
    assert perf["thresholds"]["results"]["total_latency_p95_ms"] is True
    assert perf["soak"]["passed"] is False
    assert perf["soak"]["cases_completed"] == soak_cases_completed
    assert perf["soak"]["frames"] == {
        "sent": soak_frames,
        "ok": soak_frames,
        "errors": 0,
        "latency_samples": soak_frames,
    }
    assert perf["soak"]["hz"] == pytest.approx(10.0)
    assert perf["soak"]["error_rate"] == 0.0
    assert perf["soak"]["total_latency_ms"]["p95"] == 210.0
    assert "soak_total_latency_p95_ms" in perf["soak"]["failure_reasons"]
    assert "soak_total_latency_p99_ms" in perf["soak"]["failure_reasons"]
    assert "perf: soak_total_latency_p95_ms" in report["failure_reasons"]


@pytest.mark.asyncio
async def test_soak_replay_exception_writes_enabled_failed_soak_report(
    tmp_path,
    monkeypatch,
):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"
    calls = []

    async def fake_replay_scene(**kwargs):
        calls.append(kwargs)
        if "soak" in kwargs["save_jsonl"].parts:
            raise RuntimeError("soak connection dropped")
        write_fake_jsonl(kwargs["save_jsonl"])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)
    monkeypatch.setattr(
        "tools.run_val_data_e2e.read_process_rss_mb",
        lambda pid: 512.0,
    )

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--response-timeout-ms",
            "250",
            "--soak-seconds",
            "5",
            "--server-pid",
            "1234",
        ]
    )

    assert exit_code == 1
    assert len(calls) == FULL_MATRIX_CASE_COUNT + 1
    report = json.loads((out / "report.json").read_text())
    perf = json.loads((tmp_path / "artifacts" / "perf" / "server_perf.json").read_text())
    assert report["overall_pass"] is False
    assert report["soak"]["enabled"] is True
    assert report["soak"]["passed"] is False
    assert perf["soak"] == report["soak"]
    assert "soak_exception: RuntimeError: soak connection dropped" in report["soak"][
        "failure_reasons"
    ]
    assert "perf: soak_exception: RuntimeError: soak connection dropped" in report[
        "failure_reasons"
    ]


@pytest.mark.asyncio
async def test_soak_unreadable_server_pid_fails(tmp_path, monkeypatch):
    data_dir = make_val_data_root(tmp_path)
    out = tmp_path / "artifacts" / "e2e"
    perf_clock = iter([300.0])
    calls = []

    async def fake_replay_scene(**kwargs):
        calls.append(kwargs)
        write_fake_jsonl(kwargs["save_jsonl"])
        return passing_stats(Path(kwargs["scene_dir"]).name, kwargs["head_motion"])

    monkeypatch.setattr("tools.run_val_data_e2e.replay_scene", fake_replay_scene)
    monkeypatch.setattr(
        "tools.run_val_data_e2e._perf_counter",
        lambda: next(perf_clock),
    )
    monkeypatch.setattr(
        "tools.run_val_data_e2e.read_process_rss_mb",
        lambda pid: None,
    )

    exit_code = await async_main(
        [
            "--server",
            "ws://127.0.0.1:8765/v1/stream",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--response-timeout-ms",
            "250",
            "--soak-seconds",
            "5",
            "--server-pid",
            "1234",
        ]
    )

    assert exit_code == 1
    assert len(calls) == FULL_MATRIX_CASE_COUNT
    report = json.loads((out / "report.json").read_text())
    perf = json.loads((tmp_path / "artifacts" / "perf" / "server_perf.json").read_text())
    assert report["overall_pass"] is False
    assert perf["soak"]["passed"] is False
    assert "soak_rss_unavailable" in perf["soak"]["failure_reasons"]
    assert "perf: soak_rss_unavailable" in report["failure_reasons"]
