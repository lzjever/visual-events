from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GA_PLAN = REPO_ROOT / "docs" / "ga-development-plan.md"
PROTOCOL = REPO_ROOT / "common" / "schema" / "protocol.md"
GITIGNORE = REPO_ROOT / ".gitignore"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assert_contains_all(text: str, required: list[str]) -> None:
    missing = [item for item in required if item not in text]
    assert missing == []


def test_ga_plan_requires_val_data_manifest_and_release_hashes():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "val-data/manifest.json",
            "manifest_sha256",
            "scene 名称",
            "sha256",
            "frame count",
            "fps",
            "expected event timeline source/version",
            "PC/release report",
        ],
    )


def test_ga_plan_separates_pc_regression_from_robot_dds_and_closed_loop_gates():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "PC 本地 E2E 的定位是 regression gate",
            "真实 camera DDS runtime/network",
            "板端 DDS type/QoS compatibility",
            "真实 head state topic/type/Hz/freshness",
            "真实 head/motion consumer",
            "valid tracking 时头部物理指向目标",
            "invalid/stale 后不继续动",
            "restart 后无残留动作",
        ],
    )


def test_ga_plan_pins_head_state_latency_botified_and_runtime_hard_gates():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "head_state.required=true",
            "required = true",
            "head_state_publisher_mode",
            "head_state_hz",
            "head_state_stale_count",
            "head_state_unknown_ratio",
            "stationary",
            "moving",
            "unknown",
            "capture_to_gaze_publish_p95_ms",
            "capture_to_gaze_publish_p99_ms",
            "capture_to_botified_stdout_p95_ms",
            "capture_to_botified_stdout_p99_ms",
            "`attention_target_changed` 不输出到 Botified",
            "runtime/venv/bin/visual-events-server",
            "runtime/venv/bin/visual-events-cli",
            "visual-events-server` wheel",
            "CLI 不依赖 Torch/Ultralytics",
            "aarch64/RK3588 runtime",
            "ldd",
            "readelf",
            "DDS topic allowlist",
        ],
    )
    assert "required_for_ga" not in text


def test_protocol_pins_cli_frame_id_stale_watchdog_and_botified_allowlist():
    text = read_text(PROTOCOL)

    assert_contains_all(
        text,
        [
            "per-connection monotonic transport identity",
            "CameraFrame_` DDS 输入没有源 `frame_id`",
            "不得使用 DDS `timestamp_ns`/`timestamp_ms` 作为 identity",
            "WebSocket header `timestamp_ms`",
            "server 原样回显为 `visual_state.frame_timestamp_ms`",
            "不是 frame identity",
            "`gaze_target.stale_ms` 与 `service.response_timeout_ms` 是两个计时器",
            "Botified stdout allowlist",
            "`attention_target_changed` 只保留在 `visual_state.semantic_events`",
            "stdout 写入不得阻塞 gaze stale watchdog",
            "bounded queue/drop/coalescing",
            "BrokenPipe fail behavior",
        ],
    )


def test_gitignore_blocks_large_artifacts_without_ignoring_json_samples():
    lines = {
        line.strip()
        for line in read_text(GITIGNORE).splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert "*.json" not in lines
    assert_contains_all(
        "\n".join(sorted(lines)),
        [
            "*.pt",
            "*.onnx",
            "*.engine",
            "*.rknn",
            "*.mcap",
            "*.bag",
            "*.pcap",
            "*.mp4",
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.jsonl",
            "*.log",
            "captures/",
            "logs/",
            "models/",
        ],
    )
