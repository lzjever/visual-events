from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_BRIDGE = REPO_ROOT / "native" / "dds_bridge"
TOOLS_BUILD = REPO_ROOT / "tools" / "build_dds_bridge.py"
GITIGNORE = REPO_ROOT / ".gitignore"

ALLOWED_TOPICS = {
    "/camera/image/jpeg",
    "/robot/head_state",
    "/visual_events/gaze_target",
}
ALLOWED_TYPES = {
    "unitree_camera::msg::dds_::CameraFrame_",
    "visual_events::msg::dds_::HeadStateV1_",
    "visual_events::msg::dds_::GazeTargetV1_",
}
DENIED_MOTION_TOKENS = {
    "LowCmd",
    "MotorCmd",
    "SportModeCmd",
    "MotionSwitcherClient",
    "look_at",
    "head_position",
    "yaw_velocity",
    "pitch_velocity",
    "motor_command",
    "rt/lowcmd",
    "rt/arm_sdk",
}


def _repo_native_sources() -> list[Path]:
    assert NATIVE_BRIDGE.is_dir(), "expected native/dds_bridge foundation"
    return sorted(
        path
        for path in NATIVE_BRIDGE.rglob("*")
        if path.suffix in {".cmake", ".txt", ".hpp", ".cpp", ".h", ".cc"}
    )


def _combined_native_source_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _repo_native_sources())


def _make_minimal_unitree_sdk_root(tmp_path: Path) -> Path:
    root = tmp_path / "unitree-sdk"
    (root / "lib" / "cmake" / "unitree_sdk2").mkdir(parents=True)
    (root / "lib" / "cmake" / "unitree_sdk2" / "unitree_sdk2Config.cmake").write_text(
        "# fake unitree_sdk2 package\n",
        encoding="utf-8",
    )
    for name in ["libunitree_sdk2.a", "libddsc.so", "libddscxx.so"]:
        (root / "lib" / name).write_text("", encoding="utf-8")
    return root


def _make_minimal_video_dds_publisher_dir(tmp_path: Path) -> Path:
    root = tmp_path / "video-dds-publisher"
    header = root / "include" / "unitree_camera" / "msg" / "dds" / "CameraFrame_.hpp"
    source = root / "src" / "CameraFrame_.cpp"
    header.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    header.write_text("// fake CameraFrame_ header\n", encoding="utf-8")
    source.write_text("// fake CameraFrame_ source\n", encoding="utf-8")
    return root


def _run_build_tool(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, os.fspath(TOOLS_BUILD), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_native_bridge_source_allowlist_has_only_camera_head_gaze_and_no_motion_tokens():
    text = _combined_native_source_text()

    for topic in ALLOWED_TOPICS:
        assert topic in text
    for type_name in ALLOWED_TYPES:
        assert type_name in text

    topic_literals = set()
    for part in text.split('"')[1::2]:
        if part.startswith("/"):
            topic_literals.add(part)
    assert topic_literals <= ALLOWED_TOPICS

    offenders = [token for token in sorted(DENIED_MOTION_TOKENS) if token in text]
    assert offenders == []


def test_native_bridge_cmake_uses_unitree_and_camera_frame_inputs_without_python_or_motion_sdks():
    cmake = NATIVE_BRIDGE / "CMakeLists.txt"
    assert cmake.exists()
    text = cmake.read_text(encoding="utf-8")

    required = [
        "find_package(unitree_sdk2 REQUIRED)",
        "unitree_sdk2",
        "VIDEO_DDS_PUBLISHER_DIR",
        "CameraFrame_.cpp",
        "unitree_camera/msg/dds/CameraFrame_.hpp",
        "visual_events_dds_bridge_probe",
    ]
    missing = [item for item in required if item not in text]
    assert missing == []

    forbidden = ["python_dds", "cyclonedds-python", "fastdds", "SportMode", "LowCmd", "MotorCmd"]
    offenders = [item for item in forbidden if item in text]
    assert offenders == []


def test_native_probe_status_source_contract_uses_existing_jsonl_bridge_status_frame():
    text = _combined_native_source_text()
    assert '"protocol_version":1' in text
    assert '"type":"status"' in text
    assert '"code":"probe_ok"' in text
    assert '"message":"' in text
    assert '"mode":"probe"' in text
    assert '"status":"ok"' not in text
    assert "std::cout" in text
    assert "std::cerr" in text


def test_native_probe_binary_emits_single_jsonl_status_frame_without_stdout_logs():
    binary = REPO_ROOT / "build" / "dds_bridge" / "visual_events_dds_bridge_probe"
    if not binary.exists():
        pytest.skip("probe binary not built")

    result = subprocess.run(
        [os.fspath(binary), "--probe"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    stdout_lines = result.stdout.splitlines()
    assert len(stdout_lines) == 1
    status = json.loads(stdout_lines[0])
    assert status["protocol_version"] == 1
    assert status["type"] == "status"
    assert status["code"] == "probe_ok"
    assert isinstance(status["message"], str)
    assert status["message"]
    assert status["mode"] == "probe"
    assert "log" not in status


def test_build_tool_foundation_check_does_not_require_idl_generator(tmp_path):
    assert TOOLS_BUILD.exists()
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    report_path = tmp_path / "foundation-report.json"

    env = os.environ.copy()
    env["PATH"] = ""
    result = _run_build_tool(
        [
            "--check",
            "--unitree-sdk-root",
            os.fspath(unitree_root),
            "--video-dds-publisher-dir",
            os.fspath(video_dir),
            "--out",
            os.fspath(report_path),
        ],
        env=env,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["foundation_ready"] is True
    assert report["visual_events_codegen_ready"] is False
    assert report["visual_events_codegen_error"] == "not required for foundation check"


def test_build_tool_missing_root_and_full_bridge_missing_generator_fail_fast(tmp_path):
    assert TOOLS_BUILD.exists()
    missing_root = tmp_path / "missing-unitree"
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    result = _run_build_tool(
        [
            "--check",
            "--unitree-sdk-root",
            os.fspath(missing_root),
            "--video-dds-publisher-dir",
            os.fspath(video_dir),
            "--out",
            os.fspath(tmp_path / "missing-root-report.json"),
        ],
    )
    assert result.returncode != 0
    assert "UNITREE_SDK_ROOT" in result.stderr
    assert os.fspath(missing_root) in result.stderr

    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    report_path = tmp_path / "full-bridge-report.json"
    env = os.environ.copy()
    env["PATH"] = ""
    result = _run_build_tool(
        [
            "--check",
            "--check-full-bridge",
            "--unitree-sdk-root",
            os.fspath(unitree_root),
            "--video-dds-publisher-dir",
            os.fspath(video_dir),
            "--out",
            os.fspath(report_path),
        ],
        env=env,
    )
    assert result.returncode != 0
    assert "IDL generator" in result.stderr
    assert "idlc" in result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["foundation_ready"] is True
    assert report["visual_events_codegen_ready"] is False
    assert "IDL generator" in report["visual_events_codegen_error"]


def test_run_probe_validates_complete_status_frame_abi(tmp_path):
    import tools.build_dds_bridge as build_dds_bridge

    probe = tmp_path / "visual_events_dds_bridge_probe"
    probe.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"protocol_version\":1,\"type\":\"status\",\"code\":\"probe_ok\",\"message\":\"ok\",\"mode\":\"probe\"}'\n",
        encoding="utf-8",
    )
    probe.chmod(0o755)

    report = build_dds_bridge.run_probe(tmp_path)
    assert report["probe_status"]["code"] == "probe_ok"

    probe.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"protocol_version\":1,\"code\":\"probe_ok\",\"message\":\"ok\"}'\n",
        encoding="utf-8",
    )
    with pytest.raises(build_dds_bridge.CheckError, match="type=status"):
        build_dds_bridge.run_probe(tmp_path)


def test_native_bridge_build_and_probe_artifacts_are_ignored():
    gitignore = GITIGNORE.read_text(encoding="utf-8")
    assert "build/" in gitignore
    assert "artifacts/" in gitignore
    assert "native/dds_bridge" not in gitignore
