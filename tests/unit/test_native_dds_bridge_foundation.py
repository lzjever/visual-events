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
TOOLS_PREPARE_CODEGEN = REPO_ROOT / "tools" / "prepare_dds_codegen_toolchain.py"
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


def _make_fake_idlc(tmp_path: Path, *, version: str, backends: str) -> Path:
    script = tmp_path / f"fake-idlc-{version}-{backends.replace(' ', '-')}"
    script.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  --version)\n"
        f"    printf '%s\\n' 'CycloneDDS idlc {version}'\n"
        "    ;;\n"
        "  --help|-l)\n"
        f"    printf '%s\\n' 'available backends: {backends}'\n"
        "    ;;\n"
        "  *)\n"
        "    printf '%s\\n' 'fake idlc only supports --version, --help, and -l' >&2\n"
        "    exit 64\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


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


def _run_prepare_codegen_tool(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, os.fspath(TOOLS_PREPARE_CODEGEN), *args],
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


def test_prepare_dds_codegen_toolchain_check_accepts_pinned_fake_idlc_without_writes(tmp_path):
    assert TOOLS_PREPARE_CODEGEN.exists()
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    result = _run_prepare_codegen_tool(
        ["--check", "--dry-run", "--idlc", os.fspath(fake_idlc)]
    )

    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["will_write"] is False
    assert report["cyclonedds_version"] == "0.10.2"
    assert report["cyclonedds_cxx_version"] == "0.10.2"
    assert report["toolchain_dir"] == os.fspath(
        REPO_ROOT / "build" / "tools" / "cyclonedds-cxx-idlc-0.10.2"
    )
    assert report["idlc"] == os.fspath(fake_idlc.resolve())
    assert report["idlc_version"] == "0.10.2"
    assert report["cxx_backend_available"] is True


@pytest.mark.parametrize(
    ("version", "backends", "expected_error"),
    [
        ("0.11.0", "c cxx", "expected pinned idlc version 0.10.2"),
        ("0.10.2", "c", "cxx backend"),
    ],
)
def test_prepare_dds_codegen_toolchain_check_rejects_unpinned_or_non_cxx_fake_idlc(
    tmp_path,
    version: str,
    backends: str,
    expected_error: str,
):
    fake_idlc = _make_fake_idlc(tmp_path, version=version, backends=backends)
    result = _run_prepare_codegen_tool(
        ["--check", "--dry-run", "--idlc", os.fspath(fake_idlc)]
    )

    assert result.returncode != 0
    assert expected_error in result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert expected_error in report["error"]
    assert report["idlc"] == os.fspath(fake_idlc.resolve())


def test_prepare_dds_codegen_toolchain_rejects_output_paths_outside_repo_build(tmp_path):
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    result = _run_prepare_codegen_tool(
        [
            "--check",
            "--dry-run",
            "--idlc",
            os.fspath(fake_idlc),
            "--toolchain-dir",
            os.fspath(tmp_path / "outside-repo-build"),
        ]
    )

    assert result.returncode != 0
    assert "toolchain dir must be under repo build/" in result.stderr


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
    assert "explicit --idlc or VISUAL_EVENTS_IDLC is required" in result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["foundation_ready"] is True
    assert report["visual_events_codegen_ready"] is False
    assert "explicit --idlc or VISUAL_EVENTS_IDLC is required" in report["visual_events_codegen_error"]


def test_build_tool_full_bridge_accepts_explicit_pinned_fake_idlc(tmp_path):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    report_path = tmp_path / "full-bridge-report.json"
    env = os.environ.copy()
    env["PATH"] = ""
    result = _run_build_tool(
        [
            "--check",
            "--check-full-bridge",
            "--idlc",
            os.fspath(fake_idlc),
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
    assert report["visual_events_codegen_ready"] is True
    assert report["visual_events_codegen_error"] == ""
    assert report["idl_generator"] == os.fspath(fake_idlc.resolve())
    assert report["idl_generator_version"] == "0.10.2"
    assert report["idl_generator_cxx_backend"] is True


def test_build_tool_full_bridge_accepts_visual_events_idlc_env(tmp_path):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    report_path = tmp_path / "full-bridge-env-report.json"
    env = os.environ.copy()
    env["PATH"] = ""
    env["VISUAL_EVENTS_IDLC"] = os.fspath(fake_idlc)
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

    assert result.returncode == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["visual_events_codegen_ready"] is True
    assert report["idl_generator"] == os.fspath(fake_idlc.resolve())


def test_build_tool_full_bridge_ignores_path_idlc_without_explicit_idlc(tmp_path):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    fake_path_idlc = path_bin / "idlc"
    fake_path_idlc.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  --version) printf '%s\\n' 'CycloneDDS idlc 0.10.2' ;;\n"
        "  --help|-l) printf '%s\\n' 'available backends: c cxx' ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_path_idlc.chmod(0o755)
    report_path = tmp_path / "full-bridge-path-report.json"
    env = os.environ.copy()
    env["PATH"] = os.fspath(path_bin)
    env.pop("VISUAL_EVENTS_IDLC", None)

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
    assert "explicit --idlc or VISUAL_EVENTS_IDLC is required" in result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["foundation_ready"] is True
    assert report["visual_events_codegen_ready"] is False
    assert "explicit --idlc or VISUAL_EVENTS_IDLC is required" in report["visual_events_codegen_error"]


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
