from __future__ import annotations

import importlib
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"


WRAPPERS = {
    "publish_test_dds_images": {
        "module": "tools.publish_test_dds_images",
        "path": TOOLS_DIR / "publish_test_dds_images.py",
        "binary": "visual_events_dds_bridge_publish_test_dds_images",
        "native_args": [
            "--input",
            "/tmp/frames",
            "--count",
            "3",
            "--hz",
            "10",
            "--camera-name",
            "front-left",
        ],
        "expected_native_args": [
            "--input",
            "/tmp/frames",
            "--count",
            "3",
            "--hz",
            "10",
            "--camera-name",
            "front-left",
        ],
        "topic_arg": ["--camera-topic", "/camera/front/jpeg"],
        "topic_env": "VISUAL_EVENTS_CAMERA_TOPIC",
        "topic_value": "/camera/front/jpeg",
    },
    "publish_test_head_state": {
        "module": "tools.publish_test_head_state",
        "path": TOOLS_DIR / "publish_test_head_state.py",
        "binary": "visual_events_dds_bridge_publish_test_head_state",
        "native_args": ["--state", "stationary", "--count", "4", "--hz", "9"],
        "expected_native_args": ["--state", "stationary", "--count", "4", "--hz", "9"],
        "topic_arg": ["--head-state-topic", "/robot/head_state_test"],
        "topic_env": "VISUAL_EVENTS_HEAD_STATE_TOPIC",
        "topic_value": "/robot/head_state_test",
    },
    "subscribe_test_gaze_targets": {
        "module": "tools.subscribe_test_gaze_targets",
        "path": TOOLS_DIR / "subscribe_test_gaze_targets.py",
        "binary": "visual_events_dds_bridge_subscribe_test_gaze_targets",
        "native_args": ["--count", "2", "--timeout-ms", "250"],
        "expected_native_args": ["--count", "2", "--timeout-ms", "250"],
        "topic_arg": ["--gaze-topic", "/visual_events/gaze_test"],
        "topic_env": "VISUAL_EVENTS_GAZE_TOPIC",
        "topic_value": "/visual_events/gaze_test",
    },
}


def _import_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected importable wrapper module {name}: {exc}")


def _make_build_dir(tmp_path: Path, binary_name: str) -> Path:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    binary = build_dir / binary_name
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return build_dir


def _common_args(build_dir: Path) -> list[str]:
    return [
        "--build-dir",
        os.fspath(build_dir),
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
    ]


def _call_main(module: Any, argv: list[str]) -> int:
    result = module.main(argv)
    assert isinstance(result, int)
    return result


def test_importable_wrapper_files_exist() -> None:
    helper = TOOLS_DIR / "dds_pc_tools.py"
    assert helper.is_file()
    _import_module("tools.dds_pc_tools")

    for spec in WRAPPERS.values():
        assert spec["path"].is_file()
        _import_module(spec["module"])


@pytest.mark.parametrize("spec", WRAPPERS.values(), ids=WRAPPERS.keys())
def test_missing_domain_or_network_fails_before_subprocess(
    spec: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _import_module(spec["module"])
    build_dir = _make_build_dir(tmp_path, spec["binary"])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = _call_main(
        module,
        [
            "--build-dir",
            os.fspath(build_dir),
            *spec["native_args"],
        ],
    )

    captured = capsys.readouterr()
    assert rc != 0
    assert "--dds-domain" in captured.err
    assert "--dds-network" in captured.err
    assert calls == []


@pytest.mark.parametrize("spec", WRAPPERS.values(), ids=WRAPPERS.keys())
def test_non_loopback_requires_explicit_allow_and_does_not_start_subprocess(
    spec: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _import_module(spec["module"])
    build_dir = _make_build_dir(tmp_path, spec["binary"])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = _call_main(
        module,
        [
            *spec["native_args"],
            "--build-dir",
            os.fspath(build_dir),
            "--dds-domain",
            "57",
            "--dds-network",
            "eth0",
        ],
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "--allow-non-loopback-dds" in captured.err
    assert calls == []


@pytest.mark.parametrize("spec", WRAPPERS.values(), ids=WRAPPERS.keys())
def test_build_dir_missing_binary_fails_before_subprocess(
    spec: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _import_module(spec["module"])
    build_dir = tmp_path / "empty-build"
    build_dir.mkdir()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = _call_main(module, [*spec["native_args"], *_common_args(build_dir)])

    captured = capsys.readouterr()
    assert rc == 1
    assert spec["binary"] in captured.err
    assert "not found or not executable" in captured.err
    assert calls == []


@pytest.mark.parametrize("spec", WRAPPERS.values(), ids=WRAPPERS.keys())
def test_child_command_uses_correct_native_binary_and_args(
    spec: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_module(spec["module"])
    build_dir = _make_build_dir(tmp_path, spec["binary"])
    calls: list[dict[str, Any]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, "kwargs": kwargs})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = _call_main(module, [*spec["native_args"], *_common_args(build_dir)])

    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["command"] == [
        os.fspath((build_dir / spec["binary"]).resolve()),
        *spec["expected_native_args"],
    ]
    assert calls[0]["kwargs"]["check"] is False
    assert "stdout" not in calls[0]["kwargs"]
    assert "stderr" not in calls[0]["kwargs"]


@pytest.mark.parametrize("spec", WRAPPERS.values(), ids=WRAPPERS.keys())
def test_env_domain_network_cli_overrides_existing_env_and_sets_topic(
    spec: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_module(spec["module"])
    build_dir = _make_build_dir(tmp_path, spec["binary"])
    calls: list[dict[str, Any]] = []
    monkeypatch.setenv("VISUAL_EVENTS_DDS_DOMAIN", "9")
    monkeypatch.setenv("VISUAL_EVENTS_DDS_NETWORK", "eth-test")
    monkeypatch.setenv(spec["topic_env"], "/wrong/topic")
    monkeypatch.setenv("VISUAL_EVENTS_UNRELATED", "keep")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, "kwargs": kwargs})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = _call_main(
        module,
        [*spec["native_args"], *spec["topic_arg"], *_common_args(build_dir)],
    )

    assert rc == 0
    child_env = calls[0]["kwargs"]["env"]
    assert child_env["VISUAL_EVENTS_DDS_DOMAIN"] == "57"
    assert child_env["VISUAL_EVENTS_DDS_NETWORK"] == "lo"
    assert child_env[spec["topic_env"]] == spec["topic_value"]
    assert child_env["VISUAL_EVENTS_UNRELATED"] == "keep"


def test_child_returncode_stdout_stderr_pass_through_without_wrapper_pollution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = WRAPPERS["publish_test_head_state"]
    module = _import_module(spec["module"])
    build_dir = _make_build_dir(tmp_path, spec["binary"])

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        print("native stdout line")
        print("native stderr line", file=sys.stderr)
        return subprocess.CompletedProcess(command, 23)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = _call_main(module, [*spec["native_args"], *_common_args(build_dir)])

    captured = capsys.readouterr()
    assert rc == 23
    assert captured.out == "native stdout line\n"
    assert captured.err == "native stderr line\n"


def test_wrappers_do_not_import_dds_sdk_or_visual_events_cli() -> None:
    denied_imports = [
        "visual_events_cli",
        "unitree",
        "cyclonedds",
        "fastdds",
        "rti.connext",
        "rclpy",
    ]
    paths = [TOOLS_DIR / "dds_pc_tools.py", *(spec["path"] for spec in WRAPPERS.values())]

    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in denied_imports:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports or references {token}")

    assert offenders == []
