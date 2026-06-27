from __future__ import annotations

import ast
import importlib
import json
import os
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "run_cli_local_e2e.py"


@dataclass
class FakeResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    collection_incomplete: bool = False


@dataclass
class FakeHealth:
    passed: bool
    failure_reason: str | None = None
    healthz_pid: int | None = None
    healthz_identity_verified: bool = False


class FakeProcess:
    def __init__(
        self,
        name: str,
        pid: int,
        *,
        stdout: str = "",
        stderr: str = "",
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
        collection_incomplete: bool = False,
    ) -> None:
        self.name = name
        self.pid = pid
        self.returncode: int | None = None
        self.stopped = False
        self.stdout_lines = stdout.splitlines()
        self.stderr_lines = stderr.splitlines()
        self.stdout_tail = stdout if stdout else f"{name} stdout tail"
        self.stderr_tail = stderr if stderr else f"{name} stderr tail"
        self.stdout_truncated = stdout_truncated
        self.stderr_truncated = stderr_truncated
        self.collection_incomplete = collection_incomplete

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.stopped = True
        if self.returncode is None:
            self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.stopped = True
        self.returncode = -9


class FakeRunner:
    def __init__(
        self,
        *,
        health: FakeHealth | None = None,
        sync_results: dict[str, FakeResult] | None = None,
        process_results: dict[str, FakeResult] | None = None,
    ) -> None:
        self.health = health or FakeHealth(
            passed=True,
            healthz_pid=1001,
            healthz_identity_verified=True,
        )
        self.sync_results = sync_results or {}
        self.process_results = process_results or {}
        self.events: list[tuple[str, str]] = []
        self.started: dict[str, FakeProcess] = {}
        self.commands: dict[str, list[str]] = {}
        self.health_urls: list[str] = []
        self._next_pid = 1001

    def start_process(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        name: str,
    ) -> FakeProcess:
        del cwd, env
        result = self.process_results.get(name, FakeResult(0))
        process = FakeProcess(
            name,
            self._next_pid,
            stdout=result.stdout,
            stderr=result.stderr,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            collection_incomplete=result.collection_incomplete,
        )
        self._next_pid += 1
        self.started[name] = process
        self.commands[name] = command
        self.events.append(("start", name))
        return process

    def run_sync(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        name: str,
        timeout_s: float | None = None,
    ) -> FakeResult:
        del cwd, env, timeout_s
        self.commands[name] = command
        self.events.append(("run_sync", name))
        return self.sync_results.get(name, FakeResult(0))

    def wait_healthz(
        self,
        url: str,
        process: FakeProcess,
        *,
        timeout_s: float,
        interval_s: float,
    ) -> FakeHealth:
        del process, timeout_s, interval_s
        self.health_urls.append(url)
        self.events.append(("healthz", "server"))
        return self.health

    def wait_process(
        self,
        process: FakeProcess,
        *,
        name: str,
        timeout_s: float | None = None,
    ) -> FakeResult:
        del timeout_s
        self.events.append(("wait", name))
        result = self.sync_results.get(name, FakeResult(0))
        process.returncode = result.returncode
        return result

    def sleep(self, seconds: float) -> None:
        self.events.append(("sleep", str(seconds)))

    def stop_process(self, process: FakeProcess) -> None:
        self.events.append(("stop", process.name))
        process.terminate()


def import_runner_module() -> Any:
    try:
        return importlib.import_module("tools.run_cli_local_e2e")
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected tools.run_cli_local_e2e module: {exc}")


def write_frame(path: Path, payload: bytes = b"jpeg") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def make_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_case(tmp_path: Path) -> dict[str, Path]:
    data_dir = tmp_path / "val-data"
    write_frame(data_dir / "scene-a" / "001.jpg")
    write_frame(data_dir / "scene-b" / "001.jpeg")
    out = tmp_path / "artifacts" / "cli-local-e2e.json"
    server_bin = make_executable(tmp_path / "bin" / "server")
    cli_bin = make_executable(tmp_path / "bin" / "visual-events-cli")
    build_dir = tmp_path / "build"
    dds_bridge = make_executable(build_dir / "visual_events_dds_bridge")
    return {
        "data_dir": data_dir,
        "out": out,
        "server_bin": server_bin,
        "cli_bin": cli_bin,
        "build_dir": build_dir,
        "dds_bridge": dds_bridge,
    }


def base_argv(paths: dict[str, Path], *extra: str) -> list[str]:
    return [
        "--data-dir",
        os.fspath(paths["data_dir"]),
        "--out",
        os.fspath(paths["out"]),
        "--server-bin",
        os.fspath(paths["server_bin"]),
        "--cli-bin",
        os.fspath(paths["cli_bin"]),
        "--build-dir",
        os.fspath(paths["build_dir"]),
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
        *extra,
    ]


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def successful_runner(
    *,
    cli_stdout: str = "",
    cli_stderr: str = "",
) -> FakeRunner:
    return FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            )
        },
        process_results={
            "cli": FakeResult(0, stdout=cli_stdout, stderr=cli_stderr),
        },
    )


def botified_frame(event: str = "person_waving", event_id: str = "front:evt_000456") -> str:
    payload = {
        "id": f"visual:{event_id}",
        "urgency": "normal",
        "timeout_secs": 8,
        "request": f"event={event} camera=front track_id=7 confidence=0.86",
        "expect": "ack",
    }
    inner = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"<botified>{inner}</botified>\n"


def assert_explicit_partial_smoke_fields(report: dict[str, Any], paths: dict[str, Path]) -> None:
    assert report["server_bin"] == os.fspath(paths["server_bin"])
    assert report["cli_bin"] == os.fspath(paths["cli_bin"])
    assert report["build_dir"] == os.fspath(paths["build_dir"])
    assert report["dds_bridge_bin"] == os.fspath(paths["dds_bridge"])
    assert report["head_state"]["required"] is True
    assert report["head_state"]["publisher_mode"] == "required"


def test_importable_and_source_audit_for_forbidden_imports() -> None:
    module = import_runner_module()
    assert callable(module.main)

    tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    denied = {
        "visual_events_cli",
        "visual_events_server",
        "unitree",
        "cyclonedds",
        "fastdds",
        "rti",
        "rclpy",
        "torch",
        "ultralytics",
    }
    assert imported_roots.isdisjoint(denied)


def test_argparse_rejects_missing_required_domain_or_network(capsys: pytest.CaptureFixture[str]) -> None:
    module = import_runner_module()

    with pytest.raises(SystemExit) as exc:
        module.parse_args(
            [
                "--data-dir",
                "/tmp/val-data",
                "--out",
                "/tmp/out.json",
                "--server-bin",
                "/tmp/server",
                "--cli-bin",
                "/tmp/cli",
            ]
        )

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "--dds-domain" in captured.err
    assert "--dds-network" in captured.err


def test_preflight_rejects_non_loopback_without_allow_and_writes_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)

    rc = module.main(base_argv(paths, "--dds-network", "eth0"), runner=FakeRunner())

    captured = capsys.readouterr()
    assert rc != 0
    assert captured.out == ""
    assert paths["out"].exists()
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any("--allow-non-loopback-dds" in reason for reason in report["failure_reasons"])


def test_preflight_rejects_out_under_data_dir_without_writing(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    paths["out"] = paths["data_dir"] / "reports" / "bad.json"

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    assert not paths["out"].exists()


def test_command_construction_uses_required_args_and_wrapper_paths(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout=(
                    '{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n'
                ),
            )
        }
    )

    rc = module.main(
        base_argv(
            paths,
            "--server",
            "ws://127.0.0.1:8877/v1/stream",
            "--scene",
            "scene-b",
            "--frame-count",
            "7",
            "--image-hz",
            "12.5",
            "--head-state",
            "moving",
            "--head-state-hz",
            "8.5",
            "--gaze-count",
            "1",
            "--gaze-timeout-ms",
            "1234",
        ),
        runner=runner,
    )

    assert rc == 0
    tools_dir = REPO_ROOT / "tools"
    assert runner.commands["server"] == [
        os.fspath(paths["server_bin"]),
        "--host",
        "127.0.0.1",
        "--port",
        "8877",
    ]
    assert runner.commands["cli"] == [
        os.fspath(paths["cli_bin"]),
        "--dds-runtime",
        "bridge",
        "--dds-bridge-bin",
        os.fspath(paths["dds_bridge"]),
        "--server",
        "ws://127.0.0.1:8877/v1/stream",
        "--camera",
        "front",
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
        "--image-topic",
        "/camera/image/jpeg",
        "--head-state-topic",
        "/robot/head_state",
        "--gaze-topic",
        "/visual_events/gaze_target",
    ]
    assert runner.commands["gaze_subscriber"] == [
        os.fspath(module.sys.executable),
        os.fspath(tools_dir / "subscribe_test_gaze_targets.py"),
        "--build-dir",
        os.fspath(paths["build_dir"]),
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
        "--count",
        "1",
        "--timeout-ms",
        "1234",
        "--gaze-topic",
        "/visual_events/gaze_target",
    ]
    assert runner.commands["image_publisher"] == [
        os.fspath(module.sys.executable),
        os.fspath(tools_dir / "publish_test_dds_images.py"),
        "--build-dir",
        os.fspath(paths["build_dir"]),
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
        "--input",
        os.fspath(paths["data_dir"] / "scene-b"),
        "--count",
        "7",
        "--hz",
        "12.5",
        "--camera-name",
        "image",
        "--camera-topic",
        "/camera/image/jpeg",
    ]
    assert runner.commands["head_publisher"] == [
        os.fspath(module.sys.executable),
        os.fspath(tools_dir / "publish_test_head_state.py"),
        "--build-dir",
        os.fspath(paths["build_dir"]),
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
        "--state",
        "moving",
        "--count",
        "7",
        "--hz",
        "8.5",
        "--head-state-topic",
        "/robot/head_state",
    ]


def test_fake_runner_success_writes_partial_pass_and_cleans_up(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout=(
                    '{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n'
                    '{"type":"gaze_target","camera":"front","state":"lost","valid":false}\n'
                    '{"type":"other","camera":"front","state":"tracking"}\n'
                ),
            )
        }
    )

    rc = module.main(base_argv(paths, "--gaze-count", "2"), runner=runner)

    assert rc == 0
    assert runner.events == [
        ("start", "server"),
        ("healthz", "server"),
        ("start", "gaze_subscriber"),
        ("start", "cli"),
        ("sleep", "0.5"),
        ("run_sync", "head_publisher"),
        ("run_sync", "image_publisher"),
        ("wait", "gaze_subscriber"),
        ("stop", "cli"),
        ("stop", "server"),
    ]
    assert runner.started["cli"].stopped is True
    assert runner.started["server"].stopped is True
    report = load_report(paths["out"])
    assert report["report_type"] == "cli_local_e2e_smoke_v1"
    assert report["slice_pass"] is True
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["pc_local_e2e_status"] == "partial_smoke_pass"
    assert report["failure_reasons"] == []
    assert report["manifest_source"] == "generated"
    assert report["scene_count"] == 2
    assert report["selected_scene"] == "scene-a"
    assert_explicit_partial_smoke_fields(report, paths)
    assert report["gaze"]["expected_count"] == 2
    assert report["gaze"]["received_lines"] == 3
    assert report["gaze"]["accepted_count"] == 2
    assert report["gaze"]["rejected_count"] == 1
    assert report["gaze"]["valid_count"] == 1
    assert report["gaze"]["invalid_count"] == 1
    assert report["gaze"]["state_counts"] == {"lost": 1, "tracking": 1}
    assert report["gaze"]["first_sample"]["state"] == "tracking"
    assert report["gaze"]["last_sample"]["state"] == "lost"
    assert report["botified_stdout"] == {
        "source": "cli_stdout",
        "required_frame_count": None,
        "line_count": 0,
        "frame_count": 0,
        "allowed_frame_count": 0,
        "pollution_count": 0,
        "parse_error_count": 0,
        "forbidden_event_count": 0,
        "event_counts": {},
        "first_frame": None,
        "last_frame": None,
        "collection_truncated": False,
        "collection_incomplete": False,
        "contract_violations": [],
    }
    assert "full_scene_matrix" in report["not_covered"]


def test_botified_stdout_valid_frame_is_reported_without_required_count(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner(cli_stdout=botified_frame())

    rc = module.main(base_argv(paths), runner=runner)

    assert rc == 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is True
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    botified = report["botified_stdout"]
    assert botified["source"] == "cli_stdout"
    assert botified["required_frame_count"] is None
    assert botified["line_count"] == 1
    assert botified["frame_count"] == 1
    assert botified["allowed_frame_count"] == 1
    assert botified["event_counts"] == {"person_waving": 1}
    assert botified["pollution_count"] == 0
    assert botified["parse_error_count"] == 0
    assert botified["forbidden_event_count"] == 0
    assert botified["contract_violations"] == []
    assert botified["first_frame"] == {
        "line": 1,
        "event": "person_waving",
        "payload": {
            "id": "visual:front:evt_000456",
            "urgency": "normal",
            "timeout_secs": 8,
            "request": "event=person_waving camera=front track_id=7 confidence=0.86",
            "expect": "ack",
        },
    }
    assert botified["last_frame"] == botified["first_frame"]


@pytest.mark.parametrize(
    "cli_stdout",
    [
        '{"type":"gaze_target","camera":"front","state":"tracking"}\n',
        "status: connected to visual events service\n",
        "visual_state frame=1 gaze_target=front:track-7\n",
    ],
)
def test_botified_stdout_pollution_fails_partial_slice(
    tmp_path: Path,
    cli_stdout: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner(cli_stdout=cli_stdout)

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["pc_local_e2e_status"] == "partial_smoke_failed"
    assert "botified_stdout_pollution" in report["failure_reasons"]
    assert report["botified_stdout"]["pollution_count"] == 1
    assert report["botified_stdout"]["frame_count"] == 0


def test_botified_stdout_malformed_frame_fails_partial_slice(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner(cli_stdout="<botified>{not-json}</botified>\n")

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert "botified_stdout_parse_errors" in report["failure_reasons"]
    assert report["botified_stdout"]["parse_error_count"] == 1
    assert report["botified_stdout"]["pollution_count"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {
            "event": "person_waving",
            "id": "visual:front:evt_000456",
            "urgency": "normal",
            "timeout_secs": 8,
            "expect": "ack",
        },
        {
            "id": "front:evt_000456",
            "urgency": "normal",
            "timeout_secs": 8,
            "request": "event=person_waving camera=front",
            "expect": "ack",
        },
        {
            "id": "visual:front:evt_000456",
            "urgency": "urgent",
            "timeout_secs": 8,
            "request": "event=person_waving camera=front",
            "expect": "ack",
        },
        {
            "id": "visual:front:evt_000456",
            "urgency": "normal",
            "timeout_secs": "8",
            "request": "event=person_waving camera=front",
            "expect": "ack",
        },
        {
            "id": "visual:front:evt_000456",
            "urgency": "normal",
            "timeout_secs": 8,
            "request": "event=unknown_visual_event camera=front",
            "expect": "ack",
        },
        {
            "id": "visual:front:evt_000456",
            "urgency": "normal",
            "timeout_secs": 8,
            "request": "event=person_waving camera=front",
            "expect": "reply",
        },
    ],
)
def test_botified_stdout_contract_violations_fail_as_parse_errors(
    tmp_path: Path,
    payload: dict[str, Any],
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    inner = json.dumps(payload, separators=(",", ":"))
    runner = successful_runner(cli_stdout=f"<botified>{inner}</botified>\n")

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert "botified_stdout_parse_errors" in report["failure_reasons"]
    botified = report["botified_stdout"]
    assert botified["parse_error_count"] == 1
    assert botified["frame_count"] == 0
    assert botified["allowed_frame_count"] == 0
    assert botified["forbidden_event_count"] == 0
    assert botified["contract_violations"]


def test_botified_stdout_forbidden_event_fails_partial_slice(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner(cli_stdout=botified_frame("attention_target_changed"))

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert "botified_stdout_forbidden_event" in report["failure_reasons"]
    botified = report["botified_stdout"]
    assert botified["frame_count"] == 1
    assert botified["allowed_frame_count"] == 0
    assert botified["forbidden_event_count"] == 1
    assert botified["event_counts"] == {}


def test_cli_stderr_tail_does_not_pollute_botified_stdout(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    cli_stderr = '{"type":"visual_state","status":"debug"}\n'
    runner = successful_runner(cli_stderr=cli_stderr)

    rc = module.main(base_argv(paths), runner=runner)

    assert rc == 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is True
    assert report["botified_stdout"]["pollution_count"] == 0
    assert report["botified_stdout"]["contract_violations"] == []
    assert report["stdout_stderr_tails"]["cli"]["stderr_tail"] == cli_stderr


@pytest.mark.parametrize(
    ("process_result", "expected_reason", "expected_field"),
    [
        (
            FakeResult(0, stdout=botified_frame(), stdout_truncated=True),
            "botified_stdout_collection_truncated",
            "collection_truncated",
        ),
        (
            FakeResult(0, stdout=botified_frame(), collection_incomplete=True),
            "botified_stdout_collection_incomplete",
            "collection_incomplete",
        ),
    ],
)
def test_botified_stdout_untrustworthy_collection_fails_partial_slice(
    tmp_path: Path,
    process_result: FakeResult,
    expected_reason: str,
    expected_field: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            )
        },
        process_results={"cli": process_result},
    )

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["pc_local_e2e_status"] == "partial_smoke_failed"
    assert expected_reason in report["failure_reasons"]
    assert report["botified_stdout"][expected_field] is True


def test_botified_stdout_summary_flags_untrustworthy_collection_as_failure_reasons() -> None:
    module = import_runner_module()

    summary = module._summarize_botified_stdout(
        [botified_frame().strip()],
        line_count=3,
        collection_truncated=True,
        collection_incomplete=True,
    )

    assert summary["collection_truncated"] is True
    assert summary["collection_incomplete"] is True
    assert module._botified_stdout_failure_reasons(summary) == [
        "botified_stdout_collection_truncated",
        "botified_stdout_collection_incomplete",
    ]


def test_local_process_runner_stop_process_drains_live_stdout_and_stderr(tmp_path: Path) -> None:
    module = import_runner_module()
    script = tmp_path / "writer.py"
    script.write_text(
        "\n".join(
            [
                "import sys",
                "import time",
                "print('stdout-one', flush=True)",
                "print('stderr-one', file=sys.stderr, flush=True)",
                "print('stdout-two', flush=True)",
                "print('stderr-two', file=sys.stderr, flush=True)",
                "time.sleep(30)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner = module.LocalProcessRunner()
    process = runner.start_process(
        [sys.executable, os.fspath(script)],
        cwd=tmp_path,
        env=os.environ.copy(),
        name="writer",
    )

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if process.stdout_line_count >= 2 and process.stderr_line_count >= 2:
            break
        time.sleep(0.01)

    runner.stop_process(process)

    assert process.poll() is not None
    assert process.stdout_lines == ["stdout-one", "stdout-two"]
    assert process.stderr_lines == ["stderr-one", "stderr-two"]
    assert process.stdout_tail == "stdout-one\nstdout-two\n"
    assert process.stderr_tail == "stderr-one\nstderr-two\n"
    assert process.stdout_truncated is False
    assert process.stderr_truncated is False
    assert process.collection_incomplete is False


@pytest.mark.parametrize(
    ("health", "sync_results", "expected_reason"),
    [
        (
            FakeHealth(False, failure_reason="healthz_timeout"),
            {},
            "healthz_timeout",
        ),
        (
            None,
            {"image_publisher": FakeResult(9, stderr="image failed")},
            "image_publisher_failed",
        ),
        (
            None,
            {"gaze_subscriber": FakeResult(0, stdout="{not-json}\n")},
            "gaze_target_count_shortfall",
        ),
    ],
)
def test_fake_runner_failures_write_nonzero_report_and_cleanup(
    tmp_path: Path,
    health: FakeHealth | None,
    sync_results: dict[str, FakeResult],
    expected_reason: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(health=health, sync_results=sync_results)

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["pc_local_e2e_status"] == "partial_smoke_failed"
    assert_explicit_partial_smoke_fields(report, paths)
    assert expected_reason in report["failure_reasons"]
    if "cli" in runner.started:
        assert runner.started["cli"].stopped is True
    if "server" in runner.started:
        assert runner.started["server"].stopped is True
    if "gaze_subscriber" in runner.started and runner.started["gaze_subscriber"].poll() is None:
        assert runner.started["gaze_subscriber"].stopped is True


def test_report_never_claims_full_ga_pass(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            )
        }
    )

    rc = module.main(base_argv(paths), runner=runner)

    assert rc == 0
    report = load_report(paths["out"])
    serialized = json.dumps(report, sort_keys=True).lower()
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert '"passed": true' not in serialized
    assert '"full_pass": true' not in serialized
    assert '"ga_gate_pass": true' not in serialized
    assert '"overall_pass": true' not in serialized
