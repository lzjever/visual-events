from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from tools.prepare_dds_codegen_toolchain import (
        CodegenToolchainError,
        check_idlc_codegen_toolchain,
    )
except ModuleNotFoundError:
    from prepare_dds_codegen_toolchain import (  # type: ignore[no-redef]
        CodegenToolchainError,
        check_idlc_codegen_toolchain,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BRIDGE_DIR = REPO_ROOT / "native" / "dds_bridge"
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "dds_bridge"
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "dds_bridge" / "build_report.json"


class CheckError(RuntimeError):
    def __init__(self, message: str, *, report: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.report = report or {}


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return None


def _resolve_required_dir(path: Path | None, label: str) -> Path:
    if path is None:
        raise CheckError(f"{label} is required")
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise CheckError(f"{label} does not exist or is not a directory: {resolved}")
    return resolved


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _require_repo_build_path(path: Path, label: str) -> Path:
    resolved = _resolve_path(path)
    build_root = (REPO_ROOT / "build").resolve()
    try:
        resolved.relative_to(build_root)
    except ValueError as exc:
        raise CheckError(f"{label} must be under repo build/: {resolved}") from exc
    return resolved


def _require_repo_artifact_path(path: Path, label: str) -> Path:
    resolved = _resolve_path(path)
    artifacts_root = (REPO_ROOT / "artifacts").resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as exc:
        raise CheckError(f"{label} must be under repo artifacts/: {resolved}") from exc
    return resolved


def _check_unitree_sdk_root(path: Path) -> None:
    required = [
        path / "lib" / "cmake" / "unitree_sdk2" / "unitree_sdk2Config.cmake",
        path / "lib" / "libunitree_sdk2.a",
        path / "lib" / "libddsc.so",
        path / "lib" / "libddscxx.so",
    ]
    missing = [item for item in required if not item.exists()]
    if missing:
        formatted = ", ".join(os.fspath(item) for item in missing)
        raise CheckError(f"UNITREE_SDK_ROOT is missing required SDK2 files: {formatted}")


def _check_video_dds_publisher_dir(path: Path) -> None:
    required = [
        path / "include" / "unitree_camera" / "msg" / "dds" / "CameraFrame_.hpp",
        path / "src" / "CameraFrame_.cpp",
    ]
    missing = [item for item in required if not item.exists()]
    if missing:
        formatted = ", ".join(os.fspath(item) for item in missing)
        raise CheckError(f"VIDEO_DDS_PUBLISHER_DIR is missing CameraFrame_ inputs: {formatted}")


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def check_foundation_environment(
    unitree_sdk_root: Path,
    video_dds_publisher_dir: Path,
) -> dict[str, object]:
    _check_unitree_sdk_root(unitree_sdk_root)
    _check_video_dds_publisher_dir(video_dds_publisher_dir)
    return {
        "unitree_sdk_root": os.fspath(unitree_sdk_root),
        "video_dds_publisher_dir": os.fspath(video_dds_publisher_dir),
        "foundation_ready": True,
    }


def check_visual_events_codegen(
    idlc: Path | None,
    *,
    probe_idls: list[Path] | None = None,
    probe_output_dir: Path | None = None,
) -> dict[str, object]:
    try:
        result = check_idlc_codegen_toolchain(
            idlc,
            probe_codegen=True,
            probe_idls=probe_idls,
            probe_output_dir=probe_output_dir,
        )
    except CodegenToolchainError as exc:
        raise CheckError(str(exc), report=exc.report) from exc
    report = {
        "idl_generator": result["idlc"],
        "idl_generator_version": result["idlc_version"],
        "idl_generator_cxx_backend": result["cxx_backend_available"],
        "visual_events_codegen_ready": result["oracle_ok"] is True,
        "visual_events_codegen_error": "",
    }
    report.update(result)
    return report


def configure_and_build(
    *,
    unitree_sdk_root: Path,
    video_dds_publisher_dir: Path,
    build_dir: Path,
    target: str,
) -> dict[str, object]:
    build_dir.mkdir(parents=True, exist_ok=True)
    configure = _run(
        [
            "cmake",
            "-S",
            os.fspath(NATIVE_BRIDGE_DIR),
            "-B",
            os.fspath(build_dir),
            f"-DUNITREE_SDK_ROOT={unitree_sdk_root}",
            f"-DVIDEO_DDS_PUBLISHER_DIR={video_dds_publisher_dir}",
        ],
        cwd=REPO_ROOT,
    )
    if configure.returncode != 0:
        raise CheckError("CMake configure failed:\n" + configure.stderr)

    build = _run(["cmake", "--build", os.fspath(build_dir), "--target", target], cwd=REPO_ROOT)
    if build.returncode != 0:
        raise CheckError("CMake build failed:\n" + build.stderr)

    return {
        "build_dir": os.fspath(build_dir),
        "target": target,
        "binary": os.fspath(build_dir / target),
    }


def run_probe(build_dir: Path) -> dict[str, object]:
    binary = build_dir / "visual_events_dds_bridge_probe"
    if not binary.exists():
        raise CheckError(f"probe binary does not exist: {binary}")
    result = _run([os.fspath(binary), "--probe"], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise CheckError("probe failed:\n" + result.stderr)
    lines = result.stdout.splitlines()
    if len(lines) != 1:
        raise CheckError(f"probe stdout must be exactly one JSONL line, got {len(lines)} lines")
    status = json.loads(lines[0])
    if status.get("protocol_version") != 1:
        raise CheckError("probe protocol_version must be 1")
    if status.get("type") != "status":
        raise CheckError("probe status frame must include type=status")
    if status.get("code") != "probe_ok":
        raise CheckError("probe status frame must include code=probe_ok")
    if not isinstance(status.get("message"), str) or not status["message"]:
        raise CheckError("probe status frame must include non-empty message")
    return {"probe_status": status, "probe_stderr": result.stderr}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check/build the Visual Events native DDS bridge probe")
    parser.add_argument(
        "--unitree-sdk-root",
        type=Path,
        default=_env_path("UNITREE_SDK_ROOT", "VISUAL_EVENTS_UNITREE_SDK_ROOT"),
    )
    parser.add_argument(
        "--video-dds-publisher-dir",
        type=Path,
        default=_env_path("VIDEO_DDS_PUBLISHER_DIR", "VISUAL_EVENTS_VIDEO_DDS_PUBLISHER_DIR"),
    )
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--idlc", type=Path, default=_env_path("VISUAL_EVENTS_IDLC"))
    parser.add_argument(
        "--codegen-probe-idl",
        type=Path,
        action="append",
        default=None,
        help="IDL file for the full-bridge C++ idlc codegen oracle; repeatable; defaults to repo Head/Gaze IDLs",
    )
    parser.add_argument(
        "--codegen-probe-output-dir",
        type=Path,
        default=None,
        help="repo build/ directory for the full-bridge C++ idlc codegen oracle",
    )
    parser.add_argument("--check", action="store_true", help="only validate foundation inputs")
    parser.add_argument(
        "--check-full-bridge",
        action="store_true",
        help="also require Visual Events HeadStateV1_/GazeTargetV1_ IDL codegen toolchain",
    )
    parser.add_argument("--build", action="store_true", help="run CMake configure/build after checks")
    parser.add_argument("--probe", action="store_true", help="run the built probe after checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report: dict[str, object] = {
        "ok": False,
        "timestamp_unix_ms": int(time.time() * 1000),
        "native_bridge_dir": os.fspath(NATIVE_BRIDGE_DIR),
        "foundation_ready": False,
        "visual_events_codegen_ready": False,
        "visual_events_codegen_error": "not checked",
    }
    report_path: Path | None = None

    try:
        report_path = _require_repo_artifact_path(args.out, "report path")
        build_dir = _require_repo_build_path(args.build_dir, "build dir")
        unitree_sdk_root = _resolve_required_dir(args.unitree_sdk_root, "UNITREE_SDK_ROOT")
        video_dds_publisher_dir = _resolve_required_dir(
            args.video_dds_publisher_dir,
            "VIDEO_DDS_PUBLISHER_DIR",
        )
        report.update(check_foundation_environment(unitree_sdk_root, video_dds_publisher_dir))
        report["visual_events_codegen_error"] = "not required for foundation check"

        if args.check_full_bridge:
            try:
                report.update(
                    check_visual_events_codegen(
                        args.idlc,
                        probe_idls=args.codegen_probe_idl,
                        probe_output_dir=args.codegen_probe_output_dir,
                    )
                )
            except CheckError as exc:
                report.update(exc.report)
                report["visual_events_codegen_ready"] = False
                report["visual_events_codegen_error"] = str(exc)
                raise

        should_build = args.build or args.probe
        if should_build:
            report.update(
                configure_and_build(
                    unitree_sdk_root=unitree_sdk_root,
                    video_dds_publisher_dir=video_dds_publisher_dir,
                    build_dir=build_dir,
                    target="visual_events_dds_bridge_probe",
                )
            )
        if args.probe:
            report.update(run_probe(build_dir))

        report["ok"] = True
        _write_report(report_path, report)
        return 0
    except (CheckError, json.JSONDecodeError) as exc:
        report["error"] = str(exc)
        if report_path is not None:
            _write_report(report_path, report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
