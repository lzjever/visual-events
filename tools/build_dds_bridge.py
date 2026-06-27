from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BRIDGE_DIR = REPO_ROOT / "native" / "dds_bridge"
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "dds_bridge"
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "dds_bridge" / "build_report.json"
IDL_GENERATORS = ("idlc", "cyclonedds-idlc", "fastddsgen")


class CheckError(RuntimeError):
    pass


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


def _find_idl_generator() -> str:
    for name in IDL_GENERATORS:
        found = shutil.which(name)
        if found:
            return found
    expected = ", ".join(IDL_GENERATORS)
    raise CheckError(
        "IDL generator is required before building Visual Events HeadStateV1_/GazeTargetV1_ "
        f"type support; none found on PATH ({expected})"
    )


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


def check_visual_events_codegen() -> dict[str, object]:
    generator = _find_idl_generator()
    return {
        "idl_generator": generator,
        "visual_events_codegen_ready": True,
        "visual_events_codegen_error": "",
    }


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

    try:
        unitree_sdk_root = _resolve_required_dir(args.unitree_sdk_root, "UNITREE_SDK_ROOT")
        video_dds_publisher_dir = _resolve_required_dir(
            args.video_dds_publisher_dir,
            "VIDEO_DDS_PUBLISHER_DIR",
        )
        report.update(check_foundation_environment(unitree_sdk_root, video_dds_publisher_dir))
        report["visual_events_codegen_error"] = "not required for foundation check"

        if args.check_full_bridge:
            try:
                report.update(check_visual_events_codegen())
            except CheckError as exc:
                report["visual_events_codegen_ready"] = False
                report["visual_events_codegen_error"] = str(exc)
                raise

        should_build = args.build or args.probe
        if should_build:
            report.update(
                configure_and_build(
                    unitree_sdk_root=unitree_sdk_root,
                    video_dds_publisher_dir=video_dds_publisher_dir,
                    build_dir=args.build_dir,
                    target="visual_events_dds_bridge_probe",
                )
            )
        if args.probe:
            report.update(run_probe(args.build_dir))

        report["ok"] = True
        _write_report(args.out, report)
        return 0
    except (CheckError, json.JSONDecodeError) as exc:
        report["error"] = str(exc)
        _write_report(args.out, report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
