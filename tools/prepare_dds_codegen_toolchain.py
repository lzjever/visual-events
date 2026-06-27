from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_CYCLONEDDS_VERSION = "0.10.2"
PINNED_CYCLONEDDS_CXX_VERSION = "0.10.2"
DEFAULT_TOOLCHAIN_DIR = (
    REPO_ROOT / "build" / "tools" / f"cyclonedds-cxx-idlc-{PINNED_CYCLONEDDS_VERSION}"
)


class CodegenToolchainError(RuntimeError):
    pass


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value)


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _require_repo_build_path(path: Path, label: str) -> Path:
    resolved = _resolve_path(path)
    build_root = (REPO_ROOT / "build").resolve()
    try:
        resolved.relative_to(build_root)
    except ValueError as exc:
        raise CodegenToolchainError(
            f"{label} must be under repo build/: {resolved}"
        ) from exc
    return resolved


def _require_pinned_version(value: str, expected: str, label: str) -> None:
    if value != expected:
        raise CodegenToolchainError(f"{label} must be pinned to {expected}, got {value}")


def _run_idlc(idlc: Path, arg: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.fspath(idlc), arg],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def _parse_semver(output: str) -> str:
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", output)
    if not match:
        raise CodegenToolchainError("could not parse idlc version from --version output")
    return match.group(1)


def _has_cxx_backend(output: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_+-])cxx(?![A-Za-z0-9_+-])", output, re.I) is not None


def check_idlc_codegen_toolchain(
    idlc: Path | None,
    *,
    expected_version: str = PINNED_CYCLONEDDS_VERSION,
) -> dict[str, object]:
    if idlc is None:
        raise CodegenToolchainError(
            "explicit --idlc or VISUAL_EVENTS_IDLC is required; PATH is intentionally not searched"
        )

    resolved = _resolve_path(idlc)
    if not resolved.is_file():
        raise CodegenToolchainError(f"idlc does not exist or is not a file: {resolved}")

    try:
        version_result = _run_idlc(resolved, "--version")
    except OSError as exc:
        raise CodegenToolchainError(f"failed to execute idlc: {resolved}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CodegenToolchainError(f"idlc --version timed out: {resolved}") from exc
    if version_result.returncode != 0:
        output = (version_result.stdout + version_result.stderr).strip()
        raise CodegenToolchainError(f"idlc --version failed for {resolved}: {output}")

    version = _parse_semver(version_result.stdout + version_result.stderr)
    if version != expected_version:
        raise CodegenToolchainError(
            f"expected pinned idlc version {expected_version}, got {version}"
        )

    backend_output_parts: list[str] = []
    backend_errors: list[str] = []
    for arg in ("--help", "-l"):
        try:
            result = _run_idlc(resolved, arg)
        except OSError as exc:
            backend_errors.append(f"{arg}: {exc}")
            continue
        except subprocess.TimeoutExpired:
            backend_errors.append(f"{arg}: timed out")
            continue
        output = result.stdout + result.stderr
        if result.returncode == 0:
            backend_output_parts.append(output)
        else:
            backend_errors.append(f"{arg}: {output.strip()}")

    backend_output = "\n".join(backend_output_parts)
    if not backend_output_parts:
        detail = "; ".join(part for part in backend_errors if part)
        raise CodegenToolchainError(f"could not inspect idlc backends: {detail}")
    if not _has_cxx_backend(backend_output):
        raise CodegenToolchainError("idlc cxx backend is required")

    return {
        "idlc": os.fspath(resolved),
        "idlc_version": version,
        "cxx_backend_available": True,
    }


def build_plan(
    *,
    cyclonedds_version: str,
    cyclonedds_cxx_version: str,
    toolchain_dir: Path,
    idlc: Path | None,
    dry_run: bool,
) -> dict[str, object]:
    _require_pinned_version(
        cyclonedds_version,
        PINNED_CYCLONEDDS_VERSION,
        "CycloneDDS version",
    )
    _require_pinned_version(
        cyclonedds_cxx_version,
        PINNED_CYCLONEDDS_CXX_VERSION,
        "CycloneDDS-CXX version",
    )
    resolved_toolchain_dir = _require_repo_build_path(toolchain_dir, "toolchain dir")
    report: dict[str, object] = {
        "ok": True,
        "dry_run": dry_run,
        "will_write": False,
        "repo_root": os.fspath(REPO_ROOT),
        "toolchain_dir": os.fspath(resolved_toolchain_dir),
        "cyclonedds_version": cyclonedds_version,
        "cyclonedds_cxx_version": cyclonedds_cxx_version,
        "expected_idlc_version": PINNED_CYCLONEDDS_VERSION,
    }
    report.update(check_idlc_codegen_toolchain(idlc, expected_version=cyclonedds_version))
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the pinned CycloneDDS/CycloneDDS-CXX idlc plan for Visual Events "
            "HeadStateV1_/GazeTargetV1_ C++ type support"
        )
    )
    parser.add_argument(
        "--cyclonedds-version",
        default=PINNED_CYCLONEDDS_VERSION,
        help="pinned CycloneDDS version; only 0.10.2 is supported in this slice",
    )
    parser.add_argument(
        "--cyclonedds-cxx-version",
        default=PINNED_CYCLONEDDS_CXX_VERSION,
        help="pinned CycloneDDS-CXX version; only 0.10.2 is supported in this slice",
    )
    parser.add_argument("--toolchain-dir", type=Path, default=DEFAULT_TOOLCHAIN_DIR)
    parser.add_argument("--idlc", type=Path, default=_env_path("VISUAL_EVENTS_IDLC"))
    parser.add_argument("--check", action="store_true", help="validate the local idlc/toolchain plan")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not download, build, or write files; this slice only supports dry-run checks",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dry_run = bool(args.dry_run or args.check)

    report: dict[str, object] = {
        "ok": False,
        "dry_run": dry_run,
        "will_write": False,
        "repo_root": os.fspath(REPO_ROOT),
        "toolchain_dir": os.fspath(_resolve_path(args.toolchain_dir)),
        "cyclonedds_version": args.cyclonedds_version,
        "cyclonedds_cxx_version": args.cyclonedds_cxx_version,
        "expected_idlc_version": PINNED_CYCLONEDDS_VERSION,
    }
    if args.idlc is not None:
        report["idlc"] = os.fspath(_resolve_path(args.idlc))

    try:
        if not dry_run:
            raise CodegenToolchainError(
                "this slice only supports --check/--dry-run; download/build is intentionally not implemented"
            )
        report.update(
            build_plan(
                cyclonedds_version=args.cyclonedds_version,
                cyclonedds_cxx_version=args.cyclonedds_cxx_version,
                toolchain_dir=args.toolchain_dir,
                idlc=args.idlc,
                dry_run=dry_run,
            )
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except CodegenToolchainError as exc:
        report["error"] = str(exc)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
