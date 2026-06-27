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
DEFAULT_PROBE_IDL = Path("/home/galbot/works/video_dds_publisher/idl/CameraFrame_.idl")
DEFAULT_PROBE_OUTPUT_DIR = DEFAULT_TOOLCHAIN_DIR / "codegen_probe"
CANNOT_LOAD_GENERATOR_RE = re.compile(
    r"cannot load generator(?:\s+cxx)?|libcycloneddsidlcxx",
    re.I,
)


class CodegenToolchainError(RuntimeError):
    def __init__(self, message: str, *, report: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.report = report or {}


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


def _run_idlc_args(idlc: Path, args: list[str], *, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.fspath(idlc), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _run_idlc(idlc: Path, arg: str) -> subprocess.CompletedProcess[str]:
    return _run_idlc_args(idlc, [arg])


def _parse_semver(output: str) -> str:
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", output)
    if not match:
        raise CodegenToolchainError("could not parse idlc version from --version output")
    return match.group(1)


def _has_cxx_backend(output: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_+-])cxx(?![A-Za-z0-9_+-])", output, re.I) is not None


def _compact_output(output: str) -> str:
    return " ".join(output.strip().split())


def _read_idlc_version(idlc: Path) -> tuple[str, dict[str, object]]:
    attempts: list[str] = []
    for arg in ("--version", "-v"):
        try:
            result = _run_idlc(idlc, arg)
        except OSError as exc:
            raise CodegenToolchainError(f"failed to execute idlc: {idlc}: {exc}") from exc
        except subprocess.TimeoutExpired:
            attempts.append(f"{arg}: timed out")
            continue

        output = result.stdout + result.stderr
        if CANNOT_LOAD_GENERATOR_RE.search(output):
            raise CodegenToolchainError(
                f"idlc cannot load generator while reading version: {_compact_output(output)}",
                report={
                    "idlc_version_arg": arg,
                    "idlc_version_stdout": result.stdout,
                    "idlc_version_stderr": result.stderr,
                    "cxx_backend_available": False,
                },
            )
        if result.returncode != 0:
            attempts.append(f"{arg}: rc={result.returncode} {_compact_output(output)}")
            continue
        try:
            return _parse_semver(output), {
                "idlc_version_arg": arg,
                "idlc_version_stdout": result.stdout,
                "idlc_version_stderr": result.stderr,
            }
        except CodegenToolchainError:
            attempts.append(f"{arg}: could not parse version from {_compact_output(output)}")

    detail = "; ".join(part for part in attempts if part)
    raise CodegenToolchainError(f"could not parse idlc version from --version/-v output: {detail}")


def _inspect_backend_text(idlc: Path) -> dict[str, object]:
    backend_output_parts: list[str] = []
    backend_errors: list[str] = []
    for arg in ("--help", "-h", "-l"):
        try:
            result = _run_idlc(idlc, arg)
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
            backend_errors.append(f"{arg}: {_compact_output(output)}")

    backend_output = "\n".join(backend_output_parts)
    report: dict[str, object] = {
        "idlc_backend_inspection_stdout": backend_output,
        "idlc_backend_inspection_errors": backend_errors,
    }
    if not backend_output_parts:
        detail = "; ".join(part for part in backend_errors if part)
        raise CodegenToolchainError(f"could not inspect idlc backends: {detail}", report=report)
    if CANNOT_LOAD_GENERATOR_RE.search(backend_output):
        report["cxx_backend_available"] = False
        raise CodegenToolchainError(
            f"idlc cannot load generator cxx: {_compact_output(backend_output)}",
            report=report,
        )
    if not _has_cxx_backend(backend_output):
        report["cxx_backend_available"] = False
        raise CodegenToolchainError("idlc cxx backend is required", report=report)

    report["cxx_backend_available"] = True
    return report


def _resolve_probe_idl(probe_idl: Path | None, *, require_exists: bool) -> Path | None:
    selected = probe_idl
    if selected is None and DEFAULT_PROBE_IDL.is_file():
        selected = DEFAULT_PROBE_IDL
    if selected is None:
        if require_exists:
            raise CodegenToolchainError(
                f"--probe-idl is required because default probe IDL does not exist: {DEFAULT_PROBE_IDL}"
            )
        return None

    resolved = _resolve_path(selected)
    if require_exists and not resolved.is_file():
        raise CodegenToolchainError(f"probe IDL does not exist or is not a file: {resolved}")
    return resolved


def _probe_output_dir(probe_output_dir: Path | None) -> Path:
    return _require_repo_build_path(
        probe_output_dir or DEFAULT_PROBE_OUTPUT_DIR,
        "probe output dir",
    )


def _expected_probe_files(probe_idl: Path | None) -> list[str]:
    if probe_idl is None:
        return []
    return [f"{probe_idl.stem}.hpp", f"{probe_idl.stem}.cpp"]


def _list_generated_files(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(
        os.fspath(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    )


def _remove_expected_probe_outputs(output_dir: Path, expected_files: list[str]) -> None:
    for filename in expected_files:
        path = output_dir / filename
        if path.is_dir():
            raise CodegenToolchainError(f"expected generated file path is a directory: {path}")
        if path.exists():
            path.unlink()


def _probe_idlc_codegen(
    idlc: Path,
    *,
    probe_idl: Path,
    output_dir: Path,
) -> dict[str, object]:
    expected_files = _expected_probe_files(probe_idl)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_expected_probe_outputs(output_dir, expected_files)

    try:
        result = _run_idlc_args(
            idlc,
            ["-l", "cxx", "-o", os.fspath(output_dir), os.fspath(probe_idl)],
            timeout=30,
        )
    except OSError as exc:
        raise CodegenToolchainError(f"failed to execute idlc codegen probe: {idlc}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CodegenToolchainError(f"idlc codegen probe timed out: {idlc}") from exc

    combined_output = result.stdout + result.stderr
    generated_files = _list_generated_files(output_dir)
    presence = {
        filename: (output_dir / filename).is_file()
        for filename in expected_files
    }
    expected_present = bool(expected_files) and all(presence.values())
    cannot_load_generator = CANNOT_LOAD_GENERATOR_RE.search(combined_output) is not None
    oracle_ok = result.returncode == 0 and not cannot_load_generator and expected_present

    report: dict[str, object] = {
        "probe_codegen": True,
        "probe_idl": os.fspath(probe_idl),
        "probe_output_dir": os.fspath(output_dir),
        "generated_files": generated_files,
        "expected_generated_files": expected_files,
        "expected_generated_file_presence": presence,
        "expected_generated_files_present": expected_present,
        "idlc_codegen_returncode": result.returncode,
        "idlc_codegen_stdout": result.stdout,
        "idlc_codegen_stderr": result.stderr,
        "cxx_backend_available": oracle_ok,
        "oracle_ok": oracle_ok,
    }

    if cannot_load_generator:
        raise CodegenToolchainError(
            f"idlc cannot load generator cxx: {_compact_output(combined_output)}",
            report=report,
        )
    if result.returncode != 0:
        raise CodegenToolchainError(
            f"idlc cxx codegen probe failed with rc={result.returncode}: "
            f"{_compact_output(combined_output)}",
            report=report,
        )
    if not expected_present:
        missing = [filename for filename, present in presence.items() if not present]
        raise CodegenToolchainError(
            "missing expected generated files: " + ", ".join(missing),
            report=report,
        )

    return report


def check_idlc_codegen_toolchain(
    idlc: Path | None,
    *,
    expected_version: str = PINNED_CYCLONEDDS_VERSION,
    probe_codegen: bool = False,
    probe_idl: Path | None = None,
    probe_output_dir: Path | None = None,
) -> dict[str, object]:
    if idlc is None:
        raise CodegenToolchainError(
            "explicit --idlc or VISUAL_EVENTS_IDLC is required; PATH is intentionally not searched"
        )

    resolved = _resolve_path(idlc)
    if not resolved.is_file():
        raise CodegenToolchainError(f"idlc does not exist or is not a file: {resolved}")

    resolved_probe_output_dir = _probe_output_dir(probe_output_dir)
    resolved_probe_idl = _resolve_probe_idl(probe_idl, require_exists=probe_codegen)
    expected_files = _expected_probe_files(resolved_probe_idl)

    report: dict[str, object] = {
        "idlc": os.fspath(resolved),
        "probe_codegen": probe_codegen,
        "probe_idl": os.fspath(resolved_probe_idl) if resolved_probe_idl is not None else None,
        "probe_output_dir": os.fspath(resolved_probe_output_dir),
        "generated_files": [],
        "expected_generated_files": expected_files,
        "expected_generated_file_presence": {filename: False for filename in expected_files},
        "expected_generated_files_present": False,
        "cxx_backend_available": False,
        "oracle_ok": False,
    }

    try:
        version, version_report = _read_idlc_version(resolved)
    except CodegenToolchainError as exc:
        raise CodegenToolchainError(str(exc), report={**report, **exc.report}) from exc
    report.update(version_report)
    report["idlc_version"] = version

    if version != expected_version:
        raise CodegenToolchainError(
            f"expected pinned idlc version {expected_version}, got {version}",
            report=report,
        )

    if probe_codegen:
        try:
            report.update(
                _probe_idlc_codegen(
                    resolved,
                    probe_idl=resolved_probe_idl,
                    output_dir=resolved_probe_output_dir,
                )
            )
        except CodegenToolchainError as exc:
            raise CodegenToolchainError(str(exc), report={**report, **exc.report}) from exc
        return report

    try:
        report.update(_inspect_backend_text(resolved))
    except CodegenToolchainError as exc:
        raise CodegenToolchainError(str(exc), report={**report, **exc.report}) from exc
    report["oracle_ok"] = False
    return report


def build_plan(
    *,
    cyclonedds_version: str,
    cyclonedds_cxx_version: str,
    toolchain_dir: Path,
    idlc: Path | None,
    dry_run: bool,
    probe_codegen: bool,
    probe_idl: Path | None,
    probe_output_dir: Path | None,
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
    resolved_probe_output_dir = _probe_output_dir(probe_output_dir)
    report: dict[str, object] = {
        "ok": True,
        "dry_run": dry_run,
        "will_write": probe_codegen,
        "repo_root": os.fspath(REPO_ROOT),
        "toolchain_dir": os.fspath(resolved_toolchain_dir),
        "cyclonedds_version": cyclonedds_version,
        "cyclonedds_cxx_version": cyclonedds_cxx_version,
        "expected_idlc_version": PINNED_CYCLONEDDS_VERSION,
        "probe_codegen": probe_codegen,
        "probe_output_dir": os.fspath(resolved_probe_output_dir),
    }
    report.update(
        check_idlc_codegen_toolchain(
            idlc,
            expected_version=cyclonedds_version,
            probe_codegen=probe_codegen,
            probe_idl=probe_idl,
            probe_output_dir=resolved_probe_output_dir,
        )
    )
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
    parser.add_argument(
        "--probe-codegen",
        action="store_true",
        help="write a repo-local C++ idlc probe under build/ and require .hpp/.cpp outputs",
    )
    parser.add_argument(
        "--probe-idl",
        type=Path,
        default=None,
        help=f"IDL file for --probe-codegen; defaults to {DEFAULT_PROBE_IDL} when present",
    )
    parser.add_argument(
        "--probe-output-dir",
        type=Path,
        default=None,
        help=f"repo build/ directory for --probe-codegen; defaults to {DEFAULT_PROBE_OUTPUT_DIR}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    probe_codegen = bool(args.probe_codegen)
    dry_run = bool(args.dry_run or args.check)
    invalid_probe_dry_run = probe_codegen and dry_run
    will_write = probe_codegen and not invalid_probe_dry_run

    report: dict[str, object] = {
        "ok": False,
        "dry_run": dry_run,
        "will_write": will_write,
        "repo_root": os.fspath(REPO_ROOT),
        "toolchain_dir": os.fspath(_resolve_path(args.toolchain_dir)),
        "cyclonedds_version": args.cyclonedds_version,
        "cyclonedds_cxx_version": args.cyclonedds_cxx_version,
        "expected_idlc_version": PINNED_CYCLONEDDS_VERSION,
        "probe_codegen": probe_codegen,
    }
    if args.idlc is not None:
        report["idlc"] = os.fspath(_resolve_path(args.idlc))
    if args.probe_output_dir is not None:
        report["probe_output_dir"] = os.fspath(_resolve_path(args.probe_output_dir))
    else:
        report["probe_output_dir"] = os.fspath(_resolve_path(DEFAULT_PROBE_OUTPUT_DIR))
    if args.probe_idl is not None:
        report["probe_idl"] = os.fspath(_resolve_path(args.probe_idl))

    try:
        if invalid_probe_dry_run:
            raise CodegenToolchainError(
                "--probe-codegen cannot be combined with --check or --dry-run because it writes probe output"
            )
        if not dry_run and not probe_codegen:
            raise CodegenToolchainError(
                "download/build is intentionally not implemented; use --check/--dry-run or --probe-codegen"
            )
        report.update(
            build_plan(
                cyclonedds_version=args.cyclonedds_version,
                cyclonedds_cxx_version=args.cyclonedds_cxx_version,
                toolchain_dir=args.toolchain_dir,
                idlc=args.idlc,
                dry_run=dry_run,
                probe_codegen=probe_codegen,
                probe_idl=args.probe_idl,
                probe_output_dir=args.probe_output_dir,
            )
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except CodegenToolchainError as exc:
        report.update(exc.report)
        report["error"] = str(exc)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
