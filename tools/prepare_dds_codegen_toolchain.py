from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_CYCLONEDDS_VERSION = "0.10.2"
PINNED_CYCLONEDDS_CXX_VERSION = "0.10.2"
PINNED_CYCLONEDDS_COMMIT = "9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd"
PINNED_CYCLONEDDS_CXX_COMMIT = "2a372d2c4597faea54543b925755fa2d7cdd4232"
CYCLONEDDS_REPO = "https://github.com/eclipse-cyclonedds/cyclonedds.git"
CYCLONEDDS_CXX_REPO = "https://github.com/eclipse-cyclonedds/cyclonedds-cxx.git"
DEFAULT_TOOLCHAIN_DIR = (
    REPO_ROOT / "build" / "tools" / f"cyclonedds-cxx-idlc-{PINNED_CYCLONEDDS_VERSION}"
)
DEFAULT_PROBE_IDLS = (
    REPO_ROOT / "common" / "schema" / "dds" / "head_state_v1.idl",
    REPO_ROOT / "common" / "schema" / "dds" / "gaze_target_v1.idl",
)
DEFAULT_PROBE_OUTPUT_DIR = DEFAULT_TOOLCHAIN_DIR / "codegen_probe"
REQUIRED_PREPARE_TOOLS = ("git", "cmake", "make", "gcc", "g++")
OPTIONAL_PREPARE_TOOLS = ("ninja", "bison", "flex")
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


def _shell_single_quote(value: Path | str) -> str:
    text = os.fspath(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


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


def _resolve_probe_idls(
    probe_idls: list[Path] | None,
    *,
    require_exists: bool,
) -> list[Path]:
    selected = probe_idls or list(DEFAULT_PROBE_IDLS)
    resolved = [_resolve_path(path) for path in selected]
    if require_exists:
        missing = [path for path in resolved if not path.is_file()]
        if missing:
            formatted = ", ".join(os.fspath(path) for path in missing)
            raise CodegenToolchainError(f"probe IDL does not exist or is not a file: {formatted}")
    return resolved


def _probe_output_dir(probe_output_dir: Path | None) -> Path:
    return _require_repo_build_path(
        probe_output_dir or DEFAULT_PROBE_OUTPUT_DIR,
        "probe output dir",
    )


def _expected_probe_files(probe_idl: Path) -> list[str]:
    return [f"{probe_idl.stem}.hpp", f"{probe_idl.stem}.cpp"]


def _expected_probe_files_for_idls(probe_idls: list[Path]) -> list[str]:
    return [
        filename
        for probe_idl in probe_idls
        for filename in _expected_probe_files(probe_idl)
    ]


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
    probe_idls: list[Path],
    output_dir: Path,
) -> dict[str, object]:
    expected_files = _expected_probe_files_for_idls(probe_idls)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_expected_probe_outputs(output_dir, expected_files)

    codegen_probes: list[dict[str, object]] = []
    errors: list[str] = []

    for probe_idl in probe_idls:
        idl_expected_files = _expected_probe_files(probe_idl)
        try:
            result = _run_idlc_args(
                idlc,
                ["-l", "cxx", "-o", os.fspath(output_dir), os.fspath(probe_idl)],
                timeout=30,
            )
        except OSError as exc:
            raise CodegenToolchainError(
                f"failed to execute idlc codegen probe for {probe_idl}: {idlc}: {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CodegenToolchainError(f"idlc codegen probe timed out for {probe_idl}: {idlc}") from exc

        combined_output = result.stdout + result.stderr
        generated_files = _list_generated_files(output_dir)
        presence = {
            filename: (output_dir / filename).is_file()
            for filename in idl_expected_files
        }
        expected_present = all(presence.values())
        cannot_load_generator = CANNOT_LOAD_GENERATOR_RE.search(combined_output) is not None
        idl_oracle_ok = result.returncode == 0 and not cannot_load_generator and expected_present
        generated_for_idl = [
            filename
            for filename in generated_files
            if Path(filename).name.startswith(f"{probe_idl.stem}.")
        ]
        codegen_probes.append(
            {
                "idl": os.fspath(probe_idl),
                "expected_generated_files": idl_expected_files,
                "generated_files": generated_for_idl,
                "expected_generated_file_presence": presence,
                "expected_generated_files_present": expected_present,
                "idlc_codegen_returncode": result.returncode,
                "idlc_codegen_stdout": result.stdout,
                "idlc_codegen_stderr": result.stderr,
                "cannot_load_generator": cannot_load_generator,
                "oracle_ok": idl_oracle_ok,
            }
        )
        if cannot_load_generator:
            errors.append(f"{probe_idl}: idlc cannot load generator cxx: {_compact_output(combined_output)}")
        elif result.returncode != 0:
            errors.append(
                f"{probe_idl}: idlc cxx codegen probe failed with rc={result.returncode}: "
                f"{_compact_output(combined_output)}"
            )
        elif not expected_present:
            missing = [filename for filename, present in presence.items() if not present]
            errors.append(
                f"{probe_idl}: missing expected generated files: " + ", ".join(missing)
            )

    generated_files = _list_generated_files(output_dir)
    presence = {
        filename: (output_dir / filename).is_file()
        for filename in expected_files
    }
    expected_present = bool(expected_files) and all(presence.values())
    oracle_ok = not errors and expected_present
    first_probe_idl = probe_idls[0] if len(probe_idls) == 1 else None
    returncodes = [
        probe["idlc_codegen_returncode"]
        for probe in codegen_probes
        if isinstance(probe["idlc_codegen_returncode"], int)
    ]
    aggregate_returncode = next((returncode for returncode in returncodes if returncode != 0), 0)
    report: dict[str, object] = {
        "probe_codegen": True,
        "probe_idl": os.fspath(first_probe_idl) if first_probe_idl is not None else None,
        "probe_idls": [os.fspath(path) for path in probe_idls],
        "probe_output_dir": os.fspath(output_dir),
        "generated_files": generated_files,
        "expected_generated_files": expected_files,
        "expected_generated_file_presence": presence,
        "expected_generated_files_present": expected_present,
        "codegen_probes": codegen_probes,
        "idlc_codegen_returncode": aggregate_returncode,
        "idlc_codegen_stdout": "\n".join(
            str(probe["idlc_codegen_stdout"]) for probe in codegen_probes
        ),
        "idlc_codegen_stderr": "\n".join(
            str(probe["idlc_codegen_stderr"]) for probe in codegen_probes
        ),
        "cxx_backend_available": oracle_ok,
        "oracle_ok": oracle_ok,
    }

    if errors:
        raise CodegenToolchainError("; ".join(errors), report=report)

    return report


def check_idlc_codegen_toolchain(
    idlc: Path | None,
    *,
    expected_version: str = PINNED_CYCLONEDDS_VERSION,
    probe_codegen: bool = False,
    probe_idl: Path | None = None,
    probe_idls: list[Path] | None = None,
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
    selected_probe_idls = probe_idls if probe_idls is not None else ([probe_idl] if probe_idl is not None else None)
    resolved_probe_idls = _resolve_probe_idls(selected_probe_idls, require_exists=probe_codegen)
    expected_files = _expected_probe_files_for_idls(resolved_probe_idls)
    first_probe_idl = resolved_probe_idls[0] if len(resolved_probe_idls) == 1 else None

    report: dict[str, object] = {
        "idlc": os.fspath(resolved),
        "probe_codegen": probe_codegen,
        "probe_idl": os.fspath(first_probe_idl) if first_probe_idl is not None else None,
        "probe_idls": [os.fspath(path) for path in resolved_probe_idls],
        "probe_output_dir": os.fspath(resolved_probe_output_dir),
        "generated_files": [],
        "expected_generated_files": expected_files,
        "expected_generated_file_presence": {filename: False for filename in expected_files},
        "expected_generated_files_present": False,
        "codegen_probes": [],
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
                    probe_idls=resolved_probe_idls,
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


def _prepare_tool_layout(toolchain_dir: Path) -> dict[str, Path]:
    resolved_toolchain_dir = _require_repo_build_path(toolchain_dir, "toolchain dir")
    source_dir = resolved_toolchain_dir / "src"
    build_dir = resolved_toolchain_dir / "build"
    install_dir = resolved_toolchain_dir / "install"
    return {
        "toolchain_dir": resolved_toolchain_dir,
        "source_dir": source_dir,
        "cyclonedds_source_dir": source_dir / "cyclonedds",
        "cyclonedds_cxx_source_dir": source_dir / "cyclonedds-cxx",
        "build_dir": build_dir,
        "cyclonedds_build_dir": build_dir / "cyclonedds",
        "cyclonedds_cxx_build_dir": build_dir / "cyclonedds-cxx",
        "install_dir": install_dir,
        "wrapper_idlc": resolved_toolchain_dir / "bin" / "idlc-cxx",
        "probe_output_dir": resolved_toolchain_dir / "codegen_probe",
    }


def _path_report(path: Path | None) -> dict[str, object]:
    return {
        "found": path is not None,
        "path": os.fspath(path) if path is not None else None,
    }


def _inspect_prepare_tools() -> tuple[dict[str, object], dict[str, object]]:
    required = {
        name: {**_path_report(Path(found) if found else None), "required": True}
        for name in REQUIRED_PREPARE_TOOLS
        for found in [shutil.which(name)]
    }
    optional = {
        name: {**_path_report(Path(found) if found else None), "required": False}
        for name in OPTIONAL_PREPARE_TOOLS
        for found in [shutil.which(name)]
    }
    return required, optional


def _require_prepare_tools() -> tuple[dict[str, object], dict[str, object]]:
    required, optional = _inspect_prepare_tools()
    missing = [name for name, details in required.items() if not details["found"]]
    optional_warnings = [
        f"optional tool not found: {name}"
        for name, details in optional.items()
        if not details["found"]
    ]
    if missing:
        raise CodegenToolchainError(
            "missing required tools: " + ", ".join(missing),
            report={
                "failed_step": "preflight_required_tools",
                "required_tools": required,
                "optional_tools": optional,
                "optional_tool_warnings": optional_warnings,
            },
        )
    return required, optional


def _command_record(
    *,
    step: str,
    argv: list[str],
    cwd: Path,
    returncode: int,
    stdout: str,
    stderr: str,
) -> dict[str, object]:
    return {
        "step": step,
        "argv": argv,
        "cwd": os.fspath(cwd),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _run_prepare_command(
    commands: list[dict[str, object]],
    *,
    step: str,
    argv: list[str],
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        commands.append(
            _command_record(
                step=step,
                argv=argv,
                cwd=cwd,
                returncode=-1,
                stdout="",
                stderr=str(exc),
            )
        )
        raise CodegenToolchainError(
            f"{step} failed to execute: {exc}",
            report={"failed_step": step, "commands": commands},
        ) from exc

    commands.append(
        _command_record(
            step=step,
            argv=argv,
            cwd=cwd,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    )
    if result.returncode != 0:
        raise CodegenToolchainError(
            f"{step} failed with rc={result.returncode}: "
            f"{_compact_output(result.stdout + result.stderr)}",
            report={"failed_step": step, "commands": commands},
        )
    return result


def _verify_ls_remote_tag(
    commands: list[dict[str, object]],
    *,
    step: str,
    repo: str,
    expected_commit: str,
) -> None:
    result = _run_prepare_command(
        commands,
        step=step,
        argv=[
            "git",
            "ls-remote",
            "--exit-code",
            "--tags",
            repo,
            f"refs/tags/{PINNED_CYCLONEDDS_VERSION}",
        ],
    )
    actual_commit = result.stdout.strip().split()[0] if result.stdout.strip() else ""
    if actual_commit != expected_commit:
        raise CodegenToolchainError(
            f"{step} expected tag {PINNED_CYCLONEDDS_VERSION} commit "
            f"{expected_commit}, got {actual_commit or '<empty>'}",
            report={"failed_step": step, "commands": commands},
        )


def _verify_existing_source_head(
    commands: list[dict[str, object]],
    *,
    name: str,
    source_dir: Path,
    expected_commit: str,
) -> str:
    step = f"{name}_source_head"
    result = _run_prepare_command(
        commands,
        step=step,
        argv=["git", "-C", os.fspath(source_dir), "rev-parse", "HEAD"],
    )
    actual_commit = result.stdout.strip()
    if actual_commit != expected_commit:
        raise CodegenToolchainError(
            f"{step} expected source HEAD {expected_commit}, got "
            f"{actual_commit or '<empty>'}",
            report={
                "failed_step": step,
                "commands": commands,
                f"{name}_expected_commit": expected_commit,
                f"{name}_commit": actual_commit,
            },
        )
    return actual_commit


def _clone_or_verify_source(
    commands: list[dict[str, object]],
    *,
    name: str,
    repo: str,
    source_dir: Path,
    expected_commit: str,
) -> str:
    if source_dir.exists():
        return _verify_existing_source_head(
            commands,
            name=name,
            source_dir=source_dir,
            expected_commit=expected_commit,
        )

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_prepare_command(
        commands,
        step=f"{name}_clone",
        argv=[
            "git",
            "clone",
            "--branch",
            PINNED_CYCLONEDDS_VERSION,
            "--depth",
            "1",
            repo,
            os.fspath(source_dir),
        ],
    )
    return _verify_existing_source_head(
        commands,
        name=name,
        source_dir=source_dir,
        expected_commit=expected_commit,
    )


def _cmake_build_args(build_dir: Path) -> list[str]:
    jobs = str(os.cpu_count() or 2)
    return [
        "cmake",
        "--build",
        os.fspath(build_dir),
        "--target",
        "install",
        "--",
        "-j",
        jobs,
    ]


def _configure_and_build_cyclonedds(
    commands: list[dict[str, object]],
    *,
    source_dir: Path,
    build_dir: Path,
    install_dir: Path,
) -> None:
    _run_prepare_command(
        commands,
        step="cyclonedds_configure",
        argv=[
            "cmake",
            "-S",
            os.fspath(source_dir),
            "-B",
            os.fspath(build_dir),
            "-G",
            "Unix Makefiles",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_IDLC=ON",
            "-DBUILD_DDSPERF=OFF",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_IDLC_TESTING=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_DOCS=OFF",
            "-DENABLE_SECURITY=OFF",
            "-DENABLE_SSL=OFF",
            "-DENABLE_SHM=OFF",
            "-DENABLE_TYPE_DISCOVERY=ON",
            "-DENABLE_TOPIC_DISCOVERY=ON",
            "-DENABLE_LTO=OFF",
            "-DAPPEND_PROJECT_NAME_TO_INCLUDEDIR=OFF",
            "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE",
            "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE",
        ],
    )
    _run_prepare_command(
        commands,
        step="cyclonedds_build_install",
        argv=_cmake_build_args(build_dir),
    )


def _configure_and_build_cyclonedds_cxx(
    commands: list[dict[str, object]],
    *,
    source_dir: Path,
    build_dir: Path,
    install_dir: Path,
) -> None:
    _run_prepare_command(
        commands,
        step="cyclonedds_cxx_configure",
        argv=[
            "cmake",
            "-S",
            os.fspath(source_dir),
            "-B",
            os.fspath(build_dir),
            "-G",
            "Unix Makefiles",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={install_dir}",
            f"-DCycloneDDS_DIR={install_dir / 'lib' / 'cmake' / 'CycloneDDS'}",
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_IDLLIB=ON",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_DOCS=OFF",
            "-DENABLE_SHM=OFF",
            "-DENABLE_LEGACY=OFF",
            "-DENABLE_TYPE_DISCOVERY=ON",
            "-DENABLE_TOPIC_DISCOVERY=ON",
            "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE",
            "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE",
        ],
    )
    _run_prepare_command(
        commands,
        step="cyclonedds_cxx_build_install",
        argv=_cmake_build_args(build_dir),
    )


def _write_idlc_wrapper(*, wrapper: Path, install_dir: Path) -> None:
    install_lib = install_dir / "lib"
    install_idlc = install_dir / "bin" / "idlc"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#!/bin/sh\n"
        f"install_lib={_shell_single_quote(install_lib)}\n"
        "if [ -n \"${LD_LIBRARY_PATH:-}\" ]; then\n"
        "  export LD_LIBRARY_PATH=\"$install_lib:$LD_LIBRARY_PATH\"\n"
        "else\n"
        "  export LD_LIBRARY_PATH=\"$install_lib\"\n"
        "fi\n"
        f"exec {_shell_single_quote(install_idlc)} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)


def _require_prepare_artifacts(*, install_dir: Path, wrapper: Path) -> None:
    required_files = [
        install_dir / "bin" / "idlc",
        install_dir / "lib" / "libcycloneddsidlcxx.so",
        install_dir / "lib" / "libcycloneddsidl.so",
        install_dir / "lib" / "libddsc.so",
    ]
    missing = [os.fspath(path) for path in required_files if not path.is_file()]
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        missing.append(os.fspath(wrapper))
    if missing:
        raise CodegenToolchainError(
            "missing required prepare artifacts: " + ", ".join(missing),
            report={"failed_step": "require_artifacts"},
        )


def prepare_codegen_toolchain(
    *,
    cyclonedds_version: str,
    cyclonedds_cxx_version: str,
    toolchain_dir: Path,
    probe_idls: list[Path] | None,
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
    layout = _prepare_tool_layout(toolchain_dir)
    commands: list[dict[str, object]] = []
    report: dict[str, object] = {
        "ok": True,
        "mode": "prepare",
        "prepare_toolchain": True,
        "toolchain_ready": False,
        "dry_run": False,
        "will_write": True,
        "repo_root": os.fspath(REPO_ROOT),
        "toolchain_dir": os.fspath(layout["toolchain_dir"]),
        "source_dir": os.fspath(layout["source_dir"]),
        "cyclonedds_source_dir": os.fspath(layout["cyclonedds_source_dir"]),
        "cyclonedds_cxx_source_dir": os.fspath(layout["cyclonedds_cxx_source_dir"]),
        "build_dir": os.fspath(layout["build_dir"]),
        "cyclonedds_build_dir": os.fspath(layout["cyclonedds_build_dir"]),
        "cyclonedds_cxx_build_dir": os.fspath(layout["cyclonedds_cxx_build_dir"]),
        "install_dir": os.fspath(layout["install_dir"]),
        "wrapper_idlc": os.fspath(layout["wrapper_idlc"]),
        "ld_library_path_prepend": os.fspath(layout["install_dir"] / "lib"),
        "cyclonedds_version": cyclonedds_version,
        "cyclonedds_cxx_version": cyclonedds_cxx_version,
        "cyclonedds_expected_commit": PINNED_CYCLONEDDS_COMMIT,
        "cyclonedds_cxx_expected_commit": PINNED_CYCLONEDDS_CXX_COMMIT,
        "expected_idlc_version": PINNED_CYCLONEDDS_VERSION,
        "probe_codegen": True,
        "probe_output_dir": os.fspath(layout["probe_output_dir"]),
        "commands": commands,
    }

    try:
        required_tools, optional_tools = _require_prepare_tools()
        report["required_tools"] = required_tools
        report["optional_tools"] = optional_tools
        report["optional_tool_warnings"] = [
            f"optional tool not found: {name}"
            for name, details in optional_tools.items()
            if not details["found"]
        ]

        _verify_ls_remote_tag(
            commands,
            step="cyclonedds_ls_remote",
            repo=CYCLONEDDS_REPO,
            expected_commit=PINNED_CYCLONEDDS_COMMIT,
        )
        _verify_ls_remote_tag(
            commands,
            step="cyclonedds_cxx_ls_remote",
            repo=CYCLONEDDS_CXX_REPO,
            expected_commit=PINNED_CYCLONEDDS_CXX_COMMIT,
        )
        report["cyclonedds_commit"] = _clone_or_verify_source(
            commands,
            name="cyclonedds",
            repo=CYCLONEDDS_REPO,
            source_dir=layout["cyclonedds_source_dir"],
            expected_commit=PINNED_CYCLONEDDS_COMMIT,
        )
        report["cyclonedds_cxx_commit"] = _clone_or_verify_source(
            commands,
            name="cyclonedds_cxx",
            repo=CYCLONEDDS_CXX_REPO,
            source_dir=layout["cyclonedds_cxx_source_dir"],
            expected_commit=PINNED_CYCLONEDDS_CXX_COMMIT,
        )
        _configure_and_build_cyclonedds(
            commands,
            source_dir=layout["cyclonedds_source_dir"],
            build_dir=layout["cyclonedds_build_dir"],
            install_dir=layout["install_dir"],
        )
        _configure_and_build_cyclonedds_cxx(
            commands,
            source_dir=layout["cyclonedds_cxx_source_dir"],
            build_dir=layout["cyclonedds_cxx_build_dir"],
            install_dir=layout["install_dir"],
        )
        _write_idlc_wrapper(
            wrapper=layout["wrapper_idlc"],
            install_dir=layout["install_dir"],
        )
        _require_prepare_artifacts(
            install_dir=layout["install_dir"],
            wrapper=layout["wrapper_idlc"],
        )
    except CodegenToolchainError as exc:
        raise CodegenToolchainError(
            str(exc),
            report={**report, **exc.report, "commands": commands},
        ) from exc

    try:
        report.update(
            check_idlc_codegen_toolchain(
                layout["wrapper_idlc"],
                expected_version=cyclonedds_version,
                probe_codegen=True,
                probe_idls=probe_idls,
                probe_output_dir=layout["probe_output_dir"],
            )
        )
    except CodegenToolchainError as exc:
        raise CodegenToolchainError(
            str(exc),
            report={**report, **exc.report, "failed_step": "codegen_oracle"},
        ) from exc

    report["toolchain_ready"] = True
    return report


def build_plan(
    *,
    cyclonedds_version: str,
    cyclonedds_cxx_version: str,
    toolchain_dir: Path,
    idlc: Path | None,
    dry_run: bool,
    probe_codegen: bool,
    probe_idls: list[Path] | None,
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
            probe_idls=probe_idls,
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
    parser.add_argument("--idlc", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="validate the local idlc/toolchain plan")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="prepare the repo-local CycloneDDS/CycloneDDS-CXX 0.10.2 C++ idlc toolchain",
    )
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
        action="append",
        default=None,
        help="IDL file for --probe-codegen; repeatable; defaults to the repo Head/Gaze IDLs",
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
    prepare = bool(args.prepare)
    probe_codegen = bool(args.probe_codegen)
    dry_run = False if prepare else bool(args.dry_run or args.check)
    invalid_probe_dry_run = probe_codegen and dry_run
    will_write = prepare or (probe_codegen and not invalid_probe_dry_run)
    selected_idlc = None if prepare else (args.idlc or _env_path("VISUAL_EVENTS_IDLC"))

    report: dict[str, object] = {
        "ok": False,
        "mode": "prepare" if prepare else ("probe-codegen" if probe_codegen else "check"),
        "prepare_toolchain": prepare,
        "dry_run": dry_run,
        "will_write": will_write,
        "repo_root": os.fspath(REPO_ROOT),
        "toolchain_dir": os.fspath(_resolve_path(args.toolchain_dir)),
        "cyclonedds_version": args.cyclonedds_version,
        "cyclonedds_cxx_version": args.cyclonedds_cxx_version,
        "expected_idlc_version": PINNED_CYCLONEDDS_VERSION,
        "probe_codegen": probe_codegen,
    }
    if selected_idlc is not None:
        report["idlc"] = os.fspath(_resolve_path(selected_idlc))
    if args.probe_output_dir is not None:
        report["probe_output_dir"] = os.fspath(_resolve_path(args.probe_output_dir))
    else:
        report["probe_output_dir"] = os.fspath(_resolve_path(DEFAULT_PROBE_OUTPUT_DIR))
    if args.probe_idl is not None:
        resolved_arg_probe_idls = [_resolve_path(path) for path in args.probe_idl]
        report["probe_idl"] = (
            os.fspath(resolved_arg_probe_idls[0])
            if len(resolved_arg_probe_idls) == 1
            else None
        )
        report["probe_idls"] = [os.fspath(path) for path in resolved_arg_probe_idls]

    try:
        if prepare:
            if args.check or args.dry_run or args.probe_codegen:
                raise CodegenToolchainError(
                    "--prepare cannot be combined with --check, --dry-run, or --probe-codegen"
                )
            if args.idlc is not None:
                raise CodegenToolchainError("--prepare does not accept explicit --idlc")
            report.update(
                prepare_codegen_toolchain(
                    cyclonedds_version=args.cyclonedds_version,
                    cyclonedds_cxx_version=args.cyclonedds_cxx_version,
                    toolchain_dir=args.toolchain_dir,
                    probe_idls=args.probe_idl,
                )
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
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
                idlc=selected_idlc,
                dry_run=dry_run,
                probe_codegen=probe_codegen,
                probe_idls=args.probe_idl,
                probe_output_dir=args.probe_output_dir,
            )
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except CodegenToolchainError as exc:
        report.update(exc.report)
        report["ok"] = False
        report["error"] = str(exc)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
