from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping


SERVER_SCRIPT_NAME = "visual-events-server"
CLI_SCRIPT_NAME = "visual-events-cli"
PYTHON_ENV_VARS_TO_DROP = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "VIRTUAL_ENV",
    "VIRTUAL_ENV_PROMPT",
    "__PYVENV_LAUNCHER__",
)


class RuntimeProvenanceError(Exception):
    pass


def runtime_execution_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    for key in PYTHON_ENV_VARS_TO_DROP:
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONSAFEPATH"] = "1"
    return env


def collect_runtime_provenance(
    *,
    repo_root: Path,
    server_bin: Path,
    cli_bin: Path,
    server_config: Path | None,
) -> dict[str, Any]:
    repo_root = _resolve_path(repo_root)
    expected_name, expected_version, expected_scripts = _project_distribution_contract(
        repo_root
    )
    resolved_server_bin = _resolve_path(server_bin)
    resolved_cli_bin = _resolve_path(cli_bin)
    server_shape = _runtime_script_shape(
        repo_root,
        resolved_server_bin,
        expected_script_name=SERVER_SCRIPT_NAME,
    )
    cli_shape = _runtime_script_shape(
        repo_root,
        resolved_cli_bin,
        expected_script_name=CLI_SCRIPT_NAME,
    )
    server_runtime_venv = server_shape["runtime_venv"]
    cli_runtime_venv = cli_shape["runtime_venv"]
    same_runtime_venv = (
        server_runtime_venv is not None
        and cli_runtime_venv is not None
        and server_runtime_venv == cli_runtime_venv
    )
    runtime_venv = (
        server_runtime_venv
        if server_runtime_venv is not None
        else cli_runtime_venv
    )
    if same_runtime_venv:
        runtime_venv = server_runtime_venv
    runtime_bin_dir = runtime_venv / "bin" if runtime_venv is not None else None
    runtime_venv_repo_local = (
        runtime_venv is not None and _is_relative_to(runtime_venv, repo_root)
    )

    failure_reasons: list[str] = []
    if not server_shape["is_runtime_venv"]:
        failure_reasons.append("server_bin_not_runtime_venv")
    if not cli_shape["is_runtime_venv"]:
        failure_reasons.append("cli_bin_not_runtime_venv")
    if (
        server_shape["is_runtime_venv"]
        and cli_shape["is_runtime_venv"]
        and not same_runtime_venv
    ):
        failure_reasons.append("runtime_venv_mismatch")
    failure_reasons.extend(
        _runtime_executable_failure_reasons("server", resolved_server_bin)
    )
    failure_reasons.extend(_runtime_executable_failure_reasons("cli", resolved_cli_bin))
    if runtime_venv is not None and not runtime_venv_repo_local:
        failure_reasons.append("runtime_venv_not_repo_local")

    if runtime_venv is not None and not runtime_venv_repo_local:
        distribution_report = empty_runtime_distribution_report()
        distribution_failures: list[str] = []
    else:
        (
            distribution_report,
            distribution_failures,
        ) = _runtime_distribution_provenance_report(
            runtime_venv=runtime_venv,
            expected_name=expected_name,
            expected_version=expected_version,
            expected_scripts=expected_scripts,
            server_bin=resolved_server_bin,
            cli_bin=resolved_cli_bin,
        )
    failure_reasons.extend(distribution_failures)

    config_path = _resolve_path(server_config) if server_config is not None else None
    config_hash: str | None = None
    if config_path is not None:
        if not config_path.exists():
            failure_reasons.append("server_config_not_found")
        elif not config_path.is_file():
            failure_reasons.append("server_config_not_file")
        else:
            config_hash = _sha256_file(config_path)

    report: dict[str, Any] = {
        "server_bin": os.fspath(resolved_server_bin),
        "cli_bin": os.fspath(resolved_cli_bin),
        "server_bin_is_runtime_venv": bool(server_shape["is_runtime_venv"]),
        "cli_bin_is_runtime_venv": bool(cli_shape["is_runtime_venv"]),
        "server_runtime_venv": _path_or_none(server_runtime_venv),
        "cli_runtime_venv": _path_or_none(cli_runtime_venv),
        "runtime_venv": _path_or_none(runtime_venv) if same_runtime_venv else None,
        "runtime_venv_realpath": _path_or_none(runtime_venv),
        "runtime_venv_repo_local": bool(runtime_venv_repo_local),
        "runtime_bin_dir": _path_or_none(runtime_bin_dir) if same_runtime_venv else None,
        "same_runtime_venv": same_runtime_venv,
        "server_script_sha256": _sha256_file_if_readable(resolved_server_bin),
        "cli_script_sha256": _sha256_file_if_readable(resolved_cli_bin),
        "config_path": _path_or_none(config_path),
        "config_hash": config_hash,
        "runtime_hash": None,
        "failure_reasons": _dedupe(failure_reasons),
        **distribution_report,
    }
    runtime_hash_required_keys = [
        "metadata_sha256",
        "entry_points_sha256",
        "record_sha256",
        "server_script_sha256",
        "cli_script_sha256",
    ]
    if not report["failure_reasons"] and all(
        report.get(key) is not None for key in runtime_hash_required_keys
    ):
        report["runtime_hash"] = _canonical_json_hash(
            {
                "server_bin": report["server_bin"],
                "cli_bin": report["cli_bin"],
                "runtime_venv": report["runtime_venv"],
                "runtime_bin_dir": report["runtime_bin_dir"],
                "wheel_name": report["wheel_name"],
                "wheel_version": report["wheel_version"],
                "dist_info_dir": report["dist_info_dir"],
                "metadata_sha256": report["metadata_sha256"],
                "entry_points_sha256": report["entry_points_sha256"],
                "record_sha256": report["record_sha256"],
                "server_script_sha256": report["server_script_sha256"],
                "cli_script_sha256": report["cli_script_sha256"],
            }
        )
    return report


def runtime_provenance_report_for_failure(
    *,
    repo_root: Path,
    server_bin: Path,
    cli_bin: Path,
    server_config: Path | None,
) -> dict[str, Any]:
    try:
        return collect_runtime_provenance(
            repo_root=repo_root,
            server_bin=server_bin,
            cli_bin=cli_bin,
            server_config=server_config,
        )
    except Exception as exc:
        resolved_server_bin = _resolve_path(server_bin)
        resolved_cli_bin = _resolve_path(cli_bin)
        config_path = _resolve_path(server_config) if server_config else None
        return {
            "server_bin": os.fspath(resolved_server_bin),
            "cli_bin": os.fspath(resolved_cli_bin),
            "server_bin_is_runtime_venv": False,
            "cli_bin_is_runtime_venv": False,
            "server_runtime_venv": None,
            "cli_runtime_venv": None,
            "runtime_venv": None,
            "runtime_venv_realpath": None,
            "runtime_venv_repo_local": False,
            "runtime_bin_dir": None,
            "same_runtime_venv": False,
            **empty_runtime_distribution_report(),
            "server_script_sha256": None,
            "cli_script_sha256": None,
            "runtime_hash": None,
            "config_path": _path_or_none(config_path),
            "config_hash": None,
            "failure_reasons": [f"runtime_provenance_unavailable:{type(exc).__name__}"],
        }


def runtime_provenance_not_run_report(
    *,
    repo_root: Path,
    server_bin: Path,
    cli_bin: Path,
    server_config: Path | None,
    reason: str,
) -> dict[str, Any]:
    del repo_root
    resolved_server_bin = _resolve_path(server_bin)
    resolved_cli_bin = _resolve_path(cli_bin)
    config_path = _resolve_path(server_config) if server_config else None
    return {
        "server_bin": os.fspath(resolved_server_bin),
        "cli_bin": os.fspath(resolved_cli_bin),
        "server_bin_is_runtime_venv": False,
        "cli_bin_is_runtime_venv": False,
        "server_runtime_venv": None,
        "cli_runtime_venv": None,
        "runtime_venv": None,
        "runtime_venv_realpath": None,
        "runtime_venv_repo_local": False,
        "runtime_bin_dir": None,
        "same_runtime_venv": False,
        **empty_runtime_distribution_report(),
        "server_script_sha256": None,
        "cli_script_sha256": None,
        "runtime_hash": None,
        "config_path": _path_or_none(config_path),
        "config_hash": None,
        "failure_reasons": [f"runtime_provenance_not_run:{reason}"],
    }


def runtime_provenance_report_with_exit_codes(
    runtime_report: dict[str, Any],
    *,
    server_exit_code: int | None,
    cli_exit_code: int | None,
) -> dict[str, Any]:
    report = dict(runtime_report)
    report["server_exit_code"] = server_exit_code
    report["cli_exit_code"] = cli_exit_code
    return report


def runtime_provenance_flat_aliases(
    runtime_report: dict[str, Any],
    *,
    server_exit_code: int | None,
    cli_exit_code: int | None,
) -> dict[str, Any]:
    nested_report = runtime_provenance_report_with_exit_codes(
        runtime_report,
        server_exit_code=server_exit_code,
        cli_exit_code=cli_exit_code,
    )
    return {
        "runtime_provenance": nested_report,
        "server_bin": nested_report.get("server_bin"),
        "cli_bin": nested_report.get("cli_bin"),
        "server_bin_is_runtime_venv": nested_report.get("server_bin_is_runtime_venv"),
        "cli_bin_is_runtime_venv": nested_report.get("cli_bin_is_runtime_venv"),
        "wheel_name": nested_report.get("wheel_name"),
        "wheel_version": nested_report.get("wheel_version"),
        "runtime_hash": nested_report.get("runtime_hash"),
        "config_hash": nested_report.get("config_hash"),
        "server_exit_code": server_exit_code,
        "cli_exit_code": cli_exit_code,
    }


def empty_runtime_distribution_report() -> dict[str, Any]:
    return {
        "wheel_name": None,
        "wheel_version": None,
        "dist_info_dir": None,
        "metadata_path": None,
        "metadata_sha256": None,
        "entry_points_path": None,
        "entry_points_sha256": None,
        "record_path": None,
        "record_sha256": None,
        "direct_url_path": None,
        "direct_url_sha256": None,
        "direct_url_editable": False,
        "server_entry_point": None,
        "cli_entry_point": None,
        "server_record_path": None,
        "server_record_sha256": None,
        "server_record_size": None,
        "cli_record_path": None,
        "cli_record_sha256": None,
        "cli_record_size": None,
    }


def _runtime_script_shape(
    repo_root: Path,
    path: Path,
    *,
    expected_script_name: str,
) -> dict[str, Any]:
    expected_path = _resolve_path(
        repo_root / "runtime" / "venv" / "bin" / expected_script_name
    )
    is_runtime_venv = path == expected_path
    return {
        "is_runtime_venv": is_runtime_venv,
        "runtime_venv": expected_path.parent.parent if is_runtime_venv else None,
    }


def _project_distribution_contract(repo_root: Path) -> tuple[str, str, dict[str, str]]:
    try:
        with (repo_root / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        name = project["name"]
        version = project["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeProvenanceError(
            "pyproject project name/version unavailable"
        ) from exc
    if not isinstance(name, str) or not isinstance(version, str):
        raise RuntimeProvenanceError("pyproject project name/version must be strings")
    scripts = project.get("scripts")
    if not isinstance(scripts, dict):
        raise RuntimeProvenanceError("pyproject project scripts unavailable")
    expected_scripts: dict[str, str] = {}
    for script_name in (SERVER_SCRIPT_NAME, CLI_SCRIPT_NAME):
        target = scripts.get(script_name)
        if not isinstance(target, str):
            raise RuntimeProvenanceError(f"pyproject script missing: {script_name}")
        expected_scripts[script_name] = target
    return name, version, expected_scripts


def _runtime_executable_failure_reasons(role: str, path: Path) -> list[str]:
    if not path.exists():
        return [f"{role}_executable_missing"]
    if not path.is_file():
        return [f"{role}_executable_not_file"]
    if not os.access(path, os.X_OK):
        return [f"{role}_executable_not_executable"]
    return []


def _runtime_distribution_provenance_report(
    *,
    runtime_venv: Path | None,
    expected_name: str,
    expected_version: str,
    expected_scripts: dict[str, str],
    server_bin: Path,
    cli_bin: Path,
) -> tuple[dict[str, Any], list[str]]:
    empty_report = empty_runtime_distribution_report()
    if runtime_venv is None:
        return empty_report, ["runtime_dist_info_missing"]

    metadata_paths = _runtime_metadata_paths(runtime_venv)
    if not metadata_paths:
        return empty_report, ["runtime_dist_info_missing"]

    expected_normalized = _normalize_distribution_name(expected_name)
    project_candidates: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        metadata, metadata_failure = _read_distribution_metadata(metadata_path)
        metadata_name = metadata.get("Name")
        metadata_version = metadata.get("Version")
        dir_distribution_name = _dist_info_distribution_name(metadata_path.parent)
        metadata_name_matches = (
            isinstance(metadata_name, str)
            and _normalize_distribution_name(metadata_name) == expected_normalized
        )
        dir_name_matches = (
            _normalize_distribution_name(dir_distribution_name) == expected_normalized
        )
        if not metadata_name_matches and not dir_name_matches:
            continue
        project_candidates.append(
            {
                "wheel_name": metadata_name if isinstance(metadata_name, str) else None,
                "wheel_version": metadata_version
                if isinstance(metadata_version, str)
                else None,
                "dist_info_dir": os.fspath(metadata_path.parent),
                "metadata_path": os.fspath(metadata_path),
                "metadata_sha256": _sha256_file_if_readable(metadata_path),
                "_metadata_name_matches": metadata_name_matches,
                "_metadata_failure": metadata_failure,
            }
        )

    if not project_candidates:
        return empty_report, ["runtime_dist_info_missing"]

    project_candidates.sort(key=lambda candidate: str(candidate["metadata_path"]))
    selected = project_candidates[0]
    reasons: list[str] = []
    if len(project_candidates) > 1:
        reasons.append("runtime_dist_info_ambiguous")
    if selected["_metadata_failure"] is not None:
        reasons.append(str(selected["_metadata_failure"]))
    elif (
        selected["_metadata_name_matches"] is not True
        or selected["wheel_version"] != expected_version
    ):
        reasons.append("runtime_metadata_mismatch")

    report = {
        **empty_report,
        **{
            key: value
            for key, value in selected.items()
            if not key.startswith("_")
        },
    }
    dist_info_dir = Path(str(selected["dist_info_dir"]))
    site_packages = dist_info_dir.parent

    entry_points_path = dist_info_dir / "entry_points.txt"
    report["entry_points_path"] = (
        os.fspath(entry_points_path) if entry_points_path.exists() else None
    )
    report["entry_points_sha256"] = _sha256_file_if_readable(entry_points_path)
    entry_point_report, entry_point_reasons = _runtime_entry_points_report(
        entry_points_path=entry_points_path,
        expected_scripts=expected_scripts,
    )
    report.update(entry_point_report)
    reasons.extend(entry_point_reasons)

    record_path = dist_info_dir / "RECORD"
    report["record_path"] = os.fspath(record_path) if record_path.exists() else None
    report["record_sha256"] = _sha256_file_if_readable(record_path)
    record_report, record_reasons = _runtime_record_report(
        record_path=record_path,
        site_packages=site_packages,
        script_paths={
            "server": server_bin,
            "cli": cli_bin,
        },
    )
    report.update(record_report)
    reasons.extend(record_reasons)

    direct_url_path = dist_info_dir / "direct_url.json"
    if direct_url_path.exists():
        report["direct_url_path"] = os.fspath(direct_url_path)
        report["direct_url_sha256"] = _sha256_file_if_readable(direct_url_path)
        direct_url_editable, direct_url_failure = _runtime_direct_url_editable(
            direct_url_path
        )
        report["direct_url_editable"] = direct_url_editable
        if direct_url_editable:
            reasons.append("runtime_direct_url_editable")
        if direct_url_failure is not None:
            reasons.append(direct_url_failure)

    return report, _dedupe(reasons)


def _runtime_entry_points_report(
    *,
    entry_points_path: Path,
    expected_scripts: dict[str, str],
) -> tuple[dict[str, str | None], list[str]]:
    report: dict[str, str | None] = {
        "server_entry_point": None,
        "cli_entry_point": None,
    }
    if not entry_points_path.exists():
        return report, ["runtime_entry_points_missing"]
    if not entry_points_path.is_file():
        return report, ["runtime_entry_points_missing"]

    parser = configparser.ConfigParser()
    parser.optionxform = str
    try:
        parser.read_string(entry_points_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, configparser.Error):
        return report, ["runtime_entry_points_invalid"]
    if not parser.has_section("console_scripts"):
        return report, ["runtime_entry_points_missing"]

    section = parser["console_scripts"]
    reasons: list[str] = []
    for script_name, expected_target in expected_scripts.items():
        actual_target = section.get(script_name)
        if script_name == SERVER_SCRIPT_NAME:
            report["server_entry_point"] = actual_target
        elif script_name == CLI_SCRIPT_NAME:
            report["cli_entry_point"] = actual_target
        if actual_target != expected_target:
            reasons.append(f"runtime_entry_point_mismatch:{script_name}")
    return report, reasons


def _runtime_record_report(
    *,
    record_path: Path,
    site_packages: Path,
    script_paths: dict[str, Path],
) -> tuple[dict[str, str | None], list[str]]:
    report: dict[str, str | None] = {
        "server_record_path": None,
        "server_record_sha256": None,
        "server_record_size": None,
        "cli_record_path": None,
        "cli_record_sha256": None,
        "cli_record_size": None,
    }
    if not record_path.exists():
        return report, ["runtime_record_missing"]
    if not record_path.is_file():
        return report, ["runtime_record_missing"]

    try:
        rows = list(csv.reader(record_path.read_text(encoding="utf-8").splitlines()))
    except (OSError, UnicodeDecodeError, csv.Error):
        return report, ["runtime_record_invalid"]

    reasons: list[str] = []
    for role, script_path in script_paths.items():
        matching_rows = _runtime_record_rows_for_path(
            rows,
            expected_path=script_path,
            site_packages=site_packages,
        )
        if not matching_rows:
            reasons.append(f"{role}_script_record_missing")
            continue
        if len(matching_rows) > 1:
            reasons.append(f"{role}_script_record_duplicate")

        row = matching_rows[0]
        record_relative_path = row[0]
        hash_value = row[1] if len(row) > 1 else ""
        size_value = row[2] if len(row) > 2 else ""
        report[f"{role}_record_path"] = record_relative_path
        report[f"{role}_record_sha256"] = hash_value or None
        report[f"{role}_record_size"] = size_value or None
        if Path(record_relative_path).is_absolute():
            reasons.append(f"{role}_script_record_path_absolute")
        if not hash_value:
            reasons.append(f"{role}_script_record_sha256_missing")
        elif not hash_value.startswith("sha256="):
            reasons.append(f"{role}_script_record_sha256_unsupported")
        else:
            expected_digest = _record_sha256_digest(script_path)
            if expected_digest is not None and hash_value != expected_digest:
                reasons.append(f"{role}_script_record_sha256_mismatch")
        reasons.extend(
            _runtime_record_size_failure_reasons(
                role=role,
                script_path=script_path,
                size_value=size_value,
            )
        )
    return report, reasons


def _runtime_record_rows_for_path(
    rows: list[list[str]],
    *,
    expected_path: Path,
    site_packages: Path,
) -> list[list[str]]:
    matches: list[list[str]] = []
    for row in rows:
        if not row or not row[0]:
            continue
        candidates = _runtime_record_path_candidates(
            row[0],
            site_packages=site_packages,
        )
        if expected_path in candidates:
            matches.append(row)
    return matches


def _runtime_record_path_candidates(
    record_path: str,
    *,
    site_packages: Path,
) -> list[Path]:
    raw_path = Path(record_path)
    if raw_path.is_absolute():
        return [_resolve_path(raw_path)]

    return [_resolve_path(site_packages / raw_path)]


def _runtime_record_size_failure_reasons(
    *,
    role: str,
    script_path: Path,
    size_value: str,
) -> list[str]:
    if size_value == "":
        return [f"{role}_script_record_size_missing"]
    try:
        parsed_size = int(size_value)
    except ValueError:
        return [f"{role}_script_record_size_invalid"]
    if parsed_size < 0:
        return [f"{role}_script_record_size_invalid"]
    try:
        actual_size = script_path.stat().st_size
    except OSError:
        return []
    if parsed_size != actual_size:
        return [f"{role}_script_record_size_mismatch"]
    return []


def _record_sha256_digest(path: Path) -> str | None:
    try:
        digest = hashlib.sha256(path.read_bytes()).digest()
    except OSError:
        return None
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"sha256={encoded}"


def _runtime_direct_url_editable(path: Path) -> tuple[bool, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, "runtime_direct_url_invalid"
    if not isinstance(payload, dict):
        return False, "runtime_direct_url_invalid"
    dir_info = payload.get("dir_info")
    editable = isinstance(dir_info, dict) and dir_info.get("editable") is True
    return editable, None


def _runtime_metadata_paths(runtime_venv: Path) -> list[Path]:
    paths = [
        *runtime_venv.glob("lib/python*/site-packages/*.dist-info/METADATA"),
        *runtime_venv.glob("Lib/site-packages/*.dist-info/METADATA"),
    ]
    return sorted(dict.fromkeys(paths))


def _read_distribution_metadata(metadata_path: Path) -> tuple[dict[str, str], str | None]:
    fields: dict[str, str] = {}
    try:
        text = metadata_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return fields, "runtime_metadata_invalid"
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"Name", "Version"}:
            fields[key] = value.strip()
    return fields, None


def _dist_info_distribution_name(dist_info_dir: Path) -> str:
    name = dist_info_dir.name
    if name.endswith(".dist-info"):
        name = name[: -len(".dist-info")]
    if "-" in name:
        return name.rsplit("-", 1)[0]
    return name


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _canonical_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_file_if_readable(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return _sha256_file(path)
    except OSError:
        return None


def _path_or_none(path: Path | None) -> str | None:
    return None if path is None else os.fspath(path)


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
