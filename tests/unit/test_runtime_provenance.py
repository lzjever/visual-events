from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

from tools.runtime_provenance import collect_runtime_provenance, runtime_execution_env


def write_project_contract(repo_root: Path) -> None:
    (repo_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "visual-events-server"',
                'version = "0.1.0"',
                "",
                "[project.scripts]",
                'visual-events-server = "visual_events_server.app:main"',
                'visual-events-cli = "visual_events_cli.main:main"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_digest(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"sha256={encoded}"


def record_line(
    site_packages: Path,
    path: Path,
    *,
    digest: str | None = None,
    size: int | str | None = None,
) -> str:
    rel_path = os.path.relpath(path, site_packages).replace(os.sep, "/")
    digest_value = record_digest(path) if digest is None else digest
    size_value = path.stat().st_size if size is None else size
    return f"{rel_path},{digest_value},{size_value}\n"


def rewrite_runtime_record(
    paths: dict[str, Path],
    *,
    include_server: bool = True,
    include_cli: bool = True,
    server_digest: str | None = None,
    cli_digest: str | None = None,
) -> None:
    site_packages = paths["site_packages"]
    record = paths["record"]
    lines = [
        record_line(site_packages, paths["metadata"]),
        record_line(site_packages, paths["entry_points"]),
    ]
    if include_server:
        lines.append(record_line(site_packages, paths["server_bin"], digest=server_digest))
    if include_cli:
        lines.append(record_line(site_packages, paths["cli_bin"], digest=cli_digest))
    lines.append(f"{os.path.relpath(record, site_packages).replace(os.sep, '/')},,\n")
    record.write_text("".join(lines), encoding="utf-8")


def make_runtime_distribution(
    repo_root: Path,
    *,
    name: str = "visual-events-server",
    version: str = "0.1.0",
) -> dict[str, Path]:
    write_project_contract(repo_root)
    venv = repo_root / "runtime" / "venv"
    bin_dir = venv / "bin"
    server_bin = make_executable(bin_dir / "visual-events-server")
    cli_bin = make_executable(bin_dir / "visual-events-cli")
    site_packages = (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    dist_info = site_packages / "visual_events_server-0.1.0.dist-info"
    metadata = dist_info / "METADATA"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
    entry_points = dist_info / "entry_points.txt"
    entry_points.write_text(
        "\n".join(
            [
                "[console_scripts]",
                "visual-events-server = visual_events_server.app:main",
                "visual-events-cli = visual_events_cli.main:main",
                "",
            ]
        ),
        encoding="utf-8",
    )
    record = dist_info / "RECORD"
    result = {
        "runtime_venv": venv,
        "runtime_bin_dir": bin_dir,
        "site_packages": site_packages,
        "server_bin": server_bin,
        "cli_bin": cli_bin,
        "dist_info": dist_info,
        "metadata": metadata,
        "entry_points": entry_points,
        "record": record,
        "direct_url": dist_info / "direct_url.json",
    }
    rewrite_runtime_record(result)
    return result


def collect(repo_root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    return collect_runtime_provenance(
        repo_root=repo_root,
        server_bin=paths["server_bin"],
        cli_bin=paths["cli_bin"],
        server_config=None,
    )


def test_success_provenance_covers_server_cli_and_distribution(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)

    report = collect(tmp_path, paths)

    assert report["failure_reasons"] == []
    assert report["server_bin_is_runtime_venv"] is True
    assert report["cli_bin_is_runtime_venv"] is True
    assert report["same_runtime_venv"] is True
    assert report["runtime_venv_repo_local"] is True
    assert report["runtime_venv_realpath"] == os.fspath(paths["runtime_venv"])
    assert report["runtime_venv"] == os.fspath(paths["runtime_venv"])
    assert report["runtime_bin_dir"] == os.fspath(paths["runtime_bin_dir"])
    assert report["dist_info_dir"] == os.fspath(paths["dist_info"])
    assert report["wheel_name"] == "visual-events-server"
    assert report["wheel_version"] == "0.1.0"
    assert report["metadata_sha256"] == sha256_file(paths["metadata"])
    assert report["entry_points_sha256"] == sha256_file(paths["entry_points"])
    assert report["record_sha256"] == sha256_file(paths["record"])
    assert report["server_entry_point"] == "visual_events_server.app:main"
    assert report["cli_entry_point"] == "visual_events_cli.main:main"
    assert report["server_script_sha256"] == sha256_file(paths["server_bin"])
    assert report["cli_script_sha256"] == sha256_file(paths["cli_bin"])
    assert report["server_record_sha256"] == record_digest(paths["server_bin"])
    assert report["cli_record_sha256"] == record_digest(paths["cli_bin"])
    assert re.fullmatch(r"[0-9a-f]{64}", report["runtime_hash"])


def test_runtime_execution_env_scrubs_python_import_overrides_and_preserves_home() -> None:
    env = runtime_execution_env(
        {
            "HOME": "/sentinel-home",
            "PATH": "/bin",
            "PYTHONPATH": "/dev/source",
            "PYTHONHOME": "/dev/python",
            "PYTHONUSERBASE": "/dev/userbase",
            "VIRTUAL_ENV": "/dev/venv",
            "VIRTUAL_ENV_PROMPT": "(dev)",
            "__PYVENV_LAUNCHER__": "/dev/python",
        }
    )

    assert env["HOME"] == "/sentinel-home"
    assert env["PATH"] == "/bin"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONSAFEPATH"] == "1"
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT",
        "__PYVENV_LAUNCHER__",
    ):
        assert key not in env


def test_runtime_venv_symlink_to_outside_repo_fails_provenance(tmp_path: Path) -> None:
    write_project_contract(tmp_path)
    external_root = tmp_path.parent / f"{tmp_path.name}-external-runtime"
    external_root.mkdir()
    paths = make_runtime_distribution(external_root)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "venv").symlink_to(paths["runtime_venv"], target_is_directory=True)
    server_bin = runtime_dir / "venv" / "bin" / "visual-events-server"
    cli_bin = runtime_dir / "venv" / "bin" / "visual-events-cli"

    report = collect_runtime_provenance(
        repo_root=tmp_path,
        server_bin=server_bin,
        cli_bin=cli_bin,
        server_config=None,
    )

    assert "runtime_venv_not_repo_local" in report["failure_reasons"]
    assert report["runtime_venv_repo_local"] is False
    assert report["runtime_venv_realpath"] == os.fspath(paths["runtime_venv"])
    assert report["wheel_name"] is None
    assert report["runtime_hash"] is None


def test_missing_cli_executable_fails_provenance(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    paths["cli_bin"].unlink()

    report = collect(tmp_path, paths)

    assert "cli_executable_missing" in report["failure_reasons"]
    assert report["cli_script_sha256"] is None
    assert report["runtime_hash"] is None


def test_entry_points_missing_cli_fails_provenance(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    paths["entry_points"].write_text(
        "[console_scripts]\n"
        "visual-events-server = visual_events_server.app:main\n",
        encoding="utf-8",
    )
    rewrite_runtime_record(paths)

    report = collect(tmp_path, paths)

    assert "runtime_entry_point_mismatch:visual-events-cli" in report["failure_reasons"]
    assert report["cli_entry_point"] is None
    assert report["runtime_hash"] is None


def test_record_missing_cli_script_row_fails_provenance(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    rewrite_runtime_record(paths, include_cli=False)

    report = collect(tmp_path, paths)

    assert "cli_script_record_missing" in report["failure_reasons"]
    assert report["cli_record_path"] is None
    assert report["runtime_hash"] is None


def test_record_digest_mismatch_fails_provenance(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    rewrite_runtime_record(
        paths,
        cli_digest="sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    )

    report = collect(tmp_path, paths)

    assert "cli_script_record_sha256_mismatch" in report["failure_reasons"]
    assert report["runtime_hash"] is None


def test_record_absolute_script_path_fails_provenance(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    record_self_path = os.path.relpath(paths["record"], paths["site_packages"]).replace(
        os.sep, "/"
    )
    paths["record"].write_text(
        "".join(
            [
                record_line(paths["site_packages"], paths["metadata"]),
                record_line(paths["site_packages"], paths["entry_points"]),
                (
                    f"{paths['server_bin']},{record_digest(paths['server_bin'])},"
                    f"{paths['server_bin'].stat().st_size}\n"
                ),
                record_line(paths["site_packages"], paths["cli_bin"]),
                f"{record_self_path},,\n",
            ]
        ),
        encoding="utf-8",
    )

    report = collect(tmp_path, paths)

    assert "server_script_record_path_absolute" in report["failure_reasons"]
    assert report["server_record_path"] == os.fspath(paths["server_bin"])
    assert report["runtime_hash"] is None


def test_record_script_size_mismatch_fails_provenance(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    record_self_path = os.path.relpath(paths["record"], paths["site_packages"]).replace(
        os.sep, "/"
    )
    paths["record"].write_text(
        "".join(
            [
                record_line(paths["site_packages"], paths["metadata"]),
                record_line(paths["site_packages"], paths["entry_points"]),
                record_line(
                    paths["site_packages"],
                    paths["server_bin"],
                    size=paths["server_bin"].stat().st_size + 1,
                ),
                record_line(paths["site_packages"], paths["cli_bin"]),
                f"{record_self_path},,\n",
            ]
        ),
        encoding="utf-8",
    )

    report = collect(tmp_path, paths)

    assert "server_script_record_size_mismatch" in report["failure_reasons"]
    assert report["runtime_hash"] is None


def test_record_duplicate_script_rows_fail_provenance(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    record_self_path = os.path.relpath(paths["record"], paths["site_packages"]).replace(
        os.sep, "/"
    )
    duplicate_cli_row = record_line(paths["site_packages"], paths["cli_bin"])
    paths["record"].write_text(
        "".join(
            [
                record_line(paths["site_packages"], paths["metadata"]),
                record_line(paths["site_packages"], paths["entry_points"]),
                record_line(paths["site_packages"], paths["server_bin"]),
                duplicate_cli_row,
                duplicate_cli_row,
                f"{record_self_path},,\n",
            ]
        ),
        encoding="utf-8",
    )

    report = collect(tmp_path, paths)

    assert "cli_script_record_duplicate" in report["failure_reasons"]
    assert report["runtime_hash"] is None


def test_editable_direct_url_fails_provenance(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    paths["direct_url"].write_text(
        json.dumps({"url": "file:///repo", "dir_info": {"editable": True}}),
        encoding="utf-8",
    )

    report = collect(tmp_path, paths)

    assert "runtime_direct_url_editable" in report["failure_reasons"]
    assert report["direct_url_editable"] is True
    assert report["runtime_hash"] is None


def test_multiple_matching_dist_info_fails_provenance(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    second_dist_info = paths["site_packages"] / "visual_events_server-0.1.1.dist-info"
    second_dist_info.mkdir()
    (second_dist_info / "METADATA").write_text(
        "Name: visual-events-server\nVersion: 0.1.0\n",
        encoding="utf-8",
    )

    report = collect(tmp_path, paths)

    assert "runtime_dist_info_ambiguous" in report["failure_reasons"]
    assert report["runtime_hash"] is None


def test_invalid_metadata_fails_provenance(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    paths["metadata"].write_bytes(b"Name: visual-events-server\nVersion: \xff\n")

    report = collect(tmp_path, paths)

    assert "runtime_metadata_invalid" in report["failure_reasons"]
    assert report["runtime_hash"] is None


def test_runtime_hash_changes_when_cli_script_changes(tmp_path: Path) -> None:
    paths = make_runtime_distribution(tmp_path)
    first_report = collect(tmp_path, paths)
    paths["cli_bin"].write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
    paths["cli_bin"].chmod(paths["cli_bin"].stat().st_mode | stat.S_IXUSR)
    rewrite_runtime_record(paths)

    second_report = collect(tmp_path, paths)

    assert first_report["failure_reasons"] == []
    assert second_report["failure_reasons"] == []
    assert first_report["runtime_hash"] != second_report["runtime_hash"]
