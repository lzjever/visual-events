from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest


def import_manifest_module():
    try:
        return importlib.import_module("tools.cli_local_e2e_manifest")
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected tools.cli_local_e2e_manifest module: {exc}")


def write_frame(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rich_manifest_for_data(module: Any, data_dir: Path) -> dict[str, Any]:
    inventory = module.generate_effective_manifest(data_dir)
    return {
        "schema_version": 1,
        "fps": 10.0,
        "scene_count": inventory["scene_count"],
        "frame_count": inventory["frame_count"],
        "scenes": [
            {
                "scene_name": scene["scene_name"],
                "frame_count": scene["frame_count"],
                "scene_sha256": scene["scene_sha256"],
            }
            for scene in inventory["scenes"]
        ],
        "oracle": {
            "expected_event_timeline": {
                "source": "oracle/events.json",
                "version": "events-v1",
            },
            "expected_attention_target_timeline": {
                "source": "oracle/attention.json",
                "rule": "largest_stable_person_v1",
            },
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run_tool(module: Any, argv: list[str], capsys: pytest.CaptureFixture[str]) -> int:
    result = module.main(argv)
    assert isinstance(result, int)
    return result


def test_module_is_importable():
    module = import_manifest_module()

    assert callable(module.main)


def test_missing_manifest_generates_deterministic_effective_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    write_frame(data_dir / "z_scene" / "010.jpg", b"z-ten")
    write_frame(data_dir / "z_scene" / "002.jpeg", b"z-two")
    write_frame(data_dir / "a_scene" / "b.jpg", b"a-b")
    write_frame(data_dir / "a_scene" / "a.jpeg", b"a-a")
    write_frame(data_dir / "a_scene" / "ignored.png", b"not-a-frame")

    out_one = tmp_path / "artifacts" / "manifest-one.json"
    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out_one)],
        capsys,
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    assert str(out_one) in captured.out
    first_report = load_report(out_one)
    first_summary = first_report["effective_manifest"]

    assert first_report["manifest_source"] == "generated"
    assert first_report["manifest_path"] is None
    assert first_report["manifest_sha256"]
    assert first_report["manifest_schema_version"] is None
    assert first_report["manifest_authoritative"] is False
    assert first_report["manifest_validation_errors"] == []
    assert first_report["oracle_schema_present"] is False
    assert first_report["oracle_schema_valid"] is False
    assert first_report["oracle_summary"] is None
    assert first_report["scene_count"] == 2
    assert first_report["frame_count"] == 4
    assert first_summary["scene_count"] == 2
    assert first_summary["frame_count"] == 4
    assert [scene["scene_name"] for scene in first_summary["scenes"]] == [
        "a_scene",
        "z_scene",
    ]
    assert first_summary["scenes"][0]["frame_count"] == 2
    assert first_summary["scenes"][0]["first_frame"] == "a.jpeg"
    assert first_summary["scenes"][0]["last_frame"] == "b.jpg"
    assert first_summary["scenes"][1]["frame_count"] == 2
    assert first_summary["scenes"][1]["first_frame"] == "002.jpeg"
    assert first_summary["scenes"][1]["last_frame"] == "010.jpg"

    out_two = tmp_path / "artifacts" / "manifest-two.json"
    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out_two)],
        capsys,
    )
    captured = capsys.readouterr()
    second_report = load_report(out_two)

    assert result == 0
    assert captured.err == ""
    assert second_report["effective_manifest"] == first_summary
    assert second_report["manifest_sha256"] == first_report["manifest_sha256"]

    write_frame(data_dir / "a_scene" / "a.jpeg", b"a-a-mutated")
    out_three = tmp_path / "artifacts" / "manifest-three.json"
    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out_three)],
        capsys,
    )
    captured = capsys.readouterr()
    third_report = load_report(out_three)

    assert result == 0
    assert captured.err == ""
    assert third_report["effective_manifest"]["scenes"][0]["scene_sha256"] != (
        first_summary["scenes"][0]["scene_sha256"]
    )
    assert third_report["manifest_sha256"] != first_report["manifest_sha256"]


def test_existing_manifest_uses_file_hash_and_file_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    manifest = data_dir / "manifest.json"
    manifest_bytes = b'{\n  "scenes": [{"name": "from-file"}]\n}\n'
    manifest.write_bytes(manifest_bytes)
    out = tmp_path / "artifacts" / "manifest-report.json"

    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out)],
        capsys,
    )
    captured = capsys.readouterr()
    report = load_report(out)

    assert result == 0
    assert captured.err == ""
    assert report["manifest_source"] == "file"
    assert report["manifest_path"] == str(manifest)
    assert report["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert report["manifest_schema_version"] is None
    assert report["manifest_authoritative"] is False
    assert report["manifest_validation_errors"] == []
    assert report["oracle_schema_present"] is False
    assert report["oracle_schema_valid"] is False
    assert report["oracle_summary"] is None
    assert report["scene_count"] is None
    assert report["frame_count"] is None
    assert report["effective_manifest"] == {"scenes": [{"name": "from-file"}]}


def test_existing_manifest_report_counts_use_standard_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    manifest = data_dir / "manifest.json"
    manifest_bytes = b'{\n  "scene_count": 3,\n  "frame_count": 41,\n  "scenes": []\n}\n'
    manifest.write_bytes(manifest_bytes)
    out = tmp_path / "artifacts" / "manifest-report.json"

    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out)],
        capsys,
    )
    captured = capsys.readouterr()
    report = load_report(out)

    assert result == 0
    assert captured.err == ""
    assert report["manifest_source"] == "file"
    assert report["scene_count"] == 3
    assert report["frame_count"] == 41
    assert report["effective_manifest"] == {
        "scene_count": 3,
        "frame_count": 41,
        "scenes": [],
    }
    assert report["manifest_authoritative"] is False
    assert report["oracle_schema_present"] is False


def test_schema_v1_rich_manifest_is_authoritative_and_projects_oracle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    write_frame(data_dir / "scene-a" / "001.jpg", b"a-1")
    write_frame(data_dir / "scene-a" / "002.jpg", b"a-2")
    write_frame(data_dir / "scene-b" / "001.jpeg", b"b-1")
    manifest = data_dir / "manifest.json"
    write_manifest(manifest, rich_manifest_for_data(module, data_dir))
    out = tmp_path / "artifacts" / "manifest-report.json"

    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out)],
        capsys,
    )
    captured = capsys.readouterr()
    report = load_report(out)

    assert result == 0
    assert captured.err == ""
    assert report["manifest_source"] == "file"
    assert report["manifest_schema_version"] == 1
    assert report["manifest_authoritative"] is True
    assert report["manifest_validation_errors"] == []
    assert report["oracle_schema_present"] is True
    assert report["oracle_schema_valid"] is True
    assert report["oracle_summary"] == {
        "expected_event_timeline": {
            "source": "oracle/events.json",
            "version": "events-v1",
        },
        "expected_attention_target_timeline": {
            "source": "oracle/attention.json",
            "rule": "largest_stable_person_v1",
        },
    }
    assert report["scene_count"] == 2
    assert report["frame_count"] == 3


def test_schema_v1_manifest_rejects_bool_schema_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    write_frame(data_dir / "scene" / "001.jpg", b"frame")
    manifest_payload = rich_manifest_for_data(module, data_dir)
    manifest_payload["schema_version"] = True
    write_manifest(data_dir / "manifest.json", manifest_payload)
    out = tmp_path / "artifacts" / "manifest-report.json"

    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out)],
        capsys,
    )
    captured = capsys.readouterr()
    report = load_report(out)

    assert result == 2
    assert captured.out == ""
    assert "schema_version_invalid" in report["manifest_validation_errors"]


@pytest.mark.parametrize(
    ("field_name", "bad_value", "expected_error"),
    [
        ("scene_count", 99, "scene_count_mismatch"),
        ("frame_count", 99, "frame_count_mismatch"),
    ],
)
def test_schema_v1_manifest_rejects_count_mismatches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field_name: str,
    bad_value: int,
    expected_error: str,
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    write_frame(data_dir / "scene" / "001.jpg", b"frame")
    manifest_payload = rich_manifest_for_data(module, data_dir)
    manifest_payload[field_name] = bad_value
    write_manifest(data_dir / "manifest.json", manifest_payload)
    out = tmp_path / "artifacts" / "manifest-report.json"

    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out)],
        capsys,
    )
    captured = capsys.readouterr()
    report = load_report(out)

    assert result == 2
    assert captured.out == ""
    assert expected_error in report["manifest_validation_errors"]
    assert report["manifest_authoritative"] is True
    assert report["oracle_schema_valid"] is True


def test_schema_v1_manifest_rejects_empty_authoritative_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    write_manifest(data_dir / "manifest.json", rich_manifest_for_data(module, data_dir))
    out = tmp_path / "artifacts" / "manifest-report.json"

    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out)],
        capsys,
    )
    captured = capsys.readouterr()
    report = load_report(out)

    assert result == 2
    assert captured.out == ""
    assert report["manifest_authoritative"] is True
    assert report["scene_count"] == 0
    assert report["frame_count"] == 0
    assert "scene_count_empty" in report["manifest_validation_errors"]
    assert "frame_count_empty" in report["manifest_validation_errors"]
    assert report["oracle_schema_valid"] is True


def test_schema_v1_manifest_rejects_scene_sha_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    write_frame(data_dir / "scene" / "001.jpg", b"frame")
    manifest_payload = rich_manifest_for_data(module, data_dir)
    manifest_payload["scenes"][0]["scene_sha256"] = "0" * 64
    write_manifest(data_dir / "manifest.json", manifest_payload)
    out = tmp_path / "artifacts" / "manifest-report.json"

    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out)],
        capsys,
    )
    report = load_report(out)

    assert result == 2
    assert "scene_sha256_mismatch:scene" in report["manifest_validation_errors"]


def test_schema_v1_manifest_rejects_missing_oracle_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    write_frame(data_dir / "scene" / "001.jpg", b"frame")
    manifest_payload = rich_manifest_for_data(module, data_dir)
    del manifest_payload["oracle"]["expected_event_timeline"]["version"]
    del manifest_payload["oracle"]["expected_attention_target_timeline"]["rule"]
    write_manifest(data_dir / "manifest.json", manifest_payload)
    out = tmp_path / "artifacts" / "manifest-report.json"

    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out)],
        capsys,
    )
    report = load_report(out)

    assert result == 2
    assert report["oracle_schema_present"] is True
    assert report["oracle_schema_valid"] is False
    assert "oracle.expected_event_timeline.version_missing" in report[
        "manifest_validation_errors"
    ]
    assert "oracle.expected_attention_target_timeline.rule_missing" in report[
        "manifest_validation_errors"
    ]


def test_explicit_missing_manifest_fails_fast(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    out = tmp_path / "artifacts" / "manifest-report.json"
    missing_manifest = tmp_path / "missing-manifest.json"

    result = run_tool(
        module,
        [
            "--data-dir",
            str(data_dir),
            "--manifest",
            str(missing_manifest),
            "--out",
            str(out),
        ],
        capsys,
    )
    captured = capsys.readouterr()

    assert result == 2
    assert not out.exists()
    assert captured.out == ""
    assert "manifest" in captured.err.lower()
    assert "not found" in captured.err.lower()


def test_out_inside_data_dir_fails_fast(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    out = data_dir / "reports" / "manifest-report.json"

    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out)],
        capsys,
    )
    captured = capsys.readouterr()

    assert result == 2
    assert not out.exists()
    assert captured.out == ""
    assert "--out" in captured.err
    assert "data-dir" in captured.err


def test_report_skeleton_never_claims_pc_local_e2e_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = import_manifest_module()
    data_dir = tmp_path / "val-data"
    write_frame(data_dir / "scene" / "001.jpg", b"frame")
    out = tmp_path / "artifacts" / "manifest-report.json"

    result = run_tool(
        module,
        ["--data-dir", str(data_dir), "--out", str(out)],
        capsys,
    )
    captured = capsys.readouterr()
    report = load_report(out)

    assert result == 0
    assert captured.err == ""
    assert report["overall_pass"] is False
    assert report["pc_local_e2e_status"] == "not_run"
    assert "pc_local_e2e_not_run" in report["failure_reasons"]
    serialized = json.dumps(report, sort_keys=True).lower()
    assert "dds_process" not in serialized
    assert "server_process" not in serialized
    assert "cli_process" not in serialized
    assert '"pc_local_e2e_status": "success"' not in serialized
    assert '"pc_local_e2e_status": "passed"' not in serialized


def test_source_audit_has_no_dds_cli_server_or_model_imports():
    module = import_manifest_module()
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    forbidden_roots = {
        "subprocess",
        "torch",
        "ultralytics",
        "visual_events_cli",
        "visual_events_server",
    }
    forbidden_exact = {"dds", "unitree"}

    for imported in imported_modules:
        root = imported.split(".", 1)[0]
        assert root not in forbidden_roots
        assert imported not in forbidden_exact
