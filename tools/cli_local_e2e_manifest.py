from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


_JPEG_GLOBS = ("*.jpeg", "*.jpg")
MANIFEST_SCHEMA_VERSION = 1


class PreflightError(Exception):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manifest-only PC local E2E report skeleton."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args(argv)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def preflight_data_dir(data_dir: Path) -> Path:
    resolved = resolve_path(data_dir)
    if not resolved.exists():
        raise PreflightError(f"data-dir not found: {data_dir}")
    if not resolved.is_dir():
        raise PreflightError(f"data-dir is not a directory: {data_dir}")
    return resolved


def preflight_out_path(out: Path, *, data_dir: Path) -> Path:
    resolved = resolve_path(out)
    if is_relative_to(resolved, data_dir):
        raise PreflightError("--out must not be inside --data-dir")
    return resolved


def select_manifest_path(
    *,
    data_dir: Path,
    manifest: Path | None,
) -> Path | None:
    if manifest is None:
        default_manifest = data_dir / "manifest.json"
        return default_manifest if default_manifest.exists() else None

    resolved = resolve_path(manifest)
    if not resolved.exists():
        raise PreflightError(f"manifest not found: {manifest}")
    if not resolved.is_file():
        raise PreflightError(f"manifest is not a file: {manifest}")
    return resolved


def load_manifest_file(path: Path) -> tuple[Any, str]:
    manifest_bytes = path.read_bytes()
    try:
        manifest_json = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"manifest JSON is invalid: {path}: {exc}") from exc
    return manifest_json, sha256_bytes(manifest_bytes)


def jpeg_files(scene_dir: Path) -> list[Path]:
    frames: list[Path] = []
    for pattern in _JPEG_GLOBS:
        frames.extend(path for path in scene_dir.glob(pattern) if path.is_file())
    return sorted(frames, key=lambda path: path.name)


def scene_summary(scene_dir: Path) -> dict[str, Any]:
    frame_entries: list[dict[str, str]] = []
    ordered_frames = jpeg_files(scene_dir)
    for frame in ordered_frames:
        relative_name = frame.relative_to(scene_dir).as_posix()
        frame_entries.append(
            {
                "filename": relative_name,
                "sha256": sha256_file(frame),
            }
        )

    frame_names = [entry["filename"] for entry in frame_entries]
    frames_sha256 = sha256_bytes(canonical_json_bytes(frame_entries))
    scene_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "scene_name": scene_dir.name,
                "frames": frame_entries,
            }
        )
    )
    return {
        "scene_name": scene_dir.name,
        "frame_count": len(frame_entries),
        "first_frame": frame_names[0] if frame_names else None,
        "last_frame": frame_names[-1] if frame_names else None,
        "frames_sha256": frames_sha256,
        "scene_sha256": scene_sha256,
    }


def generate_effective_manifest(data_dir: Path) -> dict[str, Any]:
    scene_dirs = sorted(
        (path for path in data_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    scenes = [scene_summary(scene_dir) for scene_dir in scene_dirs]
    return {
        "scene_count": len(scenes),
        "frame_count": sum(scene["frame_count"] for scene in scenes),
        "scenes": scenes,
    }


def manifest_count_field(manifest: Any, field_name: str) -> int | None:
    if not isinstance(manifest, dict):
        return None
    value = manifest.get(field_name)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def manifest_schema_version_field(manifest: Any) -> Any | None:
    if not isinstance(manifest, dict):
        return None
    return manifest.get("schema_version")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0.0


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def _is_hex_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _empty_manifest_contract_projection(schema_version: Any | None) -> dict[str, Any]:
    return {
        "manifest_schema_version": schema_version,
        "manifest_authoritative": False,
        "manifest_validation_errors": [],
        "oracle_schema_present": False,
        "oracle_schema_valid": False,
        "oracle_summary": None,
    }


def _oracle_summary(oracle: Any) -> dict[str, Any] | None:
    if not isinstance(oracle, dict):
        return None
    expected_event = oracle.get("expected_event_timeline")
    expected_attention = oracle.get("expected_attention_target_timeline")
    if not isinstance(expected_event, dict):
        expected_event = {}
    if not isinstance(expected_attention, dict):
        expected_attention = {}
    return {
        "expected_event_timeline": {
            "source": expected_event.get("source")
            if isinstance(expected_event.get("source"), str)
            else None,
            "version": expected_event.get("version")
            if isinstance(expected_event.get("version"), str)
            else None,
        },
        "expected_attention_target_timeline": {
            "source": expected_attention.get("source")
            if isinstance(expected_attention.get("source"), str)
            else None,
            "rule": expected_attention.get("rule")
            if isinstance(expected_attention.get("rule"), str)
            else None,
        },
    }


def _validate_oracle_schema(manifest: dict[str, Any]) -> tuple[bool, bool, dict[str, Any] | None, list[str]]:
    oracle = manifest.get("oracle")
    if not isinstance(oracle, dict):
        return False, False, None, ["oracle_missing_or_not_object"]

    errors: list[str] = []
    expected_event = oracle.get("expected_event_timeline")
    if not isinstance(expected_event, dict):
        errors.append("oracle.expected_event_timeline_missing")
        expected_event = {}
    expected_attention = oracle.get("expected_attention_target_timeline")
    if not isinstance(expected_attention, dict):
        errors.append("oracle.expected_attention_target_timeline_missing")
        expected_attention = {}

    for field_name in ("source", "version"):
        if not _non_empty_str(expected_event.get(field_name)):
            errors.append(f"oracle.expected_event_timeline.{field_name}_missing")
    for field_name in ("source", "rule"):
        if not _non_empty_str(expected_attention.get(field_name)):
            errors.append(
                f"oracle.expected_attention_target_timeline.{field_name}_missing"
            )

    return True, not errors, _oracle_summary(oracle), errors


def _actual_scenes_by_name(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenes = inventory.get("scenes")
    if not isinstance(scenes, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_name = scene.get("scene_name")
        if isinstance(scene_name, str):
            result[scene_name] = scene
    return result


def _validate_manifest_scenes(
    *,
    manifest: dict[str, Any],
    inventory: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list):
        return ["scenes_missing_or_not_list"]

    actual_by_name = _actual_scenes_by_name(inventory)
    seen_names: set[str] = set()
    manifest_names: set[str] = set()
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            errors.append(f"scene_invalid:{index}")
            continue

        raw_name = scene.get("scene_name")
        if not _non_empty_str(raw_name):
            errors.append(f"scene_name_missing:{index}")
            continue
        scene_name = str(raw_name)
        if scene_name in seen_names:
            errors.append(f"scene_duplicate:{scene_name}")
        else:
            seen_names.add(scene_name)
            manifest_names.add(scene_name)

        actual_scene = actual_by_name.get(scene_name)
        if actual_scene is None:
            errors.append(f"scene_unknown:{scene_name}")

        frame_count = scene.get("frame_count")
        if not _is_int(frame_count):
            errors.append(f"scene_frame_count_invalid:{scene_name}")
        elif actual_scene is not None and frame_count != actual_scene.get("frame_count"):
            errors.append(f"scene_frame_count_mismatch:{scene_name}")

        scene_sha256 = scene.get("scene_sha256")
        if not _is_hex_sha256(scene_sha256):
            errors.append(f"scene_sha256_invalid:{scene_name}")
        elif (
            actual_scene is not None
            and str(scene_sha256).lower() != actual_scene.get("scene_sha256")
        ):
            errors.append(f"scene_sha256_mismatch:{scene_name}")

    for scene_name in sorted(set(actual_by_name) - manifest_names):
        errors.append(f"scene_missing:{scene_name}")
    return errors


def _validate_manifest_v1(
    *,
    manifest: dict[str, Any],
    inventory: dict[str, Any],
) -> tuple[list[str], bool, bool, dict[str, Any] | None]:
    errors: list[str] = []
    schema_version = manifest.get("schema_version")
    if not _is_int(schema_version):
        errors.append("schema_version_invalid")
    elif schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if not _is_positive_number(manifest.get("fps")):
        errors.append("fps_missing_or_non_positive")

    scene_count = manifest.get("scene_count")
    if not _is_int(scene_count):
        errors.append("scene_count_invalid")
    else:
        if scene_count <= 0:
            errors.append("scene_count_empty")
        if scene_count != inventory["scene_count"]:
            errors.append("scene_count_mismatch")

    frame_count = manifest.get("frame_count")
    if not _is_int(frame_count):
        errors.append("frame_count_invalid")
    else:
        if frame_count <= 0:
            errors.append("frame_count_empty")
        if frame_count != inventory["frame_count"]:
            errors.append("frame_count_mismatch")

    errors.extend(_validate_manifest_scenes(manifest=manifest, inventory=inventory))
    oracle_present, oracle_valid, oracle_summary, oracle_errors = _validate_oracle_schema(
        manifest
    )
    errors.extend(oracle_errors)
    return errors, oracle_present, oracle_valid, oracle_summary


def manifest_contract_projection(
    *,
    manifest: Any,
    manifest_source: str,
    data_dir: Path,
) -> dict[str, Any]:
    schema_version = manifest_schema_version_field(manifest)
    if manifest_source == "generated":
        return _empty_manifest_contract_projection(schema_version)
    if not isinstance(manifest, dict):
        return _empty_manifest_contract_projection(schema_version)
    if "schema_version" not in manifest:
        return _empty_manifest_contract_projection(schema_version)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        projection = _empty_manifest_contract_projection(schema_version)
        projection["manifest_validation_errors"] = ["unsupported_manifest_schema_version"]
        return projection

    inventory = generate_effective_manifest(data_dir)
    errors, oracle_present, oracle_valid, oracle_summary = _validate_manifest_v1(
        manifest=manifest,
        inventory=inventory,
    )
    return {
        "manifest_schema_version": schema_version,
        "manifest_authoritative": True,
        "manifest_validation_errors": errors,
        "oracle_schema_present": oracle_present,
        "oracle_schema_valid": oracle_valid,
        "oracle_summary": oracle_summary,
    }


def manifest_contract_satisfied(report: dict[str, Any]) -> bool:
    validation_errors = report.get("manifest_validation_errors")
    return (
        report.get("manifest_source") == "file"
        and report.get("manifest_authoritative") is True
        and isinstance(validation_errors, list)
        and not validation_errors
        and report.get("oracle_schema_present") is True
        and report.get("oracle_schema_valid") is True
    )


def manifest_contract_failure_reasons(report: dict[str, Any]) -> list[str]:
    if manifest_contract_satisfied(report):
        return []

    reasons: list[str] = []
    if report.get("manifest_source") != "file":
        reasons.append("manifest_source_not_file")
    if report.get("manifest_authoritative") is not True:
        reasons.append("manifest_not_authoritative")

    validation_errors = report.get("manifest_validation_errors")
    if isinstance(validation_errors, list):
        if validation_errors:
            joined = ",".join(str(error) for error in validation_errors)
            reasons.append(f"manifest_validation_errors:{joined}")
    else:
        reasons.append("manifest_validation_errors_invalid")

    if report.get("oracle_schema_present") is not True:
        reasons.append("oracle_schema_missing")
    if report.get("oracle_schema_valid") is not True:
        reasons.append("oracle_schema_invalid")
    return reasons


def build_report(
    *,
    data_dir: Path,
    out: Path,
    manifest: Path | None,
) -> tuple[Path, dict[str, Any]]:
    resolved_data_dir = preflight_data_dir(data_dir)
    resolved_out = preflight_out_path(out, data_dir=resolved_data_dir)
    manifest_path = select_manifest_path(data_dir=resolved_data_dir, manifest=manifest)

    if manifest_path is None:
        manifest_source = "generated"
        effective_manifest = generate_effective_manifest(resolved_data_dir)
        manifest_sha256 = sha256_bytes(canonical_json_bytes(effective_manifest))
    else:
        manifest_source = "file"
        effective_manifest, manifest_sha256 = load_manifest_file(manifest_path)

    contract_projection = manifest_contract_projection(
        manifest=effective_manifest,
        manifest_source=manifest_source,
        data_dir=resolved_data_dir,
    )
    report = {
        "report_type": "cli_local_e2e_manifest_skeleton",
        "overall_pass": False,
        "pc_local_e2e_status": "not_run",
        "failure_reasons": ["pc_local_e2e_not_run"],
        "data_dir": str(resolved_data_dir),
        "manifest_source": manifest_source,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_sha256": manifest_sha256,
        "scene_count": manifest_count_field(effective_manifest, "scene_count"),
        "frame_count": manifest_count_field(effective_manifest, "frame_count"),
        "effective_manifest": effective_manifest,
        **contract_projection,
    }
    return resolved_out, report


def write_report(out: Path, report: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        out, report = build_report(
            data_dir=args.data_dir,
            out=args.out,
            manifest=args.manifest,
        )
        write_report(out, report)
    except (OSError, PreflightError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    validation_errors = report.get("manifest_validation_errors")
    if validation_errors:
        joined = ",".join(str(error) for error in validation_errors)
        print(f"error: manifest validation failed: {joined}", file=sys.stderr)
        return 2

    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
