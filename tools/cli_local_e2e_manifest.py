from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


_JPEG_GLOBS = ("*.jpeg", "*.jpg")


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

    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
