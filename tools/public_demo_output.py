from __future__ import annotations

import os
from pathlib import Path


PUBLIC_DEMO_ROOT = Path("artifacts") / "demo"


class PublicDemoOutputError(ValueError):
    pass


def resolve_public_demo_out(out: Path, *, repo_root: Path) -> Path:
    repo_root = Path(repo_root).resolve(strict=False)
    requested = Path(out).expanduser()
    candidate = requested if requested.is_absolute() else repo_root / requested
    candidate = Path(os.path.normpath(os.fspath(candidate)))
    demo_root = repo_root / PUBLIC_DEMO_ROOT

    try:
        relative = candidate.relative_to(repo_root)
    except ValueError as exc:
        raise PublicDemoOutputError(
            f"--out must be inside {PUBLIC_DEMO_ROOT}/<name> under the repo: {out}"
        ) from exc
    parts = relative.parts
    if len(parts) < 3 or parts[:2] != PUBLIC_DEMO_ROOT.parts:
        raise PublicDemoOutputError(
            f"--out must be inside {PUBLIC_DEMO_ROOT}/<name> under the repo: {out}"
        )

    _reject_existing_symlink_component(candidate, repo_root=repo_root, original=out)
    return candidate


def _reject_existing_symlink_component(
    candidate: Path,
    *,
    repo_root: Path,
    original: Path,
) -> None:
    current = repo_root
    for part in candidate.relative_to(repo_root).parts:
        current = current / part
        if current.is_symlink():
            raise PublicDemoOutputError(
                f"--out path must not contain symlink components: {original}"
            )
        if not current.exists():
            break
