from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import memory_teaching_evidence
from tools import run_memory_teaching_ga_e2e as runner


DEFAULT_OUT = Path("artifacts/memory-teaching-evidence")
DEFAULT_DATA_DIR = Path("val-data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate artifact-first memory teaching evidence from a runner artifact."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Source memory teaching runner artifact. Required unless --run-local-smoke is used.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--strict-source-ok",
        action="store_true",
        help="Return non-zero when the source report has ok=false.",
    )
    parser.add_argument(
        "--run-local-smoke",
        action="store_true",
        help="Delegate to tools.run_memory_teaching_ga_e2e.run_local_smoke first.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--camera", default=runner.DEFAULT_CAMERA)
    parser.add_argument(
        "--embedding-backend",
        choices=("fake", "local"),
        default="fake",
    )
    parser.add_argument("--person-model-path", type=Path)
    parser.add_argument("--scene-model-path", type=Path)
    parser.add_argument(
        "--inference-backend",
        choices=("mock", "ultralytics"),
        default="mock",
    )
    parser.add_argument("--pose-model-path", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = generate_memory_teaching_evidence_from_args(args)
    except memory_teaching_evidence.MemoryTeachingEvidenceError as exc:
        print(f"memory teaching evidence failed: {exc}", file=sys.stderr)
        return 2
    except SystemExit:
        raise

    print(f"source report: {summary['source_report_path']}")
    print(f"source gate: {summary['source_status']}")
    print(f"root index: {summary['index_html']}")
    if args.strict_source_ok and not summary["source_report_ok"]:
        return 1
    return 0


def generate_memory_teaching_evidence_from_args(
    args: argparse.Namespace,
) -> dict[str, object]:
    out = Path(args.out)
    _reject_out_inside_data_dir(out=out, data_dir=Path(args.data_dir))
    artifact = _source_artifact_from_args(args)
    _reject_bad_out(
        out=out,
        artifact=artifact,
    )
    return memory_teaching_evidence.render_memory_teaching_evidence(
        artifact=artifact,
        out=out,
    )


def _source_artifact_from_args(args: argparse.Namespace) -> Path:
    out = Path(args.out)
    if args.run_local_smoke:
        artifact = out / "runner-artifact"
        runner.run_local_smoke(
            data_dir=Path(args.data_dir),
            out=artifact,
            camera=args.camera,
            embedding_backend=args.embedding_backend,
            person_model_path=args.person_model_path,
            scene_model_path=args.scene_model_path,
            inference_backend=args.inference_backend,
            pose_model_path=args.pose_model_path,
        )
        return artifact
    if args.artifact is None:
        raise memory_teaching_evidence.MemoryTeachingEvidenceError(
            "--artifact is required unless --run-local-smoke is used"
        )
    return Path(args.artifact)


def _reject_bad_out(*, out: Path, artifact: Path) -> None:
    out_resolved = _resolve_for_compare(out)
    artifact_resolved = _resolve_for_compare(artifact)
    if out_resolved == artifact_resolved or artifact_resolved in out_resolved.parents:
        raise memory_teaching_evidence.MemoryTeachingEvidenceError(
            f"--out must not be inside the source artifact: out={out} artifact={artifact}"
        )


def _reject_out_inside_data_dir(*, out: Path, data_dir: Path) -> None:
    out_resolved = _resolve_for_compare(out)
    data_dir_resolved = _resolve_for_compare(data_dir)
    if out_resolved == data_dir_resolved or data_dir_resolved in out_resolved.parents:
        raise memory_teaching_evidence.MemoryTeachingEvidenceError(
            f"--out must not be inside data-dir: out={out} data-dir={data_dir}"
        )


def _resolve_for_compare(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


if __name__ == "__main__":
    raise SystemExit(main())
