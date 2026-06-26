from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.replay_val_data import (
        DEFAULT_SEMANTIC_EVENT_COOLDOWN_MS,
        ReplayStats,
        replay_scene,
        stats_passed,
        stats_to_summary,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - direct script execution
    if exc.name != "tools":
        raise
    from replay_val_data import (
        DEFAULT_SEMANTIC_EVENT_COOLDOWN_MS,
        ReplayStats,
        replay_scene,
        stats_passed,
        stats_to_summary,
    )


REQUIRED_SCENE_NAMES = (
    "pci_stand",
    "pic_1_l_to_r",
    "pic_1_r_to_l",
    "pic_hello",
    "pic_leave",
    "pic_persone_walk_in",
    "pic_walk_in_stop",
)

DEFAULT_OUT = Path("artifacts/e2e")
DEFAULT_CAMERA = "front"
DEFAULT_FPS = 10.0
_JPEG_GLOBS = ("*.jpeg", "*.jpg")
_HZ_MIN = 9.0
_TOTAL_LATENCY_P95_MAX_MS = 120.0
_TOTAL_LATENCY_P99_MAX_MS = 200.0
_ERROR_RATE_MAX = 0.01
_SERVER_PHASES = (
    "decode",
    "preprocess",
    "infer",
    "postprocess",
    "tracking",
    "events",
)


@dataclass(frozen=True)
class CaseResult:
    case: str
    scene: str
    head_motion: str
    gate: str
    stats: ReplayStats
    summary: dict[str, Any]
    artifacts: dict[str, str]
    failure_reasons: list[str]

    @property
    def passed(self) -> bool:
        return bool(self.summary.get("passed"))

    def report_entry(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "scene": self.scene,
            "head_motion": self.head_motion,
            "gate": self.gate,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
            "artifacts": self.artifacts,
            "frames_sent": self.stats.frames_sent,
            "frames_ok": self.stats.frames_ok,
            "errors": self.stats.errors,
            "hz": self.summary.get("hz"),
            "error_rate": self.summary.get("error_rate"),
            "latency_ms": self.summary.get("latency_ms"),
        }


class PreflightError(Exception):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full val-data E2E gate against visual-events-server."
    )
    parser.add_argument("--server", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--response-timeout-ms", type=int, default=None)
    parser.add_argument(
        "--semantic-event-cooldown-ms",
        type=int,
        default=DEFAULT_SEMANTIC_EVENT_COOLDOWN_MS,
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send the next frame as soon as a response arrives.",
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scene_dirs = preflight_val_data_root(args.data_dir)
        preflight_out_path(args.out, data_dir=args.data_dir)
        if args.fps <= 0:
            raise PreflightError("fps must be positive")
        if args.response_timeout_ms is not None and args.response_timeout_ms <= 0:
            raise PreflightError("response-timeout-ms must be positive")
        if args.semantic_event_cooldown_ms < 0:
            raise PreflightError("semantic-event-cooldown-ms must be non-negative")
    except PreflightError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 1

    cases: list[CaseResult] = []
    try:
        for scene_dir in scene_dirs:
            cases.append(
                await _run_case(
                    server=args.server,
                    scene_dir=scene_dir,
                    out=args.out,
                    camera=args.camera,
                    fps=args.fps,
                    head_motion="stationary",
                    gate="all",
                    realtime=not args.no_realtime,
                    response_timeout_ms=args.response_timeout_ms,
                    semantic_event_cooldown_ms=args.semantic_event_cooldown_ms,
                )
            )

        for scene_dir in scene_dirs:
            cases.append(
                await _run_case(
                    server=args.server,
                    scene_dir=scene_dir,
                    out=args.out,
                    camera=args.camera,
                    fps=args.fps,
                    head_motion="unknown",
                    gate="events",
                    realtime=not args.no_realtime,
                    response_timeout_ms=args.response_timeout_ms,
                    semantic_event_cooldown_ms=args.semantic_event_cooldown_ms,
                )
            )

        perf_report = build_perf_report(cases)
        report = build_e2e_report(cases, perf_report=perf_report)
        _write_json(args.out / "report.json", report)
        perf_path = args.out.parent / "perf" / "server_perf.json"
        _write_json(perf_path, perf_report)
    except Exception as exc:
        print(f"e2e failed: {exc}", file=sys.stderr)
        _write_failure_artifacts(args.out, cases=cases, exc=exc)
        return 1

    return 0 if report["overall_pass"] else 1


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))


def preflight_val_data_root(data_dir: Path) -> list[Path]:
    root = Path(data_dir)
    errors: list[str] = []
    if not root.exists():
        raise PreflightError(f"data-dir does not exist: {root}")
    if not root.is_dir():
        raise PreflightError(f"data-dir is not a directory: {root}")
    if _has_jpegs(root):
        errors.append("data-dir must be the val-data root, not a single scene directory")

    scene_dirs: list[Path] = []
    for scene in REQUIRED_SCENE_NAMES:
        scene_dir = root / scene
        if not scene_dir.exists():
            errors.append(f"missing required scene directory: {scene}")
            continue
        if not scene_dir.is_dir():
            errors.append(f"required scene path is not a directory: {scene}")
            continue
        if not _has_jpegs(scene_dir):
            errors.append(f"required scene directory has no JPEG frames: {scene}")
            continue
        scene_dirs.append(scene_dir)

    if errors:
        raise PreflightError("; ".join(errors))
    return scene_dirs


def preflight_out_path(out: Path, *, data_dir: Path) -> None:
    if out.name != "e2e" or out.parent.name != "artifacts":
        raise PreflightError("out must be an artifacts/e2e path")
    resolved_out = out.resolve()
    resolved_data_dir = data_dir.resolve()
    if resolved_out == resolved_data_dir or resolved_out.is_relative_to(
        resolved_data_dir
    ):
        raise PreflightError("out must not be inside data-dir")


async def _run_case(
    *,
    server: str,
    scene_dir: Path,
    out: Path,
    camera: str,
    fps: float,
    head_motion: str,
    gate: str,
    realtime: bool,
    response_timeout_ms: int | None,
    semantic_event_cooldown_ms: int,
) -> CaseResult:
    scene = scene_dir.name
    case = scene if head_motion == "stationary" else f"{scene}__head_{head_motion}"
    case_dir = out / case
    visual_state_jsonl = case_dir / "visual_state.jsonl"
    summary_json = case_dir / "summary.json"
    summary_md = case_dir / "summary.md"

    stats = await replay_scene(
        server=server,
        scene_dir=scene_dir,
        camera=camera,
        fps=fps,
        head_motion=head_motion,
        save_jsonl=visual_state_jsonl,
        realtime=realtime,
        response_timeout_ms=response_timeout_ms,
        semantic_event_cooldown_ms=semantic_event_cooldown_ms,
    )
    summary = _case_summary(
        stats,
        case=case,
        head_motion=head_motion,
        gate=gate,
        visual_state_jsonl=visual_state_jsonl,
    )
    artifacts = {
        "visual_state_jsonl": str(visual_state_jsonl),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    summary["artifacts"] = artifacts
    failure_reasons = failure_reasons_for(stats, summary=summary, gate=gate)
    summary["failure_reasons"] = failure_reasons
    _write_json(summary_json, summary)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    return CaseResult(
        case=case,
        scene=scene,
        head_motion=head_motion,
        gate=gate,
        stats=stats,
        summary=summary,
        artifacts=artifacts,
        failure_reasons=failure_reasons,
    )


def _case_summary(
    stats: ReplayStats,
    *,
    case: str,
    head_motion: str,
    gate: str,
    visual_state_jsonl: Path,
) -> dict[str, Any]:
    summary = dict(stats_to_summary(stats, gate=gate))
    summary["case"] = case
    summary["head_motion"] = head_motion
    summary["gate"] = gate
    summary["passed"] = stats_passed(stats, gate=gate)
    summary["hz"] = _hz(stats.frames_ok, stats.elapsed_s)
    summary["error_rate"] = _error_rate(stats.errors, stats.frames_sent)
    summary["latency_ms"] = _latency_summary(read_latency_samples(visual_state_jsonl))
    return summary


def build_e2e_report(
    cases: list[CaseResult],
    *,
    perf_report: dict[str, Any],
    extra_failure_reasons: list[str] | None = None,
) -> dict[str, Any]:
    case_entries = [case.report_entry() for case in cases]
    failure_reasons: list[str] = []
    if extra_failure_reasons:
        failure_reasons.extend(extra_failure_reasons)
    for case in cases:
        failure_reasons.extend(
            f"{case.case}: {reason}" for reason in case.failure_reasons
        )
    if not perf_report["passed"]:
        failure_reasons.extend(
            f"perf: {reason}" for reason in perf_report["failure_reasons"]
        )

    return {
        "overall_pass": all(case.passed for case in cases) and perf_report["passed"],
        "cases": case_entries,
        "failure_reasons": failure_reasons,
        "thresholds": {
            "hz_min": _HZ_MIN,
            "total_latency_p95_max_ms": _TOTAL_LATENCY_P95_MAX_MS,
            "total_latency_p99_max_ms": _TOTAL_LATENCY_P99_MAX_MS,
            "error_rate_max": _ERROR_RATE_MAX,
            "perf_passed": perf_report["passed"],
        },
    }


def build_perf_report(cases: list[CaseResult]) -> dict[str, Any]:
    stats_items = [case.stats for case in cases]
    frames_sent = sum(item.frames_sent for item in stats_items)
    frames_ok = sum(item.frames_ok for item in stats_items)
    errors = sum(item.errors for item in stats_items)
    elapsed_s = sum(max(0.0, item.elapsed_s) for item in stats_items)
    latencies: list[float] = []
    for case in cases:
        jsonl_path = Path(case.artifacts["visual_state_jsonl"])
        latencies.extend(read_latency_samples(jsonl_path))

    latency_summary = _latency_summary(latencies)
    hz = _hz(frames_ok, elapsed_s)
    error_rate = _error_rate(errors, frames_sent)
    threshold_results = {
        "hz": hz >= _HZ_MIN,
        "total_latency_p95_ms": (
            latency_summary["available"]
            and latency_summary["p95"] < _TOTAL_LATENCY_P95_MAX_MS
        ),
        "total_latency_p99_ms": (
            latency_summary["available"]
            and latency_summary["p99"] < _TOTAL_LATENCY_P99_MAX_MS
        ),
        "error_rate": error_rate < _ERROR_RATE_MAX,
    }
    failure_reasons = [
        name for name, passed in threshold_results.items() if not passed
    ]
    return {
        "passed": all(threshold_results.values()),
        "failure_reasons": failure_reasons,
        "total_latency_ms": latency_summary,
        "hz": hz,
        "error_rate": error_rate,
        "frames": {
            "sent": frames_sent,
            "ok": frames_ok,
            "errors": errors,
            "latency_samples": len(latencies),
        },
        "thresholds": {
            "hz_min": _HZ_MIN,
            "total_latency_p95_max_ms": _TOTAL_LATENCY_P95_MAX_MS,
            "total_latency_p99_max_ms": _TOTAL_LATENCY_P99_MAX_MS,
            "error_rate_max": _ERROR_RATE_MAX,
            "results": threshold_results,
        },
        "server_phase_latency_ms": {
            phase: {"available": False} for phase in _SERVER_PHASES
        },
        "vram": {"available": False},
        "memory": {"available": False},
    }


def build_failed_perf_report(failure_reasons: list[str]) -> dict[str, Any]:
    return {
        "passed": False,
        "failure_reasons": failure_reasons,
        "total_latency_ms": _latency_summary([]),
        "hz": 0.0,
        "error_rate": 0.0,
        "frames": {
            "sent": 0,
            "ok": 0,
            "errors": 0,
            "latency_samples": 0,
        },
        "thresholds": {
            "hz_min": _HZ_MIN,
            "total_latency_p95_max_ms": _TOTAL_LATENCY_P95_MAX_MS,
            "total_latency_p99_max_ms": _TOTAL_LATENCY_P99_MAX_MS,
            "error_rate_max": _ERROR_RATE_MAX,
            "results": {
                "hz": False,
                "total_latency_p95_ms": False,
                "total_latency_p99_ms": False,
                "error_rate": False,
            },
        },
        "server_phase_latency_ms": {
            phase: {"available": False} for phase in _SERVER_PHASES
        },
        "vram": {"available": False},
        "memory": {"available": False},
    }


def read_latency_samples(jsonl_path: Path) -> list[float]:
    if not jsonl_path.exists():
        return []
    samples: list[float] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        latency_ms = payload.get("latency_ms")
        if isinstance(latency_ms, bool) or not isinstance(latency_ms, int | float):
            continue
        latency = float(latency_ms)
        if math.isfinite(latency):
            samples.append(latency)
    return samples


def failure_reasons_for(
    stats: ReplayStats,
    *,
    summary: dict[str, Any],
    gate: str,
) -> list[str]:
    if summary["passed"]:
        return []
    reasons: list[str] = []
    if stats.errors:
        reasons.append(f"{stats.errors} frame errors")
    if stats.frame_id_mismatch:
        reasons.append(f"{stats.frame_id_mismatch} frame_id mismatches")
    if gate in {"tracking", "all"} and not summary.get("tracking_pass"):
        reasons.append("tracking gate failed")
    if gate in {"attention", "all"} and not summary.get("attention_pass"):
        reasons.append("attention gate failed")
    if gate in {"events", "all"} and not summary.get("events_pass"):
        if (
            stats.head_motion in {"moving", "unknown"}
            and stats.semantic_event_motion_sensitive_count > 0
        ):
            reasons.append(
                f"motion-sensitive events emitted for {stats.head_motion} head motion"
            )
        elif stats.semantic_event_expected_missing:
            reasons.append("expected semantic events missing")
        elif stats.semantic_event_unexpected_by_scene:
            reasons.append("unexpected semantic events emitted")
        else:
            reasons.append("events gate failed")
    if not reasons:
        reasons.append(f"{gate} gate failed")
    return reasons


def _summary_markdown(summary: dict[str, Any]) -> str:
    latency = summary["latency_ms"]
    return "\n".join(
        [
            f"# {summary['case']}",
            "",
            f"- scene: {summary['scene']}",
            f"- head_motion: {summary['head_motion']}",
            f"- gate: {summary['gate']}",
            f"- passed: {str(summary['passed']).lower()}",
            f"- frames_ok: {summary['frames_ok']} / {summary['frames_sent']}",
            f"- errors: {summary['errors']}",
            f"- hz: {summary['hz']}",
            (
                "- latency_ms: unavailable"
                if not latency["available"]
                else (
                    f"- latency_ms: p50={latency['p50']}, "
                    f"p95={latency['p95']}, p99={latency['p99']}"
                )
            ),
            "",
        ]
    )


def _latency_summary(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "available": False,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    return {
        "available": True,
        "p50": _percentile_nearest_rank(samples, 50),
        "p95": _percentile_nearest_rank(samples, 95),
        "p99": _percentile_nearest_rank(samples, 99),
    }


def _percentile_nearest_rank(samples: list[float], percentile: int) -> float:
    ordered = sorted(samples)
    rank = math.ceil((percentile / 100.0) * len(ordered))
    index = min(max(rank - 1, 0), len(ordered) - 1)
    return float(ordered[index])


def _hz(frames_ok: int, elapsed_s: float) -> float:
    if elapsed_s <= 0.0:
        return 0.0
    return frames_ok / elapsed_s


def _error_rate(errors: int, frames_sent: int) -> float:
    if frames_sent <= 0:
        return 1.0
    return errors / frames_sent


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_failure_artifacts(out: Path, *, cases: list[CaseResult], exc: Exception) -> None:
    reason = f"e2e exception: {type(exc).__name__}: {exc}"
    try:
        perf_report = build_failed_perf_report([reason])
        report = build_e2e_report(
            cases,
            perf_report=perf_report,
            extra_failure_reasons=[reason],
        )
        _write_json(out / "report.json", report)
        _write_json(out.parent / "perf" / "server_perf.json", perf_report)
    except Exception as write_exc:
        print(f"failed to write failure artifacts: {write_exc}", file=sys.stderr)


def _has_jpegs(path: Path) -> bool:
    return path.is_dir() and any(match for glob in _JPEG_GLOBS for match in path.glob(glob))


if __name__ == "__main__":
    main()
