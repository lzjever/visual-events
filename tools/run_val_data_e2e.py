from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
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

MOVING_SUPPRESSION_SCENE_NAMES = (
    "pci_stand",
    "pic_1_l_to_r",
    "pic_1_r_to_l",
    "pic_persone_walk_in",
    "pic_walk_in_stop",
)

DEFAULT_OUT = Path("artifacts/e2e")
DEFAULT_CAMERA = "front"
DEFAULT_FPS = 10.0
DEFAULT_SOAK_MEMORY_GROWTH_MAX_MB = 64.0
DEFAULT_SOAK_SAMPLE_INTERVAL_S = 10.0
_perf_counter = time.perf_counter
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


@dataclass(frozen=True)
class RunConfig:
    server: str
    camera: str
    fps: float
    realtime: bool
    response_timeout_ms: int | None
    semantic_event_cooldown_ms: int


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
        "--soak-seconds",
        type=float,
        default=0.0,
        help="Run complete val-data loops until this runner wall-clock target is met.",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="Server process PID used for soak VmRSS sampling.",
    )
    parser.add_argument(
        "--soak-memory-growth-max-mb",
        type=float,
        default=DEFAULT_SOAK_MEMORY_GROWTH_MAX_MB,
    )
    parser.add_argument(
        "--soak-sample-interval-s",
        type=float,
        default=DEFAULT_SOAK_SAMPLE_INTERVAL_S,
    )
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
        if args.soak_seconds < 0.0:
            raise PreflightError("soak-seconds must be non-negative")
        if args.server_pid is not None and args.server_pid <= 0:
            raise PreflightError("server-pid must be positive")
        if args.soak_memory_growth_max_mb < 0.0:
            raise PreflightError("soak-memory-growth-max-mb must be non-negative")
        if args.soak_sample_interval_s <= 0.0:
            raise PreflightError("soak-sample-interval-s must be positive")
        if args.soak_seconds > 0.0 and args.response_timeout_ms is None:
            raise PreflightError("soak requires response-timeout-ms")
        if args.soak_seconds > 0.0 and args.server_pid is None:
            raise PreflightError("soak requires server-pid")
        if args.soak_seconds > 0.0 and args.no_realtime:
            raise PreflightError("soak requires realtime playback")
        if args.semantic_event_cooldown_ms < 0:
            raise PreflightError("semantic-event-cooldown-ms must be non-negative")
    except PreflightError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 1

    cases: list[CaseResult] = []
    soak_report = build_soak_report_disabled()
    config = RunConfig(
        server=args.server,
        camera=args.camera,
        fps=args.fps,
        realtime=not args.no_realtime,
        response_timeout_ms=args.response_timeout_ms,
        semantic_event_cooldown_ms=args.semantic_event_cooldown_ms,
    )
    try:
        soak_started = False
        initial_rss_mb = None
        await _run_full_matrix(scene_dirs, out=args.out, config=config, cases=cases)

        if args.soak_seconds > 0.0:
            initial_rss_mb = read_process_rss_mb(args.server_pid)
            if initial_rss_mb is None:
                soak_report = build_failed_soak_report(
                    failure_reasons=["soak_rss_unavailable"],
                    target_seconds=args.soak_seconds,
                    server_pid=args.server_pid,
                    memory_growth_max_mb=args.soak_memory_growth_max_mb,
                )
                perf_report = build_perf_report(cases, soak_report=soak_report)
                report = build_e2e_report(cases, perf_report=perf_report)
                _write_json(args.out / "report.json", report)
                _write_json(args.out.parent / "perf" / "server_perf.json", perf_report)
                return 1
            soak_started = True
            soak_report = await _run_soak(
                scene_dirs,
                out=args.out / "soak",
                config=config,
                target_seconds=args.soak_seconds,
                server_pid=args.server_pid,
                initial_rss_mb=initial_rss_mb,
                memory_growth_max_mb=args.soak_memory_growth_max_mb,
                sample_interval_s=args.soak_sample_interval_s,
            )

        perf_report = build_perf_report(cases, soak_report=soak_report)
        report = build_e2e_report(cases, perf_report=perf_report)
        _write_json(args.out / "report.json", report)
        perf_path = args.out.parent / "perf" / "server_perf.json"
        _write_json(perf_path, perf_report)
    except Exception as exc:
        print(f"e2e failed: {exc}", file=sys.stderr)
        if soak_started:
            soak_report = build_failed_soak_report(
                failure_reasons=[f"soak_exception: {type(exc).__name__}: {exc}"],
                target_seconds=args.soak_seconds,
                server_pid=args.server_pid,
                memory_growth_max_mb=args.soak_memory_growth_max_mb,
                initial_rss_mb=initial_rss_mb,
            )
        _write_failure_artifacts(args.out, cases=cases, exc=exc, soak_report=soak_report)
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


async def _run_full_matrix(
    scene_dirs: list[Path],
    *,
    out: Path,
    config: RunConfig,
    cases: list[CaseResult] | None = None,
    include_moving: bool = True,
) -> list[CaseResult]:
    results: list[CaseResult] = [] if cases is None else cases
    for scene_dir in scene_dirs:
        results.append(
            await _run_case(
                server=config.server,
                scene_dir=scene_dir,
                out=out,
                camera=config.camera,
                fps=config.fps,
                head_motion="stationary",
                gate="all",
                realtime=config.realtime,
                response_timeout_ms=config.response_timeout_ms,
                semantic_event_cooldown_ms=config.semantic_event_cooldown_ms,
            )
        )

    for scene_dir in scene_dirs:
        results.append(
            await _run_case(
                server=config.server,
                scene_dir=scene_dir,
                out=out,
                camera=config.camera,
                fps=config.fps,
                head_motion="unknown",
                gate="events",
                realtime=config.realtime,
                response_timeout_ms=config.response_timeout_ms,
                semantic_event_cooldown_ms=config.semantic_event_cooldown_ms,
            )
        )

    if include_moving:
        scene_dirs_by_name = {scene_dir.name: scene_dir for scene_dir in scene_dirs}
        for scene_name in MOVING_SUPPRESSION_SCENE_NAMES:
            results.append(
                await _run_case(
                    server=config.server,
                    scene_dir=scene_dirs_by_name[scene_name],
                    out=out,
                    camera=config.camera,
                    fps=config.fps,
                    head_motion="moving",
                    gate="events",
                    realtime=config.realtime,
                    response_timeout_ms=config.response_timeout_ms,
                    semantic_event_cooldown_ms=config.semantic_event_cooldown_ms,
                )
            )
    return results


async def _run_soak(
    scene_dirs: list[Path],
    *,
    out: Path,
    config: RunConfig,
    target_seconds: float,
    server_pid: int | None,
    initial_rss_mb: float | None,
    memory_growth_max_mb: float,
    sample_interval_s: float,
) -> dict[str, Any]:
    rss_samples: list[float] = []
    rss_unavailable = False
    if initial_rss_mb is None:
        rss_unavailable = True
    else:
        rss_samples.append(initial_rss_mb)

    failure_reasons: list[str] = []
    loops_completed = 0
    elapsed_s = 0.0
    cases_completed = 0
    frames_sent = 0
    frames_ok = 0
    errors = 0
    stats_elapsed_s = 0.0
    latencies: list[float] = []
    last_sample_elapsed_s = 0.0
    started_s = _perf_counter()

    while True:
        loops_completed += 1
        loop_out = out / f"loop_{loops_completed:04d}"
        loop_cases = await _run_full_matrix(
            scene_dirs,
            out=loop_out,
            config=config,
            include_moving=False,
        )
        cases_completed += len(loop_cases)
        for case in loop_cases:
            frames_sent += case.stats.frames_sent
            frames_ok += case.stats.frames_ok
            errors += case.stats.errors
            stats_elapsed_s += max(0.0, case.stats.elapsed_s)
            latencies.extend(
                read_latency_samples(Path(case.artifacts["visual_state_jsonl"]))
            )
            failure_reasons.extend(
                f"loop_{loops_completed:04d}/{case.case}: {reason}"
                for reason in case.failure_reasons
            )

        elapsed_s = _perf_counter() - started_s
        if elapsed_s - last_sample_elapsed_s >= sample_interval_s:
            sample = read_process_rss_mb(server_pid)
            if sample is None:
                rss_unavailable = True
            else:
                rss_samples.append(sample)
            last_sample_elapsed_s = elapsed_s
        if elapsed_s >= target_seconds:
            break

    final_sample = read_process_rss_mb(server_pid)
    if final_sample is None:
        rss_unavailable = True
    else:
        rss_samples.append(final_sample)

    latency_summary = _latency_summary(latencies)
    hz = _hz(frames_ok, stats_elapsed_s)
    error_rate = _error_rate(errors, frames_sent)
    threshold_results = {
        "soak_hz": hz >= _HZ_MIN,
        "soak_total_latency_p95_ms": (
            latency_summary["available"]
            and latency_summary["p95"] < _TOTAL_LATENCY_P95_MAX_MS
        ),
        "soak_total_latency_p99_ms": (
            latency_summary["available"]
            and latency_summary["p99"] < _TOTAL_LATENCY_P99_MAX_MS
        ),
        "soak_error_rate": error_rate < _ERROR_RATE_MAX,
    }
    failure_reasons.extend(
        name for name, passed in threshold_results.items() if not passed
    )

    rss_mb = _rss_summary(
        rss_samples,
        available=not rss_unavailable,
        memory_growth_max_mb=memory_growth_max_mb,
    )
    if rss_unavailable:
        failure_reasons.append("soak_rss_unavailable")
    elif rss_mb["growth"] > memory_growth_max_mb:
        failure_reasons.append("soak_memory_growth_mb")

    return build_soak_report(
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
        target_seconds=target_seconds,
        elapsed_s=elapsed_s,
        loops_completed=loops_completed,
        cases_completed=cases_completed,
        frames={
            "sent": frames_sent,
            "ok": frames_ok,
            "errors": errors,
            "latency_samples": len(latencies),
        },
        hz=hz,
        error_rate=error_rate,
        total_latency_ms=latency_summary,
        server_pid=server_pid,
        rss_mb=rss_mb,
    )


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


def build_soak_report_disabled() -> dict[str, Any]:
    return {
        "enabled": False,
        "passed": True,
        "failure_reasons": [],
    }


def build_soak_report(
    *,
    passed: bool,
    failure_reasons: list[str],
    target_seconds: float,
    elapsed_s: float,
    loops_completed: int,
    cases_completed: int,
    frames: dict[str, Any],
    hz: float,
    error_rate: float,
    total_latency_ms: dict[str, Any],
    server_pid: int | None,
    rss_mb: dict[str, Any],
) -> dict[str, Any]:
    return {
        "enabled": True,
        "passed": passed,
        "failure_reasons": failure_reasons,
        "target_seconds": float(target_seconds),
        "elapsed_s": float(elapsed_s),
        "loops_completed": loops_completed,
        "cases_completed": cases_completed,
        "frames": frames,
        "hz": hz,
        "error_rate": error_rate,
        "total_latency_ms": total_latency_ms,
        "server_pid": server_pid,
        "rss_mb": rss_mb,
    }


def build_failed_soak_report(
    *,
    failure_reasons: list[str],
    target_seconds: float,
    server_pid: int | None,
    memory_growth_max_mb: float,
    initial_rss_mb: float | None = None,
) -> dict[str, Any]:
    if initial_rss_mb is None:
        rss_mb = {
            "available": False,
            "start": None,
            "end": None,
            "growth": None,
            "max_growth": float(memory_growth_max_mb),
            "samples": 0,
        }
    else:
        rss_mb = {
            "available": True,
            "start": float(initial_rss_mb),
            "end": float(initial_rss_mb),
            "growth": 0.0,
            "max_growth": float(memory_growth_max_mb),
            "samples": 1,
        }

    return build_soak_report(
        passed=False,
        failure_reasons=failure_reasons,
        target_seconds=target_seconds,
        elapsed_s=0.0,
        loops_completed=0,
        cases_completed=0,
        frames={
            "sent": 0,
            "ok": 0,
            "errors": 0,
            "latency_samples": 0,
        },
        hz=0.0,
        error_rate=1.0,
        total_latency_ms=_latency_summary([]),
        server_pid=server_pid,
        rss_mb=rss_mb,
    )


def _rss_summary(
    samples: list[float],
    *,
    available: bool,
    memory_growth_max_mb: float,
) -> dict[str, Any]:
    if not samples:
        return {
            "available": False,
            "start": None,
            "end": None,
            "growth": None,
            "max_growth": float(memory_growth_max_mb),
            "samples": 0,
        }
    start = samples[0]
    end = samples[-1]
    growth = max(samples) - start
    return {
        "available": available,
        "start": float(start),
        "end": float(end),
        "growth": float(growth),
        "max_growth": float(memory_growth_max_mb),
        "samples": len(samples),
    }


def build_e2e_report(
    cases: list[CaseResult],
    *,
    perf_report: dict[str, Any],
    extra_failure_reasons: list[str] | None = None,
) -> dict[str, Any]:
    soak_report = perf_report.get("soak", build_soak_report_disabled())
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
        "soak": soak_report,
        "failure_reasons": failure_reasons,
        "thresholds": {
            "hz_min": _HZ_MIN,
            "total_latency_p95_max_ms": _TOTAL_LATENCY_P95_MAX_MS,
            "total_latency_p99_max_ms": _TOTAL_LATENCY_P99_MAX_MS,
            "error_rate_max": _ERROR_RATE_MAX,
            "perf_passed": perf_report["passed"],
            "soak_passed": soak_report["passed"],
        },
    }


def build_perf_report(
    cases: list[CaseResult],
    *,
    soak_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if soak_report is None:
        soak_report = build_soak_report_disabled()
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
    if not soak_report["passed"]:
        failure_reasons.extend(soak_report["failure_reasons"])
    return {
        "passed": all(threshold_results.values()) and soak_report["passed"],
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
        "soak": soak_report,
    }


def build_failed_perf_report(
    failure_reasons: list[str],
    *,
    soak_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if soak_report is None:
        soak_report = build_soak_report_disabled()
    combined_failure_reasons = list(failure_reasons)
    if not soak_report["passed"]:
        combined_failure_reasons.extend(soak_report["failure_reasons"])
    return {
        "passed": False,
        "failure_reasons": combined_failure_reasons,
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
        "soak": soak_report,
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


def read_process_rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    status_path = Path("/proc") / str(pid) / "status"
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return float(parts[1]) / 1024.0
        except ValueError:
            return None
    return None


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


def _write_failure_artifacts(
    out: Path,
    *,
    cases: list[CaseResult],
    exc: Exception,
    soak_report: dict[str, Any] | None = None,
) -> None:
    reason = f"e2e exception: {type(exc).__name__}: {exc}"
    try:
        perf_report = build_failed_perf_report([reason], soak_report=soak_report)
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
