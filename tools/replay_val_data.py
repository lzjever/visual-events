from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from visual_events_server.protocol import (
    MAX_JPEG_BYTES,
    SCHEMA_VERSION,
    encode_frame_message,
)

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
_JPEG_GLOBS = ("*.jpeg", "*.jpg")
_FILENAME_NUMBER = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ReplayFrame:
    path: Path
    header: dict[str, Any]


@dataclass(frozen=True)
class ReplayStats:
    scene: str
    frames_sent: int
    frames_ok: int
    errors: int
    elapsed_s: float
    frames_with_person: int = 0
    frame_id_mismatch: int = 0

    @property
    def ok_rate(self) -> float:
        if self.frames_sent == 0:
            return 0.0
        return self.frames_ok / self.frames_sent

    @property
    def person_frame_rate(self) -> float:
        if self.frames_sent == 0:
            return 0.0
        return self.frames_with_person / self.frames_sent


def discover_scene_dirs(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir)
    if _has_jpegs(data_dir):
        return [data_dir]

    scene_dirs = [
        child
        for child in sorted(data_dir.iterdir())
        if child.is_dir() and _has_jpegs(child)
    ]
    if not scene_dirs:
        raise FileNotFoundError(f"no JPEG frames found under {data_dir}")
    return scene_dirs


def iter_scene_frames(
    scene_dir: Path,
    *,
    camera: str,
    fps: float,
    head_motion: str,
) -> list[ReplayFrame]:
    if fps <= 0:
        raise ValueError("fps must be positive")

    paths = _sorted_jpeg_paths(Path(scene_dir))
    if not paths:
        raise FileNotFoundError(f"no JPEG frames found in {scene_dir}")

    frames: list[ReplayFrame] = []
    fallback_step_ms = 1000.0 / fps
    for frame_id, path in enumerate(paths):
        timestamp_ms = _timestamp_ms_from_filename(path)
        if timestamp_ms is None:
            timestamp_ms = int(round(frame_id * fallback_step_ms))
        frames.append(
            ReplayFrame(
                path=path,
                header={
                    "type": "frame",
                    "schema_version": SCHEMA_VERSION,
                    "camera": camera,
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "encoding": "jpeg",
                    "width": DEFAULT_WIDTH,
                    "height": DEFAULT_HEIGHT,
                    "head_motion": {"state": head_motion},
                },
            )
        )
    return frames


async def replay_scene(
    *,
    server: str,
    scene_dir: Path,
    camera: str,
    fps: float,
    head_motion: str,
    save_jsonl: Path | None = None,
    append_jsonl: bool = False,
    connector: Callable[..., Any] | None = None,
    realtime: bool = True,
    response_timeout_ms: int | None = None,
) -> ReplayStats:
    frames = iter_scene_frames(
        scene_dir,
        camera=camera,
        fps=fps,
        head_motion=head_motion,
    )
    connect = connector or _default_connector()
    interval_s = 1.0 / fps
    start_s = time.perf_counter()
    frames_sent = 0
    frames_ok = 0
    errors = 0
    frames_with_person = 0
    frame_id_mismatch = 0

    jsonl_file = None
    try:
        if save_jsonl is not None:
            save_jsonl.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append_jsonl else "w"
            jsonl_file = save_jsonl.open(mode, encoding="utf-8")

        async with connect(server, max_size=None) as websocket:
            for frame in frames:
                frame_started_s = time.perf_counter()
                jpeg_bytes = frame.path.read_bytes()
                payload = encode_frame_message(frame.header, jpeg_bytes)
                await websocket.send(payload)
                frames_sent += 1

                try:
                    raw_response = await _recv_with_timeout(
                        websocket,
                        response_timeout_ms=response_timeout_ms,
                    )
                except TimeoutError:
                    errors += 1
                    if jsonl_file is not None:
                        jsonl_file.write(
                            json.dumps(
                                {
                                    "scene": Path(scene_dir).name,
                                    "frame_id": frame.header["frame_id"],
                                    "latency_ms": (
                                        time.perf_counter() - frame_started_s
                                    )
                                    * 1000.0,
                                    "response": {
                                        "type": "error",
                                        "code": "response_timeout",
                                    },
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    break
                latency_ms = (time.perf_counter() - frame_started_s) * 1000.0
                response = _decode_response(raw_response)
                if response.get("type") == "visual_state":
                    frames_ok += 1
                    scene_flags = response.get("scene_flags", {})
                    if isinstance(scene_flags, dict) and scene_flags.get("has_person"):
                        frames_with_person += 1
                    if response.get("frame_id") != frame.header["frame_id"]:
                        frame_id_mismatch += 1
                else:
                    errors += 1

                if jsonl_file is not None:
                    jsonl_file.write(
                        json.dumps(
                            {
                                "scene": Path(scene_dir).name,
                                "frame_id": frame.header["frame_id"],
                                "latency_ms": latency_ms,
                                "response": response,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

                if realtime:
                    elapsed = time.perf_counter() - frame_started_s
                    if elapsed < interval_s:
                        await asyncio.sleep(interval_s - elapsed)
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    return ReplayStats(
        scene=Path(scene_dir).name,
        frames_sent=frames_sent,
        frames_ok=frames_ok,
        errors=errors,
        elapsed_s=time.perf_counter() - start_s,
        frames_with_person=frames_with_person,
        frame_id_mismatch=frame_id_mismatch,
    )


async def replay_data_dir(
    *,
    server: str,
    data_dir: Path,
    camera: str,
    fps: float,
    head_motion: str,
    save_jsonl: Path | None,
    realtime: bool = True,
    response_timeout_ms: int | None = None,
) -> list[ReplayStats]:
    scene_dirs = discover_scene_dirs(data_dir)
    append_jsonl = False
    if save_jsonl is not None and len(scene_dirs) > 1:
        save_jsonl.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl.write_text("", encoding="utf-8")
        append_jsonl = True

    stats: list[ReplayStats] = []
    for scene_dir in scene_dirs:
        stats.append(
            await replay_scene(
                server=server,
                scene_dir=scene_dir,
                camera=camera,
                fps=fps,
                head_motion=head_motion,
                save_jsonl=save_jsonl,
                append_jsonl=append_jsonl,
                realtime=realtime,
                response_timeout_ms=response_timeout_ms,
            )
        )
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay val-data JPEG scenes into visual-events-server."
    )
    parser.add_argument("--server", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--head-motion",
        choices=("stationary", "moving", "unknown"),
        default="stationary",
    )
    parser.add_argument("--save-jsonl", type=Path)
    parser.add_argument(
        "--response-timeout-ms",
        type=int,
        default=None,
        help="Stop waiting for a frame response after this many milliseconds.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Write per-scene replay summary JSON.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send the next frame as soon as a response arrives.",
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = await replay_data_dir(
        server=args.server,
        data_dir=args.data_dir,
        camera=args.camera,
        fps=args.fps,
        head_motion=args.head_motion,
        save_jsonl=args.save_jsonl,
        realtime=not args.no_realtime,
        response_timeout_ms=args.response_timeout_ms,
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                [_stats_to_summary(item) for item in stats],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    for item in stats:
        print(
            json.dumps(
                _stats_to_summary(item),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return 0 if all(_stats_passed(item) for item in stats) else 1


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))


def _has_jpegs(path: Path) -> bool:
    return path.is_dir() and any(match for glob in _JPEG_GLOBS for match in path.glob(glob))


def _sorted_jpeg_paths(path: Path) -> list[Path]:
    paths = [match for glob in _JPEG_GLOBS for match in path.glob(glob)]
    return sorted(paths, key=lambda item: (_filename_sort_key(item), item.name))


def _filename_sort_key(path: Path) -> int:
    timestamp_ms = _timestamp_ms_from_filename(path)
    return timestamp_ms if timestamp_ms is not None else 0


def _timestamp_ms_from_filename(path: Path) -> int | None:
    match = _FILENAME_NUMBER.search(path.stem)
    if match is None:
        return None
    return int(match.group(1)) // 1_000_000


def _decode_response(raw_response: str | bytes) -> dict[str, Any]:
    if isinstance(raw_response, bytes):
        raw_response = raw_response.decode("utf-8")
    response = json.loads(raw_response)
    if not isinstance(response, dict):
        raise ValueError("server response must be a JSON object")
    return response


async def _recv_with_timeout(
    websocket: Any,
    *,
    response_timeout_ms: int | None,
) -> str | bytes:
    if response_timeout_ms is None:
        return await websocket.recv()
    if response_timeout_ms <= 0:
        raise ValueError("response_timeout_ms must be positive")
    try:
        return await asyncio.wait_for(
            websocket.recv(),
            timeout=response_timeout_ms / 1000.0,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError("timed out waiting for server response") from exc


def _stats_to_summary(item: ReplayStats) -> dict[str, Any]:
    return {
        "scene": item.scene,
        "frames_sent": item.frames_sent,
        "frames_ok": item.frames_ok,
        "errors": item.errors,
        "ok_rate": item.ok_rate,
        "frames_with_person": item.frames_with_person,
        "person_frame_rate": item.person_frame_rate,
        "frame_id_mismatch": item.frame_id_mismatch,
        "elapsed_s": item.elapsed_s,
    }


def _stats_passed(item: ReplayStats) -> bool:
    return item.errors == 0 and item.frame_id_mismatch == 0


def _default_connector() -> Callable[..., Any]:
    import websockets

    return websockets.connect


if __name__ == "__main__":
    main()
