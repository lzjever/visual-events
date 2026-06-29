from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import websockets

from tools.visual_evidence_helpers import (
    _engagement_state,
    _list_len,
    draw_visual_state,
    render_html_document,
)
from visual_events_server.protocol import encode_frame_message


JPEG_GLOBS = ("*.jpg", "*.jpeg")
DEFAULT_OUT = Path("artifacts/visual-debug")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay JPEG frames through visual-events-server and draw visual_state overlays."
    )
    parser.add_argument("--server", required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8765/v1/stream")
    parser.add_argument("--scene", required=True, type=Path, help="Directory containing JPEG frames")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--head-motion",
        choices=("stationary", "moving", "unknown"),
        default="stationary",
    )
    parser.add_argument("--response-timeout-s", type=float, default=10.0)
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")

    frames = _find_jpegs(args.scene)
    if not frames:
        raise SystemExit(f"no JPEG frames found in {args.scene}")
    frames = frames[:: args.stride][: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    frame_out_dir = args.out / "frames"
    state_out_dir = args.out / "states"
    frame_out_dir.mkdir(parents=True, exist_ok=True)
    state_out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    async with websockets.connect(args.server, max_size=8 * 1024 * 1024) as websocket:
        for index, image_path in enumerate(frames, start=1):
            jpeg = image_path.read_bytes()
            image = _decode_jpeg(jpeg)
            height, width = image.shape[:2]
            header = {
                "type": "frame",
                "schema_version": 1,
                "camera": args.camera,
                "frame_id": index,
                "timestamp_ms": _timestamp_ms_from_path(image_path, fallback_ms=index * 100),
                "encoding": "jpeg",
                "width": int(width),
                "height": int(height),
                "head_motion": {
                    "state": args.head_motion,
                    "yaw_vel_rad_s": 0.0,
                    "pitch_vel_rad_s": 0.0,
                },
            }
            await websocket.send(encode_frame_message(header, jpeg))
            raw_response = await asyncio.wait_for(
                websocket.recv(),
                timeout=float(args.response_timeout_s),
            )
            response = _decode_response(raw_response)

            annotated = draw_visual_state(
                image,
                response,
                scene=str(args.scene),
                frame_id=index,
            )
            output_name = f"{index:06d}_{image_path.stem}.jpg"
            output_image = frame_out_dir / output_name
            output_state = state_out_dir / f"{index:06d}_{image_path.stem}.json"
            _write_jpeg(output_image, annotated)
            output_state.write_text(
                json.dumps(response, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            results.append(
                {
                    "index": index,
                    "source": image_path,
                    "output_image": output_image,
                    "output_state": output_state,
                    "response": response,
                }
            )
            print(_progress_line(index, len(frames), image_path, response), flush=True)

    jsonl_path = args.out / "visual_state.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result["response"], ensure_ascii=False) + "\n")

    html_path = args.out / "index.html"
    html_path.write_text(_render_html(args, results, jsonl_path), encoding="utf-8")
    print(f"wrote {html_path}")
    return 0


def _find_jpegs(scene_dir: Path) -> list[Path]:
    if not scene_dir.is_dir():
        raise SystemExit(f"scene is not a directory: {scene_dir}")
    frames: list[Path] = []
    for pattern in JPEG_GLOBS:
        frames.extend(scene_dir.glob(pattern))
    return sorted(frames)


def _decode_jpeg(jpeg: bytes) -> Any:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - local tooling guard
        raise SystemExit(
            "OpenCV is required. Run with `uv run --extra inference ...`."
        ) from exc
    array = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("failed to decode JPEG frame")
    return image


def _write_jpeg(path: Path, image: Any) -> None:
    import cv2

    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise SystemExit(f"failed to write {path}")


def _decode_response(raw_response: str | bytes) -> dict[str, Any]:
    if isinstance(raw_response, bytes):
        raw_response = raw_response.decode("utf-8")
    response = json.loads(raw_response)
    if not isinstance(response, dict):
        raise ValueError("server response must be a JSON object")
    return response


def _timestamp_ms_from_path(path: Path, *, fallback_ms: int) -> int:
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        return int(fallback_ms)
    value = int(numbers[-1])
    if value > 10**15:
        return value // 1_000_000
    return value


def _progress_line(
    index: int,
    total: int,
    image_path: Path,
    response: dict[str, Any],
) -> str:
    if response.get("type") == "error":
        return f"[{index}/{total}] {image_path.name}: error {response.get('code')}"
    tracks = response.get("tracks")
    attention = response.get("attention")
    events = response.get("semantic_events")
    return (
        f"[{index}/{total}] {image_path.name}: "
        f"tracks={_list_len(tracks)} "
        f"attention={attention.get('target_track_id') if isinstance(attention, dict) else '-'} "
        f"engagement={_engagement_state(response.get('scene_context'))} "
        f"events={_list_len(events)}"
    )


def _render_html(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    jsonl_path: Path,
) -> str:
    frames = [
        {
            "frame_id": result["index"],
            "scene": str(args.scene),
            "source_name": result["source"].name,
            "image_path": result["output_image"],
            "state_path": result["output_state"],
            "state": result["response"],
        }
        for result in results
    ]
    return render_html_document(
        root=args.out,
        server=args.server,
        scene=args.scene,
        frames=frames,
        jsonl_path=jsonl_path,
    )


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
