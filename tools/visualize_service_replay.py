from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

import websockets

from visual_events_server.protocol import encode_frame_message


JPEG_GLOBS = ("*.jpg", "*.jpeg")
DEFAULT_OUT = Path("artifacts/visual-debug")
_EVIDENCE_KEYS = (
    "runtime_person_slot",
    "visible_duration_ms",
    "lost_duration_ms",
    "wave_duration_ms",
    "passing_speed_class",
    "dx_ratio",
    "avg_vx_px_s",
    "bbox_area_ratio",
    "area_growth_ratio",
    "stationary_duration_ms",
    "previous_track_id",
    "target_track_id",
    "switch_reason",
)
_MEMORY_EVENTS = {
    "known_person_present",
    "scene_activated",
    "familiar_unknown_present",
}


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

            annotated = _draw_visual_state(image, response)
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


def _draw_visual_state(image: Any, state: dict[str, Any]) -> Any:
    import cv2

    canvas = image.copy()
    tracks = state.get("tracks")
    if isinstance(tracks, list):
        for track in tracks:
            if isinstance(track, dict):
                _draw_track(canvas, track)

    attention = state.get("attention")
    if isinstance(attention, dict):
        target_uv = attention.get("target_uv")
        if _is_point(target_uv):
            _draw_cross(canvas, target_uv, color=(0, 255, 255), radius=16, thickness=2)
            _put_label(
                canvas,
                f"attention track={attention.get('target_track_id', '-')}",
                (int(target_uv[0]) + 8, int(target_uv[1]) - 12),
                color=(0, 255, 255),
            )

    _draw_header(canvas, state)
    _draw_events(canvas, state)
    return canvas


def _draw_track(canvas: Any, track: dict[str, Any]) -> None:
    import cv2

    bbox = track.get("bbox_xyxy")
    if not _is_bbox(bbox):
        return
    track_id = int(track.get("track_id", -1))
    color = _track_color(track_id)
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    height, width = canvas.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    confidence = _fmt_float(track.get("confidence"))
    label = f"id={track_id} conf={confidence} lost={int(track.get('lost_ms', 0))}ms"
    _put_label(canvas, label, (x1, max(18, y1 - 8)), color=color)

    head_uv = track.get("head_uv")
    if _is_point(head_uv):
        point = (int(round(float(head_uv[0]))), int(round(float(head_uv[1]))))
        cv2.circle(canvas, point, 5, color, -1)
        cv2.circle(canvas, point, 8, (255, 255, 255), 1)


def _draw_header(canvas: Any, state: dict[str, Any]) -> None:
    lines = [
        f"frame={state.get('frame_id', '-')}",
        f"camera={state.get('camera', '-')}",
    ]
    scene_flags = state.get("scene_flags")
    if isinstance(scene_flags, dict):
        lines.append(
            "persons="
            f"{scene_flags.get('person_count', '-')}"
            f" stable={scene_flags.get('largest_person_stable', '-')}"
        )
    scene_summary = _scene_context_summary(state.get("scene_context"))
    if scene_summary != "-":
        lines.append(_clip(scene_summary, 116))
    reacquire_summary = _reacquire_summary(state.get("scene_context"))
    if reacquire_summary != "-":
        lines.append(_clip(reacquire_summary, 116))
    _draw_text_block(canvas, lines, origin=(10, 24))


def _draw_events(canvas: Any, state: dict[str, Any]) -> None:
    events = state.get("semantic_events")
    if not isinstance(events, list) or not events:
        return
    lines = []
    for event in events[:4]:
        if isinstance(event, dict):
            lines.append(_clip(_event_summary(event), 116))
    if lines:
        height = canvas.shape[0]
        _draw_text_block(canvas, lines, origin=(10, max(24, height - 24 * len(lines) - 8)))


def _draw_text_block(canvas: Any, lines: list[str], *, origin: tuple[int, int]) -> None:
    import cv2

    if not lines:
        return
    x, y = origin
    line_height = 22
    max_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for line in lines)
    cv2.rectangle(
        canvas,
        (x - 6, y - 18),
        (x + max_width + 8, y + line_height * (len(lines) - 1) + 8),
        (0, 0, 0),
        -1,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _put_label(
    canvas: Any,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
) -> None:
    import cv2

    x, y = origin
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(canvas, (x - 3, y - size[1] - 5), (x + size[0] + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_cross(
    canvas: Any,
    point: list[Any] | tuple[Any, Any],
    *,
    color: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    import cv2

    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    cv2.line(canvas, (x - radius, y), (x + radius, y), color, thickness)
    cv2.line(canvas, (x, y - radius), (x, y + radius), color, thickness)
    cv2.circle(canvas, (x, y), radius, color, thickness)


def _track_color(track_id: int) -> tuple[int, int, int]:
    palette = (
        (80, 220, 255),
        (80, 255, 140),
        (255, 170, 80),
        (255, 100, 190),
        (180, 140, 255),
        (120, 210, 120),
    )
    return palette[abs(track_id) % len(palette)]


def _is_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
    )


def _is_point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "-"


def _scene_context_summary(scene_context: Any) -> str:
    if not isinstance(scene_context, dict):
        return "-"

    parts: list[str] = []
    engagement = scene_context.get("engagement_state")
    if engagement:
        parts.append(f"engagement={_short(engagement)}")
    reasons = _reasons_summary(scene_context.get("no_engage_reasons"))
    if reasons:
        parts.append(f"reasons={reasons}")
    return " ".join(parts) if parts else "-"


def _reacquire_summary(scene_context: Any) -> str:
    if not isinstance(scene_context, dict):
        return "-"
    target_reacquired = scene_context.get("target_reacquired")
    if not isinstance(target_reacquired, dict) or not target_reacquired:
        return "-"

    old_track = _short(target_reacquired.get("reacquired_from_track_id", "-"))
    new_track = _short(target_reacquired.get("reacquired_to_track_id", "-"))
    elapsed_ms = _short(target_reacquired.get("reacquire_elapsed_ms", "-"))
    return f"reacq {old_track}->{new_track} elapsed_ms={elapsed_ms}"


def _event_summary(event: Any) -> str:
    if not isinstance(event, dict):
        return "-"
    event_name = _short(event.get("event", "-"))
    track = _short(event.get("track_id", "-"))
    if event.get("event") in _MEMORY_EVENTS:
        return f"{event_name} track={track} memory={_memory_event_summary(event)}"
    evidence = _evidence_summary(event)
    return f"{event_name} track={track} evidence={evidence}"


def _memory_event_summary(event: Any) -> str:
    if not isinstance(event, dict):
        return "-"
    evidence = event.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}

    items: list[str] = []
    if "matched_id" in evidence:
        items.append(f"matched_id={_short(evidence['matched_id'])}")

    label = _memory_label(event.get("memory_context"))
    if label:
        items.append(label)

    for key in ("match_score", "top2_margin", "memory_match_id"):
        if key in evidence:
            items.append(f"{key}={_short(evidence[key])}")

    return " ".join(items) if items else "-"


def _memory_label(memory_context: Any) -> str:
    if not isinstance(memory_context, dict):
        return ""

    person = memory_context.get("person")
    if isinstance(person, dict) and person.get("display_name"):
        return f"name={_short(person['display_name'])}"

    scene = memory_context.get("scene")
    if isinstance(scene, dict) and scene.get("title"):
        return f"title={_short(scene['title'])}"

    anonymous = memory_context.get("anonymous_person")
    if isinstance(anonymous, dict):
        for key in ("display_name", "name", "title"):
            if anonymous.get(key):
                return f"name={_short(anonymous[key])}"

    return ""


def _evidence_summary(event: Any) -> str:
    if not isinstance(event, dict) or not isinstance(event.get("evidence"), dict):
        return "-"
    evidence = event["evidence"]

    items: list[str] = []
    if "runtime_person_slot" in evidence:
        items.append(f"runtime_person_slot={_short(evidence['runtime_person_slot'])}")

    if (
        "reacquired_from_track_id" in evidence
        and "reacquired_to_track_id" in evidence
    ):
        items.append(
            "reacq="
            f"{_short(evidence['reacquired_from_track_id'])}"
            f"->{_short(evidence['reacquired_to_track_id'])}"
        )

    if "reacquire_elapsed_ms" in evidence:
        items.append(f"reacquire_elapsed_ms={_short(evidence['reacquire_elapsed_ms'])}")

    for key in _EVIDENCE_KEYS:
        if len(items) >= 4:
            break
        if key in evidence and not any(item.startswith(f"{key}=") for item in items):
            items.append(f"{key}={_short(evidence[key])}")

    return " ".join(items) if items else "-"


def _reasons_summary(value: Any) -> str:
    if isinstance(value, list):
        reasons = [_short(item, 28) for item in value[:3] if item]
        if len(value) > 3:
            reasons.append(f"+{len(value) - 3}")
        return ",".join(reasons)
    if not value:
        return ""
    return _short(value, 40)


def _short(value: Any, max_chars: int = 36) -> str:
    if value is None:
        text = "-"
    elif isinstance(value, float):
        text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    elif isinstance(value, list):
        text = "[" + ",".join(_short(item, 12) for item in value[:3])
        text += ",...]" if len(value) > 3 else "]"
    else:
        text = str(value)
    return _clip(text, max_chars)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _engagement_state(scene_context: Any) -> str:
    if not isinstance(scene_context, dict):
        return "-"
    engagement = scene_context.get("engagement_state")
    return _short(engagement, 32) if engagement else "-"


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
    cards = "\n".join(_render_card(args.out, result) for result in results)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>visual-events replay debug</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 20px; background: #111; color: #eee; }}
    a {{ color: #8bd3ff; }}
    .meta {{ color: #bbb; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }}
    .card {{ background: #1b1b1b; border: 1px solid #333; padding: 10px; }}
    .card img {{ width: 100%; display: block; }}
    .caption {{ font-size: 13px; color: #ccc; margin: 8px 0; }}
    pre {{ white-space: pre-wrap; max-height: 260px; overflow: auto; background: #0a0a0a; padding: 8px; }}
  </style>
</head>
<body>
  <h1>visual-events replay debug</h1>
  <div class="meta">
    server: {html.escape(str(args.server))}<br>
    scene: {html.escape(str(args.scene))}<br>
    frames: {len(results)}<br>
    visual_state jsonl: {html.escape(_rel(args.out, jsonl_path))}
  </div>
  <div class="grid">
    {cards}
  </div>
</body>
</html>
"""


def _render_card(root: Path, result: dict[str, Any]) -> str:
    response = result["response"]
    events = response.get("semantic_events")
    event_summaries = []
    if isinstance(events, list):
        event_summaries = [_event_summary(event) for event in events if isinstance(event, dict)]
    attention = response.get("attention")
    attention_text = "-"
    if isinstance(attention, dict):
        attention_text = str(attention.get("target_track_id", "-"))
    scene_summary = _scene_context_summary(response.get("scene_context"))
    reacquire_summary = _reacquire_summary(response.get("scene_context"))
    state_json = json.dumps(response, ensure_ascii=False, indent=2)
    return f"""<div class="card">
  <img src="{html.escape(_rel(root, result["output_image"]))}" alt="frame {result["index"]}">
  <div class="caption">
    #{result["index"]} {html.escape(result["source"].name)}
    | tracks={_list_len(response.get("tracks"))}
    | attention={html.escape(attention_text)}
    | scene={html.escape(scene_summary)}
    | reacq={html.escape(reacquire_summary)}
    | events={html.escape("; ".join(event_summaries) if event_summaries else "-")}
    | <a href="{html.escape(_rel(root, result["output_state"]))}">json</a>
  </div>
  <details><summary>visual_state</summary><pre>{html.escape(state_json)}</pre></details>
</div>"""


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
