from __future__ import annotations

import argparse
import asyncio
import html
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.replay_val_data import (
    discover_scene_dirs,
    iter_scene_frames,
    replay_data_dir,
)
from tools.visual_evidence_helpers import (
    _event_summary,
    draw_visual_state,
    render_frame_card,
)


DEFAULT_OUT = Path("artifacts/visual-evidence")
WRAPPED_JSONL_KEYS = ("scene", "frame_id", "latency_ms", "response")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate artifact-first visual evidence pages from replay JSONL."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--visual-state-jsonl", type=Path)
    parser.add_argument(
        "--replay-artifact",
        type=Path,
        help="Directory containing visual_state.jsonl from tools.replay_val_data.",
    )
    parser.add_argument(
        "--run-replay",
        action="store_true",
        help="Run tools.replay_val_data.replay_data_dir before generating evidence.",
    )
    parser.add_argument("--server", help="WebSocket URL used only with --run-replay.")
    parser.add_argument("--camera", default="front")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--head-motion",
        choices=("stationary", "moving", "unknown"),
        default="stationary",
    )
    parser.add_argument(
        "--response-timeout-ms",
        type=int,
        default=None,
        help="Replay response timeout used only with --run-replay.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send replay frames as fast as responses arrive.",
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    await generate_visual_evidence_from_args(args)
    return 0


async def generate_visual_evidence_from_args(args: argparse.Namespace) -> dict[str, Any]:
    _reject_out_inside_data_dir(args.out, args.data_dir)
    jsonl_path = _input_jsonl_path(args)
    if args.run_replay:
        if not args.server:
            raise SystemExit("--server is required with --run-replay")
        jsonl_path = args.out / "visual_state.jsonl"
        await replay_data_dir(
            server=args.server,
            data_dir=args.data_dir,
            camera=args.camera,
            fps=args.fps,
            head_motion=args.head_motion,
            save_jsonl=jsonl_path,
            realtime=not args.no_realtime,
            response_timeout_ms=args.response_timeout_ms,
            continue_on_timeout=True,
        )

    records = read_wrapped_visual_state_jsonl(jsonl_path)
    source_images = map_source_images(
        args.data_dir,
        records,
        camera=args.camera,
        fps=args.fps,
        head_motion=args.head_motion,
    )
    return generate_visual_evidence(
        records=records,
        source_images=source_images,
        out=args.out,
        input_jsonl=jsonl_path,
    )


def read_wrapped_visual_state_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"visual_state.jsonl not found: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise SystemExit(f"{path}:{line_number}: expected wrapped JSON object")
        missing = [key for key in WRAPPED_JSONL_KEYS if key not in record]
        if missing:
            if record.get("type") == "visual_state":
                raise SystemExit(
                    f"{path}:{line_number}: raw visual_state-only JSONL is not "
                    "accepted; "
                    "expected wrapped JSONL produced by tools.replay_val_data with "
                    "keys scene, frame_id, latency_ms, response"
                )
            raise SystemExit(
                f"{path}:{line_number}: expected wrapped JSONL produced by "
                f"tools.replay_val_data; missing keys: {', '.join(missing)}"
            )
        if not isinstance(record["scene"], str) or not record["scene"]:
            raise SystemExit(
                f"{path}:{line_number}: wrapped record scene must be a non-empty string"
            )
        if not isinstance(record["frame_id"], int) or isinstance(
            record["frame_id"],
            bool,
        ):
            raise SystemExit(
                f"{path}:{line_number}: wrapped record frame_id must be an integer"
            )
        if not isinstance(record["latency_ms"], (int, float)) or isinstance(
            record["latency_ms"],
            bool,
        ):
            raise SystemExit(
                f"{path}:{line_number}: wrapped record latency_ms must be a number"
            )
        if not isinstance(record["response"], dict):
            raise SystemExit(
                f"{path}:{line_number}: wrapped record response must be an object"
            )
        records.append(record)

    if not records:
        raise SystemExit(f"no wrapped visual_state records found in {path}")
    return records


def map_source_images(
    data_dir: Path,
    records: list[dict[str, Any]],
    *,
    camera: str,
    fps: float,
    head_motion: str,
) -> dict[tuple[str, int], Path]:
    scene_frames: dict[str, list[Path]] = {}
    for scene_dir in discover_scene_dirs(data_dir):
        scene = scene_dir.name
        if scene in scene_frames:
            raise SystemExit(f"duplicate scene name from data-dir discovery: {scene}")
        scene_frames[scene] = [
            frame.path
            for frame in iter_scene_frames(
                scene_dir,
                camera=camera,
                fps=fps,
                head_motion=head_motion,
            )
        ]

    mapped: dict[tuple[str, int], Path] = {}
    for record in records:
        scene = record["scene"]
        frame_id = record["frame_id"]
        frames = scene_frames.get(scene)
        if frames is None:
            raise SystemExit(
                f"visual_state references unknown scene {scene!r}; "
                "scene names must come from tools.replay_val_data.discover_scene_dirs()"
            )
        if frame_id < 0 or frame_id >= len(frames):
            raise SystemExit(
                f"visual_state references frame_id {frame_id} outside scene {scene!r} "
                f"range 0..{len(frames) - 1}"
            )
        mapped[(scene, frame_id)] = frames[frame_id]
    return mapped


def generate_visual_evidence(
    *,
    records: list[dict[str, Any]],
    source_images: dict[tuple[str, int], Path],
    out: Path,
    input_jsonl: Path,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    scenes_dir = out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    scene_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        scene_records.setdefault(record["scene"], []).append(record)

    root_frames: list[dict[str, Any]] = []
    for scene, items in sorted(scene_records.items()):
        root_frames.extend(_write_scene(out, scenes_dir, scene, items, source_images))

    summary = summarize_records(records)
    _write_json(out / "summary.json", summary)
    _write_wrapped_jsonl(out / "visual_state.jsonl", records)
    (out / "index.html").write_text(
        _render_root_html(
            out=out,
            summary=summary,
            frames=root_frames,
            input_jsonl=input_jsonl,
        ),
        encoding="utf-8",
    )
    return summary


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    scene_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        scene_groups.setdefault(record["scene"], []).append(record)

    scene_keyframes = {
        scene: _scene_keyframes(items)
        for scene, items in sorted(scene_groups.items())
    }
    scene_summaries = {}
    for scene, items in sorted(scene_groups.items()):
        scene_summary = _summary_for_records(items)
        scene_summary["keyframes"] = scene_keyframes[scene]
        scene_summaries[scene] = scene_summary

    return {
        **_summary_for_records(records, include_scene_in_event_first_frame=True),
        "keyframes": scene_keyframes,
        "scenes": scene_summaries,
    }


def _summary_for_records(
    records: list[dict[str, Any]],
    *,
    include_scene_in_event_first_frame: bool = False,
) -> dict[str, Any]:
    frames_total = len(records)
    frames_ok = 0
    errors = 0
    frames_with_person = 0
    max_person_count = 0
    track_ids: set[int] = set()
    tracks_per_frame: list[int] = []
    attention_available_frames = 0
    attention_unavailable_frames = 0
    attention_target_switches = 0
    last_attention_target: int | None = None
    has_last_attention_target = False
    semantic_event_total = 0
    event_counts: dict[str, int] = {}
    first_frame_by_type: dict[str, int | dict[str, Any]] = {}

    for record in records:
        response = record["response"]
        is_visual_state = response.get("type") == "visual_state"
        if is_visual_state:
            frames_ok += 1
        else:
            errors += 1

        person_count = _person_count(response)
        max_person_count = max(max_person_count, person_count)
        if person_count > 0:
            frames_with_person += 1

        tracks = response.get("tracks")
        track_count = len(tracks) if isinstance(tracks, list) else 0
        tracks_per_frame.append(track_count)
        if isinstance(tracks, list):
            for track in tracks:
                if isinstance(track, dict) and _is_int(track.get("track_id")):
                    track_ids.add(int(track["track_id"]))

        if is_visual_state and _has_available_attention(response.get("attention")):
            attention_available_frames += 1
            target = _attention_target_id(response.get("attention"))
            if target is not None:
                if has_last_attention_target and target != last_attention_target:
                    attention_target_switches += 1
                last_attention_target = target
                has_last_attention_target = True
        elif is_visual_state:
            attention_unavailable_frames += 1

        events = response.get("semantic_events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("event", "-"))
                semantic_event_total += 1
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                if event_type not in first_frame_by_type:
                    frame_id = int(record["frame_id"])
                    first_frame_by_type[event_type] = (
                        {"scene": record["scene"], "frame_id": frame_id}
                        if include_scene_in_event_first_frame
                        else frame_id
                    )

    return {
        "frames_total": frames_total,
        "frames_ok": frames_ok,
        "errors": errors,
        "person": {
            "frames_with_person": frames_with_person,
            "person_frame_ratio": _ratio(frames_with_person, frames_total),
            "max_person_count": max_person_count,
        },
        "tracking": {
            "unique_track_count": len(track_ids),
            "track_ids": sorted(track_ids),
            "max_tracks_per_frame": max(tracks_per_frame, default=0),
            "avg_tracks_per_frame": (
                sum(tracks_per_frame) / len(tracks_per_frame)
                if tracks_per_frame
                else 0.0
            ),
        },
        "attention": {
            "available_frames": attention_available_frames,
            "available_ratio": _ratio(attention_available_frames, frames_ok),
            "null_frames": attention_unavailable_frames,
            "target_switches": attention_target_switches,
        },
        "semantic_events": {
            "total": semantic_event_total,
            "counts_by_type": dict(sorted(event_counts.items())),
            "first_frame_by_type": dict(sorted(first_frame_by_type.items())),
        },
    }


def _scene_keyframes(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame_ids = [int(record["frame_id"]) for record in records]
    first_frame = min(frame_ids) if frame_ids else None
    last_frame = max(frame_ids) if frame_ids else None
    first_person_frame: int | None = None
    first_attention_frame: int | None = None
    first_event_frame_by_type: dict[str, int] = {}

    for record in sorted(records, key=lambda item: int(item["frame_id"])):
        frame_id = int(record["frame_id"])
        response = record["response"]
        if response.get("type") != "visual_state":
            continue

        if first_person_frame is None and _person_count(response) > 0:
            first_person_frame = frame_id
        if first_attention_frame is None and _has_available_attention(
            response.get("attention")
        ):
            first_attention_frame = frame_id

        events = response.get("semantic_events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("event", "-"))
                first_event_frame_by_type.setdefault(event_type, frame_id)

    return {
        "first_frame": first_frame,
        "first_person_frame": first_person_frame,
        "first_attention_frame": first_attention_frame,
        "first_event_frame_by_type": dict(sorted(first_event_frame_by_type.items())),
        "last_frame": last_frame,
    }


def _write_scene(
    out: Path,
    scenes_dir: Path,
    scene: str,
    records: list[dict[str, Any]],
    source_images: dict[tuple[str, int], Path],
) -> list[dict[str, Any]]:
    scene_dir = scenes_dir / scene
    frame_dir = scene_dir / "frames"
    state_dir = scene_dir / "states"
    frame_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    for record in records:
        frame_id = record["frame_id"]
        source = source_images[(scene, frame_id)]
        image = _decode_image(source)
        annotated = draw_visual_state(
            image,
            record["response"],
            scene=scene,
            frame_id=frame_id,
        )
        output_image = frame_dir / f"{frame_id:06d}.jpg"
        output_state = state_dir / f"{frame_id:06d}.json"
        _write_jpeg(output_image, annotated)
        _write_json(output_state, record["response"])

        frames.append(
            {
                "frame_id": frame_id,
                "scene": scene,
                "source": source,
                "source_name": (
                    f"{source.name} path={source.as_posix()} "
                    f"latency_ms={record['latency_ms']}"
                ),
                "image_path": output_image,
                "state_path": output_state,
                "state": record["response"],
                "latency_ms": record["latency_ms"],
            }
        )

    scene_summary = _summary_for_records(records)
    scene_summary["keyframes"] = _scene_keyframes(records)
    _write_json(scene_dir / "summary.json", scene_summary)
    _write_wrapped_jsonl(scene_dir / "visual_state.jsonl", records)
    (scene_dir / "index.html").write_text(
        _render_scene_html(
            root=scene_dir,
            scene=scene,
            frames=frames,
            summary=scene_summary,
        ),
        encoding="utf-8",
    )
    return frames


def _decode_image(path: Path) -> Any:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - local tooling guard
        raise SystemExit(
            "OpenCV is required to draw visual evidence. Run with `uv run --extra inference ...`."
        ) from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"failed to decode source image: {path}")
    return image


def _write_jpeg(path: Path, image: Any) -> None:
    import cv2

    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise SystemExit(f"failed to write {path}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_wrapped_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _render_scene_html(
    *,
    root: Path,
    scene: str,
    frames: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    cards = "\n".join(
        render_frame_card(root, frame, anchor=f"frame-{frame['frame_id']}")
        for frame in frames
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>visual evidence - {html.escape(scene)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>{html.escape(scene)}</h1>
  <p class="meta">
    frames: {summary["frames_total"]} |
    ok: {summary["frames_ok"]} |
    errors: {summary["errors"]} |
    <a href="summary.json">summary.json</a> |
    <a href="visual_state.jsonl">visual_state.jsonl</a>
  </p>
  <div class="grid">
    {cards}
  </div>
</body>
</html>
"""


def _render_root_html(
    *,
    out: Path,
    summary: dict[str, Any],
    frames: list[dict[str, Any]],
    input_jsonl: Path,
) -> str:
    scenes_table = _scenes_table_html(summary["scenes"])
    keyframes = _keyframes_html(summary["keyframes"])
    timeline = _semantic_event_timeline(out, frames)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>visual evidence</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>visual evidence</h1>
  <p class="meta">
    source jsonl: {html.escape(_rel(out, input_jsonl))}<br>
    frames: {summary["frames_total"]} |
    ok: {summary["frames_ok"]} |
    errors: {summary["errors"]} |
    person_frames: {summary["person"]["frames_with_person"]} |
    tracks: {summary["tracking"]["unique_track_count"]} |
    events: {summary["semantic_events"]["total"]}<br>
    <a href="summary.json">summary.json</a> |
    <a href="visual_state.jsonl">visual_state.jsonl</a>
  </p>
  <h2>Scenes</h2>
  {scenes_table}
  <h2>Keyframes</h2>
  {keyframes}
  <h2>Semantic Event Timeline</h2>
  {timeline}
</body>
</html>
"""


def _scenes_table_html(scene_summaries: dict[str, dict[str, Any]]) -> str:
    if not scene_summaries:
        return '<p class="meta">scenes=none</p>'

    rows = []
    for scene, item in scene_summaries.items():
        person = item["person"]
        tracking = item["tracking"]
        attention = item["attention"]
        events = item["semantic_events"]
        rows.append(
            "    <tr>"
            f'<td><a href="scenes/{html.escape(scene)}/index.html">{html.escape(scene)}</a></td>'
            f'<td>{item["frames_total"]}</td>'
            f'<td>{item["frames_ok"]} / {item["errors"]}</td>'
            f'<td>{person["frames_with_person"]} / {person["max_person_count"]}</td>'
            f'<td>{_tracks_cell(tracking)}</td>'
            f'<td>{_attention_cell(attention)}</td>'
            f'<td>{_events_cell(events)}</td>'
            f'<td>{_scene_keyframes_cell(scene, item.get("keyframes"))}</td>'
            "</tr>"
        )

    return (
        '<table class="scenes">\n'
        "  <thead><tr>"
        "<th>Scene</th>"
        "<th>Frames</th>"
        "<th>OK / Errors</th>"
        "<th>Person Frames / Max</th>"
        "<th>Tracks</th>"
        "<th>Attention</th>"
        "<th>Events</th>"
        "<th>Keyframes</th>"
        "</tr></thead>\n"
        "  <tbody>\n"
        + "\n".join(rows)
        + "\n  </tbody>\n"
        "</table>"
    )


def _tracks_cell(tracking: dict[str, Any]) -> str:
    return (
        f'unique={tracking["unique_track_count"]} '
        f'ids={html.escape(_compact_list(tracking["track_ids"]))}'
    )


def _attention_cell(attention: dict[str, Any]) -> str:
    return (
        f'available={attention["available_frames"]} '
        f'null={attention["null_frames"]} '
        f'switches={attention["target_switches"]}'
    )


def _events_cell(events: dict[str, Any]) -> str:
    counts = events.get("counts_by_type")
    if not isinstance(counts, dict) or not counts:
        return f'total={events["total"]}'
    count_text = ", ".join(
        f"{html.escape(str(event_type))}={count}"
        for event_type, count in counts.items()
    )
    return f'total={events["total"]} {count_text}'


def _scene_keyframes_cell(scene: str, keyframes: Any) -> str:
    if not isinstance(keyframes, dict):
        return "-"
    links = []
    for label in ("first_frame", "first_person_frame", "first_attention_frame"):
        frame_id = keyframes.get(label)
        if frame_id is not None:
            links.append(_inline_keyframe_link(scene, label, int(frame_id)))
    first_events = keyframes.get("first_event_frame_by_type")
    if isinstance(first_events, dict):
        for event_type, frame_id in sorted(first_events.items()):
            if frame_id is not None:
                links.append(
                    _inline_keyframe_link(scene, f"{event_type}_event", int(frame_id))
                )
    frame_id = keyframes.get("last_frame")
    if frame_id is not None:
        links.append(_inline_keyframe_link(scene, "last_frame", int(frame_id)))
    return " ".join(links) if links else "-"


def _inline_keyframe_link(scene: str, label: str, frame_id: int) -> str:
    href = f"scenes/{scene}/index.html#frame-{frame_id}"
    return (
        f'<a href="{html.escape(href)}">'
        f"{html.escape(label)}={frame_id}"
        "</a>"
    )


def _compact_list(values: Any) -> str:
    if not isinstance(values, list):
        return "[]"
    return "[" + ",".join(str(value) for value in values) + "]"


def _keyframes_html(keyframes_by_scene: dict[str, dict[str, Any]]) -> str:
    items: list[str] = []
    for scene, keyframes in keyframes_by_scene.items():
        for label in ("first_frame", "first_person_frame", "first_attention_frame"):
            frame_id = keyframes.get(label)
            if frame_id is not None:
                items.append(_keyframe_link(scene, str(label), int(frame_id)))

        first_events = keyframes.get("first_event_frame_by_type")
        if isinstance(first_events, dict):
            for event_type, frame_id in sorted(first_events.items()):
                if frame_id is not None:
                    items.append(
                        _keyframe_link(
                            scene,
                            f"{event_type} first_event",
                            int(frame_id),
                        )
                    )

        frame_id = keyframes.get("last_frame")
        if frame_id is not None:
            items.append(_keyframe_link(scene, "last_frame", int(frame_id)))

    if not items:
        return '<p class="meta">keyframes=none</p>'
    return "<ul>\n" + "\n".join(items) + "\n</ul>"


def _keyframe_link(scene: str, label: str, frame_id: int) -> str:
    href = f"scenes/{scene}/index.html#frame-{frame_id}"
    return (
        "<li>"
        f'<a href="{html.escape(href)}">'
        f"{html.escape(scene)} {html.escape(label)}"
        "</a>"
        f" frame={frame_id}"
        "</li>"
    )


def _semantic_event_timeline(out: Path, frames: list[dict[str, Any]]) -> str:
    items: list[str] = []
    for frame in frames:
        events = frame["state"].get("semantic_events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            href = _rel(out, Path("scenes") / frame["scene"] / "index.html")
            href = f"{href}#frame-{frame['frame_id']}"
            items.append(
                "<li>"
                f'<a href="{html.escape(href)}">'
                f'{html.escape(frame["scene"])} '
                f'frame={html.escape(str(frame["frame_id"]))}'
                "</a> "
                f"{html.escape(_event_summary(event))}"
                "</li>"
            )
    if not items:
        return '<p class="meta">events=none</p>'
    return "<ol>\n" + "\n".join(items) + "\n</ol>"


def _input_jsonl_path(args: argparse.Namespace) -> Path:
    inputs = [
        bool(args.visual_state_jsonl),
        bool(args.replay_artifact),
        bool(args.run_replay),
    ]
    if sum(inputs) != 1:
        raise SystemExit(
            "choose exactly one input: --visual-state-jsonl, --replay-artifact, or --run-replay"
        )
    if args.visual_state_jsonl:
        return args.visual_state_jsonl
    if args.replay_artifact:
        return args.replay_artifact / "visual_state.jsonl"
    return args.out / "visual_state.jsonl"


def _reject_out_inside_data_dir(out: Path, data_dir: Path) -> None:
    data_root = data_dir.resolve()
    out_root = out.resolve(strict=False)
    if out_root == data_root or out_root.is_relative_to(data_root):
        raise SystemExit("--out must not be inside --data-dir")


def _person_count(response: dict[str, Any]) -> int:
    scene_flags = response.get("scene_flags")
    if isinstance(scene_flags, dict):
        value = scene_flags.get("person_count")
        if _is_int(value):
            return max(0, int(value))
        if scene_flags.get("has_person") is True:
            return 1
    tracks = response.get("tracks")
    return len(tracks) if isinstance(tracks, list) else 0


def _has_available_attention(attention: Any) -> bool:
    if not isinstance(attention, dict):
        return False
    return _attention_target_id(attention) is not None or _is_point(
        attention.get("target_uv")
    )


def _attention_target_id(attention: Any) -> int | None:
    if not isinstance(attention, dict):
        return None
    target = attention.get("target_track_id")
    if not _is_int(target):
        return None
    return int(target)


def _is_point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


_CSS = """
body { font-family: system-ui, sans-serif; margin: 20px; background: #111; color: #eee; }
a { color: #8bd3ff; }
.meta { color: #bbb; margin-bottom: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
.card { background: #1b1b1b; border: 1px solid #333; padding: 10px; }
.card img { width: 100%; display: block; }
.caption { font-size: 13px; color: #ccc; margin: 8px 0; }
pre { white-space: pre-wrap; max-height: 260px; overflow: auto; background: #0a0a0a; padding: 8px; }
""".strip()


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
