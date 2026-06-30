from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


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


def draw_visual_state(
    image: Any,
    state: dict[str, Any],
    *,
    scene: Any = None,
    frame_id: Any = None,
) -> Any:
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

    _draw_header(canvas, state, scene=scene, frame_id=frame_id)
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


def frame_header_lines(
    state: dict[str, Any],
    *,
    scene: Any = None,
    frame_id: Any = None,
) -> list[str]:
    lines = [
        f"frame={_short(frame_id if frame_id is not None else state.get('frame_id', '-'))}",
        f"camera={state.get('camera', '-')}",
    ]
    if scene is not None:
        lines.append(f"scene={_short(scene, 80)}")
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
    overlay_summary = _identity_overlay_summary(state.get("identity_context"))
    if overlay_summary != "-":
        lines.append(_clip(overlay_summary, 116))
    return lines


def _draw_header(
    canvas: Any,
    state: dict[str, Any],
    *,
    scene: Any = None,
    frame_id: Any = None,
) -> None:
    lines = frame_header_lines(state, scene=scene, frame_id=frame_id)
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
    if not isinstance(scene_context, dict) or not scene_context:
        return "scene_context=none"

    parts: list[str] = []
    engagement = scene_context.get("engagement_state")
    if engagement:
        parts.append(f"engagement={_short(engagement)}")
    reasons = _reasons_summary(scene_context.get("no_engage_reasons"))
    if reasons:
        parts.append(f"reasons={reasons}")
    return " ".join(parts) if parts else "scene_context=none"


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
        return (
            f"{event_name} track={track} memory={_memory_event_summary(event)}"
            f"{_event_identity_suffix(event)}"
        )
    evidence = _evidence_summary(event)
    return f"{event_name} track={track}{_event_identity_suffix(event)} evidence={evidence}"


def _event_identity_suffix(event: dict[str, Any]) -> str:
    identity = event.get("identity_context")
    summary = _identity_summary(identity)
    return f" identity={summary}" if summary else ""


def _identity_overlay_summary(identity_context: Any) -> str:
    if not isinstance(identity_context, dict):
        return "-"
    status = identity_context.get("overlay_status")
    tracks = identity_context.get("tracks")
    identity = None
    if isinstance(tracks, list):
        for item in tracks:
            if isinstance(item, dict) and isinstance(item.get("identity"), dict):
                identity = item["identity"]
                break
    identity_text = _identity_summary(identity)
    parts = []
    if status:
        parts.append(f"overlay={_short(status)}")
    if identity_text:
        parts.append(f"identity={identity_text}")
    return " ".join(parts) if parts else "-"


def _identity_summary(identity: Any) -> str:
    if not isinstance(identity, dict):
        return ""
    person = identity.get("person")
    if isinstance(person, dict):
        display_name = person.get("display_name")
        if display_name:
            return _short(display_name)
        person_id = person.get("person_id")
        if person_id:
            return _short(person_id)
    anonymous = identity.get("anonymous_person")
    if isinstance(anonymous, dict):
        anonymous_id = anonymous.get("anonymous_id")
        return f"familiar:{_short(anonymous_id)}" if anonymous_id else "familiar"
    status = identity.get("status")
    source = identity.get("source")
    if status and source:
        return f"{_short(status)}/{_short(source)}"
    if status:
        return _short(status)
    return ""


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


def _track_ids_summary(tracks: Any) -> str:
    if not isinstance(tracks, list):
        return "[]"
    track_ids = []
    for track in tracks:
        if isinstance(track, dict) and "track_id" in track:
            track_ids.append(_short(track["track_id"], 24))
    return "[" + ",".join(track_ids) + "]"


def _frame_timestamp_summary(response: dict[str, Any]) -> str:
    if "frame_timestamp_ms" in response:
        return f"frame_timestamp_ms={_short(response['frame_timestamp_ms'])}"
    return "timestamp=-"


def _server_timestamp_summary(response: dict[str, Any]) -> str:
    return f"server_timestamp_ms={_short(response.get('server_timestamp_ms'))}"


def _latency_summary(frame_evidence: dict[str, Any]) -> str:
    return f"latency_ms={_short(frame_evidence.get('latency_ms'))}"


def _engagement_state(scene_context: Any) -> str:
    if not isinstance(scene_context, dict):
        return "-"
    engagement = scene_context.get("engagement_state")
    return _short(engagement, 32) if engagement else "-"


def render_html_document(
    *,
    root: Path,
    server: Any,
    scene: Any,
    frames: list[dict[str, Any]],
    jsonl_path: Path,
) -> str:
    cards = "\n".join(render_frame_card(root, frame) for frame in frames)
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
    server: {html.escape(str(server))}<br>
    scene: {html.escape(str(scene))}<br>
    frames: {len(frames)}<br>
    visual_state jsonl: {html.escape(_rel(root, jsonl_path))}
  </div>
  <div class="grid">
    {cards}
  </div>
</body>
</html>
"""


def render_frame_card(
    root: Path,
    frame_evidence: dict[str, Any],
    *,
    anchor: str | None = None,
) -> str:
    response = frame_evidence["state"]
    events = response.get("semantic_events")
    event_summaries = []
    if isinstance(events, list):
        event_summaries = [_event_summary(event) for event in events if isinstance(event, dict)]
    attention = response.get("attention")
    attention_text = "-"
    if isinstance(attention, dict):
        attention_text = str(attention.get("target_track_id", "-"))
    scene_summary = _scene_context_summary(response.get("scene_context"))
    scene_context_text = (
        scene_summary
        if scene_summary == "scene_context=none"
        else f"scene_context={scene_summary}"
    )
    reacquire_summary = _reacquire_summary(response.get("scene_context"))
    overlay_summary = _identity_overlay_summary(response.get("identity_context"))
    events_text = "; ".join(event_summaries) if event_summaries else "none"
    source_name = str(frame_evidence.get("source_name", "-"))
    scene = frame_evidence.get("scene")
    scene_text = str(scene) if scene is not None else "-"
    frame_id = frame_evidence.get("frame_id", response.get("frame_id", "-"))
    track_ids_text = _track_ids_summary(response.get("tracks"))
    frame_timestamp_text = _frame_timestamp_summary(response)
    server_timestamp_text = _server_timestamp_summary(response)
    latency_text = _latency_summary(frame_evidence)
    state_json = json.dumps(response, ensure_ascii=False, indent=2)
    anchor_attr = f' id="{html.escape(anchor)}"' if anchor else ""
    return f"""<div class="card"{anchor_attr}>
  <img src="{html.escape(_rel(root, frame_evidence["image_path"]))}" alt="frame {html.escape(str(frame_id))}">
  <div class="caption">
    frame={html.escape(str(frame_id))} {html.escape(source_name)}
    | scene={html.escape(scene_text)}
    | tracks={_list_len(response.get("tracks"))}
    | track_ids={html.escape(track_ids_text)}
    | {html.escape(frame_timestamp_text)}
    | {html.escape(server_timestamp_text)}
    | {html.escape(latency_text)}
    | attention={html.escape(attention_text)}
    | {html.escape(scene_context_text)}
    | {html.escape(overlay_summary)}
    | reacq={html.escape(reacquire_summary)}
    | events={html.escape(events_text)}
    | <a href="{html.escape(_rel(root, frame_evidence["state_path"]))}">json</a>
  </div>
  <details><summary>visual_state</summary><pre>{html.escape(state_json)}</pre></details>
</div>"""


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
