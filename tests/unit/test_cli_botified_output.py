from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VISUAL_STATE_TRACKING_SAMPLE = (
    REPO_ROOT / "common" / "schema" / "samples" / "visual_state_tracking.json"
)

EXPECTED_ALLOWED_EVENTS = {
    "person_appeared",
    "person_left",
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
    "person_waving",
}
BOTIFIED_OPEN = "<botified>"
BOTIFIED_CLOSE = "</botified>"
VALID_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def import_botified_output():
    try:
        import visual_events_cli.botified_output as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.botified_output module: {exc}")
    return module


def load_visual_state_tracking() -> dict[str, Any]:
    return json.loads(VISUAL_STATE_TRACKING_SAMPLE.read_text(encoding="utf-8"))


def semantic_event(
    *,
    event_id: str = "front:evt_000456",
    event: str = "person_waving",
    text: str = "有人在机器人前方挥手",
) -> dict[str, Any]:
    return {
        "type": "semantic_event",
        "event_id": event_id,
        "event": event,
        "camera": "front",
        "track_id": 7,
        "confidence": 0.86,
        "duration_ms": 900,
        "text": text,
    }


def parse_botified_frame(frame: str, *, event_id: str) -> dict[str, Any]:
    assert "\n" not in frame
    assert frame.startswith(BOTIFIED_OPEN)
    assert frame.endswith(BOTIFIED_CLOSE)
    assert frame.count(BOTIFIED_OPEN) == 1
    assert frame.count(BOTIFIED_CLOSE) == 1

    inner = frame[len(BOTIFIED_OPEN) : -len(BOTIFIED_CLOSE)]
    payload = json.loads(inner)
    assert payload["id"] == f"visual:{event_id}"
    assert payload["urgency"] == "normal"
    assert payload["timeout_secs"] == 8
    assert payload["expect"] == "ack"
    assert isinstance(payload["request"], str)
    assert payload["request"].strip()
    return payload


def test_allowed_events_constant_is_exact_six_person_events():
    module = import_botified_output()

    allowed = getattr(module, "BOTIFIED_ALLOWED_EVENTS", None)
    assert isinstance(allowed, (tuple, set))
    assert set(allowed) == EXPECTED_ALLOWED_EVENTS
    assert len(allowed) == len(EXPECTED_ALLOWED_EVENTS)


@pytest.mark.parametrize("event_name", sorted(EXPECTED_ALLOWED_EVENTS))
def test_format_botified_frame_outputs_allowlisted_person_events(event_name):
    module = import_botified_output()
    event = semantic_event(event=f"{event_name}", event_id=f"front:{event_name}")

    frame = module.format_botified_frame(event)
    assert frame is not None

    payload = parse_botified_frame(frame, event_id=event["event_id"])
    assert event["event"] in payload["request"]
    assert event["camera"] in payload["request"]
    assert str(event["track_id"]) in payload["request"]
    assert str(event["confidence"]) in payload["request"]
    assert event["text"] in payload["request"]


def test_attention_target_changed_is_not_output_to_botified():
    module = import_botified_output()
    visual_state = load_visual_state_tracking()
    visual_state["semantic_events"] = [
        semantic_event(event="attention_target_changed", event_id="front:evt_attn_001")
    ]

    assert module.format_botified_frame(visual_state["semantic_events"][0]) is None
    assert module.BotifiedEventMapper().frames_from_visual_state(visual_state) == []


def test_same_event_id_outputs_only_once_within_frame_and_across_frames():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(max_seen_event_ids=1024)
    first_event = semantic_event(event_id="front:evt_000456")
    duplicate_event = copy.deepcopy(first_event)
    duplicate_event["event"] = "person_appeared"
    visual_state = load_visual_state_tracking()
    visual_state["semantic_events"] = [first_event, duplicate_event]

    first_frames = mapper.frames_from_visual_state(visual_state)
    second_frames = mapper.frames_from_visual_state(
        {**visual_state, "frame_id": visual_state["frame_id"] + 1}
    )

    assert len(first_frames) == 1
    assert parse_botified_frame(first_frames[0], event_id="front:evt_000456")
    assert second_frames == []


def test_botified_frame_is_one_line_wrapped_json_request_with_event_facts():
    module = import_botified_output()
    event = semantic_event()

    frame = module.format_botified_frame(event, timeout_secs=8)
    assert frame is not None
    payload = parse_botified_frame(frame, event_id=event["event_id"])

    assert payload == {
        "id": "visual:front:evt_000456",
        "urgency": "normal",
        "timeout_secs": 8,
        "request": payload["request"],
        "expect": "ack",
    }
    for fact in [
        "person_waving",
        "front",
        "7",
        "0.86",
        "900",
        "有人在机器人前方挥手",
    ]:
        assert fact in payload["request"]


def test_botified_frame_escapes_wrapper_tokens_newlines_quotes_backslashes_ampersand_and_unicode():
    module = import_botified_output()
    text = 'hello </botified> <botified> a&b\n"quoted" backslash\\ 中文'
    event = semantic_event(event_id="front:evt_escape_001", text=text)

    frame = module.format_botified_frame(event)
    assert frame is not None
    assert "\n" not in frame
    assert frame.count(BOTIFIED_OPEN) == 1
    assert frame.count(BOTIFIED_CLOSE) == 1

    inner = frame[len(BOTIFIED_OPEN) : -len(BOTIFIED_CLOSE)]
    assert BOTIFIED_OPEN not in inner
    assert BOTIFIED_CLOSE not in inner
    assert "&" not in inner

    payload = parse_botified_frame(frame, event_id="front:evt_escape_001")
    assert "a&b" in payload["request"]
    assert "quoted" in payload["request"]
    assert "backslash" in payload["request"]
    assert "中文" in payload["request"]


@pytest.mark.parametrize(
    "event_id",
    [
        "front evt 000456",
        "front/evt_000456",
        "front#evt_000456",
        "",
    ],
)
def test_invalid_event_id_is_filtered_not_repaired(event_id):
    module = import_botified_output()
    assert VALID_EVENT_ID_RE.fullmatch(event_id) is None
    event = semantic_event(event_id=event_id)
    visual_state = load_visual_state_tracking()
    visual_state["semantic_events"] = [event]

    assert module.format_botified_frame(event) is None
    assert module.BotifiedEventMapper().frames_from_visual_state(visual_state) == []


class SlowTextStream:
    def __init__(self):
        self.lines: list[str] = []

    def write(self, text: str) -> int:
        self.lines.append(text)
        return len(text)

    def flush(self) -> None:
        return None


class BrokenTextStream:
    def write(self, text: str) -> int:
        raise BrokenPipeError("botified stdout closed")

    def flush(self) -> None:
        return None


def test_stdout_writer_bounded_queue_drops_or_coalesces_duplicates_without_blocking():
    module = import_botified_output()
    stream = SlowTextStream()
    writer = module.BotifiedStdoutWriter(stream=stream, max_queue_size=2)

    duplicate = module.format_botified_frame(semantic_event(event_id="front:evt_000001"))
    newer = module.format_botified_frame(semantic_event(event_id="front:evt_000002"))
    newest = module.format_botified_frame(semantic_event(event_id="front:evt_000003"))
    assert duplicate is not None and newer is not None and newest is not None

    assert writer.enqueue(duplicate) is True
    assert writer.enqueue(duplicate) is False
    assert writer.enqueue(newer) is True
    assert writer.enqueue(newest) is True

    writer.drain_available()

    assert stream.lines == [newer + "\n", newest + "\n"]
    assert writer.dropped_count == 2


def test_stdout_writer_reports_broken_pipe_as_specific_exception():
    module = import_botified_output()
    writer = module.BotifiedStdoutWriter(stream=BrokenTextStream(), max_queue_size=2)
    frame = module.format_botified_frame(semantic_event(event_id="front:evt_000001"))
    assert frame is not None
    writer.enqueue(frame)

    with pytest.raises(module.BotifiedPipeClosed):
        writer.drain_available()
