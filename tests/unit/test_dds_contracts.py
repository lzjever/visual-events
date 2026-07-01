from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DDS_DIR = REPO_ROOT / "common" / "schema" / "dds"

CAMERA_CONTRACT = DDS_DIR / "camera_jpeg_contract.md"
GAZE_IDL = DDS_DIR / "gaze_target_v1.idl"
GAZE_MD = DDS_DIR / "gaze_target_v1.md"
HEAD_IDL = DDS_DIR / "head_state_v1.idl"
HEAD_MD = DDS_DIR / "head_state_v1.md"
GAZE_TRACKING_SAMPLE = DDS_DIR / "samples" / "gaze_target_tracking.json"
GAZE_STALE_SAMPLE = DDS_DIR / "samples" / "gaze_target_stale.json"
GAZE_LOST_SAMPLE = DDS_DIR / "samples" / "gaze_target_lost.json"
GAZE_DISABLED_SAMPLE = DDS_DIR / "samples" / "gaze_target_disabled.json"
SCHEMA_SAMPLES_DIR = REPO_ROOT / "common" / "schema" / "samples"
VISUAL_STATE_SAMPLE = SCHEMA_SAMPLES_DIR / "visual_state_tracking.json"
SEMANTIC_EVENT_SAMPLE = SCHEMA_SAMPLES_DIR / "semantic_event_person_waving.json"
LEGACY_DOCS = REPO_ROOT / "docs" / "legacy"
NO_MOTION_AUDIT = LEGACY_DOCS / "no-motion-sdk-audit.md"
GA_PLAN = LEGACY_DOCS / "ga-development-plan.md"
README = REPO_ROOT / "README.md"
PROTOCOL = REPO_ROOT / "common" / "schema" / "protocol.md"

GAZE_FIELDS = [
    ("schema_version", "uint32"),
    ("camera", "string"),
    ("frame_id", "int64"),
    ("frame_timestamp_ms", "int64"),
    ("publish_timestamp_ms", "int64"),
    ("valid", "bool"),
    ("state", "string"),
    ("target_track_id", "int64"),
    ("target_u", "float32"),
    ("target_v", "float32"),
    ("target_norm_x", "float32"),
    ("target_norm_y", "float32"),
    ("image_width", "uint32"),
    ("image_height", "uint32"),
    ("confidence", "float32"),
    ("reason", "string"),
    ("stale_after_ms", "uint32"),
]

HEAD_FIELDS = [
    ("schema_version", "uint32"),
    ("timestamp_ms", "int64"),
    ("valid", "bool"),
    ("yaw_rad", "float64"),
    ("pitch_rad", "float64"),
    ("yaw_vel_rad_s", "float64"),
    ("pitch_vel_rad_s", "float64"),
]

GAZE_FORBIDDEN_TOKENS = {
    "yaw",
    "pitch",
    "velocity",
    "vel",
    "position",
    "pose",
    "motor",
    "command",
    "look_at",
    "joint",
    "trajectory",
    "setpoint",
}

HEAD_FORBIDDEN_TOKENS = {
    "motor",
    "command",
    "look_at",
    "trajectory",
    "setpoint",
    "target_u",
    "target_v",
}

TYPE_ALIASES = {
    "unsigned long": "uint32",
    "long long": "int64",
    "double": "float64",
    "float": "float32",
    "boolean": "bool",
    "string": "string",
}


def parse_idl_struct_fields(path: Path, struct_name: str) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"\bstruct\s+{re.escape(struct_name)}\s*\{{(?P<body>.*?)\}};", text, re.S)
    assert match is not None, f"{path} does not define struct {struct_name}"

    fields: list[tuple[str, str]] = []
    for raw_line in match.group("body").splitlines():
        line = re.sub(r"//.*$", "", raw_line).strip()
        if not line or line.startswith("@"):
            continue
        line = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", line).strip()
        field_match = re.fullmatch(
            r"(?P<type>unsigned\s+long|long\s+long|double|float|boolean|string(?:<[^>]+>)?)"
            r"\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;",
            line,
        )
        assert field_match is not None, f"{path}:{struct_name} has unparseable IDL member line: {raw_line!r}"
        raw_type = re.sub(r"\s+", " ", field_match.group("type"))
        raw_type = re.sub(r"<[^>]+>", "", raw_type)
        fields.append((field_match.group("name"), TYPE_ALIASES[raw_type]))
    return fields


def parse_md_metadata_table(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, str] = {}
    in_table = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table:
                break
            continue
        if not stripped.startswith("|"):
            if in_table:
                break
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        lowered = [cell.lower() for cell in cells]
        if lowered[:2] == ["key", "value"]:
            in_table = True
            continue
        if in_table and all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if in_table and len(cells) >= 2:
            metadata[cells[0]] = cells[1]

    assert metadata, f"{path} does not contain a top metadata table"
    return metadata


def parse_md_field_names(path: Path) -> list[str]:
    field_names: list[str] = []
    current_headers: list[str] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            current_headers = None
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        lowered = [cell.lower() for cell in cells]
        if lowered and lowered[0] in {"field", "name"}:
            current_headers = lowered
            continue
        if current_headers is None:
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if cells:
            field_names.append(cells[0])

    return field_names


def parse_md_field_table(path: Path) -> dict[str, dict[str, str]]:
    fields: dict[str, dict[str, str]] = {}
    current_headers: list[str] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            current_headers = None
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        lowered = [cell.lower() for cell in cells]
        if lowered and lowered[0] in {"field", "name"}:
            current_headers = lowered
            continue
        if current_headers is None:
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) >= len(current_headers):
            fields[cells[0]] = dict(zip(current_headers[1:], cells[1:]))

    assert fields, f"{path} does not contain a field table"
    return fields


def assert_no_forbidden_keys(keys: list[str], forbidden_tokens: set[str]) -> None:
    violations: dict[str, list[str]] = {}
    for key in keys:
        normalized = key.lower()
        key_tokens = set(re.split(r"[^a-z0-9]+", normalized))
        matched = sorted(
            token
            for token in forbidden_tokens
            if normalized == token or token in key_tokens
        )
        if matched:
            violations[key] = matched

    assert violations == {}


def test_step1_dds_contract_files_exist():
    for path in [
        CAMERA_CONTRACT,
        GAZE_IDL,
        GAZE_MD,
        HEAD_IDL,
        HEAD_MD,
        GAZE_TRACKING_SAMPLE,
        GAZE_STALE_SAMPLE,
    ]:
        assert path.is_file(), f"missing contract file: {path}"
        assert path.stat().st_size > 0, f"empty contract file: {path}"


def test_gaze_target_idl_has_exact_target_fields_and_no_motion_commands():
    fields = parse_idl_struct_fields(GAZE_IDL, "GazeTargetV1_")

    assert fields == GAZE_FIELDS
    assert_no_forbidden_keys([name for name, _type in fields], GAZE_FORBIDDEN_TOKENS)
    assert_no_forbidden_keys(parse_md_field_names(GAZE_MD), GAZE_FORBIDDEN_TOKENS)

    for sample_path in [GAZE_TRACKING_SAMPLE, GAZE_STALE_SAMPLE]:
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        assert_no_forbidden_keys(list(sample.keys()), GAZE_FORBIDDEN_TOKENS)


def test_head_state_idl_is_read_only_head_state_schema():
    fields = parse_idl_struct_fields(HEAD_IDL, "HeadStateV1_")

    assert fields == HEAD_FIELDS
    assert_no_forbidden_keys([name for name, _type in fields], HEAD_FORBIDDEN_TOKENS)


def test_dds_markdown_metadata_pins_topics_and_qos():
    expected_by_file = {
        CAMERA_CONTRACT: {
            "topic": "/camera/image/jpeg",
            "dds_type": "unitree_camera::msg::dds_::CameraFrame_",
            "encoding": "JPEG",
            "reliability": "best_effort",
            "durability": "volatile",
            "history": "keep_last_1",
            "deadline_ms": "150",
            "lifespan_ms": "300",
            "liveliness_lease_ms": "1000",
        },
        HEAD_MD: {
            "topic": "/robot/head_state",
            "dds_type": "visual_events::msg::dds_::HeadStateV1_",
            "reliability": "best_effort",
            "durability": "volatile",
            "history": "keep_last_1",
            "deadline_ms": "150",
            "lifespan_ms": "250",
            "liveliness_lease_ms": "500",
        },
        GAZE_MD: {
            "topic": "/visual_events/gaze_target",
            "dds_type": "visual_events::msg::dds_::GazeTargetV1_",
            "reliability": "best_effort",
            "durability": "volatile",
            "history": "keep_last_1",
            "deadline_ms": "150",
            "lifespan_ms": "250",
            "liveliness_lease_ms": "500",
        },
    }

    for path, expected in expected_by_file.items():
        metadata = parse_md_metadata_table(path)
        for key, value in expected.items():
            assert metadata.get(key) == value


def test_camera_jpeg_contract_field_table_matches_camera_frame_exactly():
    assert parse_md_field_names(CAMERA_CONTRACT) == [
        "timestamp_ns",
        "camera_name",
        "width",
        "height",
        "encoding",
        "step",
        "data",
    ]


def test_camera_jpeg_contract_pins_exact_timestamp_type_and_encoding():
    fields = parse_md_field_table(CAMERA_CONTRACT)
    problems: list[str] = []

    if fields["timestamp_ns"]["type"] != "unsigned long long":
        problems.append("timestamp_ns type must be unsigned long long")

    encoding_notes = fields["encoding"]["notes"]
    accepted_encodings = re.findall(r"`([^`]+)`", encoding_notes)
    if accepted_encodings != ["JPEG"]:
        problems.append("encoding notes must only list `JPEG` as accepted")
    if "jpeg" in encoding_notes:
        problems.append("encoding notes must not include lowercase `jpeg` fallback text")

    assert problems == []


def test_gaze_target_state_enum_is_closed_in_docs_and_samples():
    allowed_states = {"tracking", "lost", "stale", "disabled"}
    state_notes = parse_md_field_table(GAZE_MD)["state"]["notes"]
    documented_states = set(re.findall(r"`([^`]+)`", state_notes))

    assert "another documented observation state" not in state_notes
    assert documented_states == allowed_states

    for sample_path in sorted((DDS_DIR / "samples").glob("gaze_target_*.json")):
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        assert sample["state"] in allowed_states, f"{sample_path} has unknown state"


def test_gaze_target_invalid_samples_lock_zero_target_semantics():
    for sample_path in [GAZE_STALE_SAMPLE, GAZE_LOST_SAMPLE, GAZE_DISABLED_SAMPLE]:
        assert sample_path.is_file(), f"missing invalid sample: {sample_path}"

    for state, sample_path in {
        "stale": GAZE_STALE_SAMPLE,
        "lost": GAZE_LOST_SAMPLE,
        "disabled": GAZE_DISABLED_SAMPLE,
    }.items():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        assert sample["valid"] is False
        assert sample["state"] == state
        assert sample["target_track_id"] == -1
        assert sample["target_u"] == 0.0
        assert sample["target_v"] == 0.0
        assert sample["target_norm_x"] == 0.0
        assert sample["target_norm_y"] == 0.0
        assert sample["confidence"] == 0.0
        assert sample["reason"] == state
        assert sample["stale_after_ms"] == 250


def test_gaze_target_samples_capture_tracking_and_stale_semantics():
    tracking = json.loads(GAZE_TRACKING_SAMPLE.read_text(encoding="utf-8"))
    stale = json.loads(GAZE_STALE_SAMPLE.read_text(encoding="utf-8"))

    assert tracking["schema_version"] == 1
    assert tracking["valid"] is True
    assert tracking["state"] == "tracking"
    assert tracking["target_track_id"] != -1
    assert 0.0 <= tracking["target_u"] <= tracking["image_width"]
    assert 0.0 <= tracking["target_v"] <= tracking["image_height"]
    assert tracking["target_norm_x"] == tracking["target_u"] / tracking["image_width"] - 0.5
    assert tracking["target_norm_y"] == tracking["target_v"] / tracking["image_height"] - 0.5
    assert 0.0 <= tracking["confidence"] <= 1.0
    assert tracking["reason"] == "largest_stable_person"
    assert tracking["stale_after_ms"] == 250

    assert stale["valid"] is False
    assert stale["state"] == "stale"
    assert stale["target_track_id"] == -1
    assert stale["target_u"] == 0.0
    assert stale["target_v"] == 0.0
    assert stale["target_norm_x"] == 0.0
    assert stale["target_norm_y"] == 0.0
    assert stale["confidence"] == 0.0
    assert stale["reason"] == "stale"
    assert stale["stale_after_ms"] == 250


def test_step1_visual_state_and_semantic_event_samples_exist_and_parse():
    visual_state = json.loads(VISUAL_STATE_SAMPLE.read_text(encoding="utf-8"))
    semantic_event = json.loads(SEMANTIC_EVENT_SAMPLE.read_text(encoding="utf-8"))

    assert visual_state["type"] == "visual_state"
    assert visual_state["schema_version"] == 1
    assert "target_track_id" in visual_state["attention"]
    assert isinstance(visual_state["semantic_events"], list)

    assert semantic_event["type"] == "semantic_event"
    assert semantic_event["event"] == "person_waving"
    for key in ["event_id", "camera", "track_id", "confidence", "text"]:
        assert key in semantic_event


def test_ga_plan_uses_exact_dds_type_literals():
    text = GA_PLAN.read_text(encoding="utf-8")
    required = {
        "visual_events::msg::dds_::HeadStateV1_",
        "visual_events::msg::dds_::GazeTargetV1_",
    }

    problems: list[str] = []
    for type_name in sorted(required):
        if type_name not in text:
            problems.append(f"missing {type_name}")
    for forbidden in [
        "visual_events::msg::HeadStateV1",
        "visual_events::msg::GazeTargetV1",
    ]:
        if forbidden in text:
            problems.append(f"forbidden non-DDS type literal {forbidden}")

    assert problems == []


def test_readme_and_protocol_link_dds_contract_entrypoints():
    required_paths = {
        "common/schema/dds/camera_jpeg_contract.md",
        "common/schema/dds/gaze_target_v1.md",
        "common/schema/dds/head_state_v1.md",
    }

    for path in [README, PROTOCOL]:
        text = path.read_text(encoding="utf-8")
        linked_paths = set(re.findall(r"common/schema/dds/[A-Za-z0-9_./-]+", text))
        missing = sorted(required_paths - linked_paths)
        assert missing == [], f"{path} is missing DDS contract entrypoints"


def test_no_motion_sdk_audit_documents_blacklist_and_allowed_contracts():
    text = NO_MOTION_AUDIT.read_text(encoding="utf-8")

    for token in [
        "rt/lowcmd",
        "rt/arm_sdk",
        "LowCmd",
        "MotorCmd",
        "SportModeCmd",
        "MotionSwitcherClient",
        "look_at",
        "head_position",
        "yaw_velocity",
        "pitch_velocity",
        "motor_command",
    ]:
        assert token in text

    for contract_name in ["CameraFrame_", "HeadStateV1_", "GazeTargetV1_"]:
        assert contract_name in text
