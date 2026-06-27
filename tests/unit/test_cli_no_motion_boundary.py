from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_SRC = REPO_ROOT / "src" / "visual_events_cli"
DENIED_MOTION_TOKENS = {
    "LowCmd",
    "MotorCmd",
    "SportModeCmd",
    "MotionSwitcherClient",
    "look_at",
    "head_position",
    "yaw_velocity",
    "pitch_velocity",
    "motor_command",
    "rt/lowcmd",
    "rt/arm_sdk",
}


def test_cli_source_contains_no_motion_sdk_tokens():
    assert CLI_SRC.is_dir(), "expected src/visual_events_cli package for CLI audit"

    offenders: list[str] = []
    for path in sorted(CLI_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in sorted(DENIED_MOTION_TOKENS):
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")

    assert offenders == []
