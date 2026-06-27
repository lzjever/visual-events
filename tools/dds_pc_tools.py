from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "dds_bridge-pc-e2e-tools-main"

DOMAIN_ENV = "VISUAL_EVENTS_DDS_DOMAIN"
NETWORK_ENV = "VISUAL_EVENTS_DDS_NETWORK"
CAMERA_TOPIC_ENV = "VISUAL_EVENTS_CAMERA_TOPIC"
HEAD_STATE_TOPIC_ENV = "VISUAL_EVENTS_HEAD_STATE_TOPIC"
GAZE_TOPIC_ENV = "VISUAL_EVENTS_GAZE_TOPIC"

DEFAULT_CAMERA_TOPIC = "/camera/image/jpeg"
DEFAULT_HEAD_STATE_TOPIC = "/robot/head_state"
DEFAULT_GAZE_TOPIC = "/visual_events/gaze_target"


class PcDdsToolError(RuntimeError):
    pass


def _domain_arg(value: str) -> int:
    try:
        domain = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--dds-domain must be a non-negative integer") from exc
    if domain < 0:
        raise argparse.ArgumentTypeError("--dds-domain must be a non-negative integer")
    return domain


def _network_arg(value: str) -> str:
    if value == "":
        raise argparse.ArgumentTypeError("--dds-network must be non-empty")
    return value


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=f"native bridge build directory, default: {DEFAULT_BUILD_DIR}",
    )
    parser.add_argument("--dds-domain", type=_domain_arg, required=True)
    parser.add_argument("--dds-network", type=_network_arg, required=True)
    parser.add_argument(
        "--allow-non-loopback-dds",
        action="store_true",
        help="allow native DDS participants to use a network other than lo",
    )


def parse_args_or_return(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> object:
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)


def validate_domain_network(
    *,
    dds_domain: int,
    dds_network: str,
    allow_non_loopback_dds: bool,
) -> None:
    if dds_domain < 0:
        raise PcDdsToolError("--dds-domain must be a non-negative integer")
    if dds_network == "":
        raise PcDdsToolError("--dds-network must be non-empty")
    if dds_network != "lo" and not allow_non_loopback_dds:
        raise PcDdsToolError(
            f"--dds-network {dds_network!r} requires --allow-non-loopback-dds"
        )


def resolve_native_binary(build_dir: Path, binary_name: str) -> Path:
    binary = build_dir.expanduser().resolve() / binary_name
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise PcDdsToolError(f"native binary not found or not executable: {binary}")
    return binary


def build_child_env(
    *,
    dds_domain: int,
    dds_network: str,
    topic_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env[DOMAIN_ENV] = str(dds_domain)
    env[NETWORK_ENV] = dds_network
    for name, value in (topic_env or {}).items():
        if value == "":
            raise PcDdsToolError(f"{name} must be non-empty")
        env[name] = value
    return env


def run_native_tool(
    *,
    binary_name: str,
    build_dir: Path,
    dds_domain: int,
    dds_network: str,
    allow_non_loopback_dds: bool,
    native_args: Sequence[str],
    topic_env: Mapping[str, str] | None = None,
) -> int:
    try:
        validate_domain_network(
            dds_domain=dds_domain,
            dds_network=dds_network,
            allow_non_loopback_dds=allow_non_loopback_dds,
        )
        binary = resolve_native_binary(build_dir, binary_name)
        env = build_child_env(
            dds_domain=dds_domain,
            dds_network=dds_network,
            topic_env=topic_env,
        )
    except PcDdsToolError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    completed = subprocess.run(
        [os.fspath(binary), *native_args],
        env=env,
        check=False,
    )
    return int(completed.returncode)
