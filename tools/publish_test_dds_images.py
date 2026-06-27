from __future__ import annotations

import argparse
from typing import Sequence

try:
    from tools.dds_pc_tools import (
        CAMERA_TOPIC_ENV,
        DEFAULT_CAMERA_TOPIC,
        add_common_arguments,
        parse_args_or_return,
        run_native_tool,
    )
except ModuleNotFoundError:
    from dds_pc_tools import (  # type: ignore[no-redef]
        CAMERA_TOPIC_ENV,
        DEFAULT_CAMERA_TOPIC,
        add_common_arguments,
        parse_args_or_return,
        run_native_tool,
    )


BINARY_NAME = "visual_events_dds_bridge_publish_test_dds_images"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--count", required=True)
    parser.add_argument("--hz", required=True)
    parser.add_argument("--camera-name")
    parser.add_argument("--camera-topic", default=DEFAULT_CAMERA_TOPIC)
    add_common_arguments(parser)
    return parser


def native_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--input",
        args.input,
        "--count",
        args.count,
        "--hz",
        args.hz,
    ]
    if args.camera_name is not None:
        command.extend(["--camera-name", args.camera_name])
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parsed = parse_args_or_return(build_parser(), argv)
    if isinstance(parsed, int):
        return parsed
    args = parsed
    return run_native_tool(
        binary_name=BINARY_NAME,
        build_dir=args.build_dir,
        dds_domain=args.dds_domain,
        dds_network=args.dds_network,
        allow_non_loopback_dds=args.allow_non_loopback_dds,
        native_args=native_args(args),
        topic_env={CAMERA_TOPIC_ENV: args.camera_topic},
    )


if __name__ == "__main__":
    raise SystemExit(main())
