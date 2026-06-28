from __future__ import annotations

import argparse
import sys
from typing import Sequence

try:
    from tools.dds_pc_tools import (
        DEFAULT_GAZE_TOPIC,
        GAZE_TOPIC_ENV,
        add_common_arguments,
        parse_args_or_return,
        run_native_tool,
    )
except ModuleNotFoundError:
    from dds_pc_tools import (  # type: ignore[no-redef]
        DEFAULT_GAZE_TOPIC,
        GAZE_TOPIC_ENV,
        add_common_arguments,
        parse_args_or_return,
        run_native_tool,
    )


BINARY_NAME = "visual_events_dds_bridge_subscribe_test_gaze_targets"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--count")
    mode.add_argument("--duration-ms")
    parser.add_argument("--min-count")
    parser.add_argument("--timeout-ms")
    parser.add_argument("--gaze-topic", default=DEFAULT_GAZE_TOPIC)
    add_common_arguments(parser)
    return parser


def _mode_error(args: argparse.Namespace) -> str | None:
    if args.count is not None:
        if args.timeout_ms is None:
            return "--count requires --timeout-ms"
        if args.min_count is not None:
            return "--min-count can only be used with --duration-ms"
        return None
    if args.min_count is None:
        return "--duration-ms requires --min-count"
    if args.timeout_ms is not None:
        return "--timeout-ms can only be used with --count"
    return None


def native_args(args: argparse.Namespace) -> list[str]:
    if args.count is not None:
        return [
            "--count",
            args.count,
            "--timeout-ms",
            args.timeout_ms,
        ]
    return [
        "--duration-ms",
        args.duration_ms,
        "--min-count",
        args.min_count,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parsed = parse_args_or_return(build_parser(), argv)
    if isinstance(parsed, int):
        return parsed
    args = parsed
    error = _mode_error(args)
    if error is not None:
        parser = build_parser()
        parser.prog = "subscribe_test_gaze_targets.py"
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        return 2
    return run_native_tool(
        binary_name=BINARY_NAME,
        build_dir=args.build_dir,
        dds_domain=args.dds_domain,
        dds_network=args.dds_network,
        allow_non_loopback_dds=args.allow_non_loopback_dds,
        native_args=native_args(args),
        topic_env={GAZE_TOPIC_ENV: args.gaze_topic},
    )


if __name__ == "__main__":
    raise SystemExit(main())
