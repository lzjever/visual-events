from __future__ import annotations

import argparse
from typing import Sequence

try:
    from tools.dds_pc_tools import (
        DEFAULT_HEAD_STATE_TOPIC,
        HEAD_STATE_TOPIC_ENV,
        add_common_arguments,
        parse_args_or_return,
        run_native_tool,
    )
except ModuleNotFoundError:
    from dds_pc_tools import (  # type: ignore[no-redef]
        DEFAULT_HEAD_STATE_TOPIC,
        HEAD_STATE_TOPIC_ENV,
        add_common_arguments,
        parse_args_or_return,
        run_native_tool,
    )


BINARY_NAME = "visual_events_dds_bridge_publish_test_head_state"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--state", required=True)
    parser.add_argument("--count", required=True)
    parser.add_argument("--hz", required=True)
    parser.add_argument("--head-state-topic", default=DEFAULT_HEAD_STATE_TOPIC)
    add_common_arguments(parser)
    return parser


def native_args(args: argparse.Namespace) -> list[str]:
    return [
        "--state",
        args.state,
        "--count",
        args.count,
        "--hz",
        args.hz,
    ]


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
        topic_env={HEAD_STATE_TOPIC_ENV: args.head_state_topic},
    )


if __name__ == "__main__":
    raise SystemExit(main())
