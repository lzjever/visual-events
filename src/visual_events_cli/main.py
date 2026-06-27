from __future__ import annotations

import argparse
import sys

from visual_events_cli.config import ConfigError, apply_overrides, load_config


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        config = load_config(args.config)
        apply_overrides(config, _overrides_from_args(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.check_config:
        return 0

    print(
        "visual-events-cli runtime is not implemented yet; use --check-config "
        "to validate configuration.",
        file=sys.stderr,
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visual-events-cli")
    parser.add_argument("--config")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--server")
    parser.add_argument("--camera")
    parser.add_argument("--dds-domain", type=int)
    parser.add_argument("--dds-network")
    parser.add_argument("--image-topic")
    parser.add_argument("--head-state-topic")
    parser.add_argument("--gaze-topic")
    parser.add_argument("--log-jsonl")
    return parser


def _overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "server": args.server,
        "camera": args.camera,
        "dds_domain": args.dds_domain,
        "dds_network": args.dds_network,
        "image_topic": args.image_topic,
        "head_state_topic": args.head_state_topic,
        "gaze_topic": args.gaze_topic,
        "log_path": args.log_jsonl,
    }


if __name__ == "__main__":
    raise SystemExit(main())
