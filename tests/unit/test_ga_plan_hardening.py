from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GA_PLAN = REPO_ROOT / "docs" / "ga-development-plan.md"
PROTOCOL = REPO_ROOT / "common" / "schema" / "protocol.md"
GITIGNORE = REPO_ROOT / ".gitignore"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assert_contains_all(text: str, required: list[str]) -> None:
    missing = [item for item in required if item not in text]
    assert missing == []


def test_ga_plan_requires_val_data_manifest_and_release_hashes():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "val-data/manifest.json",
            "manifest_sha256",
            "scene 名称",
            "sha256",
            "frame count",
            "fps",
            "expected event timeline source/version",
            "PC/release report",
        ],
    )


def test_ga_plan_separates_pc_regression_from_robot_dds_and_closed_loop_gates():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "PC 本地 E2E 的定位是 regression gate",
            "真实 camera DDS runtime/network",
            "板端 DDS type/QoS compatibility",
            "真实 head state topic/type/Hz/freshness",
            "真实 head/motion consumer",
            "valid tracking 时头部物理指向目标",
            "invalid/stale 后不继续动",
            "restart 后无残留动作",
        ],
    )


def test_ga_plan_pins_head_state_latency_botified_and_runtime_hard_gates():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "head_state.required=true",
            "required = true",
            "head_state_publisher_mode",
            "head_state_hz",
            "head_state_stale_count",
            "head_state_unknown_ratio",
            "stationary",
            "moving",
            "unknown",
            "capture_to_gaze_publish_p95_ms",
            "capture_to_gaze_publish_p99_ms",
            "capture_to_botified_stdout_p95_ms",
            "capture_to_botified_stdout_p99_ms",
            "`attention_target_changed` 不输出到 Botified",
            "runtime/venv/bin/visual-events-server",
            "runtime/venv/bin/visual-events-cli",
            "--server-bin runtime/venv/bin/visual-events-server",
            "--cli-bin runtime/venv/bin/visual-events-cli",
            "runtime provenance",
            "server_exit_code",
            "cli_exit_code",
            "dev console script 或手工预启动 server",
            "visual-events-server` wheel",
            "CLI 不依赖 Torch/Ultralytics",
            "aarch64/RK3588 runtime",
            "ldd",
            "readelf",
            "DDS topic allowlist",
        ],
    )
    assert "required_for_ga" not in text


def test_ga_plan_freezes_head_state_contract_and_dds_stack_decision_record():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "`/robot/head_state` + `visual_events::msg::dds_::HeadStateV1_` 是 canonical GA 合同",
            "adapter 映射",
            "canonical internal schema",
            "PC publisher 使用的权威类型",
            "compatibility report 字段",
            "DDS stack decision record",
            "SDK/bridge 名称和版本",
            "IDL codegen 命令",
            "native bridge ABI",
            "runtime env vars",
            "PC 安装方式",
            "板端 probe 命令",
        ],
    )
    assert "topic | `/robot/head_state`，实现前需与运控 owner 固化最终名称" not in text
    assert "DDS type | `visual_events::msg::dds_::HeadStateV1_` 或运控 owner 已有等价类型；Step 1 必须固化" not in text


def test_ga_plan_defines_ga_thresholds_and_equivalent_closed_loop():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "## 7. GA 阈值",
            "Botified 每 track/event/min 上限",
            "同一 `track_id` 的同一 event type 每分钟 <=1 条",
            "Botified 全局事件/min 上限",
            "合计每分钟 <=12 条",
            "Head state unknown ratio",
            "stationary/moving segment <=1%",
            "Target switch dwell/jitter",
            "dwell >=750ms",
            "target_norm jitter P95 <=0.04",
            "现场 head pointing 误差",
            "P95 <=8 度",
            "Invalid/stale 停止延迟",
            "250ms 内发布 `valid=false,state=stale`",
            "500ms 内停止继续跟随旧目标",
            "等价闭环必须消费同一个 `/visual_events/gaze_target` DDS topic",
            "真实或 HIL head_state >=9Hz",
            "由运控 owner sign-off",
            "shadow consumer/logs 只能作为 preflight",
        ],
    )


def test_ga_plan_pins_botified_backpressure_and_broken_pipe_semantics():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "Botified stdout writer：bounded queue/drop/coalescing + BrokenPipe exception unit core",
            "slow stdout bounded queue/drop/coalescing 不阻塞 gaze stale",
            "stdout bounded queue/drop/coalescing 不阻塞 gaze stale",
            "RuntimeCoordinator/main wiring unit core",
            "main runtime_runner 注入和默认 Step 4 DDS adapters fail-fast",
            "exact stale deadline",
            "slow Botified drain 不阻塞 stale",
            "BrokenPipe publish stale then nonzero unit core",
            "Botified stdout BrokenPipe 时必须尽力发布一次 stale，然后受控非 0 退出",
            'broken_pipe = "publish_stale_then_exit_nonzero"',
            "BrokenPipe 受控非 0 退出",
        ],
    )
    assert 'broken_pipe = "fail_fast"' not in text
    assert "Runtime coordinator 对 BrokenPipe 的 publish stale then nonzero exit 处理" not in text
    assert "runtime coordinator 对 BrokenPipe 的 publish stale then nonzero exit 处理" not in text


def test_ga_plan_pins_final_review_contracts_without_expanding_scope():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "测试 runner 必须显式传入 domain/network",
            "标准 PC 值是 `DDS_NETWORK=lo`、`DDS_DOMAIN=57`",
            "缺少显式 `--dds-domain` 或 `--dds-network` 必须 fail fast",
            "非 loopback 网络必须显式传入 `--allow-non-loopback-dds`",
            "Step 3 不实现真实 DDS adapters",
            "Step 4 真实 DDS adapters",
            "expected attention target timeline/rule（target label/track、allowed switch windows、no-target windows）",
            "现场 checklist 必须验证 expected attention target timeline/rule",
            "进入、路过、靠近、停留、挥手等每类事件有 expected occurrence 和允许延迟窗口",
            "负例不得触发",
            "同一物理人短暂 lost/恢复期间不得产生新的招呼型事件序列",
            "`track_id` 变化时也必须通过 scene/person label 发现",
            "Botified owner 完整产品闭环 artifact",
            "Visual Events semantic event -> Botified 决策/回应动作 -> 冷却不重复招呼",
            "不要求也不允许把 Botified 会话逻辑实现回本 repo",
        ],
    )
    assert "测试 runner 默认必须使用 `DDS_NETWORK=lo`" not in text
    assert "E2E 默认使用 `DDS_NETWORK=lo`" not in text
    assert "真实 DDS adapters：DDS image/head state/gaze target 的 runtime adapter" not in text


def test_ga_plan_baseline_and_team_review_match_current_cli_state():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "当前 repo 已完成 DDS contract/schema Step 1 的主要产物",
            "`common/schema/dds/camera_jpeg_contract.md`",
            "`common/schema/dds/gaze_target_v1.idl` 和 `gaze_target_v1.md`",
            "`common/schema/dds/head_state_v1.idl` 和 `head_state_v1.md`",
            "gaze target samples",
            "no-motion-SDK audit 覆盖",
            "Step 1 仍未完成 DDS stack decision record",
            "尚未实际确定 SDK/bridge runtime choice",
            "当前 repo 已完成 CLI Step 3A skeleton + Step 3B pure logic",
            "### Step 3：实现正式 CLI core（3A skeleton/3B pure logic 和 RuntimeCoordinator/main wiring unit core 已完成；production runner/lifecycle 依赖 Step 4 DDS adapters）",
            "CLI package",
            "`visual-events-cli` entrypoint",
            "配置 skeleton",
            "`target_mapper` 纯逻辑",
            "`botified_output` 纯逻辑",
            "并已完成 CLI unit core 的 `service_client` WebSocket wire client",
            "`frame_pump` deterministic core/stale watchdog",
            "RuntimeCoordinator/main wiring unit core",
            "main runtime_runner 注入和默认 Step 4 DDS adapters fail-fast",
            "exact stale deadline",
            "slow Botified drain 不阻塞 stale",
            "Botified stdout bounded queue/drop/coalescing 与 BrokenPipe exception 单元核心",
            "BrokenPipe publish stale then nonzero unit core",
            "`service_client`：WebSocket wire/pack-unpack、连接复用/关闭、timeout、invalid response、frame_id mismatch、retryable/non-retryable error handling 的单元核心",
            "`frame_pump`：one in-flight coordination、keep-latest frame slot/backpressure、gaze stale watchdog、Botified enqueue 的 deterministic unit core",
            "main runtime_runner 注入和默认 Step 4 DDS adapters fail-fast",
            "exact stale deadline",
            "slow Botified drain 不阻塞 stale",
            "Botified stdout writer：bounded queue/drop/coalescing + BrokenPipe exception unit core",
            "BrokenPipe publish stale then nonzero unit core",
            "production runner/lifecycle 依赖 Step 4 真实 DDS adapters 实例化 RuntimeCoordinator",
            "正式 start/run/shutdown/reconnect、metrics/logging",
            "真实 DDS adapters",
            "PC 本地 DDS E2E tools",
            "release/runtime 编排",
            "真机 smoke",
            "closed-loop handoff",
            "剩余是 Step 4 真实 DDS adapters、PC E2E tools、release/runtime 编排、真机 smoke 和 closed-loop handoff",
            "DDS contract/schema Step 1 主要产物已完成",
            "DDS runtime stack 和板端 compatibility probe 仍必须补齐",
        ],
    )
    assert "尚未完成 CLI runtime loop、service_client、frame_pump、真实 DDS adapters、Botified stdout writer/backpressure" not in text
    assert "仍需完成 runtime loop、真实 DDS adapters、Botified stdout writer、PC E2E tools 和 release/runtime 编排" not in text
    assert "仍未完成 Step 3 formal CLI runtime loop/main wiring" not in text
    assert "仍需完成 Step 3 formal CLI runtime loop/main wiring" not in text
    assert "production runner/lifecycle 剩余" not in text
    assert "生产 runtime runner/lifecycle 编排需要 Step 4 真实 DDS adapters 实例化 RuntimeCoordinator" not in text
    assert "runtime coordinator 对 BrokenPipe 的 publish stale then nonzero exit 处理" not in text
    assert "Runtime coordinator 对 BrokenPipe 的 publish stale then nonzero exit 处理" not in text


def test_ga_plan_baseline_and_team_review_match_current_server_state():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "server GA 收口 Step 2 已完成",
            "frame/timestamp regression reset",
            "JPEG dimensions validation",
            "`scene_flags.someone_near_center`",
            "shared backend inference serialization",
            "metrics sink write error count/stderr",
            "server/CLI contract tests",
            "Step 2：完成 server GA 收口改进（已完成，后续只防回归）",
            "这些 GA 前 server 收口项已完成；后续只防回归和继续跑 gates",
            "已 reset 当前连接 tracker/event state，并补测试",
            "已校验 header 与 decode 尺寸一致",
            "已实现真实计算",
            "已串行化共享 backend 推理",
            "已记录 write error count 并输出 stderr",
            "已固化 attention、semantic events、head_motion suppression 行为",
            "frame/timestamp reset、JPEG dimensions validation、`someone_near_center`、shared backend inference serialization、metrics sink write error count/stderr 和 server/CLI contract tests 已完成",
            "后续只防回归/跑 gates",
        ],
    )
    assert "Server S8 baseline 不重写，只补 frame/timestamp reset、尺寸策略、`someone_near_center`、多连接推理和 metrics error 收口" not in text
    assert "这些改进是 GA 前 server 必做收口项" not in text


def test_protocol_pins_cli_frame_id_stale_watchdog_and_botified_allowlist():
    text = read_text(PROTOCOL)

    assert_contains_all(
        text,
        [
            "per-connection monotonic transport identity",
            "CameraFrame_` DDS 输入没有源 `frame_id`",
            "不得使用 DDS `timestamp_ns`/`timestamp_ms` 作为 identity",
            "WebSocket header `timestamp_ms`",
            "server 原样回显为 `visual_state.frame_timestamp_ms`",
            "不是 frame identity",
            "`gaze_target.stale_ms` 与 `service.response_timeout_ms` 是两个计时器",
            "Botified stdout allowlist",
            "`attention_target_changed` 只保留在 `visual_state.semantic_events`",
            "stdout 写入不得阻塞 gaze stale watchdog",
            "bounded queue/drop/coalescing",
            "固定 BrokenPipe 行为",
            "受控非 0 退出",
            "不实现 Botified 业务 rate limiter",
            "由 server semantic event engine 的 rising-edge、cooldown、dedupe 规则产生",
            "由 PC/现场 report gate 验收",
        ],
    )


def test_gitignore_blocks_large_artifacts_without_ignoring_json_samples():
    lines = {
        line.strip()
        for line in read_text(GITIGNORE).splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert "*.json" not in lines
    assert_contains_all(
        "\n".join(sorted(lines)),
        [
            "*.pt",
            "*.onnx",
            "*.engine",
            "*.rknn",
            "*.mcap",
            "*.bag",
            "*.pcap",
            "*.mp4",
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.jsonl",
            "*.log",
            "captures/",
            "logs/",
            "models/",
        ],
    )
