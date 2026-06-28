from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GA_PLAN = REPO_ROOT / "docs" / "ga-development-plan.md"
DDS_STACK_DECISION = REPO_ROOT / "docs" / "dds-stack-decision-record.md"
NO_MOTION_AUDIT = REPO_ROOT / "docs" / "no-motion-sdk-audit.md"
PROTOCOL = REPO_ROOT / "common" / "schema" / "protocol.md"
VAL_DATA_MANIFEST_SCHEMA = REPO_ROOT / "common" / "schema" / "val_data_manifest_v1.md"
GITIGNORE = REPO_ROOT / ".gitignore"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assert_contains_all(text: str, required: list[str]) -> None:
    missing = [item for item in required if item not in text]
    assert missing == []


def section_between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def paragraph_containing(text: str, marker: str) -> str:
    for paragraph in text.split("\n\n"):
        if marker in paragraph:
            return paragraph
    raise AssertionError(f"missing paragraph containing {marker!r}")


def line_containing(text: str, marker: str) -> str:
    for line in text.splitlines():
        if marker in line:
            return line
    raise AssertionError(f"missing line containing {marker!r}")


def test_dds_stack_decision_record_freezes_ga_handoff_fields():
    assert DDS_STACK_DECISION.exists()
    text = read_text(DDS_STACK_DECISION)

    assert_contains_all(
        text,
        [
            "docs/dds-stack-decision-record.md",
            "Unitree SDK2 2.0.0",
            "CycloneDDS 0.10.2",
            "C++ native DDS helper/bridge",
            "VISUAL_EVENTS_DDS_DOMAIN",
            "VISUAL_EVENTS_DDS_NETWORK",
            "VISUAL_EVENTS_UNITREE_SDK_ROOT",
            "VISUAL_EVENTS_DDS_BRIDGE_BIN",
            "protocol_version=1",
            "line-delimited JSON",
            "not implemented by this decision record",
            "RK3588",
            "explicit unsupported fail-fast",
            "不链接、import 或调用运控 SDK",
            "data_base64",
            "data_size_bytes",
            "dds_timestamp_ns",
            "received_monotonic_ns",
            "camera_name",
            "width",
            "height",
            "encoding",
            "step",
            "track_id",
            "target_norm_x",
            "target_norm_y",
            "valid",
            "state",
            "stale_after_ms",
            "schema_version",
            "camera",
            "frame_id",
            "frame_timestamp_ms",
            "publish_timestamp_ms",
            "target_track_id",
            "target_u",
            "target_v",
            "image_width",
            "image_height",
            "confidence",
            "reason",
            "single-line UTF-8 JSON object",
            "raw bytes are forbidden",
            "Unitree Channel API is the only DDS pub/sub API used by the bridge",
            "time.monotonic_ns()",
            "same OS monotonic clock domain",
            "CLOCK_MONOTONIC",
            "Claiming board compatibility still requires an RK3588/board probe",
            "not part of the repo-local PC DDS emulation delivery gate",
        ],
    )
    assert "box_norm_width" not in text
    assert "box_norm_height" not in text


def test_ga_plan_requires_val_data_manifest_and_release_hashes():
    text = read_text(GA_PLAN)
    step9 = section_between(text, "### Step 9：Release 和 handoff", "## 10. Server 改进清单")

    assert_contains_all(
        step9,
        [
            "当前 PC gate evidence/hash",
            "server baseline",
            "CLI unit/integration",
            "PC local E2E",
            "runtime smoke",
            "no-motion-SDK audit",
            "val-data/manifest.json",
            "manifest_sha256",
            "scene 名称",
            "sha256",
            "frame count",
            "fps",
            "expected event timeline source/version",
            "runtime server/CLI provenance",
            "stationary head state required-mode",
            "Botified stdout allowlist",
            "stdout pollution",
            "fresh gaze Hz/rate",
            "basic finite latency sanity",
            "P95/P99 latency report 属于 non-blocking evidence / 后续 handoff evidence",
            "Post-GA hardware/field rollout evidence",
            "真机 smoke",
            "camera DDS owner",
            "gaze consumer/运控 owner",
            "Botified owner sign-off",
            "真实闭环/现场 owner sign-off",
            "不阻塞 GA",
            "不得由 PC evidence 声称通过",
        ],
    )
    assert "Artifact hash：server baseline、CLI unit/integration、PC local E2E、真机 smoke" not in step9
    assert "Camera DDS owner、gaze consumer/运控 owner、Botified owner sign-off。" not in step9


def test_val_data_manifest_v1_schema_is_oracle_contract_skeleton_only():
    assert VAL_DATA_MANIFEST_SCHEMA.exists()
    schema_text = read_text(VAL_DATA_MANIFEST_SCHEMA)
    plan_text = read_text(GA_PLAN)

    assert_contains_all(
        schema_text,
        [
            "schema_version: 1",
            "fps",
            "scene_count",
            "frame_count",
            "positive integer",
            "scenes",
            "scene_name",
            "scene_sha256",
            "expected_event_timeline.source",
            "expected_event_timeline.version",
            "expected_attention_target_timeline.source",
            "expected_attention_target_timeline.rule",
            "does not evaluate event correctness",
            "does not complete the full PC GA gate",
        ],
    )
    assert_contains_all(
        plan_text,
        [
            "authoritative manifest/oracle schema skeleton",
            "`common/schema/val_data_manifest_v1.md`",
            "`val-data/manifest.json` 仍不进 Git",
            "`tools/run_val_data_e2e.py` 支持 `--manifest` 和 `--require-authoritative-manifest`",
            "server replay gate report 记录 manifest/oracle contract evidence",
            "默认无 manifest 仍使用 generated inventory 且不破坏本机 replay gate",
            "server replay oracle 迁移不是 current PC core gate 的阻塞项",
            "不直接做 oracle evaluation",
            "Step 5 `run_cli_local_e2e` current PC core functional gate 已完成",
        ],
    )
    assert "full PC GA gate is complete" not in schema_text


def test_ga_plan_defines_current_pc_delivery_gate_and_deferred_hardware_gate():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "PC 本地 DDS 仿真是当前 PC 本地核心功能门禁",
            "pass/fail authority 是 `tools/run_cli_local_e2e.py --full-scene --all-scenes --head-state stationary`",
            "`val-data` full-scene matrix + Botified event oracle",
            "真实 runtime server/CLI",
            "DDS gaze subscriber/Botified stdout collector",
            "`overall_scope=current_pc_core_gate`",
            "`current_pc_core_gate_pass`",
            "RK3588/board/real robot/field validation 是 GA 之后的硬件适配/现场验证，不是 deferred current PC core gate，也不阻塞当前 PC 本地核心功能门禁",
            "PC evidence 只能声称 PC-simulated GA passed",
            "不得声称 `real robot validated`、`board compatible`、`RK supported`、`field GA passed` 或 release audit passed",
            "valid tracking 时头部物理指向目标",
            "invalid/stale 后不继续动",
            "restart 后无残留动作",
        ],
    )
    assert "PC 本地 E2E 的定位是 regression gate" not in text
    assert "PC green 不等价于板端 green" not in text


def test_ga_plan_pins_head_state_latency_botified_and_runtime_hard_gates():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "head_state.required=true",
            "required = true",
            "head_state_publisher_mode",
            "head_state_hz",
            "current PC core gate 标准命令为 `--head-state stationary`",
            "只要求 stationary head state 新鲜",
            "stationary/moving/unknown 三段覆盖保留为历史 partial smoke / unit-integration evidence",
            "moving/unknown 下运动敏感事件会被 suppression，不作为 Botified oracle core pass",
            "basic finite latency sanity",
            "non_blocking_gaps",
            "不阻塞本次 current PC core gate",
            "除非后续单独实现 blocking latency gate",
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


def test_ga_plan_documents_current_runtime_smoke_without_stale_local_failure():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "`tools/run_runtime_smoke.py`",
            "`uv sync --frozen --no-dev --no-editable --extra inference --reinstall-package visual-events-server`",
            "repo-local `runtime/cache/uv`",
            "repo-local `runtime/venv`",
            "server + CLI runtime provenance",
            "CLI import check",
            "server `/healthz` identity",
            "`runtime_hash=null`",
            "`runtime_provenance_not_run:sync_failed`",
        ],
    )
    assert "runtime/venv/bin/visual-events-cli` 缺失失败" not in text
    assert "当前本机真实 preflight 用 `runtime/venv/bin/...`" not in text


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
            "`docs/dds-stack-decision-record.md`",
            "DDS stack decision record 已冻结",
            "Unitree SDK2 2.0.0 + CycloneDDS 0.10.2 + C++ native DDS helper/bridge",
            "IDL codegen gate",
            "native bridge ABI",
            "runtime env vars",
            "PC loopback smoke plan",
            "板端 probe plan",
            "不代表板端兼容或真机闭环已完成",
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
            "Post-GA head pointing 误差",
            "P95 <=8 度",
            "Invalid/stale 停止延迟",
            "250ms 内发布 `valid=false,state=stale`",
            "500ms 内停止继续跟随旧目标",
            "等价闭环消费同一个 `/visual_events/gaze_target` DDS topic",
            "真实或 HIL head_state >=9Hz",
            "由运控 owner sign-off",
            "shadow consumer/logs 只能作为 preflight",
            "GA 硬 gate 只包含 PC runtime path 可验证的阈值",
            "不是 GA pass/fail authority",
        ],
    )


def test_ga_plan_pins_review_followup_contract_boundaries():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "nominal 10Hz DDS image 输入",
            "nominal 10Hz 输出 gaze target",
            "目标 >=9Hz 且 <=10Hz",
            "不承诺断线期间继续 >=9Hz heartbeat",
            "`CameraFrame_.camera_name` 是 DDS source camera name",
            "`[camera].name` 是发给 server、Botified frame、gaze target 和 report 的逻辑相机名",
            "默认仍为 `front`",
            "`dds_source_camera_name`",
            "`logical_camera_name`/`camera.name`",
            "manifest/oracle 标注的短遮挡、lost hold/cooldown 和空间邻近窗口",
            "不承诺 ReID/长期记忆",
            "重复招呼风险控制仍由 server semantic event rules",
            "Production runner/lifecycle unit core 已覆盖",
            "`RuntimeFactories/run_runtime`",
            "start/run/shutdown lifecycle unit core",
            "stop_requested cleanup seam",
            "Botified task 启停",
            "required head_state 模式",
            "默认 DDS factories fail-fast",
            "PC DDS E2E/over-wire gate",
            "sync/async resource close",
            "service client public close",
            "coordinator shutdown",
            "Botified drain daemon-thread bounded shutdown",
            "shutdown observe timeout 内观察到的 BrokenPipe publish stale then nonzero",
            "start failure cleanup",
            "DDS over-wire 和 serialization/QoS 行为必须被测试",
            "release report",
            "真机 smoke 不能替代 production runner/lifecycle unit core",
            "native DDS bridge/helper",
            "aarch64/RK3588 build/probe",
            "explicit unsupported fail-fast",
            "不能把 bridge 兼容问题藏到 server backend",
            "Botified request frame contract test",
            "wrapper",
            "JSON 字段",
            "ttl/timeout 语义",
            "错误/ack 期望",
            "manifest 中所有 GA scene",
            "计划不硬编码 scene 数量",
            "Current PC core gate config 必须显式验证 target dwell >=750ms",
            "当前 server 默认 500ms 不能作为 PC core gate pass 证据",
        ],
    )
    assert "正式 start/run/shutdown/reconnect 单元核心" not in text
    assert "signal/process cleanup path" not in text
    assert "BrokenPipe during shutdown publish stale then nonzero" not in text
    assert "同一物理人短暂 lost/恢复期间不得产生新的招呼型事件序列" not in text


def test_ga_plan_pins_botified_backpressure_and_broken_pipe_semantics():
    text = read_text(GA_PLAN)

    assert_contains_all(
        text,
        [
            "Botified stdout writer：bounded queue/drop/coalescing + BrokenPipe exception unit core",
            "slow stdout bounded queue/drop/coalescing 不阻塞 gaze stale",
            "stdout bounded queue/drop/coalescing 不阻塞 gaze stale",
            "RuntimeCoordinator/main wiring unit core",
            "main runtime_runner 注入和默认 DDS factories fail-fast",
            "`RuntimeFactories/run_runtime` production runner/lifecycle unit core",
            "测试注入 factories",
            "start image/head/gaze",
            "head current_motion wiring",
            "coordinator shutdown",
            "sync/async resource close",
            "service client public close",
            "Botified drain daemon-thread bounded shutdown",
            "exact stale deadline",
            "slow Botified drain 不阻塞 stale",
            "BrokenPipe publish stale then nonzero unit core",
            "shutdown observe timeout 内观察到的 BrokenPipe publish stale then nonzero",
            "start failure cleanup",
            "Botified stdout BrokenPipe 时必须尽力发布一次 stale，然后受控非 0 退出",
            'broken_pipe = "publish_stale_then_exit_nonzero"',
            "BrokenPipe 受控非 0 退出",
        ],
    )
    assert 'broken_pipe = "fail_fast"' not in text
    assert "BrokenPipe during shutdown publish stale then nonzero" not in text
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
            "Step 3 不直接链接真实 DDS SDK",
            "native bridge runtime/wiring 属于 Step 4",
            "expected attention target timeline/rule（target label/track、allowed switch windows、no-target windows）",
            "现场 checklist 必须验证 expected attention target timeline/rule",
            "进入、路过、靠近、停留、挥手等每类事件有 expected occurrence 和允许延迟窗口",
            "负例不得触发",
            "manifest/oracle 标注的短遮挡、lost hold/cooldown 和空间邻近窗口",
            "同一物理人不得产生新的招呼型事件序列",
            "`track_id` 变化时也必须通过 scene/person label 发现",
            "不承诺 ReID/长期记忆",
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
    current_step4_status = paragraph_containing(text, "当前 repo 已完成 Step 4 first slice/unit core")

    assert_contains_all(
        current_step4_status,
        [
            "Step 4 native generated DDS type/ABI mapping construction slice 已完成",
            "`CameraFrame_ -> CameraJpegFrame`",
            "`HeadStateV1_ -> HeadStateFrame`",
            "`GazeTargetFrame -> GazeTargetV1_`",
            "camera/head/gaze field mapping",
            "Step 4 native Unitree Channel construction harness/smoke slice 已完成",
            "Step 4 native runtime loop core/full-bridge wiring/fake harness/build include fix slice 已完成",
            "`runtime_loop` core",
            "full-bridge `visual_events_dds_bridge` 无参数路径进入 Unitree DDS runtime loop",
            "只证明 native runtime loop；current PC core gate 的 full-scene/Botified 证据见 Step 5，不证明 release report、RK/board 或真机闭环",
        ],
    )

    assert_contains_all(
        text,
        [
            "当前 repo 已完成 DDS contract/schema Step 1 的主要产物",
            "Step 1 的 DDS stack decision record 已冻结",
            "当前验收口径：PC 本地核心功能门禁 pass 是 repo-local PC DDS emulation",
            "硬件/现场/release audit 证据属于 post-GA validation 或交付审计层",
            "当前 repo 已完成 CLI Step 3A skeleton + Step 3B pure logic",
            "### Step 3：实现正式 CLI core（3A skeleton/3B pure logic、RuntimeCoordinator/main wiring 和 production runner/lifecycle unit core 已完成；current PC core gate 已由 Step 5 编排）",
            "`RuntimeFactories/run_runtime` production runner/lifecycle unit core 已完成",
            "### Step 4：实现 DDS adapters（first slice/unit core、native runtime loop 和 current PC over-wire core gate 已完成）",
            "Step 5 manifest reader/report skeleton slice 已完成",
            "Step 5 `run_cli_local_e2e` current PC core functional gate 已完成",
            "`tools/run_cli_local_e2e.py --full-scene --all-scenes --head-state stationary`",
            "`val-data` full-scene matrix",
            "Botified event oracle",
            "`overall_scope=current_pc_core_gate`",
            "`current_pc_core_gate_pass`",
            "`ga_gate_pass=true` 且 `ga_gate_status=pc_simulated_ga_pass`",
            "失败时 `ga_gate_pass=false` 且 `ga_gate_status=pc_simulated_ga_fail`",
            "partial smoke/preflight 为 `ga_gate_status=not_evaluated`",
            "real robot/field/RK/release audit 未覆盖写入 `post_ga_not_covered`",
            "current PC core gate 已覆盖 PC 本地 DDS over-wire 核心路径",
            "CLI 当前 PC gate 必须有 unit、integration、PC local E2E 和必要轻量 fault checks",
            "真机 smoke/closed-loop 是 post-GA hardware/field validation",
        ],
    )
    assert "Step 1 仍未完成 DDS stack decision record" not in text
    assert "尚未完成 CLI runtime loop、service_client、frame_pump、真实 DDS adapters、Botified stdout writer/backpressure" not in text
    assert "完整 PC 本地 DDS E2E GA gate 仍未完成" not in text
    assert "runner 仍不使用 val-data oracle 作为 full matrix" not in text
    assert "不证明 full PC gate、full event oracle、Botified 端到端业务" not in text
    assert "CLI 必须有 unit、integration、PC local E2E、fault matrix、真机 smoke" not in text
    assert "`overall_pass=true`" not in text
    assert "RK/board probe 已完成" not in text
    assert "真机 smoke 已完成" not in text


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


def test_ga_plan_repeats_formal_cli_bridge_opt_in_status_in_three_summaries():
    text = read_text(GA_PLAN)
    required = [
        "formal CLI bridge runtime opt-in slice 已完成",
        "默认仍 fail_fast，不因 env 隐式切 bridge",
        '显式 `[dds].runtime="bridge"`/`--dds-runtime bridge` 才走 `bridge_runtime_factories()`',
        "该 slice 不覆盖真实 DDS runtime",
        "未完成边界统一见 Step 4 剩余缺口",
    ]
    snippets = [
        paragraph_containing(text, "当前 repo 已完成 Step 4 first slice/unit core"),
        section_between(text, "### Step 4：实现 DDS adapters", "剩余缺口："),
        paragraph_containing(text, "- Step 4 first slice/unit core 已完成"),
    ]

    for snippet in snippets:
        assert_contains_all(snippet, required)


def test_ga_plan_keeps_native_runtime_loop_done_and_one_remaining_boundary():
    text = read_text(GA_PLAN)
    required_done = [
        "Step 4 native Unitree Channel construction harness/smoke slice 已完成",
        "`visual_events_dds_bridge_construction_harness`",
        "full-bridge only construction harness",
        "`runtime_options` pure env parser",
        "`--construct-once` 解析 env",
        "Unitree ChannelFactory Init(domain/network)",
        "构造 `CameraFrame_` subscriber",
        "构造 `HeadStateV1_` subscriber",
        "构造 `GazeTargetV1_` publisher",
        "Step 4 native runtime loop core/full-bridge wiring/fake harness/build include fix slice 已完成",
        "`runtime_loop` core",
        "`visual_events_dds_bridge_runtime_loop_harness` fake harness",
        "full-bridge `visual_events_dds_bridge` 无参数路径进入 Unitree DDS runtime loop",
        "stdin 读取 canonical `gaze_target` JSONL 并经 backend 发布 DDS gaze",
        "只证明 native runtime loop；current PC core gate 的 full-scene/Botified 证据见 Step 5，不证明 release report、RK/board 或真机闭环",
    ]
    snippets = [
        paragraph_containing(text, "当前 repo 已完成 Step 4 first slice/unit core"),
        section_between(text, "### Step 4：实现 DDS adapters", "剩余缺口："),
        paragraph_containing(text, "- Step 4 first slice/unit core 已完成"),
    ]

    for snippet in snippets:
        assert_contains_all(snippet, required_done)

    remaining_gap = section_between(text, "剩余缺口：", "验收：\n\n- 无 publisher")
    assert_contains_all(
        remaining_gap,
        [
            "`run_cli_local_e2e` current PC core functional gate 已完成",
            "当前 PC core gate 用正式 CLI + real server 覆盖 `/camera/image/jpeg` -> bridge/CLI -> server -> `/visual_events/gaze_target` over wire",
            "`val-data` full-scene matrix 和 Botified event oracle",
            "partial smoke 只作为历史/兼容模式保留",
            "当前 PC core gate 的 pass/fail authority 是 `tools/run_cli_local_e2e.py --full-scene --all-scenes --head-state stationary`",
            "RK/board probe、真机 smoke/closed-loop handoff、field validation、release report/handoff audit、full fault matrix 和 long soak 属于 post-GA validation 或交付审计层",
            "不阻塞当前 PC 本地核心功能门禁",
        ],
    )

    assert "完整 PC 本地 DDS E2E GA gate 仍未完成" not in text
    assert "DDS discovery/real serialization over wire/QoS behavior 的 PC runtime path 测试仍未完成" not in text
    assert "runner 仍不使用 val-data oracle 作为 full matrix" not in text
    assert "真机 smoke/closed-loop 已完成" not in text


def test_ga_plan_records_step5_native_participants_and_current_pc_core_gate():
    text = read_text(GA_PLAN)
    step5 = section_between(text, "### Step 5：实现 PC 本地测试工具", "### Step 6：CLI 单元与集成测试")

    assert_contains_all(
        step5,
        [
            "Python `tools/*.py` 是 runner/wrapper",
            "真实 DDS participants 由 native full-bridge binaries 承担",
            "Step 5 native PC DDS over-wire test participant slice 已完成",
            "`visual_events_dds_bridge_publish_test_dds_images`",
            "`visual_events_dds_bridge_publish_test_head_state`",
            "`visual_events_dds_bridge_subscribe_test_gaze_targets`",
            "loopback over-wire smoke",
            "Step 5 Python native participant wrappers slice 已完成",
            "`tools/dds_pc_tools.py`",
            "`tools/publish_test_dds_images.py`",
            "`tools/publish_test_head_state.py`",
            "`tools/subscribe_test_gaze_targets.py`",
            "Python wrappers 不 import DDS SDK/visual_events_cli/server",
            "Step 5 manifest reader/report skeleton slice 已完成",
            "`tools/cli_local_e2e_manifest.py`",
            "manifest skeleton report 固定 `overall_pass=false`",
            "`pc_local_e2e_status=not_run`",
            "Step 5 server replay manifest contract evidence slice 已完成",
            "`tools/run_val_data_e2e.py`",
            "`manifest_contract_required`",
            "`manifest_contract_satisfied`",
            "server replay oracle 迁移不是 current PC core gate 的阻塞项",
            "Step 5 mock visual_state server slice 已完成",
            "Step 5 CLI runtime + mock visual_state server integration slice 已完成",
            "`tests/integration/test_cli_bridge_runtime_mock_server.py`",
            "不使用 DDS participant、`val-data` 或 real server",
            "不能替代 current PC core gate",
            "Step 5 `run_cli_local_e2e` current PC core functional gate 已完成",
            "`tools/run_cli_local_e2e.py --full-scene --all-scenes --head-state stationary`",
            "启动正式 server binary",
            "正式 CLI binary with bridge runtime",
            "gaze subscriber wrapper",
            "head/image publisher wrappers",
            "跑 `val-data` full-scene matrix",
            "Botified event oracle 判定 stdout 事件",
            "`overall_scope=current_pc_core_gate`",
            "`current_pc_core_gate_pass`",
            "`ga_gate_pass=true` 且 `ga_gate_status=pc_simulated_ga_pass`",
            "失败时 `ga_gate_pass=false` 且 `ga_gate_status=pc_simulated_ga_fail`",
            "partial smoke/preflight 为 `ga_gate_status=not_evaluated`",
            "`post_ga_not_covered`",
            "历史 partial smoke 证据保留为背景",
            "不能再被解释为当前只有 partial smoke",
            "current PC core gate 的 manifest/oracle 必须列出 scene 名称",
        ],
    )

    assert "runner 仍不使用 val-data oracle 作为 full matrix" not in text
    assert "当前只声明 partial smoke" not in text
    assert "不是 full PC gate" not in text
    assert "不证明 full PC gate、full event oracle、Botified 端到端业务" not in text
    assert "完整 PC 本地 DDS E2E GA gate 仍未完成" not in text
    assert "`overall_pass=true`" not in text
    assert "fault matrix 已完成" not in text
    assert "release report 已完成" not in text


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


def test_no_motion_audit_does_not_claim_hardware_motion_validation():
    text = read_text(NO_MOTION_AUDIT)

    assert_contains_all(
        text,
        [
            "no-motion audit + PC DDS emulation 不验证硬件运动/真实头部执行",
            "does not validate hardware motion or real head execution",
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
