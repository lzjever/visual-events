from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GA_PLAN = REPO_ROOT / "docs" / "ga-development-plan.md"
DDS_STACK_DECISION = REPO_ROOT / "docs" / "dds-stack-decision-record.md"
PROTOCOL = REPO_ROOT / "common" / "schema" / "protocol.md"
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
        ],
    )
    assert "box_norm_width" not in text
    assert "box_norm_height" not in text


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
            "`docs/dds-stack-decision-record.md`",
            "DDS stack decision record 已冻结",
            "Unitree SDK2 2.0.0 + CycloneDDS 0.10.2 + C++ native DDS helper/bridge",
            "IDL codegen gate",
            "native bridge ABI",
            "runtime env vars",
            "PC loopback smoke plan",
            "板端 probe gate",
            "不代表真实 DDS factories/adapters、PC E2E、板端兼容或真机闭环已完成",
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
            "真实 DDS factories/adapters wiring",
            "sync/async resource close",
            "service client public close",
            "coordinator shutdown",
            "Botified drain daemon-thread bounded shutdown",
            "shutdown observe timeout 内观察到的 BrokenPipe publish stale then nonzero",
            "start failure cleanup",
            "真实 serialization/QoS construction tests",
            "release/runtime 真跑",
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
            "GA release config/gate 必须显式设置并验证满足 750ms",
            "当前 server 默认 500ms 不能被当作 GA pass 证据",
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
            "Step 3 不实现真实 DDS factories/adapters",
            "Step 4 真实 DDS factories/adapters",
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

    assert_contains_all(
        text,
        [
            "当前 repo 已完成 DDS contract/schema Step 1 的主要产物",
            "`common/schema/dds/camera_jpeg_contract.md`",
            "`common/schema/dds/gaze_target_v1.idl` 和 `gaze_target_v1.md`",
            "`common/schema/dds/head_state_v1.idl` 和 `head_state_v1.md`",
            "gaze target samples",
            "no-motion-SDK audit 覆盖",
            "`docs/dds-stack-decision-record.md`：DDS stack decision record 已冻结",
            "选择 Unitree SDK2 2.0.0 + CycloneDDS 0.10.2 + C++ native DDS helper/bridge",
            "IDL codegen/toolchain、真实 bridge/factories 和验证仍属于 Step 4",
            "Step 1 的 DDS stack decision record 已冻结",
            "不代表真实 DDS factories/adapters、PC E2E、板端兼容或真机闭环已完成",
            "当前 repo 已完成 CLI Step 3A skeleton + Step 3B pure logic",
            "### Step 3：实现正式 CLI core（3A skeleton/3B pure logic、RuntimeCoordinator/main wiring 和 production runner/lifecycle unit core 已完成；真实 DDS factories/adapters 剩余）",
            "CLI package",
            "`visual-events-cli` entrypoint",
            "配置 skeleton",
            "`target_mapper` 纯逻辑",
            "`botified_output` 纯逻辑",
            "并已完成 CLI unit core 的 `service_client` WebSocket wire client",
            "`frame_pump` deterministic core/stale watchdog",
            "RuntimeCoordinator/main wiring unit core",
            "main runtime_runner 注入和默认 DDS factories fail-fast",
            "exact stale deadline",
            "slow Botified drain 不阻塞 stale",
            "Botified stdout bounded queue/drop/coalescing 与 BrokenPipe exception 单元核心",
            "BrokenPipe publish stale then nonzero unit core",
            "`RuntimeFactories/run_runtime` production runner/lifecycle unit core 已完成",
            "默认 DDS factories fail-fast 且不 import DDS/native",
            "测试注入 factories",
            "start image/head/gaze",
            "head current_motion wiring",
            "coordinator shutdown",
            "sync/async resource close",
            "service client public close",
            "Botified drain daemon-thread bounded shutdown",
            "shutdown observe timeout 内观察到的 BrokenPipe publish stale then nonzero",
            "start failure cleanup",
            "当前 repo 已完成 Step 4 first slice/unit core",
            "纯 Python SDK-neutral DDS adapter core/fakes",
            "`visual_events_cli.dds.qos`",
            "`visual_events_cli.dds.types`",
            "`visual_events_cli.dds.protocols`",
            "`visual_events_cli.dds.fake`",
            "QoS constants",
            "CameraJpegMessage JPEG SOF dimension validation",
            "fake image latest-only",
            "Fake DDS adapters lifecycle unit core（start/close idempotent；close 后拒绝使用/重启）",
            "HeadStateSample stationary/moving/unknown stale/future timestamp mapping",
            "FakeDdsGazeTargetPublisher lifecycle",
            "protocol names",
            "no-motion/no-real-DDS import audit",
            "不 import 真实 DDS SDK/ML/运控依赖",
            "Step 4 Python JSONL bridge client/facade slice 已完成",
            "`bridge_protocol.py`",
            "`bridge_process.py`",
            "`bridge_adapters.py`",
            "`runtime_factories.py`",
            "explicit `bridge_runtime_factories()`",
            "JSONL protocol/base64/canonical gaze fields",
            "subprocess lifecycle",
            "no DDS/native import audit tests",
            "Step 4 Python JSONL bridge runtime integration slice 已完成",
            "formal CLI bridge runtime opt-in slice 已完成",
            "真实 subprocess fake JSONL child + `bridge_runtime_factories()`/`run_runtime`",
            "camera/head -> service -> gaze stdin",
            "logical camera",
            "stale/cleanup",
            "child nonzero/fatal",
            "Step 4 native DDS bridge build/probe foundation slice 已完成",
            "`native/dds_bridge` CMake project",
            "`visual_events_dds_bridge_probe` probe target",
            "Unitree SDK2 + `CameraFrame_` build inputs",
            "camera/head/gaze topic/type/QoS constants",
            "单行 JSONL status frame（`protocol_version=1,type=status,code=probe_ok,message=...`）",
            "`tools/build_dds_bridge.py` split gate",
            "foundation check/build/probe 只要求 SDK root、video publisher dir 和 `CameraFrame_` inputs",
            "可在无 IDL generator 时成功",
            "`foundation_ready=true`",
            "`visual_events_codegen_ready=false`",
            '`visual_events_codegen_error="not required for foundation check"`',
            "Step 4 DDS C++ idlc repo-local prepare/oracle hardening slice 已完成",
            "`tools/prepare_dds_codegen_toolchain.py`",
            "默认 pinned CycloneDDS/CycloneDDS-CXX 0.10.2",
            "`build/tools/cyclonedds-cxx-idlc-0.10.2/`",
            "不下载、不构建、不写系统或用户目录",
            "只做版本、路径、显式 `idlc` 和 cxx backend 文本检查",
            "`probe_codegen=false`、`oracle_ok=false`",
            "`--probe-codegen` 是显式非 dry-run oracle",
            "`idlc -l cxx -o <probe-output-dir> <probe-idl>`",
            "`cannot load generator`/`cannot load generator cxx`",
            "`generated_files`、`expected_generated_files_present`、`cxx_backend_available` 和 `oracle_ok`",
            "`--prepare` 是显式非 dry-run toolchain 编排",
            "固定 ignored `build/tools/cyclonedds-cxx-idlc-0.10.2/` layout",
            "验证 git tag commit",
            "创建 `bin/idlc-cxx` wrapper",
            "它不接受 `--idlc` 且不使用 `VISUAL_EVENTS_IDLC`",
            "fake git/cmake/idlc 覆盖成功生成",
            "clone/artifact/oracle failure",
            "`tools/build_dds_bridge.py --check-full-bridge` 不再搜索 PATH",
            "只接受显式 `--idlc` 或 `VISUAL_EVENTS_IDLC`",
            "复用同一个 codegen probe",
            "expected `.hpp/.cpp` 都写出",
            "`visual_events_codegen_ready=true`",
            "外部源码/build/install/probe 输出不进 Git",
            "Head/Gaze generated type support 仍未生成或接入",
            "full bridge runtime",
            "real serialization/QoS",
            "真实 DDS/C++/IDL/QoS/PC E2E/RK/真机仍未完成",
            "`service_client`：WebSocket wire/pack-unpack、连接复用/关闭、timeout、invalid response、frame_id mismatch、retryable/non-retryable error handling 的单元核心",
            "`frame_pump`：one in-flight coordination、keep-latest frame slot/backpressure、gaze stale watchdog、Botified enqueue 的 deterministic unit core",
            "main runtime_runner 注入和默认 DDS factories fail-fast",
            "exact stale deadline",
            "slow Botified drain 不阻塞 stale",
            "Botified stdout writer：bounded queue/drop/coalescing + BrokenPipe exception unit core",
            "BrokenPipe publish stale then nonzero unit core",
            "真实 DDS factories/adapters 仍未完成",
            "当前默认 factories 只 fail-fast，不是真实 DDS runtime",
            "Step 3 不实现真实 DDS factories/adapters",
            "### Step 4：实现 DDS adapters（first slice/unit core 已完成；真实 runtime adapters 剩余）",
            "真实 DDS runtime adapters",
            "按 `docs/dds-stack-decision-record.md` 已冻结决策实现 C++ native DDS helper/bridge",
            "Head/Gaze generated type support 接入",
            "real serialization/QoS tests",
            "`native/dds_bridge` CMake project 可构建 very small `visual_events_dds_bridge_probe`",
            "`tools/build_dds_bridge.py` 拆分 foundation gate 和 full-bridge gate",
            "`foundation_ready`、`visual_events_codegen_ready`、`visual_events_codegen_error`",
            "`tools/prepare_dds_codegen_toolchain.py` 默认 pinned CycloneDDS/CycloneDDS-CXX 0.10.2",
            "`build/tools/cyclonedds-cxx-idlc-0.10.2/codegen_probe/`",
            "`--prepare` 显式准备 ignored repo-local CycloneDDS/CycloneDDS-CXX 0.10.2 C++ idlc toolchain",
            "`tools/build_dds_bridge.py --check-full-bridge` 不再搜索 PATH",
            "缺显式 idlc、版本不是 0.10.2、backend 不能实际加载或缺 expected `.hpp/.cpp` 时 fail-fast",
            "`oracle_ok=true`",
            "只证明 repo-local prepare 编排和 C++ codegen probe 会拒绝假阳性",
            "只证明 camera/probe foundation",
            "不实现 `HeadStateV1_`/`GazeTargetV1_` C++ type support",
            "不实现 full bridge runtime loop",
            "不证明真实 serialization/QoS、PC E2E、board/RK 或真机闭环",
            "PC E2E",
            "native bridge ABI implementation",
            "real serialization/QoS construction tests",
            "real serialization/QoS tests",
            "板端 compatibility probe",
            "board/RK probe",
            "真实 DDS factories/adapters",
            "PC 本地 DDS E2E tools",
            "PC E2E",
            "release/runtime 真跑",
            "真机 smoke",
            "closed-loop handoff",
            "剩余是 Step 4 真实 DDS factories/adapters",
            "DDS contract/schema Step 1 主要产物已完成",
            "DDS runtime stack 和板端 compatibility probe 仍必须补齐",
            "真实 DDS factories/adapters、Visual Events `HeadStateV1_`/`GazeTargetV1_` generated C++ type support 接入、full bridge runtime、real serialization/QoS 和 native bridge ABI implementation 仍未完成",
            "`visual_events_dds_bridge_probe` 可输出既有 JSONL bridge ABI status frame（`protocol_version=1,type=status,code=probe_ok,message=...`）",
            "`tools/build_dds_bridge.py` foundation gate 可在无 IDL generator 时通过并报告 `foundation_ready=true`",
            "`tools/prepare_dds_codegen_toolchain.py --prepare` 可在 ignored `build/tools/cyclonedds-cxx-idlc-0.10.2/` 下准备 pinned idlc wrapper 并复用 codegen oracle",
            "`--probe-codegen` 和 `tools/build_dds_bridge.py --check-full-bridge` 会实际运行 C++ idlc probe",
            "拒绝版本不是 0.10.2、`cannot load generator cxx`、只生成 `.hpp` 或缺 `.cpp` 的假阳性",
            "这只证明 repo-local prepare 编排和 generator oracle hardening 完成",
            "真实 DDS factories/adapters、Visual Events `HeadStateV1_`/`GazeTargetV1_` generated C++ type support 接入、full bridge runtime、real serialization/QoS tests / construction tests、板端 compatibility probe、PC E2E tools、board/RK probe、release/runtime 真跑、真机 smoke/closed-loop handoff 仍未完成",
        ],
    )
    assert "Step 1 仍未完成 DDS stack decision record" not in text
    assert "尚未实际确定 SDK/bridge runtime choice" not in text
    assert "SDK/bridge decision record：SDK/bridge 名称和版本" not in text
    assert "真实 DDS factories/adapters、SDK/bridge decision record" not in text
    assert "尚未完成 CLI runtime loop、service_client、frame_pump、真实 DDS adapters、Botified stdout writer/backpressure" not in text
    assert "仍需完成 runtime loop、真实 DDS adapters、Botified stdout writer、PC E2E tools 和 release/runtime 编排" not in text
    assert "仍未完成 Step 3 formal CLI runtime loop/main wiring" not in text
    assert "仍需完成 Step 3 formal CLI runtime loop/main wiring" not in text
    assert "production runner/lifecycle 剩余" not in text
    assert "production runner/lifecycle 依赖 Step 4 真实 DDS adapters 实例化 RuntimeCoordinator" not in text
    assert "生产 runtime runner/lifecycle 编排需要 Step 4 真实 DDS adapters 实例化 RuntimeCoordinator" not in text
    assert "runtime coordinator 对 BrokenPipe 的 publish stale then nonzero exit 处理" not in text
    assert "Runtime coordinator 对 BrokenPipe 的 publish stale then nonzero exit 处理" not in text
    assert "Step 4 真实 DDS adapters 全部未开始" not in text
    assert "真实 DDS adapters 全部未开始" not in text
    assert "真实 DDS factories/adapters 已完成" not in text
    assert "Step 4 完全完成" not in text
    assert "full bridge ready" not in text
    assert "Step 4 DDS C++ IDL codegen toolchain proof slice 已完成" not in text
    assert "但这只证明 pinned generator route" not in text
    assert "真实 CycloneDDS 0.10.2 toolchain 已准备完成" not in text
    assert "检查 SDK root、video publisher dir 和 IDL generator/toolchain，缺失时 fail-fast" not in text
    assert "随便依赖 PATH 上的 idlc" not in text
    assert "Step 4 Python JSONL bridge runtime integration slice 仍未完成" not in text
    assert "真实 DDS/C++ bridge/PC E2E/RK probe 已完成" not in text
    assert "Step 4 Python JSONL bridge runtime integration slice 完成代表真实 DDS" not in text
    assert "Fake DDS adapters started/closed lifecycle unit core" not in text
    assert "可重复 start/stop" not in text


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
        "真实 DDS/C++/IDL/QoS/PC E2E/RK/真机仍未完成",
    ]
    snippets = [
        paragraph_containing(text, "当前 repo 已完成 Step 4 first slice/unit core"),
        section_between(text, "### Step 4：实现 DDS adapters", "剩余缺口："),
        paragraph_containing(text, "- Step 4 first slice/unit core 已完成"),
    ]

    for snippet in snippets:
        assert_contains_all(snippet, required)


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
