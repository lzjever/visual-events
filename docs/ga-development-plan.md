# Visual Events GA 后续开发计划

日期：2026-06-27

## 1. 目标

术语口径：本文里的 manifest 指 `val-data` 测试数据清单；oracle 指测试标准答案；attention oracle / expected attention target timeline 指注视目标标准答案，也就是 PC 测试里“应该看谁、什么时候不该看”。它们只属于 PC 本地 E2E 验收/测试报告，不是产品运行功能，不是治理平台，不是审计平台，也不是 release audit 层。authoritative 只表示“这份测试数据清单/标准答案是 PC 测试采用的版本”，不表示组织级治理、审计或发布批准；当前不新增 manifest builder、schema 审计、release audit、handoff audit 或 strict gate 扩张。

本计划覆盖 `visual-events` 从当前 server S0-S8 baseline 走到 GA 的剩余工作和当前收口状态。当前 PC 本地核心功能门禁（current PC core functional gate）已经收敛到真实 server、正式 CLI runtime、DDS image/head/gaze、`val-data` full-scene matrix 和 Botified event oracle；直接保护这条路径的必要轻量检查仍是当前核心工作。release report skeleton、handoff audit、manifest/schema 审计扩张、full fault matrix、long soak、field/real robot validation 属于 GA 后交付审计/硬件层，不在当前核心实现切片中扩张。

首个 GA 场景是商店门口揽客机器人。系统必须在 nominal 10Hz DDS image 输入下稳定观察前方画面，识别人、追踪人、生成低频语义事件，并以 nominal 10Hz 输出 gaze target：每个新鲜 `visual_state` 派生一条 valid 或 invalid gaze target（实际目标 >=9Hz 且 <=10Hz），让机器人可以注视画面中最大且稳定的人。CLI/bridge 是 keep-latest + one-in-flight，等待期间可丢弃旧 DDS 输入帧，不补发旧帧、不用重复旧 target 凑频率。断线或状态过期后，CLI 在 250ms 内发布一次 invalid/stale gaze target，然后依赖 DDS lifespan 失效，不承诺断线期间继续 >=9Hz heartbeat。

核心边界：

- `visual-events-server` 只做视觉推理、追踪、attention 和语义事件生成。
- `visual-events-cli` 订阅 DDS JPEG，调用 server，发布 DDS gaze target，输出 Botified request frame。
- `visual-events-cli` 不直接操纵运控，不调用头部速度/位置/`look_at` API，不链接运动控制 SDK。
- 运控/头控模块订阅 DDS gaze target，并在自己的安全边界内执行真实头部动作；该模块不在本 repo 实现。
- Botified agent 只接收低频语义事件并决定后续响应，不参与 10Hz 注视闭环。
- PC 本地 DDS 仿真是当前 PC 本地核心功能门禁；GA pass/fail authority 只要求 `tools/run_cli_local_e2e.py --full-scene --all-scenes --head-state stationary --server-config configs/pc-ga-server.toml` 在 PC 本地模拟完整 E2E 跑通：本地开发 PC 模拟 robot 发送 DDS camera/head-state，真实 runtime server/CLI 跑通，DDS gaze subscriber/Botified stdout collector 收到预期输出，并用 `val-data` full-scene matrix + Botified event oracle 做当前 PC 模拟核心功能判定。report 可记录 `overall_scope=current_pc_core_gate` 和 `current_pc_core_gate_pass`。
- RK3588/board/real robot/field validation 是 GA 之后的硬件适配/现场验证，不是 deferred current PC core gate，也不阻塞当前 PC 本地核心功能门禁；GA 不要求真机实际运行、真机测试、现场测试、RK3588/board 验证或真实头部闭环。PC evidence 只能声称 PC-simulated GA passed，不得声称 `real robot validated`、`board compatible`、`RK supported`、`field GA passed` 或 release audit passed。
- 当前阶段 PC gate 必须有可用、新鲜的 head state；缺失 head state 只能算 degraded 运行，不能算 current phase pass / PC gate pass。
- `attention_target_changed` 只保留在 `visual_state` 和诊断 artifact 中，不输出到 Botified stdout。

## 2. 开发原则

这些原则必须写进实现 review checklist：

- KISS：一个 DDS 图像输入，一个 WebSocket 推理协议，一个 DDS gaze target 输出，一个 Botified 事件出口。
- DRY：事件规则只在 server 实现；CLI 只做事件幂等输出，不重新判断 passing by、approaching、stopped 或 waving。
- YAGNI：不训练模型，不做人脸识别，不做 ReID，不做多摄像头融合，不做数据库，不做后台治理平台，不引入第二套 streaming 协议。
- 本体安全边界清晰：CLI 只发布目标事实和有效期，不发布速度/位置命令，不直接控制硬件。
- 可测试优先：PC 本地 DDS 端到端测试是 current PC core functional gate，不是 demo；硬件/现场证据属于 post-GA validation。
- 运行隔离：继续使用 `uv`；开发环境和 release/runtime 环境分开；不污染系统和用户目录；模型、cache、artifacts 不进 Git。
- Artifact 可追溯：`val-data/` 不进 Git；当前只保留能直接保护核心 runtime path 的有限 evidence，如 manifest 身份、runtime/config hash、Botified event oracle 和关键 PC E2E 摘要。不要把 release report/handoff audit 当成当前实现切片。
- 输出隔离：CLI stdout 是 Botified allowlist 输出，不是调试通道；高频 attention/gaze 状态不得泄漏到 Botified。
- 治理克制：governance/report/audit/gate 层只能在直接保护核心运行边界或用户明确要求时添加。开发顺序以 server/CLI core runtime path 和 PC local E2E 为先；release report skeleton、manifest/schema 审计层和 handoff audit 不早于 core server/CLI MVP。已完成的 manifest/evidence/strict gate 工作保留为有限 evidence，不再继续扩张，不能替代 MVP runtime path。
- TDD 边界：只对核心功能和高风险集成做 TDD；不要为测试工具、报告骨架、文档文字再堆“测试测试”。

## 3. 当前基线

当前 repo 已完成 server S0-S8 baseline 和部分 GA 收口产物：

- WebSocket `/v1/stream` 协议。
- `YOLOv8n-pose` 推理 backend 和 mock backend。
- 项目内 ByteTrack-style IoU/TTL tracker baseline。
- attention selector。
- semantic events：`person_appeared`、`person_left`、`person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot`、`person_waving`、`attention_target_changed`。
- `val-data/` full matrix、semantic event timeline gate、runtime smoke、300s soak、opt-in server metrics evidence。
- server GA 收口 Step 2 已完成：frame/timestamp regression reset、JPEG dimensions validation、`scene_flags.someone_near_center`、shared backend inference serialization、metrics sink write error count/stderr、server/CLI contract tests。

当前 repo 已完成 DDS contract/schema Step 1 的主要产物：

- `common/schema/dds/camera_jpeg_contract.md`
- `common/schema/dds/gaze_target_v1.idl` 和 `gaze_target_v1.md`
- `common/schema/dds/head_state_v1.idl` 和 `head_state_v1.md`
- gaze target samples
- no-motion-SDK audit 覆盖
- `docs/dds-stack-decision-record.md`：DDS stack decision record 已冻结，选择 Unitree SDK2 2.0.0 + CycloneDDS 0.10.2 + C++ native DDS helper/bridge；IDL codegen/toolchain、真实 bridge/factories 和验证仍属于 Step 4。

Step 1 的 DDS stack decision record 已冻结；这只固化 SDK/bridge runtime choice、IDL codegen gate、native bridge ABI、runtime env vars、PC 安装/loopback smoke plan 和板端 probe plan，不代表板端兼容或真机闭环已完成。当前验收口径：PC 本地核心功能门禁 pass 是 repo-local PC DDS emulation；硬件/现场/release audit 证据属于 post-GA validation 或交付审计层。

当前 repo 已完成 CLI Step 3A skeleton + Step 3B pure logic + RuntimeCoordinator/main wiring unit core：CLI package、`visual-events-cli` entrypoint、配置 skeleton、`target_mapper` 纯逻辑和 `botified_output` 纯逻辑；并已完成 CLI unit core 的 `service_client` WebSocket wire client、`frame_pump` deterministic core/stale watchdog、RuntimeCoordinator/main wiring unit core、main runtime_runner 注入和默认 DDS factories fail-fast、exact stale deadline、slow Botified drain 不阻塞 stale、Botified stdout bounded queue/drop/coalescing 与 BrokenPipe exception 单元核心、BrokenPipe publish stale then nonzero unit core。`RuntimeFactories/run_runtime` production runner/lifecycle unit core 已完成：默认 DDS factories fail-fast 且不 import DDS/native，测试注入 factories，start image/head/gaze，head current_motion wiring，coordinator shutdown，sync/async resource close，service client public close，Botified drain daemon-thread bounded shutdown，shutdown observe timeout 内观察到的 BrokenPipe publish stale then nonzero，start failure cleanup。
Step 5 native PC DDS over-wire test participant slice 已完成：image publisher、head publisher、gaze subscriber 和 `pc_test_tools` 已通过 loopback over-wire smoke。Step 5 Python native participant wrappers slice 已完成：`tools/dds_pc_tools.py` 和三个 wrapper 已接入，显式 `--dds-domain`/`--dds-network`，非 `lo` 必须 `--allow-non-loopback-dds`，默认只从 `--build-dir` 找 native binary，Python wrappers 不 import DDS SDK/visual_events_cli/server，wrapper-level PC loopback smoke 已通过。Step 5 manifest reader/report skeleton slice 已完成：`tools/cli_local_e2e_manifest.py` 读取 `--data-dir`，默认探测 `<data-dir>/manifest.json`（标准 `--data-dir val-data` 时为 `val-data/manifest.json`），也支持显式 `--manifest`；manifest（`val-data` 测试数据清单）缺失时生成 deterministic effective manifest，记录 scene 名称、per-scene sha256、frame count、first/last frame，并记录 aggregate `scene_count` / `frame_count` 和 `manifest_sha256`；manifest skeleton report 固定 `overall_pass=false`、`pc_local_e2e_status=not_run`，只针对 `tools/cli_local_e2e_manifest.py` 的报告骨架；只做数据集身份和报告骨架，不能替代 runtime E2E；`--out` 不能写入 data dir；本机 ignored `val-data` 当前可识别 7 scene / 576 frames。Step 5 authoritative manifest/oracle schema skeleton 已完成：`common/schema/val_data_manifest_v1.md` 固定 `schema_version=1`、fps、scene/frame count、scene sha256 和 oracle source/version/rule 字段；这里的 oracle 是测试标准答案来源，authoritative 只表示 PC 测试采用的清单/答案版本，不表示组织级治理、审计或发布批准；`tools/cli_local_e2e_manifest.py` 只验证 manifest/oracle schema 并投影 `manifest_authoritative`、`manifest_validation_errors`、`oracle_schema_present`、`oracle_schema_valid`、`oracle_summary`，不直接做 oracle evaluation；`val-data/manifest.json` 仍不进 Git。Step 5 server replay manifest contract evidence slice 已完成：`tools/run_val_data_e2e.py` 支持 `--manifest` 和 `--require-authoritative-manifest`，server replay gate report 记录 manifest/oracle contract evidence；默认无 manifest 仍使用 generated inventory 且不破坏本机 replay gate；manifest 文件 parse/I/O 错误始终 preflight fail；只有显式要求 authoritative manifest 时才把缺失、schema/contract-invalid but parseable 或 non-authoritative manifest 作为 authoritative manifest contract preflight failure；semantic event oracle 仍来自 `tools.replay_val_data.hardcoded_scene_expectations`，server replay oracle 迁移不是当前 PC core gate 的阻塞项。本段不新增 manifest builder、schema 审计、release audit、handoff audit 或 strict gate 扩张。
Step 5 `run_cli_local_e2e` current PC core functional gate 状态详见 Step 5 canonical 状态段；历史 partial smoke 说明保留为兼容/回归背景，但当前 pass/fail authority 是 full-scene/all-scenes + Botified event oracle，不应再读成“当前只有 partial smoke”。
Step 5 mock visual_state server slice 已完成：`tools/mock_visual_state_server.py` 提供 `/healthz` 和 `/v1/stream`，复用现有 `visual_events_server.protocol` decode/serialize，不定义第二套协议；支持 `tracking/lost/event` profile、`--delay-ms`、`--disconnect-after`，用于 CLI deterministic attention/event、slow response、disconnect 测试；不 import DDS/CLI runtime/模型，不启动 DDS participant，不读取 val-data，不做 runner/report。当前 PC 本地核心功能门禁由正式 CLI + real server + `val-data` full-scene matrix + Botified event oracle 覆盖；RK/board probe、真机 smoke 和 closed-loop handoff 属于 post-GA validation。

Step 5 CLI runtime + mock visual_state server integration slice 已完成：`tests/integration/test_cli_bridge_runtime_mock_server.py` 使用真实 `bridge_runtime_factories()`/`run_runtime`、fake JSONL bridge child 和 mock server subprocess，覆盖 event Botified stdout allowlist、tracking/lost/stale gaze；不使用 DDS participant、`val-data` 或 real server，不做 runner/report；不能替代 full E2E 或正式 CLI + real server + `val-data` GA gate。

当前 repo 已完成 Step 4 first slice/unit core：纯 Python SDK-neutral DDS adapter core/fakes 位于 `visual_events_cli.dds.qos`、`visual_events_cli.dds.types`、`visual_events_cli.dds.protocols`、`visual_events_cli.dds.fake`，覆盖 QoS constants、CameraJpegMessage JPEG SOF dimension validation、fake image latest-only、Fake DDS adapters lifecycle unit core（start/close idempotent；close 后拒绝使用/重启）、HeadStateSample stationary/moving/unknown stale/future timestamp mapping、FakeDdsGazeTargetPublisher lifecycle、protocol names，并有 no-motion/no-real-DDS import audit；该层不 import 真实 DDS SDK/ML/运控依赖。Step 4 Python JSONL bridge client/facade slice 已完成：`bridge_protocol.py`、`bridge_process.py`、`bridge_adapters.py`、`runtime_factories.py` 和 explicit `bridge_runtime_factories()` 覆盖 JSONL protocol/base64/canonical gaze fields、subprocess lifecycle、three thin facade wiring 和 no DDS/native import audit tests。Step 4 Python JSONL bridge runtime integration slice 已完成，formal CLI bridge runtime opt-in slice 已完成：默认仍 fail_fast，不因 env 隐式切 bridge；显式 `[dds].runtime="bridge"`/`--dds-runtime bridge` 才走 `bridge_runtime_factories()`。真实 subprocess fake JSONL child + `bridge_runtime_factories()`/`run_runtime` 覆盖 camera/head -> service -> gaze stdin、logical camera、stale/cleanup、child nonzero/fatal；该 slice 不覆盖真实 DDS runtime，未完成边界统一见 Step 4 剩余缺口。Step 4 native DDS bridge build/probe foundation slice 已完成：新增 `native/dds_bridge` CMake project、`visual_events_dds_bridge_probe` probe target、Unitree SDK2 + `CameraFrame_` build inputs、camera/head/gaze topic/type/QoS constants、单行 JSONL status frame（`protocol_version=1,type=status,code=probe_ok,message=...`）和 `tools/build_dds_bridge.py` split gate；foundation check/build/probe 只要求 SDK root、video publisher dir 和 `CameraFrame_` inputs，可在无 IDL generator 时成功，并在 report 写入 `foundation_ready=true`、`visual_events_codegen_ready=false`、`visual_events_codegen_error="not required for foundation check"`。Step 4 DDS C++ idlc repo-local prepare/oracle hardening slice 已完成：`tools/prepare_dds_codegen_toolchain.py` 保持 CycloneDDS/CycloneDDS-CXX 0.10.2 pinned 和 repo-local ignored `build/tools/cyclonedds-cxx-idlc-0.10.2/`；`--check`/`--dry-run` 不下载、不构建、不写系统或用户目录，只做版本、路径、显式 `idlc` 和 cxx backend 文本检查，并报告 `probe_codegen=false`、`oracle_ok=false`；`--probe-codegen` 是显式非 dry-run oracle，默认验证 repo Head/Gaze IDL codegen oracle only，会在 repo `build/` 下分别运行 `idlc -l cxx -o <probe-output-dir> common/schema/dds/head_state_v1.idl` 和 `common/schema/dds/gaze_target_v1.idl`，拒绝 `cannot load generator`/`cannot load generator cxx`、任一 IDL 缺 `.hpp` 或缺 `.cpp`，并报告每个 probed IDL、expected `.hpp/.cpp`、`generated_files`、per-IDL presence、`expected_generated_files_present`、`cxx_backend_available` 和 `oracle_ok`。`--prepare` 是显式非 dry-run toolchain 编排，固定 ignored `build/tools/cyclonedds-cxx-idlc-0.10.2/` layout，验证 git tag commit、运行 CMake Makefiles install、创建 `bin/idlc-cxx` wrapper、要求 installed idlc/idlcxx/ddsc artifacts，并自动复用同一个 Head/Gaze codegen oracle；它不接受 `--idlc` 且不使用 `VISUAL_EVENTS_IDLC`。fake git/cmake/idlc 覆盖成功生成、0.11.0 fail、missing cxx generator 但 rc=0 fail、只生成 `.hpp` fail、clone/artifact/oracle failure；`tools/build_dds_bridge.py --check-full-bridge` 不再搜索 PATH，只接受显式 `--idlc` 或 `VISUAL_EVENTS_IDLC`，并复用同一个 codegen probe，只有 Head/Gaze expected `.hpp/.cpp` 都写出时才报告 `visual_events_codegen_ready=true`。Step 4 native full-bridge generated Head/Gaze C++ type-support compile/probe slice 已完成：`tools/build_dds_bridge.py --check --check-full-bridge --build --probe` 会运行 Head/Gaze IDL codegen oracle，CMake full-bridge 编译 `head_state_v1.hpp/.cpp` 和 `gaze_target_v1.hpp/.cpp`，native probe 检查 `CameraFrame_`、`HeadStateV1_`、`GazeTargetV1_` type props 并输出一行 JSONL status；Foundation 路径仍然 CameraFrame-only。Step 4 native JSONL ABI/runtime skeleton slice 已完成：`visual_events_dds_bridge` target 存在，`--probe` 单行 JSONL status；ABI-only 不带参数运行仍 explicit fatal `dds_runtime_not_implemented`；`visual_events_dds_bridge_abi_harness` test harness 复用同一 core 产出 fake camera/head，并消费 Python canonical `gaze_target`；parser 严格 canonical fields + state 闭集。Step 4 native generated DDS type/ABI mapping construction slice 已完成：覆盖 `CameraFrame_ -> CameraJpegFrame`、`HeadStateV1_ -> HeadStateFrame`、`GazeTargetFrame -> GazeTargetV1_`、camera/head/gaze field mapping、head state derived stationary/moving/unknown、gaze valid/state consistency 和 finite/range checks；mapping harness 不启 DDS 网络、不调用 Unitree Channel，不证明真实 DDS over-wire 端到端发布订阅 gate、PC E2E、RK/真机。Step 4 native Unitree Channel construction harness/smoke slice 已完成：`visual_events_dds_bridge_construction_harness` 是 full-bridge only construction harness；`runtime_options` pure env parser；`--print-options` 单行 JSONL，`--print-options` 不启 DDS；`--construct-once` 解析 env，执行 Unitree ChannelFactory Init(domain/network)，构造 `CameraFrame_` subscriber、构造 `HeadStateV1_` subscriber、构造 `GazeTargetV1_` publisher，CloseChannel 后 Release。Step 4 native runtime loop core/full-bridge wiring/fake harness/build include fix slice 已完成：`runtime_loop` core；`visual_events_dds_bridge_runtime_loop_harness` fake harness；full-bridge `visual_events_dds_bridge` 无参数路径进入 Unitree DDS runtime loop；ABI-only 路径仍 explicit fatal `dds_runtime_not_implemented`；stdout emitter latest-slot 输出 camera_jpeg/head_state；stdin 读取 canonical `gaze_target` JSONL 并经 backend 发布 DDS gaze；async backend fatal 不被 stdin 阻塞；shutdown late fatal 仍输出 fatal JSONL；full-bridge 构建显式传入 repo-local CycloneDDS C++ include dir。该 slice 只证明 native runtime loop；current PC core gate 的 full-scene/Botified 证据见 Step 5，不证明 release report、RK/board 或真机闭环。外部源码/build/install/probe 输出不进 Git；未完成边界统一见 Step 4 剩余缺口。

后续开发不能重写 server 主线。Server 已完成的 GA 收口只做防回归和继续跑 gates；CLI 工作从现有 package/entrypoint/config/纯逻辑模块继续补齐 runtime 能力。后续实现切片必须优先推进实际运行能力；没有当前验收收益的治理/report overhead 延后。

## 4. 产品边界

| 模块 | 做 | 不做 |
| --- | --- | --- |
| `visual-events-server` | 接收 JPEG frame，输出 `visual_state`，生成 semantic events，选择 attention target | 不接 DDS，不输出 Botified，不发布 gaze DDS，不控制机器人 |
| `visual-events-cli` | 订阅 DDS JPEG，发送 WebSocket frame，接收 `visual_state`，发布 DDS gaze target，输出 Botified frame | 不跑大模型，不做事件规则，不做 agent 决策，不直接操纵运控 |
| PC test tools | 发布测试 DDS 图像/头部状态，订阅 gaze target，编排本地 E2E | 不替代正式 CLI，不作为产品运行单元 |
| Botified agent | 接收低频事件并决定是否回应 | 不接收 10Hz 视觉状态，不做注视闭环 |
| 运控/头控模块 | 订阅 DDS gaze target 并执行动作 | 不在本 repo 内实现 |

GA 非目标：

- 不训练模型。
- 不做身份识别、人脸识别、长期记忆。
- 不做多摄像头。
- 不做云端部署、管理后台、可视化大屏。
- 不承诺 RK3588 已 GA；只保留 backend 边界和单独 spike 计划。
- 不把会话管理、跨轮对话冷却或“是否再次招呼同一个人”的业务策略塞进本 repo。

商店揽客重复招呼边界：

- Botified agent 负责会话状态、跨事件业务冷却和是否开口。
- Visual Events 必须保证 `event_id` 幂等、同 track 同事件 cooldown、短遮挡/lost hold 不刷出新招呼。
- 在 manifest/oracle 标注的短遮挡、lost hold/cooldown 和空间邻近窗口内，同一物理人不得产生新的招呼型事件序列；即使 `track_id` 变化，也必须通过 scene/person label 的验收 oracle 发现重复招呼风险。这不是 ReID/长期记忆承诺，超过 oracle 窗口不做身份延续。
- 重复招呼风险控制仍由 server semantic event rules 和验收 oracle 覆盖；CLI 不实现 ReID、长期记忆或业务冷却策略。
- PC GA report 必须量化每事件类型、每 track、每分钟输出上限；超过上限即 fail，不能交给 agent 兜底。
- 完整产品闭环属于 post-GA/field rollout evidence，可由 Botified owner 证明至少一次 `Visual Events semantic event -> Botified 决策/回应动作 -> 冷却不重复招呼`；该 artifact 不要求也不允许把 Botified 会话逻辑实现回本 repo，且不阻塞 GA。

## 5. 运行拓扑

真机拓扑：

```text
DDS /camera/image/jpeg @10Hz
  -> visual-events-cli
  -> WebSocket ws://<server>:<port>/v1/stream
  -> visual-events-server
  -> visual_state @10Hz
  -> visual-events-cli
      -> DDS /visual_events/gaze_target @<=10Hz
      -> stdout Botified request frame for semantic_events
  -> head/motion owner subscribes gaze target and moves safely
```

PC 本地拓扑：

```text
val-data JPEG sequence
  -> tools/publish_test_dds_images
  -> real visual-events-cli
  -> real visual-events-server
  -> tools/subscribe_test_gaze_targets
  -> stdout collector / Botified frame assert
  -> tools/run_cli_local_e2e report
```

PC 本地 E2E 必须使用真实 DDS participant 和正式 CLI。Mock server 只能用于 CLI failure-path 和 deterministic unit/integration tests；不能替代 real server + `val-data/` PC-simulated GA gate。

PC 本地 E2E 的定位是 current PC core functional gate：证明 repo 内 CLI/server/DDS/Botified event oracle 在可复现数据上稳定，并作为当前 GA acceptance / PC-simulated GA pass/fail authority。它不等于 real robot/field/RK/release audit pass；硬件/现场验证属于 post-GA validation，覆盖真实 camera DDS runtime/network、板端 DDS type/QoS compatibility、真实 head state topic/type/Hz/freshness，以及真实 head/motion consumer 或等价闭环验收。

## 6. DDS 契约

### 6.1 图像输入

复用 `/home/galbot/works/image-capture` 的 JPEG DDS 约定：

| 字段 | 值 |
| --- | --- |
| topic | `/camera/image/jpeg` |
| DDS type | `unitree_camera::msg::dds_::CameraFrame_` |
| camera_name | DDS source camera name；示例/当前 image-capture 来源可为 `image` |
| encoding | `JPEG` |
| data | 完整 JPEG bytes，包含 SOI/EOI marker |
| 默认 domain/network | `DDS_DOMAIN=0`，`DDS_NETWORK=eth0` |
| QoS | best effort、volatile、keep last 1、deadline 150ms、lifespan 300ms、automatic liveliness lease 1000ms |

CLI 只保留最新合法 JPEG frame。非法 JPEG、非 JPEG encoding、空 data、width/height 非法必须丢弃并计数到 stderr/metrics。

`CameraFrame_.camera_name` 是 DDS source camera name，来自 image publisher；CLI 配置 `[camera].name` 是发给 server、Botified frame、gaze target 和 report 的逻辑相机名，默认仍为 `front`。二者可以不同，不能把 DDS source `camera_name` 当成 CLI logical `camera.name`，也不能用 CLI logical name 覆盖 source 观测值。PC GA report 必须同时记录 synthetic DDS `dds_source_camera_name` 和 `logical_camera_name`/`camera.name`；真机 report 属于 post-GA validation。

PC 本地 E2E 不使用默认真机网络。测试 runner 必须显式传入 domain/network；标准 PC 值是 `DDS_NETWORK=lo`、`DDS_DOMAIN=57`。缺少显式 `--dds-domain` 或 `--dds-network` 必须 fail fast；非 loopback 网络必须显式传入 `--allow-non-loopback-dds`。

`CameraFrame_` 不携带源 `frame_id`。CLI 必须为每条 WebSocket connection 生成 per-connection monotonic transport `frame_id`，用于 request/response 对齐和 server state reset；不得把 DDS `timestamp_ns`/`timestamp_ms` 当作 identity。源 timestamp 只允许作为 `frame_timestamp_ms` 和 freshness fallback 输入；缺失、重复、倒退或跨时钟跳变时以 CLI receive monotonic time 判 stale。

### 6.2 头部状态输入

当前阶段 PC gate 要求 CLI 能订阅头部状态 DDS，用于生成 WebSocket frame header 的 `head_motion`。缺失头部状态时系统可以降级运行，但 passing by、approaching、stopped 三类运动敏感事件会被 server suppression；因此缺失头部状态不能算 current phase pass / PC gate pass。

| 字段 | 说明 |
| --- | --- |
| topic | `/robot/head_state` |
| DDS type | `visual_events::msg::dds_::HeadStateV1_` |
| QoS | best effort、volatile、keep last 1、deadline 150ms、lifespan 250ms、automatic liveliness lease 500ms |
| timestamp | `timestamp_ms`，Unix epoch milliseconds；CLI 以本机 monotonic clock 判断 stale |
| 角度单位 | radians |
| 角速度单位 | radians/second |
| 最小字段 | `schema_version:uint32`、`timestamp_ms:int64`、`valid:bool`、`yaw_rad:float64`、`pitch_rad:float64`、`yaw_vel_rad_s:float64`、`pitch_vel_rad_s:float64` |
| 映射 | 速度低于阈值且状态新鲜时 `stationary`，速度超过阈值时 `moving`，缺失/过期/invalid 时 `unknown` |

如果头部状态不可用，CLI 必须发送 `head_motion.state=unknown`。Server 会暂停 `person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot` 的条件累积和触发。

`/robot/head_state` + `visual_events::msg::dds_::HeadStateV1_` 是 canonical GA 合同。如果未来运控 owner 只能提供已有等价类型，post-GA 硬件适配需要产出 adapter 映射、canonical internal schema、PC publisher 使用的权威类型、compatibility report 字段和 owner sign-off；GA PC gate 不等待真实 head-state source。

当前 PC core gate 的标准命令只要求 `--head-state stationary`，并要求 stationary head state 新鲜且 Hz 达标；report 字段包含 `head_state.required=true`、`head_state_publisher_mode=required`、`head_state_hz` 和必要 freshness/rate evidence。`stationary/moving/unknown` 三段覆盖保留为历史 partial smoke / unit-integration evidence；moving/unknown 下运动敏感事件会被 suppression，不作为 Botified oracle core pass。缺少 stationary head state、Hz 不达标或 freshness 不达标时，report 只能标记 degraded/fail，不能标记 current PC core gate pass。

### 6.3 Gaze Target 输出

GA 固化一个高频 DDS 输出，不发布完整 `visual_state` DDS：

| 字段 | 值 |
| --- | --- |
| topic | `/visual_events/gaze_target` |
| DDS type | `visual_events::msg::dds_::GazeTargetV1_`，Step 1 必须产出 IDL 或等价权威类型定义 |
| QoS | best effort、volatile、keep last 1、deadline 150ms、lifespan 250ms、automatic liveliness lease 500ms |
| 频率 | nominal 10Hz 输出；有新鲜 `visual_state` 的区间每帧发布一条 sample，目标 >=9Hz 且 <=10Hz；valid 和 invalid sample 都计入 |
| 消费方 | 运控/头控 owner |

消息语义是 target，不是 command。坐标使用输入图像像素坐标系，原点左上，`u/x` 向右，`v/y` 向下。`target_norm_x = target_u / image_width - 0.5`，`target_norm_y = target_v / image_height - 0.5`。Timestamp 字段使用 Unix epoch milliseconds；CLI 内部 stale 判断使用 monotonic clock。

字段级合同：

| 字段 | 类型 | 规则 |
| --- | --- | --- |
| `schema_version` | `uint32` | 固定 `1` |
| `camera` | string | 与 WebSocket frame header camera 一致，即 CLI logical `[camera].name`，不是 DDS source `camera_name` |
| `frame_id` | `int64` | 来自产生该 target 的 `visual_state.frame_id`；invalid sample 使用最后一帧 id 或 `-1` |
| `frame_timestamp_ms` | `int64` | 来自 `visual_state.frame_timestamp_ms`；未知为 `0` |
| `publish_timestamp_ms` | `int64` | CLI 发布时 wall clock |
| `valid` | `bool` | 只有新鲜 attention target 可用时为 `true` |
| `state` | enum/string | `tracking`、`lost`、`stale`、`disabled` |
| `target_track_id` | `int64` | valid 时为 track id；invalid 时固定 `-1` |
| `target_u` | `float32` | valid 时有限且在 `[0,image_width]`；invalid 时固定 `0.0` |
| `target_v` | `float32` | valid 时有限且在 `[0,image_height]`；invalid 时固定 `0.0` |
| `target_norm_x` | `float32` | valid 时约在 `[-0.5,0.5]`；invalid 时固定 `0.0` |
| `target_norm_y` | `float32` | valid 时约在 `[-0.5,0.5]`；invalid 时固定 `0.0` |
| `image_width` | `uint32` | valid 时 >0；invalid 时最后一帧尺寸或 `0` |
| `image_height` | `uint32` | valid 时 >0；invalid 时最后一帧尺寸或 `0` |
| `confidence` | `float32` | `[0,1]`；invalid 时 `0.0` |
| `reason` | string | server attention reason；invalid 时 `lost`、`stale` 或 `disabled` |
| `stale_after_ms` | `uint32` | GA 默认 `250` |

禁止字段：

- `yaw_velocity`
- `pitch_velocity`
- `head_position`
- `motor_command`
- 任何直接表达运控命令的字段

失效语义：

- Server 仍返回新鲜 `visual_state` 但无 attention、无人或 target 坐标非法时，CLI 每帧发布 `valid=false` sample，保持与 `visual_state` 同频。
- Server 超时、断线或 `visual_state` 过期时，CLI 必须在 250ms 内发布一次 `valid=false,state=stale` sample；之后不承诺断线期间继续 >=9Hz heartbeat，并依赖 DDS lifespan 让下游失效。
- `gaze_target.stale_ms` 是独立 watchdog，不依赖 WebSocket `response_timeout_ms`。即使当前 one in-flight request 还没 timeout，只要最近可用 gaze target 到达 stale deadline，CLI 也必须立即发布一次 `valid=false,state=stale`。
- 下游运控必须尊重 `valid=false`、`stale_after_ms` 和 DDS lifespan；这是假设 CLI 不直接控运仍然安全的前提。

## 7. GA 阈值

| 验收项 | GA 阈值 |
| --- | --- |
| Botified 每 track/event/min 上限 | 同一 `track_id` 的同一 event type 每分钟 <=1 条 |
| Botified 全局事件/min 上限 | 所有 Botified allowlist event 合计每分钟 <=12 条，1 秒 burst <=3 条 |
| Head state unknown ratio | required 模式下 stationary/moving segment <=1%；专门 unknown segment >=95% 映射为 `unknown` |
| Target switch dwell/jitter | 除目标丢失外，同一目标 dwell >=750ms；稳定 tracking 时 target_norm jitter P95 <=0.04 |
| Post-GA head pointing 误差 | valid tracking 稳定 800ms 后，物理指向误差 P95 <=8 度，max <=15 度 |
| Invalid/stale 停止延迟 | GA 要求 CLI 在 250ms 内发布 `valid=false,state=stale`；post-GA 真机/HIL 验证可检查头部在 500ms 内停止继续跟随旧目标 |

GA 硬 gate 只包含 PC runtime path 可验证的阈值。head pointing 误差、真机/HIL 停止延迟和现场 owner sign-off 属于 post-GA 硬件/现场验证，不是 GA pass/fail authority。

当前阶段 PC gate 的 target dwell gate 是 750ms。GA release config/gate 必须显式设置并验证 `attention.switch_confirm_ms >=750`；当前 server 默认 500ms 不能被当作 PC gate pass 证据，除非后续计划明确决定调整默认值并补对应证据。

## 8. 代码结构

目标结构：

```text
src/
  visual_events_server/
    ...
  visual_events_cli/
    __init__.py
    main.py
    config.py
    service_client.py
    frame_pump.py
    botified_output.py
    target_mapper.py
    dds/
      __init__.py
      image_subscriber.py
      head_state_subscriber.py
      gaze_target_publisher.py
      qos.py
      types.py
common/
  schema/
    protocol.md
    dds/
      camera_jpeg_contract.md
      gaze_target_v1.idl
      gaze_target_v1.md
      head_state_v1.idl
      head_state_v1.md
tools/
  build_dds_bridge.py
  publish_test_dds_images.py
  publish_test_head_state.py
  subscribe_test_gaze_targets.py
  run_cli_local_e2e.py
  mock_visual_state_server.py
tests/
  unit/
    test_cli_*.py
  integration/
    test_cli_*.py
```

`pyproject.toml` 增加：

```toml
[project.scripts]
visual-events-server = "visual_events_server.app:main"
visual-events-cli = "visual_events_cli.main:main"
```

按 KISS，GA 不改 distribution name。继续保留 `visual-events-server` wheel 名称，并在同一个 wheel 中新增 `visual-events-cli` console script。Release/runtime smoke 和 handoff 必须证明以下两个入口都存在且来自 `runtime/venv`：

- `runtime/venv/bin/visual-events-server`
- `runtime/venv/bin/visual-events-cli`

PC GA gate 必须从 `runtime/venv` 启动 server 和 CLI；orchestration 工具可以从 dev env 跑，但不能用 dev console script 或手工预启动 server 冒充 current phase pass / PC gate pass。标准 runner/report 必须校验 `--server-bin runtime/venv/bin/visual-events-server`、`--cli-bin runtime/venv/bin/visual-events-cli` 两个路径存在且可执行，证明二者来自同一个 release wheel/runtime provenance，并记录 server/CLI 子进程 exit code。

CLI core 优先使用 Python/`uv` 管理，与 server 共用 repo-local release/runtime 规则。如果 Unitree DDS runtime 只能通过 C++ SDK 接入，可以在本 repo 内实现一个很小的 native DDS bridge 或 helper，但对 CLI core 暴露的接口仍是 `DdsImageSubscriber`、`DdsHeadStateSubscriber`、`DdsGazeTargetPublisher`，并且同样受单元测试、PC E2E 和 no-motion-SDK audit 约束。

RK3588 compatibility 不作为本轮 GA 承诺，但实现必须保护未来迁移边界：server 推理 backend 仍通过 `InferBackend` 替换；CLI 不依赖 Torch/Ultralytics；aarch64/RK3588 runtime 对 CUDA extra 必须 fail fast 或选择 explicit unsupported/backend placeholder；post-GA 声称 native DDS bridge/helper 板端支持前必须有 aarch64/RK3588 build/probe 或 explicit unsupported fail-fast，不能把 bridge 兼容问题藏到 server backend；未来 RKNN 插入点只落在 server backend/package/config 边界，不侵入 CLI/DDS/Botified 合同。

## 9. 实现步骤

### Step 1：固化合同文档和 schema（主要产物已完成，DDS stack decision record 已冻结）

已完成：

- `common/schema/dds/camera_jpeg_contract.md`
- `common/schema/dds/gaze_target_v1.idl` 和 `gaze_target_v1.md`
- `common/schema/dds/head_state_v1.idl` 和 `head_state_v1.md`，冻结 `/robot/head_state` + `visual_events::msg::dds_::HeadStateV1_` canonical GA 合同。
- gaze target JSON samples。
- no-motion-SDK audit 覆盖，确认 CLI 不依赖、链接、import 或调用运控 SDK。
- `docs/dds-stack-decision-record.md`，冻结 Unitree SDK2 2.0.0 + CycloneDDS 0.10.2 + C++ native DDS helper/bridge，Python CLI 只通过受控进程/IPC 调用 bridge；记录 IDL codegen gate、line-delimited JSON `protocol_version=1` ABI、runtime env vars、PC loopback smoke plan 和 post-GA RK3588/板端 validation/probe plan。

后续 guard：

- 如果未来运控 owner 使用等价类型：post-GA adapter 映射、canonical internal schema、PC publisher 使用的权威类型、compatibility report 字段和 owner sign-off。
- DDS stack decision record 已冻结，但不实现真实 DDS factories/adapters、native bridge binary、PC E2E 或板端 compatibility pass。
- README、产品文档、开发计划全部移除“CLI 直接头控”表述。

验收：

- 文档明确 CLI 不直接操纵运控。
- 对实现代码持续做 no-motion-SDK audit：`visual_events_cli` 和 DDS bridge 不得依赖、链接、import 或调用运控 SDK；不得发布 motor/head control command topic；gaze target 输出不得含 velocity/position command 字段。允许只读 head state 字段和 `head_motion` 映射代码。
- DDS topic、QoS、字段、stale/lifespan 语义可由 PC 工具和真机实现复用。
- DDS stack decision record 随 Step 1 冻结；Step 4 缺少 full bridge runtime、真实 bridge/factories 或 PC 安装/loopback smoke 任一实际证据时，不能进入 PC E2E；板端 probe 属于 post-GA validation。

### Step 2：完成 server GA 收口改进（已完成，后续只防回归）

已完成：

- Frame `frame_id` 或 `timestamp_ms` 倒退时 reset 当前连接 tracker/event state，并有集成测试。
- JPEG header `width/height` 与实际解码尺寸一致性校验。
- `scene_flags.someone_near_center` 真实计算。
- shared backend inference serialization，保持每连接状态隔离并避免真实 backend 并发推理状态不确定。
- Metrics sink 写失败计数并输出 stderr，不改变 `visual_state` wire protocol。
- Server 与 CLI 合同测试：`visual_state.attention`、`semantic_events`、`head_motion` suppression 对 CLI 期望稳定。

验收：

- 现有 `pytest -q` 全绿。
- Runtime smoke 通过。
- `val-data/` full matrix、semantic timeline gate、300s soak、metrics aggregation 继续通过。
- 性能仍满足 P95 < 120ms、P99 < 200ms、输出 >=9Hz、显存 <4GiB。

### Step 3：实现正式 CLI core（3A skeleton/3B pure logic、RuntimeCoordinator/main wiring 和 production runner/lifecycle unit core 已完成；current PC core gate 已由 Step 5 编排）

已完成：

- `visual_events_cli.main` 和 `visual-events-cli` 入口。
- 配置读取：server URL、camera name、DDS domain/network、image topic、head state topic、gaze topic、stale thresholds、log path。
- `target_mapper`：`visual_state.attention` -> gaze target DDS payload；无有效 target 时生成 invalid sample。
- `botified_output`：`semantic_events` -> `<botified>...</botified>` request frame；按 `event_id` 幂等；stdout 只允许 Botified allowlist frame。
- Botified stdout allowlist 固定为 `person_appeared`、`person_left`、`person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot`、`person_waving`；`attention_target_changed` 不输出到 Botified。
- `service_client`：WebSocket wire/pack-unpack、连接复用/关闭、timeout、invalid response、frame_id mismatch、retryable/non-retryable error handling 的单元核心。
- `frame_pump`：one in-flight coordination、keep-latest frame slot/backpressure、gaze stale watchdog、Botified enqueue 的 deterministic unit core。
- RuntimeCoordinator/main wiring unit core。
- main runtime_runner 注入和默认 DDS factories fail-fast。
- exact stale deadline。
- slow Botified drain 不阻塞 stale。
- Botified stdout writer：bounded queue/drop/coalescing + BrokenPipe exception unit core。
- BrokenPipe publish stale then nonzero unit core。
- `RuntimeFactories/run_runtime` production runner/lifecycle unit core：默认 DDS factories fail-fast 且不 import DDS/native；测试注入 factories；start image/head/gaze；head current_motion wiring；coordinator shutdown；sync/async resource close；service client public close；Botified drain daemon-thread bounded shutdown；shutdown observe timeout 内观察到的 BrokenPipe publish stale then nonzero；start failure cleanup。

剩余缺口：

- 真实 DDS SDK 不直接链接进 Python CLI；正式路径是显式 bridge runtime。
- Step 3 不直接链接真实 DDS SDK；native bridge runtime/wiring 属于 Step 4。
- Step 5 manifest reader/report skeleton slice、mock visual_state server slice、CLI runtime + mock visual_state server integration slice、历史 partial smoke 和 `run_cli_local_e2e` current PC core functional gate（含 live bounded CLI stdout/stderr collector、additive `report["botified_stdout"]`、full-scene matrix 和 Botified event oracle）已完成；RK/board、real robot、field validation 和 release audit 后置。

验收：

- CLI 不依赖 Ultralytics/Torch，不跑模型。
- CLI 不链接、不 import、不调用任何运控 SDK。
- `visual_state` 不进入 stdout。
- 所有日志、metrics、debug 输出走 stderr 或 ignored artifact。
- Server 慢、断线、超时、错误响应或 slow Botified stdout 背压时，CLI 不无界排队、不刷 Botified，并按 `gaze_target.stale_ms` 准时发布 stale。Botified stdout BrokenPipe 时必须尽力发布一次 stale，然后受控非 0 退出。

### Step 4：实现 DDS adapters（first slice/unit core、native runtime loop 和 current PC over-wire core gate 已完成）

已完成：

- 纯 Python SDK-neutral DDS adapter core/fakes：`visual_events_cli.dds.qos`、`visual_events_cli.dds.types`、`visual_events_cli.dds.protocols`、`visual_events_cli.dds.fake`。
- QoS constants 覆盖 camera/head/gaze 的 depth、deadline、lifespan、liveliness lease 等合同值。
- `CameraJpegMessage` 做 JPEG SOI/EOI 和 SOF dimension validation；fake image subscriber 保持 latest-only。
- `HeadStateSample` 覆盖 stationary/moving/unknown、stale 和 future timestamp mapping。
- `FakeDdsGazeTargetPublisher` lifecycle：start/close idempotent；close 后拒绝使用/重启，并保护未 start publish。
- protocol names 固化，fake/unit layer 没有真实 DDS、ML 或运控 SDK import。
- Step 4 Python JSONL bridge client/facade slice 已完成：`bridge_protocol.py`、`bridge_process.py`、`bridge_adapters.py`、`runtime_factories.py` 和 explicit `bridge_runtime_factories()`；覆盖 JSONL protocol/base64/canonical gaze fields、subprocess lifecycle、camera/head/gaze 三个 thin facade、service/Botified factory wiring，以及 no DDS/native import audit tests。该 slice 不实现真实 DDS adapters。
- Step 4 Python JSONL bridge runtime integration slice 已完成，formal CLI bridge runtime opt-in slice 已完成：默认仍 fail_fast，不因 env 隐式切 bridge；显式 `[dds].runtime="bridge"`/`--dds-runtime bridge` 才走 `bridge_runtime_factories()`。真实 subprocess fake JSONL child + `bridge_runtime_factories()`/`run_runtime` 覆盖 camera/head -> service -> gaze stdin、logical camera、stale/cleanup、child nonzero/fatal；该 slice 不覆盖真实 DDS runtime，未完成边界统一见 Step 4 剩余缺口。
- Step 4 native DDS bridge build/probe foundation slice 已完成：`native/dds_bridge` CMake project 可构建 very small `visual_events_dds_bridge_probe`，链接 Unitree SDK2 和 video DDS `CameraFrame_` type props，固定 camera/head/gaze topic/type/QoS constants，并只向 stdout 输出单行 JSONL status frame（`protocol_version=1,type=status,code=probe_ok,message=...`）；`tools/build_dds_bridge.py` 拆分 foundation gate 和 full-bridge gate，foundation check/build/probe 只要求 SDK root、video publisher dir 和 `CameraFrame_` inputs，可在无 IDL generator 时成功，并把 `foundation_ready`、`visual_events_codegen_ready`、`visual_events_codegen_error` 写入 ignored `artifacts/` report。该 slice 只证明 camera/probe foundation；Foundation 路径仍然 CameraFrame-only，不证明 PC 本地 DDS E2E、board/RK 或真机闭环。
- Step 4 DDS C++ idlc repo-local prepare/oracle hardening slice 已完成：`tools/prepare_dds_codegen_toolchain.py` 默认 pinned CycloneDDS/CycloneDDS-CXX 0.10.2，默认 repo-local ignored probe 输出目录为 `build/tools/cyclonedds-cxx-idlc-0.10.2/codegen_probe/`；`--check`/`--dry-run` 不下载、不构建、不写系统或用户目录，只验证路径、版本、显式 `idlc` 和 cxx backend 文本，并报告 `oracle_ok=false`；`--probe-codegen` 默认实际调用 `idlc -l cxx -o <probe-output-dir>` 分别验证 `common/schema/dds/head_state_v1.idl` 和 `common/schema/dds/gaze_target_v1.idl`，遇到 `cannot load generator`/`cannot load generator cxx`、任一 IDL 缺预期 `.hpp` 或缺预期 `.cpp` 必须 fail，即使 idlc return code 是 0，并报告每个 probed IDL、expected `.hpp/.cpp`、generated files 和 per-IDL presence。`--prepare` 显式准备 ignored repo-local CycloneDDS/CycloneDDS-CXX 0.10.2 C++ idlc toolchain，验证 tag commit、CMake Makefiles install、`bin/idlc-cxx` wrapper 和 installed artifacts，并自动复用同一个 Head/Gaze codegen oracle；它不接受 `--idlc` 且不使用 `VISUAL_EVENTS_IDLC`。fake git/cmake/idlc 覆盖成功生成、0.11.0 fail、missing cxx generator 但 rc=0 fail、只生成 `.hpp` fail、clone/artifact/oracle failure。`tools/build_dds_bridge.py --check-full-bridge` 不再搜索 PATH，只接受显式 `--idlc` 或 `VISUAL_EVENTS_IDLC`，并复用同一个 codegen probe；缺显式 idlc、版本不是 0.10.2、backend 不能实际加载或缺 Head/Gaze expected `.hpp/.cpp` 时 fail-fast，通过时报告 `visual_events_codegen_ready=true` 和 `oracle_ok=true`。该 slice 只证明 repo-local prepare 编排和 C++ codegen probe 会拒绝假阳性；外部源码/build/install/probe 输出不进 Git，不实现 full bridge/runtime/PC E2E/RK/真机。
- Step 4 native full-bridge generated Head/Gaze C++ type-support compile/probe slice 已完成：`tools/build_dds_bridge.py --check --check-full-bridge --build --probe` 会运行 Head/Gaze IDL codegen oracle，CMake full-bridge 编译 `head_state_v1.hpp/.cpp` 和 `gaze_target_v1.hpp/.cpp`，native probe 检查 `CameraFrame_`、`HeadStateV1_`、`GazeTargetV1_` type props 并输出一行 JSONL status；Foundation 路径仍然 CameraFrame-only。该 slice 不证明 PC 本地 DDS E2E、board/RK 或真机闭环。
- Step 4 native JSONL ABI/runtime skeleton slice 已完成：`visual_events_dds_bridge` target 存在，`--probe` 单行 JSONL status；ABI-only 不带参数运行仍 explicit fatal `dds_runtime_not_implemented`；`visual_events_dds_bridge_abi_harness` 复用同一 ABI core 产出 fake camera/head，并消费 Python canonical `gaze_target`；parser 严格 canonical fields + state 闭集。
- Step 4 native generated DDS type/ABI mapping construction slice 已完成：`CameraFrame_ -> CameraJpegFrame`；foundation camera mapping 可无 generated Head/Gaze；full-bridge mapping harness 编译 generated Head/Gaze，并验证 `HeadStateV1_ -> HeadStateFrame`、`GazeTargetFrame -> GazeTargetV1_`、camera/head/gaze field mapping、head state derived stationary/moving/unknown、gaze valid/state consistency 和 finite/range checks。mapping harness 不启 DDS 网络、不调用 Unitree Channel；该 slice 只证明 mapping construction，未完成边界统一见 Step 4 剩余缺口。
- Step 4 native Unitree Channel construction harness/smoke slice 已完成：`visual_events_dds_bridge_construction_harness` 是 full-bridge only construction harness；`runtime_options` pure env parser；`--print-options` 单行 JSONL，`--print-options` 不启 DDS；`--construct-once` 解析 env，执行 Unitree ChannelFactory Init(domain/network)，构造 `CameraFrame_` subscriber、构造 `HeadStateV1_` subscriber、构造 `GazeTargetV1_` publisher，CloseChannel 后 Release。该 slice 只覆盖 construction smoke。
- Step 4 native runtime loop core/full-bridge wiring/fake harness/build include fix slice 已完成：`runtime_loop` core；`visual_events_dds_bridge_runtime_loop_harness` fake harness；full-bridge `visual_events_dds_bridge` 无参数路径进入 Unitree DDS runtime loop；ABI-only 路径仍 explicit fatal `dds_runtime_not_implemented`；stdout emitter latest-slot 输出 camera_jpeg/head_state；stdin 读取 canonical `gaze_target` JSONL 并经 backend 发布 DDS gaze；async backend fatal 不被 stdin 阻塞；shutdown late fatal 仍输出 fatal JSONL；full-bridge 构建显式传入 repo-local CycloneDDS C++ include dir。该 slice 只证明 native runtime loop；current PC core gate 的 full-scene/Botified 证据见 Step 5，不证明 release report、RK/board 或真机闭环。
- ABI harness 复用同一 core 产出 fake camera/head，作为 native JSONL ABI/runtime skeleton 的最小回归证据。

剩余缺口：

- native PC DDS over-wire test participants、Python native participant wrappers、Step 5 manifest reader/report skeleton slice、mock visual_state server slice、CLI runtime + mock server integration、历史 partial smoke 和 `run_cli_local_e2e` current PC core functional gate 已完成。当前 PC core gate 用正式 CLI + real server 覆盖 `/camera/image/jpeg` -> bridge/CLI -> server -> `/visual_events/gaze_target` over wire、`val-data` full-scene matrix 和 Botified event oracle；partial smoke 只作为历史/兼容模式保留，不能再代表当前门禁能力。
- manifest skeleton 仍只做数据集身份和报告骨架；server replay manifest contract evidence 仍是 server-level evidence，不是 current PC core gate 的唯一 authority。当前 PC core gate 的 pass/fail authority 是 `tools/run_cli_local_e2e.py --full-scene --all-scenes --head-state stationary --server-config configs/pc-ga-server.toml`。
- 若真实 adapter 通过 native DDS bridge/helper 接入，post-GA 声称 aarch64/RK3588 支持前，bridge 自身必须有板端 build/probe，或在不支持时 explicit unsupported fail-fast；不能只用 server backend 可替换来掩盖 bridge 兼容缺口。
- 板端 DDS compatibility probe 和 board/RK probe：在目标系统镜像或等价容器内验证 camera/head/gaze 三个 topic 的 type name、serialization、QoS、domain/network 和 Hz/freshness。
- RK/board probe、真机 smoke/closed-loop handoff、field validation、release report/handoff audit、full fault matrix 和 long soak 属于 post-GA validation 或交付审计层，不阻塞当前 PC 本地核心功能门禁，也不在当前核心实现切片中扩张。

验收：

- 无 publisher 时 CLI 可诊断并保持可恢复。
- Invalid JPEG 不发送给 server。
- DDS gaze target payload 坐标有限且在图像范围内；invalid sample 明确。
- DDS resource lifecycle 必须 start/close idempotent；close 后拒绝使用/重启；无泄漏、无后台线程悬挂。
- PC GA evidence 必须包含 repo-local DDS type/QoS compatibility 结果；RK/board compatibility 属于 post-GA validation，不能由 PC gate pass 外推。

### Step 5：实现 PC 本地测试工具

产出：

- Python `tools/*.py` 是 runner/wrapper；真实 DDS participants 由 native full-bridge binaries 承担，Python CLI 不直接 import/link DDS SDK。
- Step 5 native PC DDS over-wire test participant slice 已完成：`visual_events_dds_bridge_publish_test_dds_images`、`visual_events_dds_bridge_publish_test_head_state`、`visual_events_dds_bridge_subscribe_test_gaze_targets` 和 `pc_test_tools` 已接入；这些 target 都是 full-bridge only，并使用 Unitree Channel 真实 DDS participant。
- 已完成 loopback over-wire smoke：同一 domain/network=58/lo 下启动 `visual_events_dds_bridge`、image publisher、head publisher 和 gaze subscriber；bridge stdout 收到 `camera_jpeg` 和 `head_state`；向 bridge stdin 写 canonical gaze 后，gaze subscriber 收到 DDS `gaze_target`，包含 frame_id=77、track_id=12。
- Step 5 Python native participant wrappers slice 已完成：`tools/dds_pc_tools.py`、`tools/publish_test_dds_images.py`、`tools/publish_test_head_state.py`、`tools/subscribe_test_gaze_targets.py` 已接入；wrappers 要求显式 `--dds-domain`/`--dds-network`，非 `lo` 必须 `--allow-non-loopback-dds`，默认只从 `--build-dir` 找 native binary，不搜索 PATH，argv domain/network 覆盖 env，stdout/stderr 直接透传 child，child return code 原样返回。Python wrappers 不 import DDS SDK/visual_events_cli/server，不做 manifest/report/mock server/full runner。
- wrapper-level PC loopback smoke 已通过：Python wrappers 启动 native image/head/gaze participants，`visual_events_dds_bridge` 收到 `camera_jpeg`/`head_state`；向 bridge stdin 写 canonical gaze 后 wrapper subscriber 收到 DDS gaze，包含 frame_id=88、track_id=13。
- Step 5 manifest reader/report skeleton slice 已完成：`tools/cli_local_e2e_manifest.py` 读取 `--data-dir`，默认探测 `val-data/manifest.json`（实际为 `<data-dir>/manifest.json`，标准数据目录是 `val-data`），也支持显式 `--manifest`。manifest（`val-data` 测试数据清单）缺失时生成 deterministic effective manifest：scene 名称、per-scene sha256、frame count、first/last frame，并记录 aggregate `scene_count` / `frame_count` 和 `manifest_sha256`。manifest skeleton report 固定 `overall_pass=false`、`pc_local_e2e_status=not_run`，只针对 `tools/cli_local_e2e_manifest.py` 的报告骨架；该 skeleton 只做数据集身份和报告骨架，不能替代 runtime E2E；`--out` 不能写入 data dir。本机 ignored `val-data` 当前可识别 7 scene / 576 frames。authoritative manifest/oracle schema skeleton 已完成：`common/schema/val_data_manifest_v1.md` 固定 `schema_version=1` 的 fps、scene_count、frame_count、scene_name、scene_sha256 和 oracle source/version/rule 最小合同；oracle 是测试标准答案来源，authoritative 只表示 PC 测试采用的清单/答案版本，不表示组织级治理、审计或发布批准；generated inventory 和 legacy file manifest 继续可用但 `manifest_authoritative=false`、`oracle_schema_present=false`，schema v1 manifest valid 时 `manifest_validation_errors=[]`，invalid v1 会被拒绝并在 report 投影错误；`val-data/manifest.json` 仍不进 Git。本段不新增 manifest builder、schema 审计、release audit、handoff audit 或 strict gate 扩张。
- Step 5 server replay manifest contract evidence slice 已完成：`tools/run_val_data_e2e.py` 支持 `--manifest` 和 `--require-authoritative-manifest`，server replay gate report 记录 `manifest_source`、`manifest_sha256`、`manifest_authoritative`、`manifest_validation_errors`、`oracle_schema_present`、`oracle_schema_valid`、`oracle_summary`、`manifest_contract_required`、`manifest_contract_satisfied`、`oracle_evaluated=false` 和 `oracle_evaluation_passed=null`。默认无 manifest 时 report 使用 generated inventory、`manifest_authoritative=false`、`oracle_schema_present=false`，且不破坏本机 server replay gate；manifest 文件 parse/I/O 错误始终 preflight fail；只有开启 `--require-authoritative-manifest` 时，才把缺失、schema/contract-invalid but parseable 或 non-authoritative manifest 作为 preflight failure。semantic event oracle source 仍记录为 `tools.replay_val_data.hardcoded_scene_expectations`；server replay oracle 迁移不是 current PC core gate 的阻塞项。
- `tools/publish_test_dds_images.py`：native image publisher 的 runner/wrapper，按显式 domain/network 调用 build-dir 下的 native binary。
- `tools/publish_test_head_state.py`：native head publisher 的 runner/wrapper，发布 `stationary|moving|unknown` 头部状态。
- `tools/subscribe_test_gaze_targets.py`：native gaze subscriber 的 runner/wrapper，订阅 gaze target 并透传 native stdout/stderr。
- Step 5 mock visual_state server slice 已完成：`tools/mock_visual_state_server.py` 提供 `/healthz` 和 `/v1/stream`，复用现有 `visual_events_server.protocol` decode/serialize，不定义第二套协议；支持 `tracking/lost/event` profile、`--delay-ms`、`--disconnect-after`，用于 CLI deterministic attention/event、slow response、disconnect 测试；不 import DDS/CLI runtime/模型，不启动 DDS participant，不读取 val-data，不做 runner/report；不能替代 current PC core gate。
- Step 5 CLI runtime + mock visual_state server integration slice 已完成：`tests/integration/test_cli_bridge_runtime_mock_server.py` 使用真实 `bridge_runtime_factories()`/`run_runtime`、fake JSONL bridge child 和 mock server subprocess，覆盖 event Botified stdout allowlist、tracking/lost/stale gaze；不使用 DDS participant、`val-data` 或 real server，不做 runner/report；不能替代 current PC core gate。
- Step 5 `run_cli_local_e2e` current PC core functional gate 已完成：`tools/run_cli_local_e2e.py --full-scene --all-scenes --head-state stationary --server-config configs/pc-ga-server.toml` 复用 manifest skeleton，启动正式 server binary、正式 CLI binary with bridge runtime、gaze subscriber wrapper、head/image publisher wrappers，跑 `val-data` full-scene matrix，并用 Botified event oracle 判定 stdout 事件。`tools/run_cli_local_e2e.py` 支持 `--require-authoritative-manifest`；manifest contract judgment 集中在 `tools/cli_local_e2e_manifest.py` shared helper，并被 `tools/run_val_data_e2e.py` 复用，避免 server replay 与 CLI runner rules drift；CLI report 投影 `manifest_contract_required`、`manifest_contract_satisfied`、`manifest_contract_failure_reasons`、`oracle_evaluated` 和 `oracle_evaluation_passed`。report 必须区分 PC-simulated GA gate 与 GA 后范围：`overall_scope=current_pc_core_gate`，`current_pc_core_gate_pass` 是当前 PC 模拟核心 pass/fail authority；标准命令通过时 `ga_gate_pass=true` 且 `ga_gate_status=pc_simulated_ga_pass`，失败时 `ga_gate_pass=false` 且 `ga_gate_status=pc_simulated_ga_fail`，partial smoke/preflight 为 `ga_gate_status=not_evaluated`；real robot/field/RK/release audit 未覆盖写入 `post_ga_not_covered`，不阻塞 GA，也不能由 PC evidence 外推为通过。
- 历史 partial smoke 证据保留为背景：既有本机三段 partial smoke 证明 real server + CLI bridge runtime + DDS participant plumbing 可连通，并证明 observed CLI stdout 的 bounded Botified stdout collector/checks；它曾使用 generated 7 scene / 576 frames、scene `pci_stand`、`frame_count=3` 和 stationary/moving/unknown 三段 head publisher。该历史说明不能再被解释为当前只有 partial smoke，也不能否定 full-scene/all-scenes + Botified event oracle 已成为 current PC core gate。
- current PC core gate 的 manifest/oracle（测试数据清单/标准答案）必须列出 scene 名称、scene sha256、frame count、fps、expected event timeline source/version、expected attention target timeline/rule（注视目标标准答案：target label/track、allowed switch windows、no-target windows）；manifest 文件可位于 ignored `val-data/` 下，但 report 必须记录 `manifest_sha256` 和 manifest 副本摘要。这不是 release audit、handoff audit 或产品运行 artifact。

后置项：

- RK/board probe、真机 smoke/closed-loop handoff、field validation、release report/handoff audit、full fault matrix 和 long soak 属于 post-GA validation 或交付审计层；不阻塞 current PC core gate，也不在当前核心实现切片中扩张。
- Mock visual_state server、CLI runtime + mock server integration 和历史 partial smoke 只作为单元/集成/回归背景；current PC core gate 的 pass/fail authority 是正式 CLI + real server + `val-data` full-scene matrix + Botified event oracle。

验收：

- PC 无真机时可以跑完整 DDS image -> CLI -> server -> DDS gaze/Botified E2E。
- PC E2E 必须使用真实 DDS participant、synthetic `val-data` publisher、正式 CLI 和 real server；fake/in-memory DDS 只用于单元、集成和 fault matrix。
- Runner 必须显式传入 domain/network；标准 PC 值是 `DDS_NETWORK=lo`、`DDS_DOMAIN=57`。缺少显式 domain/network 时 fail fast；非 loopback 网络必须显式传 `--allow-non-loopback-dds`。
- 工具异常退出时清理子进程。
- Report 包含 `manifest_sha256`、frame count、image Hz、server response Hz、fresh gaze Hz/rate、expected attention target timeline/rule 摘要、stationary `head_state.required`、`head_state_publisher_mode`、`head_state_hz`、freshness evidence、Botified event count、stdout pollution count、basic finite latency sanity、reconnect count、stale/invalid sample count。`head_state_segments` 可作为历史 partial smoke / unit-integration evidence 保留，但不是 current PC core gate 的 blocking 字段。
- Report 必须包含 `dds_source_camera_name` 和 `logical_camera_name`/`camera.name`，避免把 DDS `CameraFrame_.camera_name` 与 CLI `[camera].name` 混同。
- Report 必须按进入、路过、靠近、停留、挥手等低频事件记录 expected occurrence、允许延迟窗口和负例不得触发，并包含 scene-level 断言：在 manifest/oracle 标注的短遮挡、lost hold/cooldown 和空间邻近窗口内，同一物理人不得产生新的招呼型事件序列；如果 `track_id` 变化也要通过 scene/person label 判定。
- Report 包含 runtime provenance：`server_bin`、`cli_bin`、`server_bin_is_runtime_venv`、`cli_bin_is_runtime_venv`、`wheel_name`、`wheel_version`、`runtime_hash`、`config_hash`、`server_exit_code`、`cli_exit_code`。任一 runtime bin 不是 `runtime/venv/bin/...`、wheel/runtime provenance 不一致或 exit code 不符合场景预期时，runner 必须返回非 0。
- Current PC core gate 只阻塞 basic finite latency sanity：latency 数值必须有限、方向正确且没有明显跨时钟异常。端到端 P95/P99 latency report（`capture_to_gaze_publish_p95_ms`、`capture_to_gaze_publish_p99_ms`、`capture_to_botified_stdout_p95_ms`、`capture_to_botified_stdout_p99_ms`）是 `non_blocking_gaps` / 后续 handoff evidence，不阻塞本次 current PC core gate，除非后续单独实现 blocking latency gate。
- Runner exit code：所有 gate pass 返回 `0`；schema/rate/stdout/timeout/process cleanup 任一失败返回非 0。
- 标准命令：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/run_cli_local_e2e.py \
  --full-scene \
  --all-scenes \
  --head-state stationary \
  --data-dir val-data \
  --manifest val-data/manifest.json \
  --server-config configs/pc-ga-server.toml \
  --server-bin runtime/venv/bin/visual-events-server \
  --cli-bin runtime/venv/bin/visual-events-cli \
  --dds-domain 57 \
  --dds-network lo \
  --server ws://127.0.0.1:8767/v1/stream \
  --out artifacts/cli-e2e
```

### Step 6：CLI 单元与集成测试

CLI 单元测试必须覆盖：

- 配置解析和默认值。
- DDS image 字段校验和 JPEG SOI/EOI 校验。
- Head state -> `head_motion` 映射。
- WebSocket one in-flight、keep-latest、timeout、reconnect、retryable error、non-retryable error。
- `visual_state.attention` -> gaze target payload。
- `valid=false`、`state=lost|stale|disabled` 失效输出。
- `gaze_target.stale_ms` watchdog 独立于 `service.response_timeout_ms`；one in-flight request 未 timeout 时仍准时 stale。
- Botified JSON/XML escaping、`event_id` 幂等、stdout/stderr 分离。
- Botified allowlist 排除 `attention_target_changed`；stdout bounded queue/drop/coalescing 不阻塞 gaze stale；BrokenPipe 尽力发布一次 stale 后受控非 0 退出。
- Botified request frame contract test 必须覆盖权威 contract 或 repo 内 schema 文档中固定的 wrapper、JSON 字段、ttl/timeout 语义、allowlist、错误/ack 期望。
- 不复刻 server event rules。

集成测试必须覆盖：

- In-memory DDS image -> CLI core -> mock server -> in-memory gaze target，用于 fault matrix，不作为 GA E2E 替代。
- Mock server semantic event -> Botified stdout。
- Server down/restart -> CLI reconnect -> gaze invalid/stale。
- DDS publisher down/restart -> CLI recovery。
- Slow server -> old frames dropped, no unbounded queue。
- Slow stdout -> DDS gaze stale 仍准时，stdout pollution count 为 0；Botified pipe close -> 尽力发布一次 `valid=false,state=stale` 后受控非 0 退出。
- Production runner/lifecycle unit core 已覆盖 `RuntimeFactories/run_runtime`、start/run/shutdown lifecycle unit core、stop_requested cleanup seam、Botified task 启停、required head_state 模式、默认 DDS factories fail-fast、coordinator shutdown、sync/async resource close、service client public close、Botified drain daemon-thread bounded shutdown、shutdown observe timeout 内观察到的 BrokenPipe publish stale then nonzero、start failure cleanup。
- PC DDS E2E/over-wire core gate 已由 Step 5 current PC core functional gate 承担；真机 smoke 不能替代这些单元/集成测试，也不阻塞当前 PC core gate。

验收：

- `uv run --group dev pytest -q` 全绿。
- CLI tests 不依赖真实机器人，但 DDS over-wire 和 serialization/QoS 行为必须被测试。
- 真机 smoke 不能替代 production runner/lifecycle unit core、PC DDS E2E/over-wire gate 和 serialization/QoS 测试。

### Step 7：PC 本地核心功能门禁（current PC core pass/fail authority）

产出：

- 使用 `tools/run_cli_local_e2e.py --full-scene --all-scenes --head-state stationary --server-config configs/pc-ga-server.toml` 跑 `val-data/` CLI local E2E。
- 使用 real server runtime 跑 server S8 gates。
- 使用 mock server 跑必要轻量 CLI fault checks。
- 使用 ignored `val-data/manifest.json` 或等价 manifest 固定数据集身份，并在 PC GA evidence 写入 `manifest_sha256`。
- 从 `runtime/venv/bin/visual-events-server` 和 `runtime/venv/bin/visual-events-cli` 启动被测 runtime；只允许编排工具从 dev env 运行。

验收：

- manifest（`val-data` 测试数据清单）中所有 GA scene 全量通过，且 manifest 中 scene 名称、sha256、frame count、fps、expected event timeline source/version、expected attention target timeline/rule（注视目标标准答案：应该看谁、什么时候不该看）与 report 一致。当前数据集若为 7 个 scene，由 manifest 记录；计划不硬编码 scene 数量。
- 标准命令必须显式传入 `--server-bin runtime/venv/bin/visual-events-server` 和 `--cli-bin runtime/venv/bin/visual-events-cli`；runner/report 校验 runtime bin 路径、wheel/runtime provenance 和 server/CLI exit code。使用 dev console script 或手工预启动 server 的结果不能标记 current phase pass / PC gate pass。
- Stationary/unknown/moving suppression 与 server gate 一致。
- Head state required 模式通过：current PC core gate 标准命令为 `--head-state stationary`，只要求 stationary head state 新鲜、`head_state.required=true`、`head_state_publisher_mode=required`、`head_state_hz` >=9。stationary/moving/unknown 三段覆盖保留为历史 partial smoke / unit-integration evidence；moving/unknown 下运动敏感事件会被 suppression，不作为 Botified oracle core pass。
- Gaze correctness gate 通过：多人大小目标、交叉、短遮挡恢复、target switch 抖动上限、目标点语义均被验收；oracle（测试标准答案）必须覆盖“注视最大且稳定的人”的 target label/track、allowed switch windows 和 no-target windows。目标点优先 `head_uv`/face keypoints，fallback 到 bbox 近似头部点，不引入新模型能力。
- Semantic event gate 通过：进入、路过、靠近、停留、挥手等每类事件有 expected occurrence 和允许延迟窗口；负例不得触发。
- Scene-level duplicate greeting gate 通过：在 manifest/oracle 标注的短遮挡、lost hold/cooldown 和空间邻近窗口内，同一物理人不得产生新的招呼型事件序列；`track_id` 变化时也必须通过 scene/person label 发现；不承诺 ReID/长期记忆。
- 有新鲜 `visual_state` 的区间内 gaze target sample >=9Hz 且 <=10Hz，valid 和 invalid 都计入；rate gate 只基于非 `stale` 的 fresh `visual_state` 派生样本，启动/收尾 stale watchdog sample 不计入；连续 5 分钟无无界队列。
- Server 断开后 250ms 内发布一次 `valid=false,state=stale`，随后不再发布过期有效 target，也不承诺断线期间继续 >=9Hz heartbeat；恢复后重新 tracking。
- Current PC core gate config 是 tracked `configs/pc-ga-server.toml`，必须显式设置并验证 `attention.switch_confirm_ms >=750` / target dwell >=750ms；当前 server 默认 500ms 不能作为 PC core gate pass 证据。
- stdout 只包含 Botified allowlist frame；任何 debug/status JSON 到 stdout、任何 `attention_target_changed` Botified 输出都 fail。
- basic finite latency sanity 必须通过；capture->gaze publish 与 capture->Botified stdout P95/P99 latency 写入 `non_blocking_gaps` 或后续 handoff evidence，不阻塞本次 current PC core gate，除非后续单独实现 blocking latency gate。
- `val-data/`、artifacts、metrics JSONL、DDS captures 不进 Git。

### Step 8：Post-GA hardware/field validation

产出：

- Botified 后台 task 启动 CLI 的命令和配置。
- 真机 DDS camera runtime/network 订阅验证。
- 板端 DDS type/QoS compatibility 验证。
- 真实 head state topic/type/Hz/freshness 验证。
- 真机 report 必须同时记录 DDS source `camera_name` 和 CLI logical `camera.name`。
- Gaze DDS shadow consumer preflight，以及真实 head/motion consumer 或等价闭环验收 artifact。
- 现场场景记录：空场、进入、路过、靠近、停留、挥手、多人大小目标、交叉、遮挡恢复、目标切换、目标丢失、server restart、head moving/unknown，并记录 expected attention target timeline/rule、低频事件 expected occurrence/允许延迟窗口/负例不得触发。
- Botified owner 完整产品闭环 artifact：至少一次 `Visual Events semantic event -> Botified 决策/回应动作 -> 冷却不重复招呼`，作为 post-GA/field rollout 证据，不作为本 repo 实现项。
- 30 分钟 soak report。
- Camera owner、gaze consumer/运控 owner、Botified owner 的验收 sign-off。

验收：

这些验证全部 out of GA scope，不阻塞 GA pass/fail。

- CLI 可由 Botified 启动、停止、重启。
- 真实 DDS camera runtime/network 输入 >=9Hz，且 topic/type/QoS 与 handoff 表一致。
- 真实 head state 输入 >=9Hz，freshness 达标；缺失或 stale 只能标记 post-GA hardware/field validation fail。
- 有新鲜 `visual_state` 的区间内 Gaze DDS sample >=9Hz，且只通过 DDS 输出；CLI 不直接调用运控。
- 真机 gaze 不能只用 shadow consumer 证明发消息。Post-GA hardware/field validation 需要真实 head/motion consumer 或等价闭环验收：等价闭环消费同一个 `/visual_events/gaze_target` DDS topic，使用真实或 HIL head_state >=9Hz，证明 valid tracking 时头部物理指向目标并按现场 head pointing 误差阈值跟随，invalid/stale 后不继续动并按停止延迟阈值停止，并由运控 owner sign-off；shadow consumer/logs 只能作为 preflight，不能单独算等价闭环。server/CLI restart 后无残留动作；恢复后重新 tracking。
- Gaze correctness 现场通过：大/小多人目标、交叉、遮挡恢复、target switch jitter 上限、目标点语义与 PC gate 一致。
- 现场 checklist 必须验证 expected attention target timeline/rule：target label/track、allowed switch windows、no-target windows。
- 现场低频事件验收必须覆盖 false-negative/时序：进入、路过、靠近、停留、挥手等每类事件有 expected occurrence 和允许延迟窗口，负例不得触发。
- Botified 只收到低频语义事件，不收到 10Hz 状态。
- Botified 收到的事件符合 allowlist；每事件类型/每 track/每分钟输出上限达标，无重复招呼刷屏。
- 短遮挡/track 切换不得重复招呼：在 manifest/oracle 标注的短遮挡、lost hold/cooldown 和空间邻近窗口内，同一物理人不得产生新的招呼型事件序列；`track_id` 变化时也必须通过 scene/person label 发现；不承诺 ReID/长期记忆。
- Post-GA/field validation 可记录 capture->gaze publish 与 capture->Botified stdout P95/P99 latency；这是现场/handoff evidence，不回溯阻塞本次 current PC core gate。
- 30 分钟无 crash、无明显 RSS 增长、无 stdout 污染、无事件刷屏。
- 现场失败恢复行为符合文档。
- 真机/现场 rollout evidence 不能只有本 repo 自测通过；应有 camera DDS、gaze consumer/运控、Botified 三方 owner sign-off artifact。

### Step 9：Release 和 handoff

Release/handoff 是交付审计层。当前核心阶段只要求 Step 7 的 PC 本地核心功能门禁跑通，并保留直接验证核心路径的 PC local DDS E2E、必要轻量稳定性和 basic finite latency sanity；P95/P99 latency report 是 non-blocking handoff evidence。不要为 release report skeleton、handoff audit、full fault matrix、long soak 或 field/real robot validation 单独扩张代码、schema 或测试。

产出：

- `docs/ga-handoff.md`
- Release/runtime sync 命令。
- Server/CLI 启动命令。
- DDS domain/topic/QoS 表。
- Botified task command。
- Runtime smoke：`tools/run_runtime_smoke.py` 会执行 `uv sync --frozen --no-dev --no-editable --extra inference --reinstall-package visual-events-server`，并强制使用 repo-local `runtime/cache/uv` 和 repo-local `runtime/venv`；随后验证 server + CLI runtime provenance、CLI import check 和 server `/healthz` identity。sync 失败时 provenance 明确 not-run：`runtime_hash=null`、failure reason `runtime_provenance_not_run:sync_failed`，不得采样旧 runtime。
- 当前 PC gate evidence/hash：server baseline、CLI unit/integration、PC local E2E、runtime smoke、no-motion-SDK audit、val-data manifest hash、runtime server/CLI provenance、stationary head state required-mode、Botified stdout allowlist、stdout pollution、fresh gaze Hz/rate 和 basic finite latency sanity。P95/P99 latency report 属于 non-blocking evidence / 后续 handoff evidence。
- `val-data/manifest.json` 或等价 manifest 摘要：scene 名称、sha256、frame count、fps、expected event timeline source/version，以及 `manifest_sha256`。manifest 若位于 ignored `val-data/` 下，handoff artifact 必须包含它的 sha256 和副本摘要。
- Expected attention target timeline/rule 摘要：target label/track、allowed switch windows、no-target windows，并链接 PC report 中对应 oracle 验收结果。
- Post-GA hardware/field rollout evidence：真机 smoke、camera DDS owner sign-off、gaze consumer/运控 owner sign-off、Botified owner sign-off、真实闭环/现场 owner sign-off。这些不阻塞 GA，不得由 PC evidence 声称通过。
- Botified owner 完整产品闭环 artifact：至少一次 `Visual Events semantic event -> Botified 决策/回应动作 -> 冷却不重复招呼`；这是 post-GA/field rollout evidence，不是本 repo 的 Botified 会话逻辑实现，不阻塞 GA。
- Model manifest 和 license owner sign-off。
- no-motion-SDK audit artifact：Python import/dependency denylist、native bridge `ldd`/`readelf` 结果、DDS topic allowlist、report artifact/hash。
- Rollback 操作。

验收：

- Fresh checkout 准备 `val-data/`、模型权重、runtime config 后可复现 PC gate。
- Release/runtime 使用 repo-local `runtime/venv` 和 `runtime/cache/uv`。
- 不依赖用户目录 cache，不改 `HOME`。
- PC GA evidence 记录 `manifest_sha256`、model/runtime/config hash、basic finite latency sanity、non-blocking P95/P99 latency evidence（如已采集）、stationary head state 字段、synthetic DDS `dds_source_camera_name`、`logical_camera_name`/`camera.name` 和 PC DDS compatibility 结果。
- PC handoff report 记录低频事件 false-negative/时序验收、scene-level duplicate greeting gate；Botified owner 完整产品闭环 artifact hash 属 post-GA/field rollout evidence，不得由 PC evidence 声称通过。
- Git 中没有大资源、模型、cache、DDS captures、现场日志。

## 10. Server 改进清单

这些 GA 前 server 收口项已完成；后续只防回归和继续跑 gates：

| 项 | 当前状态 |
| --- | --- |
| frame/timestamp 倒退 | 已 reset 当前连接 tracker/event state，并补测试 |
| JPEG 尺寸策略 | 已校验 header 与 decode 尺寸一致 |
| `someone_near_center` | 已实现真实计算 |
| shared backend inference serialization | 已串行化共享 backend 推理，保持每连接状态隔离 |
| metrics sink error | 已记录 write error count 并输出 stderr |
| CLI 合同测试 | 已固化 attention、semantic events、head_motion suppression 行为 |

## 11. 测试矩阵

| Gate | 覆盖 | 通过标准 |
| --- | --- | --- |
| Server regression | 现有 unit/integration、runtime smoke、`val-data` full matrix、metrics、必要轻量稳定性检查 | 全 pass；P95 <120ms；P99 <200ms；Hz >=9；无无界队列/RSS 明显增长 |
| CLI unit | WebSocket、DDS adapters、target mapper、Botified output、config | 全 pass；stdout/stderr 分离；无运控 API |
| DDS image input | topic/domain、JPEG 校验、jitter/drop、无 publisher、invalid JPEG、source camera_name | nominal 10Hz；异常可恢复；无无界队列；report 同时记录 DDS source camera_name 和 CLI logical camera.name |
| Head state | `/robot/head_state` required mode、stationary Hz/freshness；stationary/moving/unknown segment 作为历史 partial smoke / unit-integration evidence | current PC core gate 只阻塞 stationary head state 新鲜且 Hz >=9；moving/unknown 下运动敏感事件 suppression，不作为 Botified oracle core pass；report 写 `head_state_hz` 和 freshness evidence |
| DDS gaze output | target schema、rate、lifespan、stale/lost/disabled | 每个新鲜 `visual_state` 派生一条 valid/invalid target，新鲜区间内 <=10Hz 且 >=9Hz；keep-latest 可丢旧 DDS 输入帧，不补发旧帧；断线后 250ms 内发布一次 invalid/stale；断线期间不承诺 >=9Hz heartbeat；旧状态不发布有效 target |
| Gaze correctness | 多人大小目标、交叉、遮挡恢复、target switch jitter、`head_uv`/face keypoint/fallback 语义 | PC GA 必须通过；manifest/report 有 expected attention target timeline/rule；不新增模型能力；jitter 超上限 fail |
| Semantic event oracle | 进入、路过、靠近、停留、挥手、负例 scene、短遮挡/track 切换 | 每类事件 expected occurrence 和允许延迟窗口通过；负例不得触发；manifest/oracle 标注的短遮挡、lost hold/cooldown 和空间邻近窗口内不得产生新的招呼型事件序列；不承诺 ReID/长期记忆 |
| Botified stdout | semantic event -> request frame allowlist | 高频状态不进 stdout；`attention_target_changed` 不进 stdout；escaping 正确；同 `event_id` 不重复；slow stdout bounded queue/drop/coalescing 不阻塞 gaze stale；BrokenPipe 尽力发布 stale 后受控非 0 退出 |
| PC local E2E | real DDS participant + synthetic `val-data` image/head state publisher -> runtime CLI -> runtime server -> real DDS gaze sink/stdout collector + Botified event oracle | `tools/run_cli_local_e2e.py --full-scene --all-scenes --head-state stationary --server-config configs/pc-ga-server.toml` 作为 current PC core gate；阻塞 full-scene matrix、Botified oracle、stdout pollution、fresh gaze Hz/rate 和 basic finite latency sanity；P95/P99 latency report 是 non-blocking evidence；不声称 real robot/field/RK/release audit pass |
| Fault checks | server down/restart、slow server、DDS down/restart、bad JPEG、Botified pipe close | 非 BrokenPipe 故障可恢复；BrokenPipe 受控非 0 退出；不刷屏；stale watchdog 准时 |
| Post-GA 真机/板端 DDS compatibility | 真实 camera DDS runtime/network、板端 DDS type/QoS、真实 head state topic/type/Hz/freshness | Post-GA validation；三类 DDS contract 全部兼容只能证明硬件/现场验证通过，不能由 PC evidence 代替 |
| Post-GA 真机闭环 | live camera DDS、real server、CLI、真实 head/motion consumer 或等价闭环、Botified | 30 分钟稳定；valid tracking 物理指向；invalid/stale 不继续动；restart 无残留动作；恢复后重新 tracking |
| no-motion-SDK audit | Python import/dependency denylist、native bridge `ldd`/`readelf`、DDS topic allowlist | artifact/hash 入 handoff；发现运控 SDK 依赖或 command topic 即 fail |

## 12. 配置

GA CLI 配置示例：

```toml
[dds]
domain = 0
network = "eth0"

[camera]
name = "front"
image_topic = "/camera/image/jpeg"
hz = 10

[head_state]
enabled = true
required = true
topic = "/robot/head_state"
stationary_yaw_vel_rad_s = 0.03
stationary_pitch_vel_rad_s = 0.03
stale_ms = 250
report_segments = ["stationary", "moving", "unknown"]

[service]
url = "ws://127.0.0.1:8765/v1/stream"
response_timeout_ms = 1000
reconnect_min_ms = 200
reconnect_max_ms = 3000

[gaze_target]
enabled = true
topic = "/visual_events/gaze_target"
stale_ms = 250
publish_invalid_on_loss = true

[botified]
enabled = true
stdout = true
event_ttl_secs = 8
allowed_events = [
  "person_appeared",
  "person_left",
  "person_passing_by",
  "person_approaching_robot",
  "person_stopped_near_robot",
  "person_waving",
]
stdout_queue_max = 32
stdout_drop_policy = "drop_oldest_duplicate_event"
broken_pipe = "publish_stale_then_exit_nonzero"

[logging]
stderr_level = "info"
jsonl_path = "artifacts/cli/cli_metrics.jsonl"
```

不为每个事件规则开放 CLI 配置。事件阈值仍由 server 管理。

`gaze_target.stale_ms` 和 `service.response_timeout_ms` 是两个不同计时器：前者决定多久内必须让 gaze target 失效，GA 默认 250ms；后者决定 WebSocket 请求多久未返回后关闭连接并重连，GA 建议 1000ms。长时间 E2E 工具可以覆盖更大的 response timeout，但正式 CLI runtime 不应等 30s 才让 gaze 失效。stale watchdog 必须能在 stdout 背压和 one in-flight WebSocket 等待期间继续运行。

## 13. Git 和运行产物策略

允许进入 Git：

- Server/CLI 源码。
- 测试工具源码。
- schema/contract 文档。
- 单元测试和集成测试源码。
- `uv.lock`。

禁止进入 Git：

- `val-data/`
- `runtime/`
- `.venv/`
- `.uv-cache/`
- `.cache/`
- `artifacts/`
- 模型权重和模型 cache。
- DDS captures、bag、pcap、现场图片/视频/日志。
- Botified 本地环境、token、机器人私有 IP、设备私有配置。
- 编译产物、core dump、generated cache。

`val-data/` 整体仍然禁止进入 Git。`val-data/manifest.json` 如果放在 `val-data/` 下也不进 Git；PC GA evidence 必须包含 manifest sha256 和副本摘要，并记录 `manifest_sha256`。不要在 `.gitignore` 写 `*.json`，避免误伤 `common/schema/**/*.json` 等小型 schema/sample 文件。

`.gitignore` 必须覆盖通用大资源和本地捕获输出，包括模型权重、ONNX/TensorRT/RKNN engine、MCAP/bag/pcap、视频、现场图片、JSONL metrics、日志、capture/cache 目录；源码中必须跟踪的小 JSON schema/sample 文件不受影响。

## 14. Team Review 结论

产品 review：

- GA 最大边界是 CLI 只发布 DDS gaze target，不直接控制头部。
- PC 本地 DDS E2E 是 current PC core functional gate；真机/板端 DDS compatibility 和真实闭环验收是 GA 之后的硬件适配/现场验证，不是 deferred current PC core gate。
- 不发布完整高频 `visual_state` DDS；GA 只做 gaze target 一个高频 DDS 输出。

研发 review：

- CLI package、entrypoint、config skeleton、`target_mapper` 和 `botified_output` 纯逻辑已存在；`service_client` WebSocket wire client unit core、`frame_pump` deterministic unit core/stale watchdog、RuntimeCoordinator/main wiring unit core、main runtime_runner 注入和默认 DDS factories fail-fast、exact stale deadline、slow Botified drain 不阻塞 stale、Botified stdout bounded queue/drop/coalescing 与 BrokenPipe exception unit core、BrokenPipe publish stale then nonzero unit core 已完成。`RuntimeFactories/run_runtime` production runner/lifecycle unit core 已完成：测试注入 factories，start image/head/gaze，head current_motion wiring，coordinator shutdown，sync/async resource close，service client public close，Botified drain daemon-thread bounded shutdown，shutdown observe timeout 内观察到的 BrokenPipe publish stale then nonzero，start failure cleanup。
  Step 5 native PC DDS over-wire test participant slice 已完成：image publisher、head publisher、gaze subscriber 和 `pc_test_tools` 已通过 loopback over-wire smoke。Step 5 Python native participant wrappers slice 已完成：`tools/dds_pc_tools.py` 和三个 wrapper 已接入，显式 `--dds-domain`/`--dds-network`，非 `lo` 必须 `--allow-non-loopback-dds`，默认只从 `--build-dir` 找 native binary，Python wrappers 不 import DDS SDK/visual_events_cli/server，wrapper-level PC loopback smoke 已通过。Step 5 manifest reader/report skeleton slice 已完成：`tools/cli_local_e2e_manifest.py` 读取 `--data-dir`，默认探测 `<data-dir>/manifest.json`（标准 `--data-dir val-data` 时为 `val-data/manifest.json`），也支持显式 `--manifest`；manifest 缺失时生成 deterministic effective manifest，记录 scene 名称、per-scene sha256、frame count、first/last frame，并记录 aggregate `scene_count` / `frame_count` 和 `manifest_sha256`；manifest skeleton report 固定 `overall_pass=false`、`pc_local_e2e_status=not_run`，只针对 `tools/cli_local_e2e_manifest.py` 的报告骨架；只做数据集身份和报告骨架，不能替代 runtime E2E；`--out` 不能写入 data dir；本机 ignored `val-data` 当前可识别 7 scene / 576 frames。
  Step 5 `run_cli_local_e2e` current PC core functional gate 已完成：`tools/run_cli_local_e2e.py --full-scene --all-scenes --head-state stationary --server-config configs/pc-ga-server.toml` 跑正式 CLI + real server + `val-data` full-scene matrix，并用 Botified event oracle 判定 stdout 事件；历史 partial smoke 只作为兼容/回归背景，不能再代表当前门禁能力。
  Step 5 mock visual_state server slice 已完成：`tools/mock_visual_state_server.py` 提供 `/healthz` 和 `/v1/stream`，复用现有 `visual_events_server.protocol` decode/serialize，不定义第二套协议；支持 `tracking/lost/event` profile、`--delay-ms`、`--disconnect-after`，用于 CLI deterministic attention/event、slow response、disconnect 测试；不 import DDS/CLI runtime/模型，不启动 DDS participant，不读取 val-data，不做 runner/report；RK/board probe、真机 smoke 和 closed-loop handoff 属于 post-GA validation。
- Step 5 CLI runtime + mock visual_state server integration slice 已完成：`tests/integration/test_cli_bridge_runtime_mock_server.py` 使用真实 `bridge_runtime_factories()`/`run_runtime`、fake JSONL bridge child 和 mock server subprocess，覆盖 event Botified stdout allowlist、tracking/lost/stale gaze；不使用 DDS participant、`val-data` 或 real server，不做 runner/report；不能替代 current PC core gate。
- DDS contract/schema Step 1 主要产物已完成；DDS stack decision record 已冻结在 `docs/dds-stack-decision-record.md`，选择 Unitree SDK2 2.0.0 + CycloneDDS 0.10.2 + C++ native DDS helper/bridge，并记录 IDL codegen gate、native bridge ABI、runtime env vars、PC loopback smoke plan 和 post-GA 板端 validation/probe plan。
- Server S8 baseline 不重写；frame/timestamp reset、JPEG dimensions validation、`someone_near_center`、shared backend inference serialization、metrics sink write error count/stderr 和 server/CLI contract tests 已完成，后续只防回归/跑 gates。
- Step 4 first slice/unit core 已完成：纯 Python SDK-neutral DDS adapter core/fakes 覆盖 QoS constants、CameraJpegMessage JPEG SOF dimension validation、fake image latest-only、Fake DDS adapters lifecycle unit core（start/close idempotent；close 后拒绝使用/重启）、HeadStateSample stationary/moving/unknown stale/future timestamp mapping、FakeDdsGazeTargetPublisher lifecycle、protocol names 和 no-motion/no-real-DDS import audit。Step 4 Python JSONL bridge client/facade slice 已完成：`bridge_protocol.py`、`bridge_process.py`、`bridge_adapters.py`、`runtime_factories.py` 和 explicit `bridge_runtime_factories()` 覆盖 JSONL protocol/base64/canonical gaze fields、subprocess lifecycle 和 no DDS/native import audit tests。Step 4 Python JSONL bridge runtime integration slice 已完成，formal CLI bridge runtime opt-in slice 已完成：默认仍 fail_fast，不因 env 隐式切 bridge；显式 `[dds].runtime="bridge"`/`--dds-runtime bridge` 才走 `bridge_runtime_factories()`。真实 subprocess fake JSONL child + `bridge_runtime_factories()`/`run_runtime` 覆盖 camera/head -> service -> gaze stdin、logical camera、stale/cleanup、child nonzero/fatal；该 slice 不覆盖真实 DDS runtime，未完成边界统一见 Step 4 剩余缺口。Step 4 native DDS bridge build/probe foundation slice 已完成：`visual_events_dds_bridge_probe` 可输出既有 JSONL bridge ABI status frame（`protocol_version=1,type=status,code=probe_ok,message=...`），`tools/build_dds_bridge.py` foundation gate 可在无 IDL generator 时通过并报告 `foundation_ready=true`。Step 4 DDS C++ idlc repo-local prepare/oracle hardening slice 已完成：`tools/prepare_dds_codegen_toolchain.py --prepare` 可在 ignored `build/tools/cyclonedds-cxx-idlc-0.10.2/` 下准备 pinned idlc wrapper 并复用 Head/Gaze codegen oracle；`--probe-codegen` 和 `tools/build_dds_bridge.py --check-full-bridge` 会实际运行 C++ idlc probe，默认验证 `head_state_v1.idl` 和 `gaze_target_v1.idl` 都生成 expected `.hpp/.cpp`，拒绝版本不是 0.10.2、`cannot load generator cxx`、任一 IDL 只生成 `.hpp` 或缺 `.cpp` 的假阳性；fake git/cmake/idlc 成功生成 expected `.hpp/.cpp` 时报告 `visual_events_codegen_ready=true` 和 `oracle_ok=true`。Step 4 native full-bridge generated Head/Gaze C++ type-support compile/probe slice 已完成：`tools/build_dds_bridge.py --check --check-full-bridge --build --probe` 会运行 Head/Gaze IDL codegen oracle，CMake full-bridge 编译 `head_state_v1.hpp/.cpp` 和 `gaze_target_v1.hpp/.cpp`，native probe 检查 `CameraFrame_`、`HeadStateV1_`、`GazeTargetV1_` type props 并输出一行 JSONL status；Foundation 路径仍然 CameraFrame-only。Step 4 native JSONL ABI/runtime skeleton slice 已完成：`visual_events_dds_bridge` target 存在，`--probe` 单行 JSONL status；ABI-only 不带参数运行仍 explicit fatal `dds_runtime_not_implemented`；`visual_events_dds_bridge_abi_harness` 复用同一 ABI core 产出 fake camera/head，消费 Python canonical `gaze_target`，parser 严格 canonical fields + state 闭集。Step 4 native generated DDS type/ABI mapping construction slice 已完成：native generated DDS type/ABI mapping construction slice 验证 camera/head/gaze field mapping、head state derived stationary/moving/unknown、gaze valid/state consistency + finite/range checks。Step 4 native Unitree Channel construction harness/smoke slice 已完成：`visual_events_dds_bridge_construction_harness` 是 full-bridge only construction harness；`runtime_options` pure env parser；`--print-options` 单行 JSONL，`--print-options` 不启 DDS；`--construct-once` 解析 env，Unitree ChannelFactory Init(domain/network)，构造 `CameraFrame_` subscriber、构造 `HeadStateV1_` subscriber、构造 `GazeTargetV1_` publisher，CloseChannel 后 Release。Step 4 native runtime loop core/full-bridge wiring/fake harness/build include fix slice 已完成：`runtime_loop` core；`visual_events_dds_bridge_runtime_loop_harness` fake harness；full-bridge `visual_events_dds_bridge` 无参数路径进入 Unitree DDS runtime loop；ABI-only 路径仍 explicit fatal `dds_runtime_not_implemented`；stdout emitter latest-slot 输出 camera_jpeg/head_state；stdin 读取 canonical `gaze_target` JSONL 并经 backend 发布 DDS gaze；async backend fatal 不被 stdin 阻塞；shutdown late fatal 仍输出 fatal JSONL；full-bridge 构建显式传入 repo-local CycloneDDS C++ include dir。该 slice 只证明 native runtime loop；current PC core gate 的 full-scene/Botified 证据见 Step 5，不证明 release report、RK/board 或真机闭环。外部源码/build/install/probe 输出不进 Git；未完成边界统一见 Step 4 剩余缺口。
- DDS topic/type/QoS 合同已固化在 schema 文档；current PC core gate 已覆盖 PC 本地 DDS over-wire 核心路径。板端 compatibility probe 属于 post-GA hardware validation；没有该验证不得声称 board compatible、RK supported 或真机 E2E 等价。

QA/release review：

- 保留 server S8 gates。
- CLI 当前 PC gate 必须有 unit、integration、PC local E2E 和必要轻量 fault checks；真机 smoke/closed-loop 是 post-GA hardware/field validation。
- Release handoff 必须证明 CLI 不链接/不调用运控 SDK，只通过 DDS 输出 gaze target。
- GA evidence 必须包含 `val-data` manifest hash、expected attention target timeline/rule 验收、低频事件 false-negative/时序验收、runtime server/CLI 入口证明、stationary head state PC gate required-mode 证明、Botified allowlist/stdout pollution 证明、fresh gaze Hz/rate、basic finite latency sanity 和 no-motion-SDK audit artifact/hash；端到端 latency P95/P99 是 non-blocking evidence / 后续 handoff evidence；Botified owner 完整产品闭环 artifact 属于 post-GA/field rollout evidence。
