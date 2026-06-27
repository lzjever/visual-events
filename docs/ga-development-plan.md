# Visual Events GA 后续开发计划

日期：2026-06-27

## 1. 目标

本计划覆盖 `visual-events` 从当前 server S0-S8 baseline 走到 GA 的剩余工作：服务端收口改进、正式机器人端 CLI、DDS gaze 输出契约、PC 完全本地化端到端测试、真机验证和 release handoff。

首个 GA 场景是商店门口揽客机器人。系统必须在 10Hz 输入下稳定观察前方画面，识别人、追踪人、生成低频语义事件，并持续发布高频 gaze 目标，让机器人可以注视画面中最大且稳定的人。

核心边界：

- `visual-events-server` 只做视觉推理、追踪、attention 和语义事件生成。
- `visual-events-cli` 订阅 DDS JPEG，调用 server，发布 DDS gaze target，输出 Botified request frame。
- `visual-events-cli` 不直接操纵运控，不调用头部速度/位置/`look_at` API，不链接运动控制 SDK。
- 运控/头控模块订阅 DDS gaze target，并在自己的安全边界内执行真实头部动作；该模块不在本 repo 实现。
- Botified agent 只接收低频语义事件并决定后续响应，不参与 10Hz 注视闭环。
- PC 本地 E2E 是 release regression gate，不替代真机/板端 DDS 兼容 gate 和真实头部闭环验收。
- 完整 GA 必须有可用、新鲜的 head state；缺失 head state 只能算 degraded 运行，不能算完整 GA 通过。
- `attention_target_changed` 只保留在 `visual_state` 和诊断 artifact 中，不输出到 Botified stdout。

## 2. 开发原则

这些原则必须写进实现 review checklist：

- KISS：一个 DDS 图像输入，一个 WebSocket 推理协议，一个 DDS gaze target 输出，一个 Botified 事件出口。
- DRY：事件规则只在 server 实现；CLI 只做事件幂等输出，不重新判断 passing by、approaching、stopped 或 waving。
- YAGNI：不训练模型，不做人脸识别，不做 ReID，不做多摄像头融合，不做数据库，不做后台治理平台，不引入第二套 streaming 协议。
- 本体安全边界清晰：CLI 只发布目标事实和有效期，不发布速度/位置命令，不直接控制硬件。
- 可测试优先：PC 本地 DDS 端到端测试是 GA release regression gate，不是 demo，也不是现场/板端 gate 的替代品。
- 运行隔离：继续使用 `uv`；开发环境和 release/runtime 环境分开；不污染系统和用户目录；模型、cache、artifacts 不进 Git。
- Artifact 可追溯：`val-data/` 不进 Git，但每次 PC/release report 必须记录数据 manifest hash、模型/runtime/config hash 和关键 report hash。
- 输出隔离：CLI stdout 是 Botified allowlist 输出，不是调试通道；高频 attention/gaze 状态不得泄漏到 Botified。

## 3. 当前基线

当前 repo 已完成 server S0-S8 baseline：

- WebSocket `/v1/stream` 协议。
- `YOLOv8n-pose` 推理 backend 和 mock backend。
- 项目内 ByteTrack-style IoU/TTL tracker baseline。
- attention selector。
- semantic events：`person_appeared`、`person_left`、`person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot`、`person_waving`、`attention_target_changed`。
- `val-data/` full matrix、semantic event timeline gate、runtime smoke、300s soak、opt-in server metrics evidence。

后续开发不能重写 server 主线。Server 工作只做 GA 收口改进和与 CLI 的合同测试。

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
- PC 和现场验收 report 必须量化每事件类型、每 track、每分钟输出上限；超过上限即 fail，不能交给 agent 兜底。

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

PC 本地 E2E 必须使用真实 DDS participant 和正式 CLI。Mock server 只能用于 CLI failure-path 和 deterministic unit/integration tests；不能替代 real server + `val-data/` release gate。

PC 本地 E2E 的定位是 regression gate：证明 repo 内 CLI/server/DDS 合同在可复现数据上稳定。完整 GA 还必须通过真机/板端 DDS 兼容 gate，覆盖真实 camera DDS runtime/network、板端 DDS type/QoS compatibility、真实 head state topic/type/Hz/freshness，以及真实 head/motion consumer 或等价闭环验收。

## 6. DDS 契约

### 6.1 图像输入

复用 `/home/galbot/works/image-capture` 的 JPEG DDS 约定：

| 字段 | 值 |
| --- | --- |
| topic | `/camera/image/jpeg` |
| DDS type | `unitree_camera::msg::dds_::CameraFrame_` |
| camera_name | `image` |
| encoding | `JPEG` |
| data | 完整 JPEG bytes，包含 SOI/EOI marker |
| 默认 domain/network | `DDS_DOMAIN=0`，`DDS_NETWORK=eth0` |
| QoS | best effort、volatile、keep last 1、deadline 150ms、lifespan 300ms、automatic liveliness lease 1000ms |

CLI 只保留最新合法 JPEG frame。非法 JPEG、非 JPEG encoding、空 data、width/height 非法必须丢弃并计数到 stderr/metrics。

PC 本地 E2E 不使用默认真机网络。测试 runner 默认必须使用 `DDS_NETWORK=lo` 和非 0 domain，例如 `DDS_DOMAIN=57`，除非显式传入 `--allow-non-loopback-dds`。

`CameraFrame_` 不携带源 `frame_id`。CLI 必须为每条 WebSocket connection 生成 per-connection monotonic transport `frame_id`，用于 request/response 对齐和 server state reset；不得把 DDS `timestamp_ns`/`timestamp_ms` 当作 identity。源 timestamp 只允许作为 `frame_timestamp_ms` 和 freshness fallback 输入；缺失、重复、倒退或跨时钟跳变时以 CLI receive monotonic time 判 stale。

### 6.2 头部状态输入

完整 GA 要求 CLI 能订阅头部状态 DDS，用于生成 WebSocket frame header 的 `head_motion`。缺失头部状态时系统可以降级运行，但 passing by、approaching、stopped 三类运动敏感事件会被 server suppression；因此缺失头部状态不能算完整 GA 通过。

| 字段 | 说明 |
| --- | --- |
| topic | `/robot/head_state`，实现前需与运控 owner 固化最终名称 |
| DDS type | `visual_events::msg::dds_::HeadStateV1_` 或运控 owner 已有等价类型；Step 1 必须固化 |
| QoS | best effort、volatile、keep last 1、deadline 150ms、lifespan 250ms、automatic liveliness lease 500ms |
| timestamp | `timestamp_ms`，Unix epoch milliseconds；CLI 以本机 monotonic clock 判断 stale |
| 角度单位 | radians |
| 角速度单位 | radians/second |
| 最小字段 | `schema_version:uint32`、`timestamp_ms:int64`、`valid:bool`、`yaw_rad:float64`、`pitch_rad:float64`、`yaw_vel_rad_s:float64`、`pitch_vel_rad_s:float64` |
| 映射 | 速度低于阈值且状态新鲜时 `stationary`，速度超过阈值时 `moving`，缺失/过期/invalid 时 `unknown` |

如果头部状态不可用，CLI 必须发送 `head_motion.state=unknown`。Server 会暂停 `person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot` 的条件累积和触发。

完整 GA 的 CLI 配置必须把 head state 设为 required 模式。PC E2E 标准命令、report 字段和验收必须包含 `head_state.required=true`、`head_state_publisher_mode=required`、`head_state_hz`、`head_state_stale_count`、`head_state_unknown_ratio`，并覆盖 `stationary`、`moving`、`unknown` 三类 segment。缺少 head state、Hz 不达标、freshness 不达标或 unknown ratio 超阈值时，report 只能标记 degraded/fail，不能标记 GA pass。

### 6.3 Gaze Target 输出

GA 固化一个高频 DDS 输出，不发布完整 `visual_state` DDS：

| 字段 | 值 |
| --- | --- |
| topic | `/visual_events/gaze_target` |
| DDS type | `visual_events::msg::dds_::GazeTargetV1_`，Step 1 必须产出 IDL 或等价权威类型定义 |
| QoS | best effort、volatile、keep last 1、deadline 150ms、lifespan 250ms、automatic liveliness lease 500ms |
| 频率 | 有新鲜 `visual_state` 时每帧发布一条 sample，<=10Hz，目标 >=9Hz；valid 和 invalid sample 都计入 |
| 消费方 | 运控/头控 owner |

消息语义是 target，不是 command。坐标使用输入图像像素坐标系，原点左上，`u/x` 向右，`v/y` 向下。`target_norm_x = target_u / image_width - 0.5`，`target_norm_y = target_v / image_height - 0.5`。Timestamp 字段使用 Unix epoch milliseconds；CLI 内部 stale 判断使用 monotonic clock。

字段级合同：

| 字段 | 类型 | 规则 |
| --- | --- | --- |
| `schema_version` | `uint32` | 固定 `1` |
| `camera` | string | 与 WebSocket frame header camera 一致 |
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
- Server 超时、断线或 `visual_state` 过期时，CLI 必须在 250ms 内发布一次 `valid=false,state=stale` sample；之后不要求继续保持 >=9Hz heartbeat，并依赖 DDS lifespan 让下游失效。
- `gaze_target.stale_ms` 是独立 watchdog，不依赖 WebSocket `response_timeout_ms`。即使当前 one in-flight request 还没 timeout，只要最近可用 gaze target 到达 stale deadline，CLI 也必须立即发布一次 `valid=false,state=stale`。
- 下游运控必须尊重 `valid=false`、`stale_after_ms` 和 DDS lifespan；这是假设 CLI 不直接控运仍然安全的前提。

## 7. 代码结构

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

PC release gate 必须从 `runtime/venv` 启动 server 和 CLI；orchestration 工具可以从 dev env 跑，但不能用 dev console script 代替 runtime 入口验收。

CLI core 优先使用 Python/`uv` 管理，与 server 共用 repo-local release/runtime 规则。如果 Unitree DDS runtime 只能通过 C++ SDK 接入，可以在本 repo 内实现一个很小的 native DDS bridge 或 helper，但对 CLI core 暴露的接口仍是 `DdsImageSubscriber`、`DdsHeadStateSubscriber`、`DdsGazeTargetPublisher`，并且同样受单元测试、PC E2E 和 no-motion-SDK audit 约束。

RK3588 compatibility 不作为本轮 GA 承诺，但实现必须保护未来迁移边界：server 推理 backend 仍通过 `InferBackend` 替换；CLI 不依赖 Torch/Ultralytics；aarch64/RK3588 runtime 对 CUDA extra 必须 fail fast 或选择 explicit unsupported/backend placeholder；未来 RKNN 插入点只落在 server backend/package/config 边界，不侵入 CLI/DDS/Botified 合同。

## 8. 实现步骤

### Step 1：固化合同文档和 schema

产出：

- `common/schema/dds/camera_jpeg_contract.md`
- `common/schema/dds/gaze_target_v1.idl` 或等价权威类型定义，以及 `gaze_target_v1.md`
- `common/schema/dds/head_state_v1.idl` 或运控 owner 已有等价类型引用，以及 `head_state_v1.md`
- `visual_state`、`semantic_event`、gaze target 的 JSON sample。
- README、产品文档、开发计划全部移除“CLI 直接头控”表述。

验收：

- 文档明确 CLI 不直接操纵运控。
- 对实现代码做 no-motion-SDK audit：`visual_events_cli` 和 DDS bridge 不得依赖、链接、import 或调用运控 SDK；不得发布 motor/head control command topic；gaze target 输出不得含 velocity/position command 字段。允许只读 head state 字段和 `head_motion` 映射代码。
- DDS topic、QoS、字段、stale/lifespan 语义可由 PC 工具和真机实现复用。

### Step 2：完成 server GA 收口改进

产出：

- Frame `frame_id` 或 `timestamp_ms` 倒退时 reset 当前连接 tracker/event state，并补集成测试。
- 明确 JPEG header `width/height` 与实际解码尺寸的策略：优先校验一致；如果暂不校验，文档必须声明 server 以解码尺寸为准并记录 warning。
- 实现 `scene_flags.someone_near_center` 或从协议中弱化为 reserved/optional，避免长期硬编码。
- 明确单 camera/多连接策略：GA 先允许多连接但每连接独立状态，真实 inference backend 用串行锁或单 worker 避免并发推理状态不确定。
- Metrics sink 写失败计数到 stderr 或 health/debug 状态，不改变 `visual_state` wire protocol。
- Server 与 CLI 合同测试：`visual_state.attention`、`semantic_events`、`head_motion` suppression 对 CLI 期望稳定。

验收：

- 现有 `pytest -q` 全绿。
- Runtime smoke 通过。
- `val-data/` full matrix、semantic timeline gate、300s soak、metrics aggregation 继续通过。
- 性能仍满足 P95 < 120ms、P99 < 200ms、输出 >=9Hz、显存 <4GiB。

### Step 3：实现正式 CLI core

产出：

- `visual_events_cli.main` 和 `visual-events-cli` 入口。
- 配置读取：server URL、camera name、DDS domain/network、image topic、head state topic、gaze topic、stale thresholds、log path。
- WebSocket client：one in-flight frame、keep-latest backpressure、response timeout、断线重连、retryable/non-retryable error 处理。
- Frame pump：DDS image -> WebSocket header/JPEG；等待 response 期间只保留最新 DDS frame。
- `target_mapper`：`visual_state.attention` -> gaze target DDS payload；无有效 target 时生成 invalid sample。
- `botified_output`：`semantic_events` -> `<botified>...</botified>` request frame；按 `event_id` 幂等；stdout 只允许 Botified allowlist frame。
- Botified stdout allowlist 固定为 `person_appeared`、`person_left`、`person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot`、`person_waving`；`attention_target_changed` 不输出到 Botified。
- stdout writer 使用 bounded queue/drop/coalescing 或明确 BrokenPipe fail behavior；stdout 写阻塞不得影响 gaze stale watchdog。

验收：

- CLI 不依赖 Ultralytics/Torch，不跑模型。
- CLI 不链接、不 import、不调用任何运控 SDK。
- `visual_state` 不进入 stdout。
- 所有日志、metrics、debug 输出走 stderr 或 ignored artifact。
- Server 慢、断线、超时、错误响应或 Botified stdout 背压时，CLI 不崩溃、不无界排队、不刷 Botified，并按 `gaze_target.stale_ms` 准时发布 stale。

### Step 4：实现 DDS adapters

产出：

- `DdsImageSubscriber`：订阅 `/camera/image/jpeg`，校验 `CameraFrame_` 字段和 JPEG SOI/EOI，latest-only buffer。
- `DdsHeadStateSubscriber`：订阅头部状态，输出 `stationary|moving|unknown`。
- `DdsGazeTargetPublisher`：发布 `/visual_events/gaze_target`，best effort、volatile、keep last 1、deadline 150ms、lifespan 250ms。
- DDS adapter 的 fake/in-memory implementation，用于单元测试。
- 真机路径的 topic/type/QoS/serialization 单元测试；不能只有 mock path。
- 板端 DDS compatibility probe：在目标系统镜像或等价容器内验证 camera/head/gaze 三个 topic 的 type name、serialization、QoS、domain/network 和 Hz/freshness。

验收：

- 无 publisher 时 CLI 可诊断并保持可恢复。
- Invalid JPEG 不发送给 server。
- DDS gaze target payload 坐标有限且在图像范围内；invalid sample 明确。
- DDS resource lifecycle 可重复 start/stop，无泄漏、无后台线程悬挂。
- PC release report 和真机 smoke report 都必须包含 DDS type/QoS compatibility 结果；PC green 不等价于板端 green。

### Step 5：实现 PC 本地测试工具

产出：

- `tools/publish_test_dds_images.py`：从 `val-data/` 或 JPEG 目录按 10Hz 发布 DDS image；支持 loop、jitter、drop、invalid JPEG、timestamp skew。
- `tools/publish_test_head_state.py`：发布 `stationary|moving|unknown` 头部状态。
- `tools/subscribe_test_gaze_targets.py`：订阅 gaze target，写 ignored JSONL，校验 rate、schema、stale、坐标范围和 invalid sample。
- `tools/mock_visual_state_server.py`：给 CLI 做确定性 attention/event、slow response、disconnect 测试。
- `tools/run_cli_local_e2e.py`：编排 server、CLI、DDS publishers、gaze sink、stdout collector，生成 ignored report。
- `val-data/manifest.json` 或等价 manifest reader：列出 scene 名称、scene sha256、frame count、fps、expected event timeline source/version；manifest 文件可位于 ignored `val-data/` 下，但 report 必须记录 `manifest_sha256` 和 manifest 副本摘要。

验收：

- PC 无真机时可以跑完整 DDS image -> CLI -> server -> DDS gaze/Botified E2E。
- PC E2E 必须使用真实 DDS participant、synthetic `val-data` publisher、正式 CLI 和 real server；fake/in-memory DDS 只用于单元、集成和 fault matrix。
- E2E 默认使用 `DDS_NETWORK=lo`、`DDS_DOMAIN=57`；缺少显式 domain/network 时 fail fast；非 loopback 网络必须显式传 `--allow-non-loopback-dds`。
- 工具异常退出时清理子进程。
- Report 包含 `manifest_sha256`、frame count、image Hz、server response Hz、gaze Hz、`head_state.required`、`head_state_publisher_mode`、`head_state_hz`、`head_state_stale_count`、`head_state_unknown_ratio`、`head_state_segments`、Botified event count、stdout pollution count、reconnect count、stale/invalid sample count。
- Report 包含端到端 latency 字段：`capture_to_gaze_publish_p95_ms`、`capture_to_gaze_publish_p99_ms`、`capture_to_botified_stdout_p95_ms`、`capture_to_botified_stdout_p99_ms`。GA 预算：capture->gaze publish P95 <= 250ms、P99 <= 400ms；capture->Botified stdout P95 <= 300ms、P99 <= 500ms。
- Runner exit code：所有 gate pass 返回 `0`；schema/rate/stdout/timeout/process cleanup 任一失败返回非 0。
- 标准命令：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/run_cli_local_e2e.py \
  --data-dir val-data \
  --manifest val-data/manifest.json \
  --dds-domain 57 \
  --dds-network lo \
  --head-state-mode required \
  --head-state-segments stationary,moving,unknown \
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
- Botified allowlist 排除 `attention_target_changed`；stdout bounded queue/drop/coalescing 或 BrokenPipe fail behavior 不阻塞 gaze stale。
- 不复刻 server event rules。

集成测试必须覆盖：

- In-memory DDS image -> CLI core -> mock server -> in-memory gaze target，用于 fault matrix，不作为 GA E2E 替代。
- Mock server semantic event -> Botified stdout。
- Server down/restart -> CLI reconnect -> gaze invalid/stale。
- DDS publisher down/restart -> CLI recovery。
- Slow server -> old frames dropped, no unbounded queue。
- Botified pipe close/slow stdout -> DDS gaze stale 仍准时，stdout pollution count 为 0。

验收：

- `uv run --group dev pytest -q` 全绿。
- CLI tests 不依赖真实机器人，但真实 DDS adapter 的 serialization/QoS construction 必须被测试。

### Step 7：PC 本地 GA gate

产出：

- 使用 `val-data/` 跑 CLI local E2E。
- 使用 real server runtime 跑 server S8 gates。
- 使用 mock server 跑 CLI fault matrix。
- 使用 ignored `val-data/manifest.json` 或等价 manifest 固定数据集身份，并在 PC release report 写入 `manifest_sha256`。
- 从 `runtime/venv/bin/visual-events-server` 和 `runtime/venv/bin/visual-events-cli` 启动被测 runtime；只允许编排工具从 dev env 运行。

验收：

- 7 个 `val-data/` scene 全量通过，且 manifest 中 scene 名称、sha256、frame count、fps、expected event timeline source/version 与 report 一致。
- Stationary/unknown/moving suppression 与 server gate 一致。
- Head state required 模式通过：`head_state.required=true`、`head_state_publisher_mode=required`、`head_state_hz` >=9、`head_state_stale_count` 为 0、`head_state_unknown_ratio` 在 stationary/moving segment 中为 0，并且 report 覆盖 stationary/moving/unknown segments。
- Gaze correctness gate 通过：多人大小目标、交叉、短遮挡恢复、target switch 抖动上限、目标点语义均被验收；目标点优先 `head_uv`/face keypoints，fallback 到 bbox 近似头部点，不引入新模型能力。
- 有新鲜 `visual_state` 的区间内 gaze target sample >=9Hz，valid 和 invalid 都计入；连续 5 分钟无无界队列。
- Server 断开后 250ms 内发布 `valid=false,state=stale`，随后不再发布过期有效 target；恢复后重新 tracking。
- stdout 只包含 Botified allowlist frame；任何 debug/status JSON 到 stdout、任何 `attention_target_changed` Botified 输出都 fail。
- capture->gaze publish 与 capture->Botified stdout P95/P99 latency 符合预算，并写入 report。
- `val-data/`、artifacts、metrics JSONL、DDS captures 不进 Git。

### Step 8：真机 smoke 和现场 GA 验证

产出：

- Botified 后台 task 启动 CLI 的命令和配置。
- 真机 DDS camera runtime/network 订阅验证。
- 板端 DDS type/QoS compatibility 验证。
- 真实 head state topic/type/Hz/freshness 验证。
- Gaze DDS shadow consumer preflight，以及真实 head/motion consumer 或等价闭环验收 artifact。
- 现场场景记录：空场、进入、路过、靠近、停留、挥手、多人大小目标、交叉、遮挡恢复、目标切换、目标丢失、server restart、head moving/unknown。
- 30 分钟 soak report。
- Camera owner、gaze consumer/运控 owner、Botified owner 的验收 sign-off。

验收：

- CLI 可由 Botified 启动、停止、重启。
- 真实 DDS camera runtime/network 输入 >=9Hz，且 topic/type/QoS 与 handoff 表一致。
- 真实 head state 输入 >=9Hz，freshness 达标；缺失或 stale 只能 degraded/fail，不能完整 GA pass。
- 有新鲜 `visual_state` 的区间内 Gaze DDS sample >=9Hz，且只通过 DDS 输出；CLI 不直接调用运控。
- 真机 gaze 不能只用 shadow consumer 证明发消息。GA 必须有真实 head/motion consumer 或等价闭环验收：valid tracking 时头部物理指向目标；invalid/stale 后不继续动；server/CLI restart 后无残留动作；恢复后重新 tracking。
- Gaze correctness 现场通过：大/小多人目标、交叉、遮挡恢复、target switch jitter 上限、目标点语义与 PC gate 一致。
- Botified 只收到低频语义事件，不收到 10Hz 状态。
- Botified 收到的事件符合 allowlist；每事件类型/每 track/每分钟输出上限达标，无重复招呼刷屏。
- capture->gaze publish 与 capture->Botified stdout P95/P99 latency 符合预算。
- 30 分钟无 crash、无明显 RSS 增长、无 stdout 污染、无事件刷屏。
- 现场失败恢复行为符合文档。
- 真机 handoff 不能只有本 repo 自测通过；必须有 camera DDS、gaze consumer/运控、Botified 三方 owner sign-off artifact。

### Step 9：Release 和 handoff

产出：

- `docs/ga-handoff.md`
- Release/runtime sync 命令。
- Server/CLI 启动命令。
- DDS domain/topic/QoS 表。
- Botified task command。
- Runtime smoke 证明 `runtime/venv/bin/visual-events-server` 和 `runtime/venv/bin/visual-events-cli` 均存在且来自 `visual-events-server` wheel。
- Artifact hash：server baseline、CLI unit/integration、PC local E2E、真机 smoke、30 分钟 soak、runtime smoke、no-motion-SDK audit。
- `val-data/manifest.json` 或等价 manifest 摘要：scene 名称、sha256、frame count、fps、expected event timeline source/version，以及 `manifest_sha256`。manifest 若位于 ignored `val-data/` 下，handoff artifact 必须包含它的 sha256 和副本摘要。
- Model manifest 和 license owner sign-off。
- Camera DDS owner、gaze consumer/运控 owner、Botified owner sign-off。
- no-motion-SDK audit artifact：Python import/dependency denylist、native bridge `ldd`/`readelf` 结果、DDS topic allowlist、report artifact/hash。
- Rollback 操作。

验收：

- Fresh checkout 准备 `val-data/`、模型权重、runtime config 后可复现 PC gate。
- Release/runtime 使用 repo-local `runtime/venv` 和 `runtime/cache/uv`。
- 不依赖用户目录 cache，不改 `HOME`。
- PC/release report 记录 `manifest_sha256`、model/runtime/config hash、端到端 latency P95/P99、head state 字段和 DDS compatibility 结果。
- Git 中没有大资源、模型、cache、DDS captures、现场日志。

## 9. Server 改进清单

这些改进是 GA 前 server 必做收口项：

| 项 | 处理 |
| --- | --- |
| frame/timestamp 倒退 | reset 当前连接 tracker/event state，补测试 |
| JPEG 尺寸策略 | 校验 header 与 decode 尺寸一致，或明确以 decode 尺寸为准 |
| `someone_near_center` | 实现真实计算，或协议标记为 reserved |
| 多连接推理 | 真实 backend 串行锁或单 worker，保持每连接状态隔离 |
| metrics sink error | 记录 dropped/error count 到 stderr 或 health/debug |
| CLI 合同测试 | 固化 attention、semantic events、head_motion suppression 行为 |

## 10. 测试矩阵

| Gate | 覆盖 | 通过标准 |
| --- | --- | --- |
| Server regression | 现有 unit/integration、runtime smoke、`val-data` full matrix、metrics、300s soak | 全 pass；P95 <120ms；P99 <200ms；Hz >=9；soak RSS growth <=64MB |
| CLI unit | WebSocket、DDS adapters、target mapper、Botified output、config | 全 pass；stdout/stderr 分离；无运控 API |
| DDS image input | topic/domain、JPEG 校验、jitter/drop、无 publisher、invalid JPEG | >=9Hz；异常可恢复；无无界队列 |
| Head state | `/robot/head_state` required mode、Hz、freshness、stationary/moving/unknown segment | 完整 GA 必须通过；缺失只算 degraded；report 写 `head_state_hz`、`head_state_stale_count`、`head_state_unknown_ratio` |
| DDS gaze output | target schema、rate、lifespan、stale/lost/disabled | 新鲜 `visual_state` 区间内 <=10Hz 且 >=9Hz；断线后 250ms 内 invalid/stale；旧状态不发布有效 target |
| Gaze correctness | 多人大小目标、交叉、遮挡恢复、target switch jitter、`head_uv`/face keypoint/fallback 语义 | PC 和现场都通过；不新增模型能力；jitter 超上限 fail |
| Botified stdout | semantic event -> request frame allowlist | 高频状态不进 stdout；`attention_target_changed` 不进 stdout；escaping 正确；同 `event_id` 不重复；stdout 背压不阻塞 gaze stale |
| PC local E2E | real DDS participant + synthetic `val-data` image/head state publisher -> runtime CLI -> runtime server -> real DDS gaze sink/stdout collector | 7 scene 全量；manifest hash 一致；5 分钟稳定；latency P95/P99 达标；report pass |
| Fault matrix | server down/restart、slow server、DDS down/restart、bad JPEG、Botified pipe close | CLI 不 crash；可恢复；不刷屏；stale watchdog 准时 |
| 真机/板端 DDS compatibility | 真实 camera DDS runtime/network、板端 DDS type/QoS、真实 head state topic/type/Hz/freshness | PC green 不替代；三类 DDS contract 全部兼容才 pass |
| 真机闭环 | live camera DDS、real server、CLI、真实 head/motion consumer 或等价闭环、Botified | 30 分钟稳定；valid tracking 物理指向；invalid/stale 不继续动；restart 无残留动作；恢复后重新 tracking |
| no-motion-SDK audit | Python import/dependency denylist、native bridge `ldd`/`readelf`、DDS topic allowlist | artifact/hash 入 handoff；发现运控 SDK 依赖或 command topic 即 fail |

## 11. 配置

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
broken_pipe = "fail_fast"

[logging]
stderr_level = "info"
jsonl_path = "artifacts/cli/cli_metrics.jsonl"
```

不为每个事件规则开放 CLI 配置。事件阈值仍由 server 管理。

`gaze_target.stale_ms` 和 `service.response_timeout_ms` 是两个不同计时器：前者决定多久内必须让 gaze target 失效，GA 默认 250ms；后者决定 WebSocket 请求多久未返回后关闭连接并重连，GA 建议 1000ms。长时间 E2E 工具可以覆盖更大的 response timeout，但正式 CLI runtime 不应等 30s 才让 gaze 失效。stale watchdog 必须能在 stdout 背压和 one in-flight WebSocket 等待期间继续运行。

## 12. Git 和运行产物策略

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

`val-data/` 整体仍然禁止进入 Git。`val-data/manifest.json` 如果放在 `val-data/` 下也不进 Git；GA handoff artifact 必须包含 manifest sha256 和副本摘要，PC/release report 必须记录 `manifest_sha256`。不要在 `.gitignore` 写 `*.json`，避免误伤 `common/schema/**/*.json` 等小型 schema/sample 文件。

`.gitignore` 必须覆盖通用大资源和本地捕获输出，包括模型权重、ONNX/TensorRT/RKNN engine、MCAP/bag/pcap、视频、现场图片、JSONL metrics、日志、capture/cache 目录；源码中必须跟踪的小 JSON schema/sample 文件不受影响。

## 13. Team Review 结论

产品 review：

- GA 最大边界是 CLI 只发布 DDS gaze target，不直接控制头部。
- PC 本地 DDS E2E 是正式 release regression gate，但不替代真机/板端 DDS compatibility 和真实闭环验收。
- 不发布完整高频 `visual_state` DDS；GA 只做 gaze target 一个高频 DDS 输出。

研发 review：

- 需要新增 CLI package、DDS contracts、PC test tools。
- Server S8 baseline 不重写，只补 frame/timestamp reset、尺寸策略、`someone_near_center`、多连接推理和 metrics error 收口。
- DDS topic/type/QoS 必须先固化，否则 PC 和真机 E2E 不等价。

QA/release review：

- 保留 server S8 gates。
- CLI 必须有 unit、integration、PC local E2E、fault matrix、真机 smoke。
- Release handoff 必须证明 CLI 不链接/不调用运控 SDK，只通过 DDS 输出 gaze target。
- Release handoff 必须包含 `val-data` manifest hash、runtime server/CLI 入口证明、head state 完整 GA 证明、Botified allowlist 证明、端到端 latency P95/P99 和 no-motion-SDK audit artifact/hash。
