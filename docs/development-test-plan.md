# Visual Events 开发与测试计划

日期：2026-06-26

后续 GA 开发以 [GA 后续开发计划](ga-development-plan.md) 为准。本文保留早期总体设计背景，并已同步关键边界：机器人 CLI 只发布 DDS gaze target，不直接操纵运控。

当前 GA acceptance/pass/fail authority 是 PC 本地模拟：synthetic DDS image/head-state publishers、runtime server/CLI、DDS gaze subscriber/stdout collector、`val-data` full PC E2E，以及必要轻量稳定性和 latency checks。真机实际运行、真实 robot camera DDS、真实 head-state source、physical head pointing、HIL/real closed loop、现场测试或 owner sign-off 不阻塞 GA；RK3588/board/real robot/field validation 属于 GA 之后的硬件适配/现场验证。只有直接保护运行边界或用户明确要求时才添加 governance/report/audit/gate 工作。

## 1. 开发原则

这些原则是实现约束，不是口号：

- KISS：一个 DDS 图像输入，一个 WebSocket 服务协议，一个 `visual_state` schema，一个 DDS gaze target 输出，一个 Botified 事件出口。
- DRY：schema、几何计算、事件冷却、注视目标选择只实现一次。服务端负责事件生成、rising-edge、cooldown 和同 track 去重；CLI 只按 `event_id` 做 Botified 输出幂等保护，不重新实现事件规则。
- YAGNI：没有明确验收需求前，不加训练流程、人脸识别、ReID、数据库、事件治理后台、多摄像头、多协议。
- 可替换但不抽象过度：只为推理 backend 定一个小接口，服务 RK3588 迁移；其他地方先按 V1 需求直接实现。
- 高频状态和低频事件分离：10Hz 状态用于生成 DDS gaze target 和调试，Botified frame 只承载语义事件。
- 失败时降级：服务断开、帧过期、无人、目标丢失时，CLI 不输出误导性事件，并在 250ms 内发布 invalid/stale gaze target；DDS lifespan 只作为后备失效保护。
- TDD 只覆盖核心功能和高风险集成；不要为测试工具、报告骨架或文档文字继续堆测试。

## 2. Repo 结构

计划结构：

```text
visual-events/
  README.md
  docs/
    product-design.md
    development-test-plan.md
  src/
    visual_events_server/
      api/
      inference/
      tracking/
      events/
      attention/
    visual_events_cli/
      dds_input/
      service_client/
      gaze_target_output/
      botified_output/
  common/
    schema/
      protocol.md
      samples/
```

边界：

- `visual_events_server` 和 `visual_events_cli` 都由 `uv` 管理，开发环境与 release/runtime 环境分离。
- `visual_events_cli` 复用 `/home/galbot/works/image-capture` 的 Unitree DDS JPEG topic/type/校验经验；具体 DDS adapter 必须在本 repo 内完备实现和测试。
- 如果 Unitree DDS runtime 只能通过 C++ SDK 接入，可以在本 repo 内实现一个很小的 native DDS bridge；CLI core 仍保持 Python/`uv` 入口和统一测试。
- `common/schema` 是共享协议事实来源。
- 未来 RK3588 本地化仍运行同一个 `visual-events-server --backend rknn`，机器人 CLI 连接 `ws://127.0.0.1:<port>/v1/stream`；不把 RKNN 推理嵌进 CLI。

## 3. 模块计划

### 3.1 Server

模块：

- `api`: WebSocket 接入、帧解析、连接生命周期。
- `inference`: `InferBackend` 接口和 `UltralyticsPoseBackend`。
- `tracking`: 项目内 ByteTrack-style IoU/TTL tracker baseline，只追踪 person。
- `events`: track history 和 V1 规则。
- `attention`: 最大稳定人物选择和 `target_uv` 计算。

`InferBackend` 最小接口：

```text
infer(frame) -> PoseDetections
```

`PoseDetections` 必须是项目自己的结构，不直接把 Ultralytics result 对象传到 tracking/events。

### 3.2 Robot CLI

模块：

- `dds_input`: 持续订阅 `/camera/image/jpeg`，校验 JPEG，按 10Hz 取最新帧。
- `service_client`: WebSocket 连接、二进制帧发送、`visual_state` 接收、断线重连。
- `gaze_target_output`: stale 检查、target 映射、DDS gaze target 发布、失效 sample 输出。
- `botified_output`: 语义事件去重，写 stdout `<botified>...</botified>`。

CLI 默认行为：

```text
visual-events-cli --server ws://<host>:<port>/v1/stream --camera front
```

stdout 默认只输出 Botified frame。日志、状态和调试信息走 stderr 或文件；显式 `--debug-json-stdout` 只能用于手工调试，不能用于 Botified task。

## 4. 里程碑

### M0 文档与协议

产出：

- 产品设计文档。
- 开发/测试计划。
- `visual_state`、`semantic_event`、WebSocket frame envelope 样例。

验收：

- 产品和技术评审确认一条主线方案。

### M1 Mock 端到端

产出：

- WebSocket mock server。
- Robot CLI mock input 模式，读取本地 JPEG 序列。
- 服务端返回 mock `visual_state`。

验收：

- 10Hz JPEG 序列能跑满 5 分钟。
- 断线后 CLI 自动重连。
- 旧帧被丢弃，不产生无界队列。
- 每条 WebSocket 连接最多一个 in-flight frame，超时后丢弃响应并重连。

### M2 DDS 输入

产出：

- DDS JPEG 持续订阅 adapter。
- 复用 `image-capture` 的 JPEG 字段校验、DDS domain/network 配置思路。

验收：

- 可从 `/camera/image/jpeg` 稳定取 10Hz 最新帧。
- 无 DDS 发布者时 CLI 有清晰错误并保持可恢复。

### M3 推理

产出：

- `YOLOv8n-pose` backend。
- person bbox/keypoints 输出到项目内部结构。
- 640 输入尺寸基准配置。

验收：

- 单帧和视频回放输出稳定 bbox/keypoints。
- 5090D 服务端显存 < 4GB。
- GPU 模式从 server 收到完整 frame 到发出 `visual_state`，P95 < 120ms。

### M4 追踪与注视目标

产出：

- 项目内 ByteTrack-style IoU/TTL tracker baseline。
- `attention.target_track_id` 和 `target_uv`。
- 目标滞回、短暂丢失保持。

验收：

- 单人 1s 内形成稳定 track。
- 多人面积接近时不频繁切换。
- `attention` 10Hz 返回，过期帧不用于控制。
- 10s 多人回放中，除非新目标面积超过当前目标 25% 并持续 0.5s，目标切换次数 <= 2。

### M5 事件规则与 Botified

产出：

- 事件规则：出现、离开、路过、靠近、停留、挥手、注视目标变化。
- Event cooldown 和同 track 去重。
- Botified frame 输出。

验收：

- 高频 `visual_state` 不进入 Botified。
- 同一事件不会刷屏。
- Botified frame 符合 `id/request/expect/urgency/timeout_secs` 约束。
- `head_motion.state=moving` 或 `unknown` 时，不触发路过、靠近、停留这三类运动敏感事件。

### M6 Gaze DDS Target 输出

产出：

- `gaze_target_output` DDS publisher。
- `/visual_events/gaze_target` topic contract、QoS、stale/lifespan 语义。
- `tracking|lost|stale|disabled` 状态输出。

验收：

- CLI 不直接操纵运控，不调用头部速度、位置或 `look_at` API。
- 给定固定 `target_uv` 序列，DDS gaze target payload 坐标、confidence、track id、stale time 正确。
- 目标丢失、server 超时或 frame 过期时，250ms 内发布 invalid/stale sample；DDS lifespan 只作为后备失效保护。
- frame header 标记头部运动或未知时，服务端暂停 `person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot` 这类运动敏感事件。

### M7 RK3588 Spike

产出：

- 在 RK3588 上跑 `yolov8n-pose.rknn` 或 RKNN Model Zoo 等价示例。
- E2E 性能报告，必须包含 JPEG decode、preprocess、infer、postprocess、tracking。

验收：

- 单路 640 pose E2E >= 10Hz，或明确瓶颈与降级方案。
- RKNN 输出能转换为同一份 `PoseDetections`。

## 5. 测试计划

### 5.1 单元测试

单元测试优先覆盖 runtime 核心逻辑和高风险边界。测试工具、报告骨架和纯文档 wording 不需要再做“测试测试”。

覆盖：

- bbox 面积、中心点、头部 fallback 点。
- gaze target 坐标映射、stale/invalid 语义。
- 最大人物选择和滞回。
- track history 时间窗口。
- event rising-edge、cooldown、同 track 去重。
- Botified frame JSON escape 和 id 合法性。

### 5.2 协议测试

覆盖：

- WebSocket binary envelope 解析。
- header 长度非法、JSON 非法、JPEG 非法。
- frame_id 乱序。
- 旧 timestamp 丢弃。
- 服务端慢处理时客户端只保留最新帧。
- 断线重连后事件状态不会立即刷屏。

### 5.3 回放测试

准备固定 JPEG 序列：

- 空画面。
- 单人进入和离开。
- 单人从画面一侧路过但不靠近。
- 单人从远处朝机器人/店门方向靠近。
- 单人停留。
- 单人挥手。
- 两个人交叉和面积接近。
- 头部转动导致画面整体移动。

验收：

- 同一输入序列的 `track_id` 可允许不同，但事件类型、事件数量和触发帧偏差 <= 3 帧。
- `person_appeared` 在稳定出现后 2-5 帧内触发。
- `person_left` 在 lost TTL 后 2 帧内触发。
- `person_passing_by` 在横向通过且未进入近区停留后 5 帧内触发。
- `person_approaching_robot` 在 bbox 面积或高度持续增大并向中心/近区移动 0.5s 后 5 帧内触发。
- `person_stopped_near_robot` 在满足低速停留阈值后 5 帧内触发。
- 同一事件 cooldown 内不重复输出。
- 多人面积接近回放中 10s 内 attention 切换次数 <= 2。

### 5.4 集成测试

覆盖：

- CLI mock JPEG input -> server -> visual_state -> DDS gaze target。
- CLI DDS input -> mock server。
- server 推理 -> tracking -> events。
- semantic_event -> Botified frame stdout。

Botified 集成只测试 stdout frame，不修改 Botified 服务端。

### 5.5 性能测试

服务端 GPU：

- 输入：640 JPEG，10Hz，单路。
- 指标：decode、preprocess、infer、postprocess、tracking、event、total latency。
- 目标：P95 < 120ms，P99 < 200ms，显存 < 4GB。

机器人 CLI：

- DDS 接收频率。
- WebSocket 发送频率。
- 重连时间。
- stale frame 丢弃数量。
- Botified event rate。

RK3588：

- 不只测 NPU inference。
- 必须包含 decode/preprocess/postprocess/tracking。
- 目标：E2E >= 10Hz。

### 5.6 机器人测试

覆盖：

- 单人站在画面前方时注视头部区域。
- 两人并列时注视最大稳定人物。
- 目标短暂遮挡时保持。
- 无人时保持或回中。
- 服务断开时 250ms 内发布 invalid/stale sample，并停止发送有效 gaze target。

## 6. 配置

V1 配置只保留必要项：

```yaml
camera:
  name: front
  dds_topic: /camera/image/jpeg
  hz: 10

service:
  url: ws://127.0.0.1:8765/v1/stream

model:
  name: yolov8n-pose
  image_size: 640
  confidence: 0.35

tracking:
  type: bytetrack
  lost_ttl_ms: 1000

events:
  cooldown_ms: 5000

gaze_target:
  enabled: true
  topic: /visual_events/gaze_target
  stale_ms: 250
```

不为每个规则开放大量配置。阈值先放在一个小配置块里，只有调试证明需要时再暴露。

## 7. Handoff Checklist

开发前必须确认：

- Gaze target DDS topic/type/QoS 是否已被运控/头控 owner 接受。
- 摄像头是否安装在头部，以及能否读取头部 yaw/pitch/角速度。
- Botified 启动 CLI 的命令、工作目录、环境变量和日志采集方式。
- 产品授权路径：AGPL 开源还是 Ultralytics Enterprise。

未确认时的默认策略：

- 运控/头控 owner 未确认：CLI 仍只发布 `/visual_events/gaze_target`，使用 test sink 验收，不发送任何真实头控命令。
- 摄像头是否头载或头部运动状态未知：frame header 标记 `head_motion.state=unknown`，服务端暂停运动敏感事件。
- 授权未确认：只做内部 POC 和性能验证，不进入产品发布。
- 其他模块是否需要完整高频状态未确认：不发布完整 `visual_state` DDS，只发布 gaze target。

首版完成必须满足：

- 同 repo 中有 server 和 robot CLI。
- 高频状态走 WebSocket，不走 Botified。
- 低频事件走 Botified frame。
- 注视 target 由 CLI 发布 DDS；真实动作由运控/头控 owner 本地闭环。
- V1 事件不会刷屏。
- 有回放测试和性能报告。

## 8. 评审结论

产品评审结论：

- 分层正确：高频状态用于控制，低频事件用于 agent。
- MVP 范围应保守，先做 person/pose/track/规则事件/注视最大人物。
- 人脸检测、真实 gaze、多摄像头、长期记忆都不进入 V1。

技术评审结论：

- `YOLOv8n-pose + 项目内 ByteTrack-style IoU/TTL tracker baseline` 是当前兼顾服务端可用性和 RK3588 未来迁移的最好 baseline。
- WebSocket streaming 比 gRPC 更适合 V1。
- 服务端不接 DDS；机器人 CLI 是 DDS 图像输入、DDS gaze target 输出和 Botified 事件输出的集成边界，不是运控边界。
- RK3588 迁移风险必须用 E2E spike 验证，不能只看 NPU inference benchmark。

## 9. 参考资料

- Ultralytics tracking 文档仅作背景参考；S3 不使用 `model.track()`：<https://docs.ultralytics.com/modes/track/>
- Ultralytics Rockchip RKNN 文档：<https://docs.ultralytics.com/integrations/rockchip-rknn/>
- Rockchip RKNN Model Zoo：<https://github.com/airockchip/rknn_model_zoo>
- RKNN Toolkit2：<https://github.com/rockchip-linux/rknn-toolkit2>
- Ultralytics license：<https://www.ultralytics.com/license>
- Botified interactive stdio contract：`/home/galbot/works/botified/docs/ops-manual.md`
- DDS JPEG capture reference：`/home/galbot/works/image-capture`
