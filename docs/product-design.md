# Visual Events 产品设计

日期：2026-06-26

## 1. 产品定位

Visual Events 是一个机器人视觉事件推理服务，不是通用视觉平台。

首个产品场景是商店门口揽客。目标是让机器人以 10Hz 观察眼前画面，稳定识别人、追踪人、估计简单行为，并把结果分成两类输出：

- 高频视觉状态：给机器人本体实时控制使用，例如持续注视画面中占比最大的一个人。
- 低频语义事件：通过 Botified frame 通知 agent，让 agent 根据上下文决定是否回应。

一句话原则：模型做感知，规则做事件，CLI 发布 gaze target，运控本体做实时控制，agent 做语义决策。

## 2. 目标场景

首版面向单机器人、单前向相机、局域网推理服务。

V1 验收场景：

- 有人从机器人或店门前路过。
- 有人朝机器人或店门方向靠近。
- 有人进入画面并停留。
- 有人向机器人挥手。
- 有人站在机器人前方并保持相对稳定。
- 系统持续输出画面中最大且稳定人物的 DDS gaze target，供运控/头控 owner 转头注视。
- agent 只在出现语义事件时被唤起，不接收 10Hz 视觉流。

V1.5 候选场景：

- 有人远离机器人。
- 有人可能看向机器人。
- 输出真实人脸框。

## 3. 非目标

V1 明确不做：

- 不训练模型。
- 不做通用动作识别模型。
- 不做身份识别、人脸识别、长期人员记忆。
- 不做多摄像头融合和 3D 世界建图。
- 不让推理服务直接控制机器人动作。
- 不让 Botified frame 承载 10Hz 高频状态。
- 不做事件治理后台、可视化大屏、云端部署系统。

如果真实人脸框或更准的 gaze 判断成为硬指标，再在 V1.5 增加轻量人脸检测；不进入 V1 baseline。

## 4. 系统边界

同一个 repo 包含服务端和机器人端 CLI。

```text
DDS JPEG @10Hz
  -> visual-events-cli
  -> LAN WebSocket stream
  -> visual-events-server
      -> YOLOv8n-pose
      -> ByteTrack-style IoU/TTL tracker
      -> EventEngine
      -> AttentionTarget
  -> visual_state @10Hz
  -> visual-events-cli
      -> DDS gaze target publisher
      -> Botified frame output
  -> head/motion owner
```

职责分配：

| 模块 | 职责 | 不做 |
| --- | --- | --- |
| `visual-events-cli` | 从 DDS 获取 JPEG；连接服务端；消费 `attention`；发布 DDS gaze target；输出 Botified frame | 不跑大模型；不做 agent 决策；不直接操纵运控 |
| `visual-events-server` | 推理、追踪、事件规则、注视目标选择 | 不接 DDS；不直接控制机器人 |
| Botified agent | 收低频语义事件并决定后续响应 | 不接收 10Hz 状态；不做头部实时闭环 |
| 运控/头控 owner | 订阅 DDS gaze target 并执行本地安全闭环 | 不理解语义事件；不在本 repo 实现 |

## 5. 模型与追踪决策

V1 baseline 使用 `YOLOv8n-pose + 项目内 ByteTrack-style IoU/TTL tracker baseline`。当前 S3 不使用 Ultralytics `model.track()`，也不声明接入上游 ByteTrack package。

选择原因：

- Ultralytics 模型体系成熟，开发入口简单。
- `YOLOv8n-pose` 同时提供 person bbox 和人体关键点，足够支撑挥手、停留、最大人物、头部区域注视等规则。
- Rockchip `rknn_model_zoo` 已包含 `yolov8n-pose`，并给出 RK3588 INT8 推理性能数据，作为未来 RK3588 本地化的现实起点。
- ByteTrack-style IoU/TTL baseline 简单、快、无 ReID 额外模型，适合 10Hz 和 KISS 约束；真实低置信 rescue 依赖 inference conf 覆盖到 tracking `low_conf`。

不选择 YOLO11/YOLO26 pose 作为 V1 baseline：

- 最新模型在 NVIDIA 服务端可能更好，但 RK3588 pose 迁移确定性不如 `yolov8n-pose`。
- Ultralytics 官方 RKNN 文档更偏 detection benchmark，不能直接等价为 pose E2E 可用。

可选 V1.5：

- 如果必须输出真实人脸框，可增加 `RetinaFace_mobile320` 或 YuNet 作为第二模型。
- 这会增加模型链路复杂度，只有在 pose 头部关键点无法满足产品验收时再做。

授权风险：

- Ultralytics 官方许可说明代码、模型、训练/微调模型默认受 AGPL-3.0 约束；闭源或商业产品化前需要确认 Enterprise License。

## 6. 高频与低频输出

### 6.1 高频 `visual_state`

频率：目标 10Hz。

用途：机器人本体实时控制、状态缓存、调试。

传输：服务端通过同一条 WebSocket 连接返回 JSON text message。

示例：

```json
{
  "type": "visual_state",
  "schema_version": 1,
  "camera": "front",
  "frame_id": 1024,
  "frame_timestamp_ms": 1710000000000,
  "server_timestamp_ms": 1710000000082,
  "tracks": [
    {
      "track_id": 7,
      "class": "person",
      "bbox_xyxy": [320, 120, 520, 600],
      "bbox_area_ratio": 0.18,
      "center_uv": [420, 360],
      "head_uv": [421, 205],
      "velocity_uv_s": [35, -4],
      "age_ms": 2400,
      "lost_ms": 0,
      "confidence": 0.86,
      "pose_confidence": 0.72
    }
  ],
  "attention": {
    "target_track_id": 7,
    "target_uv": [421, 205],
    "reason": "largest_stable_person",
    "confidence": 0.82
  },
  "scene_flags": {
    "has_person": true,
    "person_count": 1,
    "largest_person_stable": true,
    "someone_near_center": true
  },
  "semantic_events": []
}
```

高频状态规则：

- 只保留最新状态，允许丢帧。
- CLI 和服务端都使用 keep-latest backpressure，禁止无界排队。
- V1 WebSocket 每个连接只允许一个 in-flight frame：CLI 发送一帧后等待对应 `visual_state` 或 timeout；等待期间 DDS 输入只保留最新帧，被覆盖/丢弃的旧 DDS 输入帧不补发 gaze target。
- 超过 stale 阈值的状态不得用于头部控制。
- 不通过 Botified frame 发送。

### 6.2 低频 `semantic_event`

语义事件随 `visual_state.semantic_events` 返回。服务端负责事件生成、rising-edge、cooldown 和同 track 去重；机器人 CLI 只按 `event_id` 做 Botified 输出幂等保护。

V1 事件：

| 事件 | 触发条件 | 默认冷却 |
| --- | --- | --- |
| `person_appeared` | 新 track 稳定出现至少 2 帧 | 5s |
| `person_left` | track 丢失超过 TTL | 5s |
| `person_passing_by` | track 从画面一侧进入并以横向速度通过，未进入近区停留，且当前帧 `head_motion.state=stationary` | 5s |
| `person_approaching_robot` | bbox 面积或高度持续增大，并向中心/近区移动 0.5s 以上，且当前帧 `head_motion.state=stationary` | 5s |
| `person_stopped_near_robot` | 大 bbox 人物低速停留 1.5s 以上，且当前帧 `head_motion.state=stationary` | 5s |
| `person_waving` | 手腕高于肩部附近且横向方向变化满足阈值 | 5s |
| `attention_target_changed` | 注视目标稳定切换 | 5s |

`person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot` 是运动敏感事件。`head_motion.state` 为 `moving` 或 `unknown` 时不触发，避免机器人自己转头造成误判。

事件优先级建议：

- `person_passing_by`: 轻量招呼，例如简短问候。
- `person_approaching_robot`: 欢迎或引导进店。
- `person_stopped_near_robot`: 进入更正式的导购/问答。

示例：

```json
{
  "type": "semantic_event",
  "event_id": "front:evt_000456",
  "event": "person_waving",
  "camera": "front",
  "track_id": 7,
  "confidence": 0.86,
  "duration_ms": 900,
  "text": "有人在机器人前方挥手"
}
```

Botified frame 示例：

```text
<botified>{"id":"visual:front:evt_000456","urgency":"normal","timeout_secs":8,"request":"视觉事件：有人在机器人前方挥手。track_id=7, confidence=0.86。请根据当前上下文决定是否回应；处理完成后回复 ack。","expect":"ack"}</botified>
```

Botified frame 只表达事实，不直接要求机器人执行动作。

## 7. 注视最大人物

注视能力是本体实时 reflex，不走 agent 决策闭环。

### 7.1 目标选择

服务端为每帧选择一个 `attention.target_track_id`：

```text
score = bbox_area_ratio * confidence * stability_score
```

切换规则：

- 已有目标仍存在时保持目标。
- 新目标面积至少大于当前目标 25%，并持续 0.5s，才允许切换。
- 当前目标短暂丢失时保持 0.5-0.8s。
- 当前目标超时丢失后，选择新的最大稳定人物。

### 7.2 注视点

注视点优先级：

1. 人脸中心，如果 V1.5 接入人脸检测。
2. pose 头部关键点中心，例如 nose/eyes/ears 的可见点均值。
3. bbox fallback：`x = bbox_center_x`，`y = bbox_top + 0.28 * bbox_height`。

### 7.3 头部控制

机器人 CLI 不把 `attention.target_uv` 转成头部速度、位置或 `look_at` 命令。CLI 只把 `attention.target_uv` 发布成 DDS gaze target，真实头部控制由运控/头控 owner 订阅后执行。

若下游运控有相机内参，可在其安全边界内使用：

```text
yaw_delta = atan((target_x - cx) / fx)
pitch_delta = -atan((target_y - cy) / fy)
```

若暂时没有内参，下游可使用归一化误差：

```text
ex = target_x / image_width - 0.5
ey = target_y / image_height - 0.5
```

这些控制律不在 `visual-events-cli` 中实现。

CLI 发布的 DDS gaze target 必须包含：

- `valid`：目标是否有效。
- `state`：`tracking`、`lost`、`stale` 或 `disabled`。
- `target_uv`：图像像素坐标。
- `target_track_id`、`confidence`、`reason`。
- `frame_id`、`frame_timestamp_ms`、`publish_timestamp_ms`。
- `stale_after_ms`：下游必须尊重的过期时间。

下游运控/头控 owner 执行动作时需要在其独立验收中证明：

- deadband：小误差不动。
- low-pass filter：平滑 `target_uv`。
- velocity limit：限制头部速度。
- acceleration limit：避免抽动。
- stale frame check：旧帧丢弃。
- target hysteresis：防止多人场景频繁切换。

`visual-events-cli` 禁止发布 `yaw_velocity`、`pitch_velocity`、`head_position`、`motor_command` 等直接运控命令字段。

## 8. 协议决策

V1 只使用 WebSocket streaming。

客户端到服务端使用一个二进制 WebSocket message 表达一帧：

```text
uint32_be header_len
header_json_utf8
jpeg_bytes
```

header 示例：

```json
{
  "type": "frame",
  "schema_version": 1,
  "camera": "front",
  "frame_id": 1024,
  "timestamp_ms": 1710000000000,
  "encoding": "jpeg",
  "width": 1280,
  "height": 720,
  "head_motion": {
    "state": "stationary",
    "yaw_vel_rad_s": 0.0,
    "pitch_vel_rad_s": 0.0
  }
}
```

服务端到客户端使用 JSON text message 返回 `visual_state`。

详细字段、坐标、错误、backpressure 和断线语义见 [protocol.md](../common/schema/protocol.md)。

不使用 gRPC，不让服务端接 DDS，不通过 Botified 发送高频状态。事件语义由 server replay 逐帧验证；CLI PC E2E 验证 DDS/CLI/server/Botified/gaze 链路和不该出现的事件、重复、顺序、stdout 合同错误。GA runtime 只发布一个高频 DDS gaze target，不发布完整 `visual_state` DDS：

```text
/visual_events/gaze_target  best_effort, keep_last=1, lifespan=250ms
```

## 9. 主要风险与处理

| 风险 | 处理 |
| --- | --- |
| RK3588 pose 端到端性能不足 | 首版就保留 backend 边界；单独做 RK3588 spike，实测 decode/preprocess/infer/postprocess/tracking |
| 头部转动污染图像运动 | V1 frame header 包含可选 `head_motion`；CLI 从头部状态 DDS 生成该字段；服务端在头部运动或状态未知时暂停运动敏感规则 |
| Botified 被事件刷屏 | 事件 rising-edge、cooldown、同 track 去重；高频状态永不进入 Botified |
| CLI 越过安全边界直接控头 | CLI 只发布 DDS gaze target；release audit 属于 post-GA validation，必须证明不链接、不调用运控 SDK |
| 人脸/看向机器人判断不准 | V1 不输出看向机器人事件；必要时 V1.5 加人脸模型和弱 gaze 规则 |
| 授权风险 | 产品化前确认 Ultralytics AGPL/Enterprise 授权 |

## 10. 验收标准

性能：

- 输入 DDS JPEG：10Hz。
- 输出 `visual_state`：稳定 >= 9Hz，连续 5 分钟不断线。
- 服务端 GPU 模式延迟：从 server 收到完整 frame 到发出 `visual_state`，P95 < 120ms，P99 < 200ms。
- 服务端显存：目标 < 4GB。
- RK3588 预研目标（post-GA / non-blocking）：单路 640 pose E2E >= 10Hz。

感知：

- 单人进入画面后 1s 内形成稳定 track。
- 单人短暂漏检时 track 不立即跳变。
- 单人横向路过且未停留时，触发 `person_passing_by`，不触发 `person_stopped_near_robot`。
- 单人从远处朝机器人/店门靠近时，触发 `person_approaching_robot`。
- `head_motion.state=moving` 或 `unknown` 时，不触发 `person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot`。
- 多人场景最大人物目标在 10s 回放内切换次数 <= 2，除非最大 bbox 面积变化超过 25% 并持续 0.5s。
- 挥手事件宁可保守少报，不频繁误报。

Botified：

- 只有低频语义事件输出 Botified frame。
- 同类同 track 事件默认 5s cooldown。
- agent 只决定后续响应，不承担实时注视闭环。

注视：

- CLI 只发布 DDS gaze target，不直接操纵运控。
- 每个新鲜 `visual_state` 派生一条 valid 或 invalid gaze target，目标 >=9Hz；CLI 不用重复旧 target 凑频率。server 断线或超时时，250ms 内发布 invalid/stale sample，之后不再发布过期有效目标。
- 多人场景默认输出画面中最大稳定人物。
- 目标短暂丢失时发布 `valid=false`；DDS lifespan 是下游失效的后备保护，不允许输出过期有效目标。
- 真实头部动作验收属于 post-GA 硬件/现场验证；GA 阶段本 repo 只验收 PC 模拟下 DDS target 正确性、时效性和失效语义，不能由 PC evidence 外推为真机/RK/field/release audit 通过。

## 11. 参考资料

- Ultralytics tracking 文档仅作背景参考；S3 不使用 `model.track()`：<https://docs.ultralytics.com/modes/track/>
- Ultralytics Rockchip RKNN 文档：<https://docs.ultralytics.com/integrations/rockchip-rknn/>
- Rockchip RKNN Model Zoo：<https://github.com/airockchip/rknn_model_zoo>
- RKNN Toolkit2：<https://github.com/rockchip-linux/rknn-toolkit2>
- Ultralytics license：<https://www.ultralytics.com/license>
- Botified interactive stdio contract：`/home/galbot/works/botified/docs/ops-manual.md`
- DDS JPEG capture reference：`/home/galbot/works/image-capture`
