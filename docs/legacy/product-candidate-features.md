# Visual Events 产品候选特性

Legacy note: historical reference only. Current active entry points are `docs/ga-development-plan.md` and `docs/identity-overlay-product-development-plan.md`.

日期：2026-06-27

## 1. 背景

本文记录后续候选产品能力，重点是增强 Visual Events 平台层的事件理解能力，让商店门口揽客机器人交互更自然。

这些特性不是当前 GA baseline 承诺。它们应在现有 `YOLOv8n-pose + tracking + rule engine` 基础上演进，优先使用 track、detection metadata、keypoints、head state 和时间窗口规则，不训练新模型，不引入 ReID、人脸识别、长期记忆或数据库。

## 2. 产品目标

当前系统已经能输出基础人物事件和 gaze target。下一步候选能力的目标不是“识别更多标签”，而是让系统更稳定地理解连续视觉变化：

- 不因单帧抖动误触发。
- 不因短遮挡或 track 切换重复打招呼。
- 不在目标快速经过、太远或不稳定时打扰。
- 能向 agent 提供事件原因和上下文，而不是只给一个事件名。
- 多人场景下能更稳定地选择交互目标。

预期效果是提升机器人交互的自然感：更少抢话、更少刷屏、更少盯错人，并且能更像“看懂了门口发生的事”。

## 3. 设计原则

- 规则基于 track 和 detection metadata，不直接把每帧检测结果交给 agent。
- 所有语义事件都必须经过时间窗口确认，禁止单帧直接触发。
- 事件规则仍只在 server 实现；CLI 不复刻事件规则。
- 低频事件进入 Botified；高频 gaze target 仍只通过 DDS 输出。
- 每个新增事件或状态都必须有 evidence，方便回放验证和现场调参。
- 先做少量高价值能力，避免堆事件类型。
- 候选能力必须能通过 `val-data` replay 或等价标注数据验证。

## 4. 可用输入

候选规则主要使用以下输入：

| 输入 | 用途 |
| --- | --- |
| `track_id` | 同一短期目标的状态累积、cooldown、dedupe |
| `bbox_xyxy` | 位置、大小、中心点、面积比例、速度趋势 |
| `bbox_area_ratio` | 靠近、远离、最大目标、可交互目标判断 |
| `center_uv` | 横向通过、靠近画面中心、目标稳定性 |
| `head_uv` / keypoints | gaze target、挥手、头部近似点 |
| `confidence` / `pose_confidence` | 规则置信度和低质量抑制 |
| `age_ms` / `lost_ms` | 稳定出现、短遮挡恢复、离开判断 |
| `velocity_uv_s` | passing by、stopped、target jitter |
| `head_motion.state` | 抑制机器人头动导致的运动误判 |
| attention history | target dwell、target switch、reacquire |
| event history | cooldown、同一人短时重复招呼抑制 |

## 5. 候选能力

### 5.1 事件生命周期

把事件从“立即触发”改成生命周期状态机：

```text
candidate -> confirmed -> active -> cooldown -> ended
```

建议语义：

| 状态 | 含义 |
| --- | --- |
| `candidate` | 有初步证据，但窗口不足，不输出 Botified |
| `confirmed` | 满足触发窗口，输出一次低频事件 |
| `active` | 事件仍持续，用于内部状态和 evidence |
| `cooldown` | 事件刚结束或已输出，禁止重复触发 |
| `ended` | 事件生命周期结束，可清理状态 |

示例：

- `person_approaching_robot` 至少需要连续 800-1500ms 的面积增大和中心趋稳。
- `person_stopped_near_robot` 至少需要 1500-3000ms 的低速停留。
- `person_waving` 需要多帧手腕/手臂运动证据，不能由单帧姿态触发。

### 5.2 场景级状态

除了单个 track 事件，候选增加少量 scene-level 状态，帮助 agent 判断“此刻是否适合交互”。

第一批建议只做：

| 状态 | 说明 | 输出建议 |
| --- | --- | --- |
| `scene_attention_available` | 有稳定、足够近、未冷却的交互目标 | 保留在 `visual_state.scene_context`，必要时产生低频 transition event |
| `scene_no_engage_target` | 有人但不适合打扰，例如太远、快速路过、刚招呼过 | 保留在 `visual_state.scene_context` |
| `scene_target_reacquired` | 短遮挡或短 lost 后恢复同一物理目标 | 作为内部 evidence，也可低频输出用于调试/报告 |

这些 scene-level 状态默认不进入 Botified。只有明确成为 allowlist 低频 transition event，并通过 replay gate 证明不会刷屏后，才允许输出给 agent。

暂缓：

- `scene_person_flow_high`
- `scene_group_approaching`
- `scene_queueing`

这些更依赖场地和数据标注，等基础状态机稳定后再评估。

### 5.3 事件 evidence

每个语义事件应带机器可读 evidence。示例：

```json
{
  "event": "person_approaching_robot",
  "track_id": 12,
  "confidence": 0.82,
  "duration_ms": 1200,
  "evidence": {
    "bbox_area_ratio_start": 0.08,
    "bbox_area_ratio_end": 0.15,
    "area_ratio_delta": 0.07,
    "center_motion_px": 18.0,
    "center_stability_px_p95": 12.0,
    "head_motion_state": "stationary",
    "visible_duration_ms": 1800
  }
}
```

规则：

- Evidence 字段必须来自已有 track/detection metadata 或派生特征。
- Evidence 不应包含身份、人脸识别结果、长期记忆 ID。
- Evidence 用于 agent 决策、回放测试和现场调参，但不要求 Botified 原样展示全部字段。

### 5.4 短遮挡恢复与重复招呼抑制

候选能力应识别短遮挡或短 lost 后的同一物理人恢复，避免 track id 改变导致重复招呼。

可用规则：

- lost 时间小于阈值，例如 500-1500ms。
- 新旧 bbox 中心距离小于阈值。
- bbox 面积比例变化在合理范围内。
- 运动方向连续或接近。
- reacquire 窗口内不重新触发 `person_appeared` 或招呼型事件。

输出要求：

- 可在 evidence 中记录 `reacquired_from_track_id`。
- 如果无法确认同一物理人，不能伪造长期身份，只能进入更保守的 cooldown。

### 5.5 可交互目标判断

候选增加一个内部判断：某个 track 是否适合让 agent 交互。

建议条件：

- bbox 面积达到近区阈值。
- visible duration 达到稳定阈值。
- target dwell 达标，未频繁切换。
- 不是快速 passing by。
- 最近没有对该 track 或 scene/person label 触发过招呼型事件。
- `head_motion.state` 不是 `moving` 导致的误判窗口。

这不是新的机器人控制命令，只是给 event engine 和 agent 提供语义上下文。

## 6. 首批推荐范围

第一阶段建议只做以下候选能力：

1. 事件生命周期字段和内部状态机。
2. 对现有事件补 evidence。
3. `scene_attention_available`。
4. `scene_no_engage_target`。
5. `scene_target_reacquired` 的内部 evidence 和测试 gate。
6. 短遮挡/track 切换后的重复招呼抑制 gate。

不建议第一阶段增加大量新事件。优先把现有 `person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot`、`person_waving` 做得更稳、更可解释。

## 7. 测试与验收

候选能力必须通过 replay 测试，而不是只靠现场观察。

建议新增或扩展标注：

| 标注 | 用途 |
| --- | --- |
| expected event lifecycle timeline | 验证 candidate/confirmed/cooldown/ended 时序 |
| expected attention target timeline | 验证最大且稳定目标选择 |
| expected scene context timeline | 验证 attention_available / no_engage_target |
| expected duplicate greeting suppression | 验证短遮挡恢复不重复招呼 |
| forbidden events | 验证负例不触发 |

验收指标建议：

- 事件首次 confirmed 时间允许误差，例如 <= 3 frames 或 <= 300ms。
- forbidden event 数量必须为 0。
- 同一物理人短遮挡恢复期间不得产生新的招呼型事件序列。
- 每个 semantic event 必须包含 evidence。
- Evidence 中关键数值必须有限且来自当前窗口。
- 低频 Botified 输出仍受 allowlist 和 cooldown 约束。

## 8. 非目标

这些能力不应在本候选阶段引入：

- 不训练新模型。
- 不做人脸识别、身份识别、长期人员记忆。
- 不引入 ReID 模型。
- 不做多摄像头融合。
- 不做数据库或事件治理后台。
- 不让 CLI 判断 passing by、approaching、stopped 或 waving。
- 不把 10Hz `visual_state` 发给 Botified。

## 9. 风险与控制

| 风险 | 控制 |
| --- | --- |
| track 抖动导致误判 | 时间窗口、dwell、cooldown、evidence gate |
| 短遮挡后 track_id 改变 | scene/person label 标注和 reacquire 规则 |
| 多人场景目标频繁切换 | target dwell、allowed switch windows |
| 规则越来越复杂 | 分层实现：feature history、event lifecycle、scene context、cooldown |
| 现场调参不可控 | 所有事件输出 evidence，并用 replay gate 固化 |
| agent 被过多事件打扰 | Botified allowlist、transition-only 输出、全局/minute 上限 |

## 10. Review 结论

本候选方向合理，能明显提升机器人交互自然感。它利用现有模型和 tracking 输出，把“检测到了人”升级为“理解人在门口的连续行为”。产品收益主要来自稳定性、去抖、可解释 evidence、重复招呼抑制和更好的交互时机判断。

设计上没有必要先引入新模型。第一阶段应聚焦现有事件生命周期、scene-level 状态和 evidence 输出；只有当数据证明 pose/keypoints 无法满足目标时，再考虑轻量人脸或额外动作模型。

Review 检查：

- 没有把事件规则下放到 CLI。
- 没有把高频状态输出到 Botified。
- 没有引入训练、ReID、人脸识别、长期身份或数据库。
- 没有把候选能力写成当前 GA baseline 承诺。
- 已覆盖事件生命周期、scene-level 状态、evidence、短遮挡恢复、重复招呼抑制、测试验收和非目标。
