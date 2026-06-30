# Identity Overlay 产品与开发计划

日期：2026-06-30

## 1. 定位

本计划取代 `docs/familiar-unknown-recognition-improvement-plan.md` 作为后续身份相关功能的 source of truth。旧文档只覆盖匿名熟悉人阈值和 evidence 小增强，本计划覆盖完整产品闭环：

- `Identity Overlay`：server 维护当前画面人物的统一身份覆盖层。
- `teach_person` 自动把 active anonymous / familiar unknown 合并成正式 person，client 不再调用 merge。
- 所有非 memory 人物事件都检查事件相关 track 的身份缓存，cache miss 才触发受控 recall。
- 当前 `visual_state` 可以读取人物身份状态。
- `identify-current` API 支持 agent 主动要求识别当前画面中的人。
- 匿名熟悉人判断继续使用低频 memory 侧链，并进入同一个 overlay。

一句话目标：

```text
事件、当前画面、主动查询和示教都复用同一套 server-side 身份逻辑。
```

## 2. 产品目标

机器人在商店、展厅、办公室等私有场景中持续观看当前画面。它需要做到：

- 看到已知人物靠近、停留、挥手时，事件里能带上姓名和背景信息。
- 看到熟悉但未命名的人时，能提示 agent “这是一个经常出现但还没有命名的人”。
- 用户指向一个匿名熟悉人并告诉 agent 其姓名和信息时，server 自动把 anonymous profile 转成正式 person profile。
- 没有事件发生时，CLI 也可以读取最新 `visual_state`，并按需给 agent 一个紧凑的当前视觉摘要，让 agent 知道画面里有哪些人、是否已知、是否匿名熟悉。
- agent 明确要求“看看现在是谁”时，可以触发一次受控身份刷新。

产品上只保留三个用户可理解能力：

```text
被动事件带身份
当前状态带身份
主动识别当前人
```

不要把 anonymous、merge、embedding、bbox 或 raw track id 纠缠给 agent。CLI 只做薄桥接：可以消费 server 技术字段，但对 agent 输出紧凑语义上下文。

## 3. 设计原则

- KISS：一个 Identity Overlay，一个身份匹配路径，一个 public identity payload。
- DRY：人脸匹配、known/anonymous 查询、familiar 判断和 merge 只在 server memory/identity service 做一次。
- YAGNI：不做顾客画像平台、跨摄像头 ReID、管理后台、审计系统、复杂 session 表或长期行为分析。
- 一个功能一种做法：示教走 `teach_person`，当前识别走 `identify-current`，事件身份增强走 Identity Overlay。
- 不测试测试：测试核心行为和合同，不做像素级截图回归、测试工具内部测试或 release audit。
- 非阻塞：identity 是增强上下文，不是 10Hz detection/tracking/gaze 的前置条件。
- 私有环境应用：不把隐私合规平台作为本阶段约束，主要风险是错认、错绑、刷屏和拖慢实时链路。
- `val-data/`、`artifacts/`、模型、runtime DB、cache 不进 Git。

## 4. 当前实现事实

已有基础：

- `processor.py` 已产生 detection/tracking/attention/events 和内部 `MemoryFrameSnapshot`。
- `visual_state.tracks` 当前只包含公开 track 字段，没有 identity，也没有 keypoints。
- `MemoryFrameSnapshot` 已在内存侧保存 `TrackSnapshot`、keypoints、attention、scene_context 和 semantic_events，可用于 resolver / recall，不需要把 keypoints 暴露给 public protocol。
- `FrameCache` 已保存最近帧和 3 帧窗口，可以支撑 request-arrival snapshot、hot buffer 选最佳脸。
- `AppMemoryService` 已有低频 background recognition、known person 查询、anonymous profile、scene 查询和 memory event gate。
- CLI 已能投影 `memory_context`，并且不做人脸匹配、不写 memory DB。
- `merge-anonymous-person` API 已存在，可把 anonymous profile 合并到 person。

当前缺口：

- `teach_person` 命中 active anonymous 时当前返回 `anonymous_merge_required`，需要 client/runner 再调用 merge。这个主路径要废弃。
- `visual_state` 不能表达当前画面中 track 的身份。
- 普通事件没有身份增强，只有独立 memory events。
- 没有 `identify-current` 主动识别 API。
- 匿名熟悉人还缺 `observed_duration_ms` 和同 tick 去重等更可靠规则。
- evidence 还不能稳定展示 identity overlay、event enrichment 和 identify-current 结果。

## 5. 核心产品模型

### 5.1 Identity Overlay

Identity Overlay 是 server 维护的一层短期内存状态：

```text
connection_id + camera + track_id -> identity_context
```

它回答一个问题：

```text
这个当前可见 track 对应的人，server 目前知道什么身份？
```

公开状态只表达 agent 需要的短信息：

```text
known_person       已知人物
familiar_unknown   匿名熟悉人
unknown            当前没有可用身份
pending            正在召回
unavailable        当前无法识别，例如无可用脸、memory disabled、stale
```

内部可以继续保存 active anonymous profile，但公开时只有达到 familiar 条件的 anonymous 才作为 `familiar_unknown` 呈现。未达到 familiar 条件的 anonymous profile 是内部记忆材料，不要让 agent 把它当成稳定人物身份。

### 5.2 事件身份增强

所有“带当前可见 person track 的非 memory 人物事件”发生时都做一次轻量 identity check：

```text
event generated
-> 找事件相关 track
-> 查 Identity Overlay cache
-> cache hit: 事件附带 identity_context
-> cache miss/stale: 启动一次受控 recall
-> recall 成功: 更新 overlay，并尽可能附到当前或后续事件
```

注意：

- “所有事件检查身份”指人物事件会检查 cache，不是 scene/memory/non-person 事件都触发召回。
- 只有当前可见 person track 的 cache miss/stale 才可以触发 recall attempt。
- 第二个事件如果仍是同一 track 且 overlay 新鲜，直接复用，不再召回。
- `person_left` 这类目标已经不可见的事件可以附带已有 cache，但不启动新的重型 recall。

等待策略：

```text
高价值事件：person_approaching_robot、person_stopped_near_robot、person_waving
  可以等待已经存在的 in-flight recall 的剩余极小预算；不为新 recall 阻塞当前 frame

低价值事件：person_appeared、person_passing_by、person_left、attention_target_changed
  cache miss 时最多启动后台 recall，不等待
```

身份缺失不能阻止原事件输出。identity enrichment 是 best-effort 增强，不是事件成立条件。

### 5.3 当前视觉状态

没有事件发生时，CLI 可以通过最新 WebSocket `visual_state` 读取当前画面身份状态。本计划不新增独立 current-state REST endpoint，避免多一套状态查询语义。

产品路径固定为：

```text
server WebSocket visual_state
-> CLI 缓存最新 visual_state.identity_context
-> agent 需要观察当前画面时，CLI 生成紧凑 current visual snapshot
```

server 到 CLI 的 `visual_state` 是技术合同，可以用 `track_id` 引用同一帧 `tracks[]`。CLI 到 agent 的 current visual snapshot 是产品合同，只返回 identity summary、event summary 和 opaque `target_ref`，不暴露 bbox、keypoints、embedding、crop，也不要求 agent 知道 track id。

每条 WebSocket 图像流必须有 server 生成的 opaque `stream_ref`。CLI 缓存它，并在 `identify-current` / `teach_person` 这类 active API 调用中自动带回 server。`stream_ref` 只用于选择正确热帧和隔离同 camera 的多连接，不是人物目标 id，也不需要 agent 推理。

公开 `visual_state.identity_context.tracks[]` 增加身份 overlay，用 `track_id` 引用同一帧 `tracks[]`。不直接修改每个 track 对象的字段形状：

```json
{
  "stream_ref": "stream_front_01",
  "identity_context": {
    "overlay_status": "ready",
    "tracks": [
      {
        "track_id": 7,
        "identity": {
          "status": "known_person",
          "source": "cache",
          "fresh_ms": 900,
          "confidence": 0.87,
          "person": {
            "person_id": "person_001",
            "display_name": "张三",
            "description": "店长",
            "tags": ["staff"]
          }
        }
      }
    ],
    "active_target": {
      "track_id": 7
    }
  }
}
```

匿名熟悉人：

```json
{
  "stream_ref": "stream_front_01",
  "identity_context": {
    "overlay_status": "ready",
    "tracks": [
      {
        "track_id": 9,
        "identity": {
          "status": "familiar_unknown",
          "source": "background_recall",
          "fresh_ms": 1200,
          "confidence": 0.82,
          "anonymous_person": {
            "anonymous_id": "anon_123",
            "seen_count": 8,
            "observed_duration_ms": 16000,
            "familiar_score": 0.82
          }
        }
      }
    ]
  }
}
```

memory disabled 时，`identity_context` 使用同一个固定合同：

```json
{
  "stream_ref": "stream_front_01",
  "identity_context": {
    "overlay_status": "unavailable",
    "reason": "memory_disabled",
    "tracks": []
  }
}
```

读当前状态默认只读 cache，不同步触发重型召回。agent-facing current visual snapshot 由 CLI 从最新 `visual_state.identity_context` 投影，返回 compact identity 摘要和 opaque target ref。agent 想主动刷新时使用 `identify-current`。

### 5.4 主动 identify-current

`identify-current` 是 agent 明确要求 server 识别当前画面人物时使用的 API。

建议 endpoint：

```text
POST /v1/memory/identify-current
```

请求只允许 CLI 自动字段和 agent-facing 语义字段，不允许 `track_id`、`bbox`、`point_uv`：

```json
{
  "camera": "front",
  "stream_ref": "stream_front_01",
  "target": {
    "kind": "person",
    "intent": "identify_current",
    "referent_text": "当前这个人"
  },
  "scope": "active_target",
  "timeout_ms": 500
}
```

MVP 只支持 `scope="active_target"`。读取全部可见人物身份由 CLI 从最新 `visual_state.identity_context.tracks[]` 投影 current visual snapshot 提供；主动刷新所有可见人物会扩大同步召回成本，暂不进入本计划首版。

`stream_ref` 是 CLI 自动填充的 opaque 流引用。server 用它选择正确的 latest frame 和 hot buffer；缺失或未知时不猜测其他连接的画面。

响应：

```json
{
  "ok": true,
  "status": "identified",
  "people": [
    {
      "target_ref": "current:front:active_target",
      "identity_context": {
        "status": "known_person",
        "person": {
          "person_id": "person_001",
          "display_name": "张三"
        },
        "confidence": 0.87
      }
    }
  ],
  "evidence": {
    "source_frame_ref": "front:100:1780000000000",
    "request_snapshot_ref": "memory_frame:front:100:1780000000000"
  }
}
```

`identify-current` 不发 Botified event，不新增 semantic event。它只更新 Identity Overlay 并返回 API response。

`identify-current` 顶层状态枚举：

```text
identified          识别到 known person 或 familiar unknown
unknown             目标存在但没有可公开身份
ambiguous           当前互动目标不明确
unavailable         memory disabled、no usable face 或 embedding unavailable
stale_interaction   frame cache 过期
no_active_frame     没有可用最新画面
timeout             超过 timeout，未写入 overlay
```

除请求格式错误外，以上都作为 HTTP 200 业务状态返回。请求 schema 违法仍使用 4xx。每个 person item 仍使用统一 `identity_context.status=known_person|familiar_unknown|unknown|pending|unavailable`。`identify-current` response 不返回 public `track_id`、bbox 或 keypoints；需要定位调试时只在 evidence/report 中使用 `source_frame_ref`、`request_snapshot_ref` 等引用。

### 5.5 teach_person 自动合并 anonymous

用户主动示教时，server 应原子完成写入：

```text
resolve target
-> select latest frame by stream_ref + camera
-> extract embedding
-> query known person
-> query active anonymous
-> decide outcome
-> write store
-> update overlay
-> return response
```

`teach_person` outcome：

```text
created_person             新建已知人物
updated_existing_person    已知人物同名或同 external ref，更新 metadata
merged_anonymous_person    active anonymous / familiar unknown 被命名并转正
conflict                   像另一个已知人物，但姓名或 external ref 冲突
ambiguous                  没看清用户指的是谁
not_found                  目标不存在或 unsupported
no_usable_face             目标明确但脸不可用
```

匿名熟悉人转正主路径：

```text
用户指向某人并说：这是张三，店长
-> agent 调 teach_person
-> CLI 自动带上当前 stream_ref
-> server 在该 stream_ref 的 hot buffer 中解析目标 track
-> embedding 匹配 active anonymous anon_123
-> server 在同一事务中创建/更新 person profile
-> copy anonymous embeddings/provenance 到 person
-> mark anonymous merged
-> update overlay: track now known_person
-> 返回 merged_anonymous_person
```

示例 response：

```json
{
  "ok": true,
  "outcome": "merged_anonymous_person",
  "person_id": "person_456",
  "merged_anonymous_id": "anon_123",
  "copied_embedding_count": 3,
  "store_delta": {
    "person_profile": "created",
    "anonymous_profile": "merged",
    "embedding_count": 4
  },
  "profile": {
    "display_name": "张三",
    "description": "店长"
  },
  "evidence": {
    "source_frame_ref": "front:100:1780000000000",
    "request_snapshot_ref": "memory_frame:front:100:1780000000000",
    "match": "active_anonymous"
  }
}
```

`merge-anonymous-person` endpoint 保留为 debug / maintenance / backfill API，不作为 agent/CLI 的产品主路径。

`teach_person` 请求必须包含 `camera` 和 opaque `stream_ref`。`target` 仍只表达用户意图，例如 `kind=person`、`referent_text=这位/我/手指的这个人`；不允许 agent 传 `track_id`、bbox、point 或 crop。

### 5.6 匿名熟悉人

匿名熟悉人判断沿用并吸收旧计划：

```text
seen_count >= familiar_seen_count
observed_duration_ms >= familiar_observed_duration_ms
familiar_score >= familiar_threshold
cooldown allow
```

`observed_duration_ms` 是采样累计，不是首末时间差：

```text
delta_ms = max(0, current_frame_timestamp_ms - previous_last_seen_at_ms)
increment_ms = min(delta_ms, query_interval_ms)
observed_duration_ms += increment_ms
```

同一 memory query tick 内，同一 `anonymous_id` 只能更新一次，避免一帧多个 track 把 `seen_count` 加多次。

## 6. 事件系统设计

### 6.1 保持事件类型稳定

不新增身份组合事件。

不要新增：

```text
known_person_approaching
familiar_person_waving
anonymous_merged
identity_refreshed
face_sampled
cache_miss
```

现有非 memory events 继续存在：

```text
person_appeared
person_left
person_passing_by
person_approaching_robot
person_stopped_near_robot
person_waving
attention_target_changed
```

memory semantic event 子集继续只有：

```text
known_person_present
familiar_unknown_present
scene_activated
```

### 6.2 普通事件附带身份

普通人物事件可以附带 `identity_context`：

```json
{
  "event": "person_approaching_robot",
  "track_id": 7,
  "identity_context": {
    "status": "known_person",
    "person": {
      "person_id": "person_001",
      "display_name": "张三",
      "description": "店长"
    },
    "confidence": 0.87,
    "source": "event_recall"
  }
}
```

匿名熟悉人：

```json
{
  "event": "person_waving",
  "track_id": 9,
  "identity_context": {
    "status": "familiar_unknown",
    "anonymous_person": {
      "anonymous_id": "anon_123",
      "seen_count": 8,
      "observed_duration_ms": 16000
    },
    "confidence": 0.82,
    "source": "cache"
  }
}
```

### 6.3 memory_context 和 identity_context 边界

为了兼容现有 memory events：

- `memory_context` 继续用于 `known_person_present`、`familiar_unknown_present`、`scene_activated`。
- `identity_context` 用于当前 track 身份 overlay，以及非 memory person events 的身份增强。
- 两者字段来源必须是同一套 identity/memory service，不允许 CLI 或 EventEngine 自己计算。

后续如果需要收敛字段名，可以再做一次小迁移；本阶段不重写所有既有 memory event contract。

### 6.4 CLI notification gate

CLI 仍只做投影和限流：

- 不做人脸识别。
- 不计算 familiar score。
- 不写 DB。
- 不调用 motion/head control。
- 不新增 DDS topic。

如果同一个 person/anonymous 在短窗口内同时产生普通事件和 memory event，CLI 可以用既有 coalesce/rate-limit 机制减少重复唤醒。推荐策略：

```text
同一 person_id / anonymous_id / track_id
在 coalesce window 内
优先发送高价值普通事件 + identity_context
抑制重复的低价值 memory notification
```

这只是 notification gate 行为，不改变 server 的事件事实输出。

## 7. 技术方案

### 7.1 模块边界

新增或拆分：

```text
src/visual_events_server/memory/identity_overlay.py
```

职责：

- 保存 per-track identity cache。
- TTL / purge。
- in-flight recall 去重。
- 输出 public `identity_context`。
- 根据 track reacquire / alias 尽量继承身份。

`AppMemoryService` 负责：

- 调用 overlay。
- 发起 recall。
- known/anonymous 查询。
- teach 自动 merge。
- identify-current。

`EventEngine` 不做人脸匹配，不访问 DB。它仍只负责产生视觉事件。

`app._attach_memory_events()` 负责把 memory service 产出的 overlay/enriched events 合并进 `visual_state`。

### 7.2 IdentityOverlayEntry

建议内部结构：

```python
IdentityOverlayEntry(
    connection_id: str,
    camera: str,
    track_id: int,
    status: Literal[
        "known_person",
        "familiar_unknown",
        "unknown",
        "pending",
        "unavailable",
    ],
    person_id: str | None,
    anonymous_id: str | None,
    display_name: str | None,
    description: str | None,
    tags: tuple[str, ...],
    confidence: float | None,
    familiar_score: float | None,
    seen_count: int | None,
    observed_duration_ms: int | None,
    memory_match_id: str | None,
    source: Literal[
        "cache",
        "background_recall",
        "event_recall",
        "active_identify",
        "teach",
    ],
    source_frame_ref: str,
    frame_timestamp_ms: int,
    observed_at_ms: int,
    expires_at_ms: int,
)
```

公开字段只输出 agent 需要的短信息。不要输出：

```text
embedding vector
crop image/path
keypoints
raw face landmarks
full DB row
```

### 7.3 Cache 和 in-flight

Cache key：

```text
(connection_id, camera, track_id)
```

原因：

- 不同 WebSocket/session 的 track id 可能重复。
- camera 相同但连接不同，不能混淆身份。

公开 active API 不直接使用内部 `connection_id`。server 在 WebSocket 侧生成 opaque `stream_ref`，并维护：

```text
stream_ref -> connection_id
```

`identify-current`、`teach_person` 和 hot buffer lookup 必须通过 `stream_ref + camera` 找到正确的 frame cache。缺失、未知或过期 `stream_ref` 返回业务失败状态，不回退到 camera-only 最新帧。

TTL：

```text
identity_overlay_ttl_ms = memory.frame_cache_seconds * 1000
```

第一版不新增配置；复用 frame cache TTL。后续如果确实需要，再拆独立配置。

in-flight key：

```text
(connection_id, camera, track_id, source_frame_ref)
```

同一 source frame 上的同一 track 已经在 recall 时，后续事件不再启动第二次。future 完成时必须校验当前 overlay/cache 仍指向同一 `source_frame_ref` 或同一 track generation，避免迟到结果覆盖新目标。高价值事件只能等待已存在 future 的剩余极小预算；低价值事件直接继续。

Track re-acquire：

- 如果 `scene_context.target_reacquired` 或 EventEngine evidence 表明 old track -> new track，overlay 可以复制旧 track 的未过期身份到新 track。
- 这是 cache 继承，不是新识别，不写 DB。

### 7.4 Hot buffer 选最佳脸

Recall 不只使用事件当前帧。它从 `FrameCache.get_snapshot_window(camera, connection_id)` 取最近窗口，或在实现中显式过滤 `CachedFrame.connection_id`。active API 先把 `stream_ref` 解析为 `connection_id`，再取窗口：

```text
当前帧 + 最近若干帧
```

对同一 track 的 crop 候选按简单规则选最佳：

```text
usable face > embedding quality > bbox area > newer frame
```

第一版不要训练质量模型，不做复杂人脸质量评分。可复用 embedding backend 返回的 `quality` 和已有 face detection metadata。

### 7.5 共享 identify helper

从当前 `_query_person()` / `_query_anonymous_person()` 抽出共享 helper：

```text
identify_person_target(cached, target, reason, options)
```

调用方：

- background recognition
- event identity recall
- identify-current
- teach_person

返回统一结果：

```text
known_person
familiar_unknown
anonymous_internal
unknown
no_usable_face
error
```

策略：

- known person 查询优先。
- anonymous 查询次之。
- event recall 和 identify-current 不创建正式 person。
- background recognition 仍负责创建/更新 anonymous profile。
- 只有 background recognition 创建/更新 anonymous profile、`seen_count`、`observed_duration_ms` 和 familiar 判断。
- event recall / identify-current 只读取已有 known/anonymous，并更新短期 overlay，不推进 familiar 计数，不创建新的 anonymous profile。

### 7.6 Event enrichment 实现位置

推荐在 memory service 侧提供：

```text
enrich_visual_state(connection_id, frame, visual_state) -> visual_state
```

内部做：

1. `observe_visual_state()` 更新 frame cache 和启动 background query。
2. drain completed memory events。
3. 对当前带可见 person track 的非 memory 人物事件做 identity check。
4. cache hit 时直接附加 `identity_context`。
5. cache miss 时按事件价值启动后台 recall；只有已存在 in-flight future 才允许短等待。
6. 追加 completed memory events。
7. 在顶层 `visual_state.identity_context.tracks[]` 附加当前 overlay。

这样 `processor.py` 和 `EventEngine` 不需要知道 memory store，也不需要等待 embedding。

### 7.7 identify-current API

新增 API contract：

```text
POST /v1/memory/identify-current
```

请求模型：

```json
{
  "camera": "front",
  "stream_ref": "stream_front_01",
  "target": {
    "kind": "person",
    "intent": "identify_current",
    "referent_text": "当前这个人"
  },
  "scope": "active_target",
  "timeout_ms": 500
}
```

约束：

- `target.kind` 只支持 `person`。
- `scope` 首版只支持 top-level `active_target`，不放在 `target` 内。
- `stream_ref` 必填，由 CLI 从最新 WebSocket `visual_state.stream_ref` 自动带回。
- 禁止 `track_id`、`bbox`、`point_uv`、`source_frame`、`source_frame_ref`、`request_snapshot_ref`。
- `timeout_ms` 有上限，例如 1000ms。
- 缺失 `stream_ref` 是 schema 4xx；未知或过期 `stream_ref` 返回 `no_active_frame`。两者都不回退 camera-only latest frame。
- 没有 fresh frame 返回 `no_active_frame`。
- frame stale 返回 `stale_interaction`。
- 没有 active target 返回 `ambiguous`。
- 没有 usable face 返回 `unavailable`，`reason=no_usable_face`。
- 不发 semantic event。

### 7.8 teach_person 自动 merge

当前 409 分支要替换：

```text
anonymous_match -> anonymous_merge_required
```

改成：

```text
anonymous_match -> merged_anonymous_person
```

实现要求：

- store 层提供一个原子 helper。
- 同一事务内完成 person profile upsert、external_user_ref 检查/写入、新 fresh embedding 写入、copy anonymous embeddings、mark anonymous merged、merge history。
- 失败时不留下半写 person、external link 或 orphan embedding。
- merge 后 overlay 当前 track 立刻变成 `known_person`。
- API response 带 `merged_anonymous_id`、`person_id`、`copied_embedding_count`、`store_delta`、evidence。
- `merge-anonymous-person` endpoint 保留为 maintenance / debug / backfill，不进入产品主路径文档示例。

决策表：

| 查询结果 | profile / external ref 状态 | 行为 |
| --- | --- | --- |
| known person 命中，同名或同 external ref | 无冲突 | `updated_existing_person` |
| known person 命中，但姓名或 external ref 冲突 | 冲突 | `conflict`，不写库 |
| external ref 已绑定其他 person，但当前 face 不匹配该 person | 冲突 | `conflict`，不新建 duplicate |
| external ref 已绑定某个 person，但 known 查询没有命中该 person，即使命中 active anonymous | 不确定是否同一人 | `conflict`，不写库 |
| 没有 known conflict，命中 active anonymous | 高置信且 target resolved/face usable | `merged_anonymous_person` |
| 命中 active anonymous，但 match/margin 不达标 | 不可靠 | `ambiguous` 或 `conflict`，不写库 |
| 无 known、无 anonymous | target resolved/face usable | `created_person` |

这张表是主路径合同。runner/client 不应再把 `anonymous_merge_required` 当作正常分支处理。

### 7.9 SQLite / sqlite-vec

第一版不新增 identity overlay 持久表。原因：

- overlay 是当前视觉状态，不是长期事实。
- 长期身份已经由 person_profiles、anonymous_profiles、embedding 表表达。
- 新表会增加迁移和心智负担。

需要的 store 变更：

- `anonymous_profiles.observed_duration_ms INTEGER NOT NULL DEFAULT 0`
- `create/update/get anonymous_profile` 支持 observed duration。
- 新增原子 `promote_anonymous_to_person(...)` helper。

`promote_anonymous_to_person(...)` 必须在单个 `with self.connection:` 事务内完成 profile、external_user_ref 检查/写入、fresh embedding、copy embeddings/provenance、mark merged、merge history。不能在 service 层串多个各自提交的 store 方法，否则失败时会留下半写 person、embedding、external link 或 merge history。

external_user_ref 不能复用现有覆盖式 upsert 语义作为 auto merge 主路径。helper 必须先检查 `external_user_links`：

- external_user_ref 未绑定：在同一事务内绑定到目标 person。
- external_user_ref 已绑定同一 person：允许更新 metadata。
- external_user_ref 已绑定其他 person：返回 conflict，整个事务回滚，不覆盖旧绑定。

### 7.10 CLI

CLI 改动保持薄：

- 投影 event 的 `identity_context` 到 Botified `visual_context`。
- 保留现有 `memory_context` 投影。
- 缓存最新 server `visual_state.stream_ref` 和 `visual_state.identity_context`，在 agent 请求观察当前画面时生成紧凑 current visual snapshot。
- 不自行读取 `visual_state.identity_context.tracks[]` 生成事件；只做当前状态投影和 event context 投影。
- 不做人脸匹配、不计算 familiar、不写 DB。
- 不新增 motion/head control。

最小函数边界：

```text
build_current_visual_snapshot(latest_visual_state) -> dict
```

输出只包含当前场景摘要、人物 identity summary 和 opaque `target_ref`。`stream_ref` 由 CLI 调 active API 时内部带回 server；不需要 agent 在自然语言里理解或选择它。

建议 Botified `visual_context` 形状：

```json
{
  "event": "person_approaching_robot",
  "target_ref": "event:front:person:current",
  "identity_context": {
    "status": "known_person",
    "person": {
      "display_name": "张三",
      "description": "店长"
    }
  }
}
```

### 7.11 Visual Evidence

`generate_visual_evidence.py`：

- bbox label 可显示短 identity，例如 `id=7 张三` 或 `id=9 familiar`.
- event caption 显示 identity summary。
- HTML raw JSON 保留完整 `identity_context`。

`generate_memory_teaching_evidence.py`：

- 展示 teach auto merge anonymous。
- 展示 identify-current result。
- 展示 event identity enrichment。
- 继续复用已有 renderer，不新增第二套 evidence 工具。

## 8. 开发步骤

1. 文档和 contract 收敛：标记旧 familiar unknown 小计划 deprecated，新增本计划为 source of truth。
2. 定义 public `identity_context` / `stream_ref` schema，补协议/contract 测试。
3. 实现 `memory/identity_overlay.py`：TTL、purge、public projection、in-flight registry、stream_ref 到 connection_id 的 lookup 边界。
4. 给 `MemoryMatchingConfig` / anonymous profile 补 `familiar_observed_duration_ms` 和 `observed_duration_ms` 存储。
5. 抽出共享 `identify_person_target` helper，复用 known/anonymous 查询和 crop/embedding 输入逻辑。
6. background recognition 更新 overlay，即使 memory event 被 cooldown suppress 也要刷新 overlay。
7. 实现 event identity enrichment：所有带当前可见 person track 的非 memory 人物事件都查 overlay，cache miss 时按事件价值触发 recall。
8. 在 public `visual_state.identity_context.tracks[]` 附加 identity overlay，保持 keypoints 不公开。
9. 新增 `POST /v1/memory/identify-current` API 和 disabled memory stub；active API 必须按 `stream_ref + camera` 选择当前帧。
10. 修改 `teach_person`，将 anonymous match 从 409 改成自动 `merged_anonymous_person`，并按 `stream_ref + camera` 解析 hot buffer。
11. 保留 `merge-anonymous-person`，但从产品主路径和 runner helper 主路径中移除。
12. 更新 CLI Botified projection，支持 event `identity_context` 和按需 current visual snapshot。
13. 更新 visual evidence 和 memory teaching evidence。
14. 补核心 tests 和 deterministic evidence demo。
15. 同步更新旧合同和 runner：`docs/memory-teaching-ga-development-plan.md`、`docs/memory-teaching-ga-handoff.md`、API tests、GA runner 中的 explicit merge 主路径必须改为 `teach_person` auto merge；maintenance merge API 只保留单独测试。

## 9. 测试计划

### 9.1 Unit Tests

Identity Overlay：

- TTL 未过期时投影身份，过期后不投影。
- 当前 frame 不含该 track 时 purge。
- `stream_ref` 只能映射到自己的 `connection_id`；同 camera 多连接时 overlay/cache/hot buffer 不串帧。
- 同一 source frame 的同一 track in-flight recall 去重。
- future 迟到时 source_frame_ref / track generation 不匹配则丢弃，不覆盖新目标。
- re-acquire old track -> new track 时可继承未过期 identity。
- public projection 不含 embedding、crop、keypoints。
- top-level `identity_context.overlay_status` 固定为 `ready|unavailable`；memory disabled 固定使用 `overlay_status=unavailable` + `reason=memory_disabled` + `tracks=[]`。
- per-track public status 矩阵固定：`known_person`、`familiar_unknown`、`unknown`、`pending`、`unavailable`。
- cache miss 启动 recall 后，pending 的规则固定：如果当前 track 可见且 recall 已登记，public overlay 必须显示 `pending`；recall 成功、失败、timeout 或 source_frame_ref 失配后必须清除 pending，迟到 future 不能覆盖新身份。

Anonymous familiar：

- `familiar_observed_duration_ms` config 默认值、显式值、负数非法。
- 新旧 DB 都有 `observed_duration_ms DEFAULT 0`。
- 首次 unknown 创建 anonymous 不发 event。
- 再次命中按 `min(delta_ms, query_interval_ms)` 累计。
- `delta_ms <= 0` 不增加 duration。
- 同一 tick 多 track 命中同一 anonymous，只加一次 count/duration。
- count、duration、score、cooldown 任一不达标都不发 `familiar_unknown_present`。

Identify helper：

- known person 优先于 anonymous。
- active anonymous 达 familiar 条件时 public status 是 `familiar_unknown`。
- 非 familiar anonymous 不作为 public familiar identity。
- no usable face 返回 unavailable，不污染 overlay。

Event enrichment：

- known identity cache hit 时普通事件带 `identity_context.person`。
- familiar unknown cache hit 时普通事件带 `identity_context.anonymous_person`。
- cache miss 启动 recall，但 identity 失败不阻止事件输出。
- 低价值事件不等待；后台 recall 成功后，后续 `visual_state.identity_context` 或下一次同 track 事件能读到更新后的 overlay。
- cache miss 启动后台 recall 时，第一帧普通事件不阻塞；第二帧同 track 的 `visual_state.identity_context.tracks[]` 在 recall 完成后出现结果。
- 高价值事件只等待已存在 in-flight future 的剩余极小预算；新 recall 不阻塞当前 frame。
- recall timeout 后清理 pending，后续事件可重试。
- recall 失败不污染 cache。
- cooldown suppress memory event 时 overlay 仍更新。
- event_id、event type、lifecycle_state 不因 enrichment 改变。

identify-current：

- 请求必须带 `stream_ref`，并且只读取该 stream 的 latest frame/hot buffer。
- 缺失 `stream_ref` 是 schema 4xx；未知或过期 `stream_ref` 返回 `no_active_frame`。两者都不回退 camera-only frame，不写 overlay/DB。
- 同 camera 两个 fresh stream 时，`identify-current` 只能识别请求 `stream_ref` 指向的 active target。
- no active frame。
- stale frame。
- active_target 成功。
- no active target 返回 ambiguous。
- unknown / unavailable / timeout 返回固定业务状态。
- unknown / unavailable / timeout 都使用 HTTP 200 业务状态，不写 DB，不发 semantic event。
- active identify 与 event recall 同 source frame 去重；background recognition 完成后刷新 overlay，不强求与 active in-flight registry 做首版强去重。
- timeout 不留下 pending，也不写半截 overlay。
- 不支持 visible_people 主动刷新；读取多人的身份走最新 `visual_state.identity_context.tracks[]`。
- 禁止低层字段。
- 不发 semantic event。

teach_person auto merge：

- 请求必须带 `stream_ref`，target resolver 只看该 stream 的 hot buffer。
- 缺失 `stream_ref` 是 schema 4xx；未知或过期 `stream_ref` 返回业务失败，不写 DB。
- 同 camera 两个 fresh stream 时，示教只能绑定请求 `stream_ref` 中的目标。
- anonymous match 返回 `merged_anonymous_person`，不再返回 409。
- 主路径不再出现 `anonymous_merge_required`，runner/client 不再调用 `/v1/memory/merge-anonymous-person`。
- anonymous 被 mark merged/inactive。
- anonymous embeddings/provenance 复制到 person。
- fresh teach embedding 写入 person。
- merge history 写入。
- overlay 当前 track 变成 known person。
- known person 同名更新 metadata。
- known person 不同名返回 conflict，不写 duplicate。
- external ref 已绑定其他 person 且当前 face 不匹配该 person 时返回 conflict，不新建 duplicate。
- external ref 已绑定某个 person 但 known 查询没有命中该 person，即使命中 active anonymous 也返回 conflict，不把 anonymous 隐式并入该 external ref。
- 非 familiar 但 active anonymous 也可在 teach 显式命名时 auto merge，前提是 target resolved、face usable、anonymous match/margin 达标。
- merge 事务失败时 person/profile/external link/embedding/provenance/merge_history/anonymous status 全部回滚。

CLI：

- Botified `visual_context.identity_context` 被投影且长度受限。
- current visual snapshot 从最新 `visual_state.identity_context` 投影，包含 identity summary 和 opaque `target_ref`。
- CLI 缓存最新 `visual_state.stream_ref`，active API 调用自动带回它；agent-facing 文本不要求用户理解 stream_ref。
- agent-facing 输出不泄露 raw track 字段、bbox、embedding、crop、keypoints。
- same-key gap 可以使用 person_id / anonymous_id alias 减少重复唤醒。

### 9.2 Integration Tests

FastAPI / WebSocket：

- WebSocket `visual_state` 包含 opaque `stream_ref`。
- WebSocket `visual_state.identity_context.tracks[]` 包含 identity overlay。
- 非 memory event 可被 identity enriched。
- `/v1/memory/identify-current` route 正常。
- 多连接同 camera 的 `identify-current` / `teach_person` 不串用 frame cache。
- memory disabled 时 identity_context 固定为 `overlay_status=unavailable` + `reason=memory_disabled` + `tracks=[]`，不崩溃。
- 10Hz stream 不等待慢 embedding。

Memory E2E：

- familiar unknown present。
- event identity enrichment。
- identify-current。
- teach anonymous auto merge。
- merge 后不再输出旧 anonymous familiar event。
- 再次看到同一人输出 known_person_present。

### 9.3 Evidence / Demo

Deterministic machine-readable memory identity report 是 blocker：

```bash
uv run python tools/run_memory_teaching_ga_e2e.py \
  --data-dir val-data \
  --out artifacts/memory-identity-handoff
```

需要扩展 runner 或新增最小 deterministic 场景，证明：

- familiar unknown event 包含 `anonymous_id`、`seen_count`、`observed_duration_ms`。
- 普通事件带 `identity_context`。
- current visual_state 顶层 `identity_context.tracks[]` 带身份。
- identify-current 返回身份。
- teach_person 自动 merge anonymous。
- active API 记录包含请求使用的 `stream_ref`，且没有 camera-only fallback。
- 主路径 `api_response_records` / timeline 不出现 `/v1/memory/merge-anonymous-person`；maintenance merge API 只能出现在单独测试段。

Visual evidence 是 handoff/demo artifact，不作为 CI hard gate。只断言关键字段存在，不做像素级验证：

```bash
uv run python tools/generate_memory_teaching_evidence.py \
  --artifact artifacts/memory-identity-handoff \
  --out artifacts/memory-identity-evidence
```

验收：

- HTML 能看到 identity overlay。
- 图片 bbox 或 caption 能看到已知人姓名 / familiar unknown。
- teach auto merge 有 source frame、request snapshot、merged anonymous id、person id。
- 不要求像素级 golden image。

全量 `val-data` `generate_visual_evidence.py --run-replay` 只作为通用 demo，不作为 identity blocker，因为自然 replay 不一定触发 familiar / teach / identify-current。

## 10. Handoff 标准

开发完成后可以 handoff 的条件：

- 核心 unit/integration tests 通过。
- `teach_person` 已不要求 client 主动 merge anonymous。
- 普通事件可附带 identity_context，且 identity 缺失不阻塞事件。
- 当前 visual_state 可读人物 identity_context。
- identify-current 可主动刷新当前身份。
- `visible_people` 主动刷新不属于本期；读取多人身份走 CLI current visual snapshot。
- anonymous familiar 使用 observed duration 和同 tick 去重。
- CLI 只投影，不做身份逻辑、不写 DB、不控制运控。
- evidence 能人工看到 event identity、current identity、identify-current、teach auto merge。
- public protocol 不包含 keypoints、embedding、crop、图片路径。
- `artifacts/`、`runtime/`、模型、`val-data/` 不进 Git。

不作为 blocker：

- 真机、RK3588、现场、HIL、真实 DDS camera。
- 长时间 soak、P95/P99 latency release gate。
- 像素级截图回归。
- 管理后台、顾客画像、跨摄像头 ReID。
- manifest/oracle/release audit 平台。

## 11. 风险和处理

错绑 anonymous：

- 只在用户显式 `teach_person` 且 target resolved、face usable、anonymous match 高置信时自动 merge。
- known person 冲突返回 conflict，不写库。
- 保留 `correct-identity` 和 maintenance merge/correction API。

事件延迟：

- 高价值事件短等待有严格 timeout。
- 低价值事件不等待。
- identity miss 不阻止事件输出。

召回成本：

- 新 track 不自动 recall。
- 所有非 memory 人物事件只做 cheap cache check。
- recall 只在 cache miss/stale 且事件相关 track 上发生。
- in-flight 去重。

字段膨胀：

- public identity_context 只保留短 profile。
- raw evidence 留在 API response/report/evidence，不进 Botified 顶层。

多套身份逻辑：

- server memory/identity service 是唯一匹配来源。
- EventEngine 不查 DB。
- CLI 不计算身份。

## 12. Team Review 结论

产品 review：

- 范围应定义为“统一身份覆盖层 + 受控身份刷新”。
- `teach_person` 自动合并 anonymous 是正确主路径，显式 merge 只保留维护用途。
- Event 是发生了什么，State 是现在有什么，Active API 是主动刷新，不要混淆。
- 不新增身份组合事件，避免事件类型爆炸。

研发 review：

- Identity Overlay 第一版应是 `AppMemoryService` 内短期内存层，不需要新持久表。
- 复用现有 `FrameCache`、`MemoryFrameSnapshot`、known/anonymous store 和 memory event gate。
- 当前 `teach_person` anonymous 409 分支和新方向冲突，需要替换为自动 merge。
- Event enrichment 应在 memory/app attach 阶段做，不放进 EventEngine。

QA review：

- 验收分两层：核心行为用 pytest，evidence/demo 用 deterministic artifacts。
- 全量 val-data 普通 replay 不是 identity blocker。
- 不测试测试工具，不做像素级验证，不引入 release audit 主线。
- 注意：QA 初稿建议保留显式 merge 主流程，但这与最终产品决策冲突。本计划明确采用 server-side auto merge 作为主路径。
