# Familiar Unknown Recognition Improvement Plan

日期：2026-06-30

## 1. 目标

把当前 `familiar_unknown_present` 从“同一 anonymous profile 命中次数达到阈值”升级为更可靠的匿名熟悉人识别能力：服务端在后台低频采样人脸/人物记忆，累计匿名人物的有效观察时长，只有同一未命名人物多次低频采样且累计观察足够时，才向 agent 发出一次低频 `familiar_unknown_present`。

这个能力面向私有环境中的机器人长期运行：机器人不用知道这个人的名字，也不自动打招呼；它只把“这个人是熟悉但未命名的匿名人”作为事实通知 agent，由 agent 决定是否响应。

同时做一个很小的 visual evidence 增强：让现有图片/HTML 能看到匿名熟悉人事件的 `anonymous_id`、`seen_count`、`observed_duration_ms`、匹配分数和触发帧，方便人工 demo 和排查。不要把它扩成新的报告系统或产品 UI。

## 2. 原则

- KISS：继续使用一条 memory 侧链、一个 SQLite/sqlite-vec 存储、一个 `familiar_unknown_present` 事件。
- DRY：匿名熟悉人判断只在 server memory service 做一次；CLI 不复刻规则，不写 DB，不做人脸匹配。
- YAGNI：不做顾客画像平台、运营后台、标注系统、复杂 session/reID、长期行为分析或新事件风暴。
- 一个功能一种做法：后台 memory 采样产生 memory semantic event；CLI 只做 allowlist、rate limit 和 Botified frame 投影。
- 不测试测试：pytest 只覆盖核心规则、存储、事件投影和小范围 evidence 字段；demo 通过确定性 artifact 人眼检查。
- PC 本地验证即可；真机、RK3588、现场和长时间 soak 不作为本计划 blocker。
- `val-data/`、`artifacts/`、`runtime/`、模型、cache 不进入 Git。

## 3. 当前状态

已有能力：

- server 已有后台 memory query tick，默认 `query_interval_ms=1000`。
- person 识别路径已经是“先查 known person，再查 anonymous person”。
- 首次未知人会静默创建 anonymous profile 和 anonymous embedding。
- anonymous 再次命中后会更新 `seen_count` 和 `familiar_score`。
- 达到阈值后已经可以输出 `familiar_unknown_present`。
- CLI 已允许 `familiar_unknown_present`，并通过现有 Botified notification gate 投影。
- `generate_visual_evidence.py` 能在图片底部显示 semantic events。

主要缺口：

- 仅靠 `seen_count` 容易把短时间内多次采样误认为匿名熟悉人。
- anonymous profile 没有累计“有效观察时长”。
- `familiar_unknown_present` 的 `memory_context` 缺少能解释匿名熟悉人判断的时长字段。
- visual evidence 对匿名熟悉人事件展示不够直观。

## 4. 产品范围

### 4.1 匿名熟悉人识别

匿名熟悉人识别继续由 server 后台 memory 采样产生，不绑定普通视觉事件。

流程：

1. 每个 camera 按 `memory.query_interval_ms` 触发一次 memory query。
2. 只扫描 current recognition-eligible person track：`class_name == "person"`、`lost_ms == 0`、`hits > 0`、target quality usable，且 person embedding path 能拿到 usable face；继续遵守现有上限 `_MAX_PERSON_QUERY_TRACKS`。
3. 对每个目标先查 known person；命中则走 `known_person_present`。
4. known person 未命中时查 anonymous person。
5. anonymous 未命中时静默创建 anonymous profile，不发事件。
6. anonymous 命中时更新 `seen_count`、`observed_duration_ms`、`familiar_score`、`last_seen_at_ms`。
7. 同时满足 `seen_count`、`observed_duration_ms`、`familiar_score` 和 cooldown 后，发出既有 `familiar_unknown_present`。

普通 `passing_by`、`approaching`、`stopped` 等 semantic events 不触发采样。短暂路过主要由 usable face、后台采样频率和 observed duration 门槛过滤；如果当前 snapshot 已有可靠的 fast passing / no-engage 信号，可以作为 skip reason 复用，但不为本计划新增第二套 EventEngine 或复杂稳定性判定。

触发条件：

```text
seen_count >= familiar_seen_count
observed_duration_ms >= familiar_observed_duration_ms
familiar_score >= familiar_threshold
event gate allow(camera, familiar_unknown_present, anonymous_id)
```

推荐默认：

- `familiar_seen_count = 3`：保持现有含义。
- `familiar_observed_duration_ms = 5000`：至少累计约 5 秒有效观察。单元测试和本地 smoke 可以显式调小。
- `familiar_threshold = 0.78`：保持现有含义。

### 4.2 observed duration 规则

`observed_duration_ms` 表示对同一 anonymous profile 的有效采样累计，不是 `last_seen_at_ms - first_seen_at_ms`。

更新规则：

```text
delta_ms = max(0, current_frame_timestamp_ms - previous_last_seen_at_ms)
increment_ms = min(delta_ms, query_interval_ms)
observed_duration_ms = previous_observed_duration_ms + increment_ms
```

这样做的原因：

- 用户离开很久后再次出现，只给当前采样贡献一个 query interval，不把离线间隔算成停留时长。
- 同一帧或时间倒退不会增加时长。
- 不需要新增 session 表、visit 表或复杂 reID 状态。

同一个 memory query tick 内，同一 `anonymous_id` 只能更新一次。实现时在当前 tick 维护 `updated_anonymous_ids`；如果多个 track 在同一帧匹配到同一 anonymous profile，只按一次有效采样更新 `seen_count` 和 `observed_duration_ms`。

首次创建 anonymous profile 时：

- `seen_count = 1`
- `observed_duration_ms = 0`
- `familiar_score = 0.0`
- 不发事件。

### 4.3 事件通知

不新增 memory 事件类型。memory semantic event 子集继续只使用：

- `known_person_present`
- `scene_activated`
- `familiar_unknown_present`

`familiar_unknown_present` 继续走唯一事件路径：

```text
server background memory query
-> completed memory event queue
-> visual_state.semantic_events
-> CLI Botified allowlist/rate limit
-> Botified frame
```

事件 `memory_context.anonymous_person` 增加：

```json
{
  "anonymous_id": "anon_000001",
  "seen_count": 6,
  "observed_duration_ms": 5200,
  "familiar_score": 0.83,
  "last_seen_at_ms": 1780000000000
}
```

`duration_ms` 仍表示事件生命周期字段，不用它承载累计观察时长；累计观察时长只放在 `memory_context.anonymous_person.observed_duration_ms`。

CLI 只把 `observed_duration_ms` 作为 compact `memory_context` 字段投影，不新增 Botified 顶层字段，不新增 DDS topic，不新增任何运动、头控或运控调用。

### 4.4 Visual Evidence

只做小增强：

- `generate_visual_evidence.py` / shared helper 展示匿名熟悉人事件时，摘要包含 `anonymous_id`、`seen_count`、`observed_duration_ms`、`familiar_score`、`match_score`。
- HTML raw JSON 继续保留完整 event，图片底部只显示短摘要，避免遮挡 bbox。
- 如果 memory teaching evidence artifact 中已有 familiar/merge 样例，renderer 在对应 item 和 HTML key refs 中展示 `anonymous_id`、`seen_count`、`observed_duration_ms`、`familiar_score`；如果 source artifact 不含该样例，显示 not present，不作为失败。

不做：

- 不新建匿名熟悉人 evidence 命令。
- 不在 renderer 里重新做人脸识别或匿名熟悉人判断。
- 不把 keypoints 加入 public `visual_state.tracks`。
- 不把 pose skeleton 强行加入通用 `generate_visual_evidence.py`。pose 指向证据继续属于 memory/teaching evidence 的 `pose_visual_evidence`。

## 5. 非目标

- 不把每个 `person_appeared`、`passing_by`、`approaching`、`stopped` 都触发一次人脸识别事件。
- 不新增 `face_sampled`、`anonymous_seen`、`familiar_score_changed` 等中间事件。
- 不自动把 anonymous profile 命名为正式 person。
- 不用全身 appearance 代替人脸 identity 路径。
- 不做跨摄像头 reID、跨门店同步或复杂 visit/session 模型。
- 不做 profile 管理 UI、纠错 UI、顾客画像、报表系统。
- 不让 CLI 计算匿名熟悉人分数、做人脸匹配或写 memory DB。
- 不让 CLI 新增运动、头控或运控调用；高频 gaze 仍只走既有 DDS gaze 输出路径。
- 不把 visual evidence 变成发布审计层或质量打分系统。

## 6. 技术方案

### 6.1 配置

在 `MemoryMatchingConfig` 增加：

```python
familiar_observed_duration_ms: int = 5000
```

解析 `[memory.matching].familiar_observed_duration_ms`，要求非负整数。`0` 的语义是关闭 duration 门槛，仅用于测试或显式兼容；正式 PC 配置应使用正数。为避免“配置矩阵”，只新增这一个阈值；采样间隔继续复用 `memory.query_interval_ms`。

`AppMemoryService` 构造参数同步增加该字段。

### 6.2 SQLite Store

`anonymous_profiles` 增加一个 canonical 字段：

```sql
observed_duration_ms INTEGER NOT NULL DEFAULT 0
```

更新：

- `create_anonymous_profile(... observed_duration_ms=0)`
- `update_anonymous_profile(... observed_duration_ms=...)`
- `get_active_anonymous_profile()` 返回 `observed_duration_ms`

使用现有 `_ensure_column` 添加字段即可，不引入 schema version 框架，不维护多套旧结构。

### 6.3 Memory Service

保持现有后台 query 结构：

- 不从普通 semantic events 触发匿名熟悉人识别。
- 不在主 10Hz 处理链路同步等待 embedding。
- 不改变 known person 优先级。
- 不改变 anonymous merge 后 suppress familiar event 的行为。

更新 `_query_anonymous_person`：

1. anonymous 未命中：创建 profile，`observed_duration_ms=0`，写 embedding，不发事件。
2. anonymous 命中：
   - 读取 profile 的 `last_seen_at_ms` 和 `observed_duration_ms`。
   - 按采样累计规则更新 duration。
   - 更新 `seen_count`、`last_seen_at_ms`、`familiar_score`。
   - 同时检查 count、duration、score 和 cooldown。
3. 当前 memory query tick 内按 `anonymous_id` 去重，避免同一帧多个 track 命中同一个 anonymous 时多次增加 `seen_count`。
4. 触发后仍只返回 `build_familiar_unknown_event(...)`。

### 6.4 Event 和 CLI

`build_familiar_unknown_event`：

- 在 `memory_context.anonymous_person` 中加入 `observed_duration_ms`。
- 继续保留 `anonymous_id`、`seen_count`、`familiar_score`、`last_seen_at_ms`。
- 不新增 event evidence 中间字段，除非已有 `memory_match_id` / match score 不足以定位问题。

CLI：

- `_project_memory_context` 的 anonymous white list 增加 `observed_duration_ms`。
- allowlist、priority、same-key gap、global rate limit 不变。

### 6.5 Evidence

`tools/visual_evidence_helpers.py`：

- `_memory_event_summary()` 对 anonymous memory context 输出 `anon=<id> seen=<n> observed_ms=<ms> score=<score>` 这类短文案。
- `match_score` 继续来自 `event.evidence.match_score`；`anonymous_id`、`seen_count`、`observed_duration_ms`、`familiar_score` 来自 `event.memory_context.anonymous_person`。
- 继续从 public `semantic_events` 读取，不读取内部 memory snapshot。

memory teaching evidence：

- 如果 source artifact 已包含 familiar/merge case，在 supporting familiar/merge item 和 `visual_evidence_items_html()` 的 key refs 中展示 `anonymous_id`、`seen_count`、`observed_duration_ms`、`familiar_score`。
- 不为了匿名熟悉人单独新建 artifact schema；复用已有 report/event/visual evidence 字段。

## 7. 开发步骤

1. 更新 config 和 service 构造参数，新增 `familiar_observed_duration_ms`。
2. 更新 `anonymous_profiles` schema 和 store create/update/get 方法。
3. 更新 anonymous 命中时的 duration 累计、同 tick `anonymous_id` 去重和熟悉人触发条件。
4. 更新 `familiar_unknown_present` 的 `memory_context`。
5. 更新 CLI compact memory context 投影。
6. 更新 visual evidence memory event 摘要。
7. 补核心单元测试和一个小范围集成式 memory service 测试。
8. 生成一个确定包含 `familiar_unknown_present` 的最小 evidence artifact，人工抽查匿名熟悉人事件展示；全量 `val-data` visual evidence 作为额外 demo，不作为核心 blocker。

## 8. 测试计划

### 8.1 Unit / Integration Tests

只补核心测试：

- config：默认值、显式配置、非法负数。
- store：新库 create/update/get round trip；已有库通过 `_ensure_column` 补 `observed_duration_ms DEFAULT 0`。
- event：`familiar_unknown_present` 包含 `observed_duration_ms`。
- service：
  - `seen_count` 达标但 `observed_duration_ms` 不达标，不发事件。
  - `observed_duration_ms` 达标但 `seen_count` 不达标，不发事件。
  - count、duration、score 都达标，发一次 `familiar_unknown_present`。
  - `delta_ms <= 0` 不增加 duration；长 gap 只按 `query_interval_ms` 上限增加 duration。
  - 同一 tick 内多个 track 命中同一 `anonymous_id`，只增加一次 `seen_count` 和 duration。
  - 同一 anonymous 已达标后，cooldown 内连续 query 不重复产生 `familiar_unknown_present`。
  - merge 后同一 anonymous 不再发 familiar event，known person 路径不退化。
- CLI：Botified `visual_context.memory_context.anonymous_person` 透出 `observed_duration_ms`。
- evidence：包含 familiar event 的 wrapped `visual_state` 能在 HTML/图片摘要中看到 observed duration。
- contract：public `visual_state.tracks` 仍不包含 keypoints；keypoints 只在 memory snapshot side channel / pose evidence 中使用。

不新增像素级 golden image 测试，不测试 HTML 美术细节，不为测试工具自身建立复杂测试。

### 8.2 Local Demo Evidence

核心熟悉人 evidence 必须是确定性的，不依赖普通 PC GA server 偶然出现 memory event。推荐两种方式选一种：

- 使用 memory-enabled fake/local runner，显式构造或 replay 会触发 `familiar_unknown_present` 的样例，再调用现有 memory teaching evidence renderer。
- 使用一个最小 wrapped `visual_state` fixture，明确包含 `familiar_unknown_present`，离线调用 `generate_visual_evidence.py` 验证 HTML/图片摘要展示字段。

memory/teaching 熟悉人样例如果由 runner 产出，继续用现有命令生成：

```bash
uv run python tools/generate_memory_teaching_evidence.py \
  --artifact artifacts/memory-teaching-ga-local-smoke \
  --out artifacts/memory-teaching-evidence
```

验收点：

- artifact 中确定存在 `familiar_unknown_present`，不是“如出现”。
- 图片底部和 HTML caption 能看到 `anonymous_id`、`seen_count`、`observed_duration_ms`。
- source artifact 不含 familiar/merge case 时，页面显示 not present，但不能被当成熟悉人 evidence 通过。

全量 `val-data` visual evidence 仍然有价值，但它只证明通用 visual evidence pipeline 可用，不证明匿名熟悉人判断。需要人工 demo 全量场景时，可以启动 PC server：

```bash
uv run --extra inference visual-events-server \
  --config configs/pc-ga-server.toml \
  --port 8765
```

生成全量 `val-data` visual evidence：

```bash
uv run --extra inference python tools/generate_visual_evidence.py \
  --run-replay \
  --server ws://127.0.0.1:8765/v1/stream \
  --data-dir val-data \
  --out artifacts/familiar-unknown-visual-evidence \
  --camera front \
  --fps 10 \
  --head-motion stationary \
  --response-timeout-ms 10000 \
  --no-realtime
```

验收点：

- 命令 exit 0。
- `artifacts/familiar-unknown-visual-evidence/index.html` 存在。
- `summary.json` 中 `errors == 0` 且 `frames_ok == frames_total`。
- 没有事件的帧仍显示 `events=none`。

这些 evidence 用于人工 demo，不替代 pytest。

## 9. Handoff 标准

开发完成后必须满足：

- 核心 pytest subset 通过。
- `familiar_unknown_present` 仍走同一 memory semantic event 路径，没有新增事件通道。
- CLI 只投影新增 `observed_duration_ms` 字段，不计算匿名熟悉人规则。
- CLI 不新增运动、头控或运控调用。
- anonymous profile 新增字段不引入第二套 schema 管理。
- public `visual_state.tracks` 仍不包含 keypoints。
- 一个确定包含 `familiar_unknown_present` 的最小 evidence artifact 能生成；全量 `val-data` visual evidence 是可选 demo，不是核心 blocker。
- 不会把 `artifacts/`、模型、runtime DB 或 `val-data/` 加入 Git。

可明确不声明：

- 不声明真机通过。
- 不声明 RK3588 memory backend 通过。
- 不声明现场长期运行效果。
- 不声明复杂顾客画像、真实身份推断或跨摄像头 reID。

## 10. Team Review 结论

产品 review 结论：

- 匿名熟悉人识别应该是后台 memory 侧链，不应该从普通事件同步派生。
- `familiar_unknown_present` 是唯一需要通知前台的事件；采样、分数变化、merge/correction 结果不进 Botified 事件流。
- memory event 子集只有 3 类，不影响既有 `person_waving`、`person_stopped_near_robot` 等非 memory 事件。
- visual evidence 只展示已有事实，不重新计算算法结论。

研发 review 结论：

- `observed_duration_ms` 必须是采样累计，不能用首末时间差。
- 同一 memory query tick 内必须按 `anonymous_id` 去重，避免一帧多个 track 把 `seen_count` 加多次。
- SQLite 只加一个字段，默认 `0`，使用现有 `_ensure_column`。
- 事件和 CLI 增加字段是向前兼容的轻量合同变更。
- keypoints 继续只留在 internal memory snapshot / pose evidence，不进入 public `visual_state`。

QA review 结论：

- pytest 聚焦核心逻辑；确定性 familiar evidence artifact 用于人眼检查。
- 全量 `val-data` visual evidence 只证明通用 evidence pipeline，不作为匿名熟悉人核心 blocker。
- 真机、RK、现场、长 soak、像素级截图测试、发布审计都不作为本计划 blocker。
- handoff 只需要记录测试结果、evidence 路径和不声明的范围边界。
