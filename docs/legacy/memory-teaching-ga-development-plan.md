# Archived Stable Memory/Teaching Reference

日期：2026-06-29

> Archived / legacy reference：本文不是当前开发入口，不是 active acceptance source，不是 handoff source，也不是 QA checklist。当前唯一 active product/development source of truth 是 `docs/identity-overlay-product-development-plan.md`，其中的 `Memory/Teaching Core Contract / Compatibility Gate` 已承接本文仍有效的 blocker 级合同。本文保留历史背景、术语和参考细节；如有冲突，一律以 identity plan 为准。
>
> Archive reading rule：正文中遗留的 “must / 必须 / gate / 验收 / checklist / handoff” 表述均按历史记录阅读，不得作为当前 active gate 使用。

## 1. Legacy 产品目标记录（非当前入口）

历史阶段目标曾是把“记忆和示教”作为稳定可发布能力交付：用户通过 agent 明确示教当前画面中的人或整图场景，服务端保存记忆；后续 replay 或在线流再次看到同一人或已教学整图场景时，服务端输出低频 memory semantic event，CLI 只把 memory 事件投影成 Botified frame。`familiar_unknown_present` 曾作为 supporting contract/report gate 覆盖，不作为 core GA 发布能力宣称。人像示教覆盖的 blocker 合同已迁入 identity plan 的 compatibility gate；本段只保留历史背景。

历史交付定义（仅参考）：

- PC 本地可以启用 memory，按实际发现的 `val-data/` JPEG replay inventory 完成端到端测试并生成报告；当前本地发现是 15 个 scene 目录、2221 张 JPEG。
- REST API 是唯一正式示教入口；CLI 只消费 server memory events，不做身份判断、不写 memory DB、不提供 memory 管理命令。
- legacy core GA gate 记录曾聚焦：`teach_person` self introduction、`teach_person` third-person introduction by arm pointing、known replay `known_person_present`、whole-scene `teach_scene` / `scene_activated`、ambiguous/no-write、object negative-only、非阻塞 10Hz/gaze。当前 active gate 见 identity plan。
- legacy supporting contract/report gate 记录曾覆盖：`memory_context`、conversation summary / compact background、external user link、`familiar_unknown_present`、`teach_person` auto merge anonymous、`correct-identity`、`resolve-target` dry-run 三态预览。`merge-anonymous-person` 只保留 maintenance/backfill 单独测试。当前 active gate 见 identity plan。
- memory semantic event 子集只保留 3 个：`known_person_present`、`familiar_unknown_present`、`scene_activated`。
- merge、correction、resolve-target、teach 结果、歧义和证据只通过 API response、report 或 evidence 表达，不新增 `teach_succeeded`、`anonymous_merged`、`identity_corrected`、`target_ambiguous` 等 Botified 事件。
- 产出机器可读报告和人工 visual evidence；visual evidence 用于人工确认，不作为硬 gate。

历史交付表述仅限“PC 本地通过”，不声称真机、RK3588 或现场通过；当前表述限制以 identity plan 为准。

## 2. Legacy 原则记录

- KISS：只保留一个 memory 侧链、一个 REST 示教入口、一个 SQLite/sqlite-vec 存储路径。
- DRY：身份识别、场景检索、cooldown、Botified 投影只实现一次；CLI 不复刻 server 规则。
- YAGNI：不建设治理平台、审核后台、复杂 oracle、manifest/audit 主线或对象记忆系统。
- 一个功能一种做法：示教走 server REST API；memory event 走 `visual_state.semantic_events`；CLI 对 memory semantic event 子集只做 allowlist、rate limit 和 Botified frame 投影。
- 不测试测试工具本身：新增 runner 只作为执行器，验收断言针对 server/API/DB/event/CLI 行为。
- 内部私有场景不以隐私合规为本阶段约束；主要风险是错认、错绑、误触发、刷屏和实时链路阻塞。
- `val-data/`、runtime DB、模型、cache、artifacts 不进 Git。

manifest mismatch 可以记录为数据清单风险，但不作为本计划主线，也不要求更新 `val-data/manifest.json` 才能交付。`manifest.json` 是旧 7 scene / 576 frame 视觉 oracle 口径，不是 identity/memory/teaching 交互清单。

## 3. Legacy 当前状态和缺口记录

已有基础：

- memory API、SQLite/sqlite-vec store、fake/local embedding backend 已存在。
- fake backend 可以支撑确定性的 memory API 和 E2E 检查。
- local 模型已有 smoke 路径，但需要用真实 `val-data` teach/replay 串成可交付证据。
- CLI 已能消费 memory semantic events，并把 compact `memory_context` 投影到 Botified frame。

主要缺口：

- 还缺基于真实 `val-data/` 实际目录发现的 memory E2E runner 或现有工具扩展。
- 还缺把正式 `target.kind` API 转换为当前内部 `target.mode` 低层形态的 adapter / schema migration；在完成前，`.transcript` 映射出的 payload 不能假定可直接运行。
- 交互输入规则不在本文重复维护，按 `docs/identity-overlay-product-development-plan.md` 的同 stem `.transcript` + `.jpeg/.jpg` 规则执行。
- 还缺 teach/replay/recognize/familiar/teach auto merge/correction/resolve-target/negative cases 的统一报告；maintenance/backfill merge 单独列段。
- 还缺可交付配置文档：如何启用 fake/local backend、DB 路径、模型路径、输出 artifact 路径。
- 还缺 visual evidence 串联：从 teach payload、replay timeline、memory event、Botified frame 到截图/帧样本。
- REST teach 入口需要作为正式入口写清楚并在 E2E 中验证；CLI 不允许成为示教入口。
- schema/API docs 和 CLI Botified 投影需要稳定化，避免下游依赖临时字段。
- 还缺稳定的 intent-to-target 解析合同：agent 只提供用户意图和目标类型，server 负责把意图关联到画面目标；生产 teach API 必须内部重新 resolve，只有 `resolved` 才原子写库；解析失败或不确定时返回三态结果，不写库。
- 还缺 resolver 可用的内部短窗口 memory frame cache。必须新增内部 `MemoryFrameSnapshot` 数据结构，由 stream processor/session 在 `TrackSnapshot` 仍包含 keypoints、event result / `scene_context` 仍在内存中时写入 cache；memory 侧链读取 `MemoryFrameSnapshot`，不能从 public `visual_state` 重建 resolver 输入。public `visual_state` 仍保持精简，keypoints、track、bbox 不进入 agent-facing protocol。
- 还缺 request-arrival interaction snapshot 绑定。teach / resolve 请求到达时，server 必须从最近短窗口中选择新鲜且稳定的 interaction snapshot；只有 snapshot 的 `observed_at_ms`、frame timestamp TTL、snapshot TTL 和 N-of-M stability window 都通过时才解析，否则失败并不写库。不引入 `resolution_id`、token 或预览结果复用协议。
- 还缺 `active_interaction_target` 定义：server 从新鲜 attention target 输入、`engagement_state`、稳定可见 track 和最近互动窗口派生当前互动对象；没有 `active_interaction_target` 时，`target.intent=third_person_introduction` 必须返回 `ambiguous`，不写库。本期不引入 speaker track 或音频定位作为依赖。
- 还缺 third-person introduction 的目标解析：agent 只表达“用户在介绍这位/他/她”，server 使用 `active_interaction_target` 作为介绍人 A，并用 `YOLOv8n-pose` 关键点估算手臂指向，在其他 person tracks 中解析被介绍人 B；不确定时返回 `ambiguous`，不写库。

## 4. Legacy 功能范围记录（非当前 contract source）

### 4.1 示教和检索 API

`teach_person`

- 通过 `POST /v1/memory/teach/person` 教学当前画面目标人。
- 必须覆盖两种稳定意图：`target.intent=self_introduction` 只接受新鲜、可互动、可见的 `active_interaction_target` 解析“我”，不接受 raw attention target fallback；`target.intent=third_person_introduction` 先用 `active_interaction_target` 确定介绍人 A，再用 A 的手臂姿态解析“这位/他/她”对应的 B。
- 可通过 `resolve-target` 做 dry-run 预览，返回 `resolved`、`ambiguous` 或 `not_found`。
- 生产 teach API 必须是原子 `resolve + write`：teach 内部重新 resolve 当前画面目标，只有结果为 `resolved` 且质量通过时才写库；之前的 `resolve-target` 响应只辅助 agent 追问或调试，不作为后续写库安全承诺。
- 不引入 `resolution_id`、token 或“预览结果复用”协议。
- payload 中的用户指示、目标类型和元信息由 agent 或测试映射提供；agent-facing request 不包含也不依赖 `track_id`、`bbox` 这类视觉内部状态。
- local backend 下可用脸部身份路径；没有可用脸、目标歧义、目标过期或质量不足时返回明确错误，不写入可识别身份。
- self introduction 不允许 fallback 到普通候选、单人候选、最近候选或最大候选；没有新鲜可互动目标时返回 `ambiguous` 或 `not_found`，不写库。
- 重复示教采用最小策略：高置信匹配到已知 person 且同名或同 external ref 时只更新 metadata；高置信匹配到已知 person 但不同名时在 teach 写入阶段返回 `conflict` outcome / `error_code` / HTTP 409，不创建新 person；匹配 active anonymous profile 时按 identity overlay 计划走 `teach_person -> merged_anonymous_person` 自动合并；不自动创建 duplicate。

`teach_scene`

- 通过 `POST /v1/memory/teach/scene` 教学当前整图场景。
- 整图教学使用 `target.kind=scene`。
- `target.kind=region` 只作为测试-only fixture 验证或后续能力保留，不作为 stable agent-facing GA capability、GA 主线或硬 gate。
- 如保留 region path，它只在测试-only fixture 或后续能力路径中使用；request 仍只提供 `target.kind`、`referent_text` / `intent`、`camera` 和 memory metadata，bbox、point、test hint 等低层输入主路径只能来自 runner-only envelope。如确需 server debug/test channel，只能使用唯一配置 gate `memory.test_debug_channel.enabled=true` 和 internal-only route；默认不注册到 OpenAPI、不进入生产 handler，生产/普通 agent REST 路径不可达。
- 没有可靠 visual region hint 时，region 请求必须返回 `ambiguous` 或 `not_found`，不写库；不能声称真实用户说“这里”已可稳定解析。
- 不做场景区域编辑器，不做对象记忆，不把局部物体教学声明为可用。

`resolve-target`

- 通过 `POST /v1/memory/resolve-target` 作为 dry-run/debug/agent 追问辅助。
- 返回三态：`resolved`、`ambiguous`、`not_found`。
- 该 `status` 枚举只属于 resolver / `resolve-target`；重复示教的 `conflict` 不是第四态，只能作为 teach 写入阶段 outcome / `error_code` / HTTP 409 表达。
- `resolve-target` 结果不作为后续 teach 写库安全承诺；teach API 必须重新 resolve，且只在重新 resolve 为 `resolved` 时写库。
- 稳定输入字段只允许 `camera`、`target.kind`、`target.intent`、`target.referent_text` 和必要的 profile/memory metadata。person 示教必须提供 `target.intent`；示例可包括“我”“这位”“他/她”“这个人”“当前办公室”“手机”等 referent text。
- server 内部可以使用 `MemoryFrameSnapshot`、request-arrival interaction snapshot、attention target 输入、scene context / engagement、tracks、bbox、pose/keypoints、track freshness / stability、region hints 和 runner-only envelope 解析候选；这些不是 agent-facing REST contract 字段。
- `track_id`、`bbox`、`point_uv`、`test_hint`、`source_scene`、`source_frame` 只能出现在 runner-only envelope、report，或唯一 config gate 打开的 internal-only debug/test route 中；不能进入稳定 REST contract 示例，也不能进入生产 handler 的 agent-facing request body。
- `target.intent=third_person_introduction` 必须支持受限的手臂指向人解析：使用 `active_interaction_target` 作为介绍人 A，使用 A 的 shoulder/elbow/wrist keypoints 估算手臂方向，在其他 person tracks 中选择被介绍人 B。它只覆盖“人指向另一个人”的示教，不承诺 finger pointing、精确射线、物体指向或通用手势理解。
- response 必须返回 candidates、confidence、resolution_reason 和 evidence，便于 agent 在失败时追问用户。
- 失败 response 的最小追问合同包含：`retryable`、`ask_user_hint`、`ambiguity_type`。`ambiguity_type` 只使用固定枚举：`introducer_unclear`、`target_unclear`、`pose_unclear`、`multiple_candidates`、`stale_interaction`、`no_active_interaction_target`、`unsupported_target_kind`、`quality_too_low`。third-person core ambiguous/no-write 至少必须覆盖 `introducer_unclear|pose_unclear|multiple_candidates|no_active_interaction_target|stale_interaction`。
- `ambiguous` 或 `not_found` 时 teach API 必须拒绝或要求明确目标，且不写库。

`merge-anonymous-person`

- `POST /v1/memory/merge-anonymous-person` 只保留 maintenance/backfill API 和单独测试。
- 产品主路径固定为 `teach_person -> merged_anonymous_person`，runner/client 主 E2E、主 report 和主 timeline 不调用该 endpoint。
- maintenance/backfill merge 后同一 anonymous 不应再作为 anonymous familiar event 输出；如果身份路径足够清晰，可以输出正式 `known_person_present`。

`correct-identity`

- 通过 `POST /v1/memory/correct-identity` 接收 `memory_match_id` 和正确/否定信息。
- 记录 negative match 或 correction evidence，影响后续同错误匹配。
- 不做在线训练，不重写历史 embedding。
- correction 成功与否只在 API response/report/evidence 中体现，不新增 Botified 事件。

### 4.2 Agent-Facing API 和 Intent-To-Target 解析合同

agent 只负责把用户语言整理成目标意图，visual-events 服务负责把目标意图和当前画面关联起来。系统可以使用 attention target 输入、track、bbox、pose/keypoints、场景状态、region hints 和 runner-only envelope 做内部解析；attention target 只能用于派生 `active_interaction_target`，不能作为 self introduction 或 third-person introduction 写库的 raw fallback。

`target.kind` 是正式 agent-facing API 形态；当前代码仍存在 `target.mode` 这类低层内部形态。本计划的前置实现任务是增加 kind-to-internal-target adapter / schema migration，把 stable `target.kind=person|scene|object` 转换为内部 resolver 可执行的目标结构；`region` 只在测试-only fixture 或后续能力路径中处理。完成该 adapter 前，`.transcript` 映射出的 payload 不能被当作当前代码已可直接执行。

正式 REST request body 的稳定字段边界：

- 允许：`camera`、`target.kind`、`target.intent`、`target.referent_text`、`profile` metadata、`memory` metadata。
- 禁止进入稳定 contract：`track_id`、`bbox`、`point_uv`、`test_hint`、`source_scene`、`source_frame`。
- 强制机制：Pydantic / strict schema 必须 `extra=forbid`；低层测试输入优先使用 runner-only envelope；生产 handler 的单元测试必须断言 `track_id`、`bbox`、`point_uv`、`test_hint`、`source_scene`、`source_frame` 被拒绝或不可达。
- 如确需 server debug/test channel，只能由唯一配置 gate `memory.test_debug_channel.enabled=true` 打开，只挂 internal-only route，默认不出现在 OpenAPI 和生产 handler 中。这些字段必须在 report 中标记为 runner/debug 输入，不能伪装成 agent 提供的字段，也不能进入生产 handler 的 agent-facing request body。

agent 输入和非稳定入口：

- `target.kind=person`：用户要教某个人，例如“我”“这个人”“这是彭刚”。
- `target.intent=self_introduction`：用户要教自己，例如“请记住我，我是小李飞刀”。
- `target.intent=third_person_introduction`：用户正在介绍另一个人，例如“我给你介绍一下，这位是彭刚”。
- `target.kind=scene`、`target.intent=teach_scene`：用户要教当前整图场景，例如“这是银河通用的办公室”。
- `target.kind=object`、`target.intent=teach_object`：用户要教某个物体。本发布可以识别出该意图，但正式 `status` 仍返回 `not_found`，并带 `error_code=unsupported_target_kind` / reason；object 是 negative-only，不写 memory。
- `target.kind=region`、`target.intent=teach_region`：仅限测试-only fixture 或后续能力，不属于 stable agent-facing GA capability；生产 agent-facing 文档和示例不把 region 宣称为可用能力。

server 解析规则：

- `active_interaction_target` 是 server 内部派生字段：从 request-arrival interaction snapshot 中的新鲜 attention target 输入、`engagement_state`、稳定可见 person track 和最近互动窗口计算。它不是 agent-facing request 字段，且 attention target 不是可直接写库的目标。
- teach / resolve 到达时必须选择新鲜且稳定的 interaction snapshot；snapshot 过期、没有稳定 track、互动状态不可用或无法派生 `active_interaction_target` 时返回失败，不写库。不引入 token、`resolution_id` 或 dry-run 结果复用协议。
- 本期不使用 speaker track、麦克风阵列或音频定位来确定介绍人。
- `person` 的 `self_introduction` 只接受新鲜、可互动、可见的 `active_interaction_target`。没有该目标时返回 `ambiguous` 或 `not_found`，不走 raw attention target、单人、最近、最大 bbox 或普通候选 fallback。用户说“请记住我/我是 xxx”但 `active_interaction_target` 不稳定或多人无法确定“我”时，必须返回 `retryable=true`、`ask_user_hint=true`、`ambiguity_type=target_unclear`，并在 response/report 中记录可供 agent 追问用户重新说明、靠近或单独站位的失败事件。
- `person` 的 `third_person_introduction` 主路径是 pose pointing resolver：`active_interaction_target` 是介绍人 A；从 A 的 shoulder/elbow/wrist keypoints 估算左右手臂方向，排除 A 自己，在其他 person tracks 中选择被指向对象 B。没有 A 时必须 `ambiguous`；只有手臂关键点置信度、指向稳定性、候选几何命中和领先 margin 都满足阈值时才 `resolved`，并返回 `resolution_reason=pose_pointing_to_person`。
- `person` 的普通候选解析不能用于 self introduction 或 third-person introduction 写库。多人候选分数接近、介绍人 A 不明确、手臂关键点缺失或指向线无法稳定命中 B 时必须返回 `ambiguous`，不能选择最大的人、最近的人或 attention target 强行写库。
- `scene` 直接解析为整图。
- `region` 只支持测试-only fixture 的实验性最小矩形区域：由可靠 region hint 或 runner-only envelope 解析出来。没有可靠可解析区域时返回 `ambiguous` / `not_found`，不写库；不作为 stable agent-facing GA capability。如确需 debug/test channel，必须遵守唯一 config gate 和 internal-only route 约束。
- `object` 在本发布中明确拒绝，不降级写成 `region` 或 `scene`；只有未来 region path 或用户重新发起 `target.kind=scene` 时，才能进入对应路径。
- 所有 `resolved` 结果必须带 `resolution_reason`，例如 `active_interaction_target`、`pose_pointing_to_person`、`scene_full_frame`；runner-only envelope 或 gated internal-only debug route 可以记录 `runner_fixture` / `region_fixture`，但 self introduction 和 third-person introduction 写库不接受 raw attention target fallback。
- 所有 `ambiguous` / `not_found` 结果必须通过 API response 返回给 agent；不支持的目标类型使用 `status=not_found` 并带 `error_code=unsupported_target_kind` / reason。agent 可以据此告诉用户“我没看清你指的是谁/哪里，请重新指定”。这些失败结果不进入 memory semantic event 流，也不写库。

### 4.3 稳定 memory semantic events

事件集合只包含：

- `known_person_present`
- `familiar_unknown_present`
- `scene_activated`

`known_person_present`

- replay 或在线流再次看到已教学人物时，server 输出 confirmed memory event。
- 事件带 `memory_match_id`、匹配分数、最小 evidence 和 compact `memory_context`。
- cooldown 和同目标去重必须生效，不能刷屏。

`familiar_unknown_present`

- 这是 memory event allowlist 中的 supporting contract/report 项，不作为 core GA gate 或发布能力宣称。
- 同一未知人多次稳定出现、但还没有被命名或绑定正式 person 时输出。
- fake backend 和 local backend 至少要在 `pic_familiar_face` 或选定重复人物场景中证明只触发一次，后续受 cooldown 控制。
- merge 后不再输出同 anonymous 的 familiar event。

`scene_activated`

- replay 或在线流再次看到已教学整图场景时输出。
- 整图事件必须覆盖 `teach_scene target.kind=scene`。
- 如果实验性 region path 被保留并启用，可以复用 `scene_activated` 表达 region 命中，但这不是 GA 硬 gate；事件 evidence 必须说明 region 来源，且不能把缺少可靠 region hint 的请求写库。
- 本期不新增 anonymous/familiar scene profile，也不做“重复出现后自动熟悉化”的整图场景能力；稳定 `scene_activated` 只表示已教学整图 scene 被激活。实验性 region 命中必须标为非 GA evidence。
- 不把局部物体或手势强行映射成 scene memory。

### 4.4 memory context 和用户链接

`memory_context`

- Botified frame 的 `visual_context.memory_context` 包含 agent 需要的短背景：人物 display name、description、summary、scene label 等。实验性 region 如启用，可以在 evidence 中携带 region label，但不作为稳定 `memory_context` 必备字段。
- 不包含图片、crop、embedding、完整 tracks 或大段聊天原文。

conversation summary / compact background

- 支持把 agent 生成的短摘要挂到 `person_id`。
- 后续识别到该人时返回短摘要；不自动从所有对话原文抽取长期记忆。
- summary 作为 compact background 进入 `memory_context`，长度受限且可解析。

external user link

- 支持把外部用户引用绑定到 `person_id`。
- 支持通过 external user ref 查回 person profile 和摘要，用于视觉身份与消息用户的显式关联。

### 4.5 非目标

- 本期不做物体记忆。
- `pic_teach_item_phone` 不强行塞进 person/object 主线。它是未来 object memory 素材或负例；不能声称手机记忆可用。
- 不做通用手势理解、手指级指向、物体指向、跨摄像头 ReID、管理后台、云同步、多租户或正式隐私治理；本期只承诺 `third_person_introduction` 中受限的人指向人解析。
- 不让 server/CLI 生成最终话术或决定机器人业务动作。
- CLI 不做身份判断、不写 DB、不做 memory 管理命令。

## 5. Legacy 架构和入口记录

实时主链路保持不变：

```text
DDS JPEG -> CLI -> server /v1/stream
  -> inference / tracking / attention / semantic events
  -> visual_state @10Hz -> CLI
  -> DDS gaze target + low-frequency Botified frame
```

memory 侧链：

```text
server recent frame snapshot/history cache
  -> REST teach / query / link / summary API
  -> interaction snapshot freshness/stability check
  -> target resolver
  -> bounded embedding worker / thread pool
  -> SQLite + sqlite-vec transaction
  -> retriever / memory event generator
  -> visual_state.semantic_events
  -> CLI Botified projection
```

内部短窗口 cache 要求：

- 新增内部 `MemoryFrameSnapshot` 数据结构，至少包含 `source_frame_ref`、frame bytes/path ref、frame timestamp、image size、track refs、bbox、keypoints、track freshness / stability、attention target 输入、event result、scene_context / engagement。它由 stream processor/session 在 `TrackSnapshot` 仍包含 keypoints、event result / `scene_context` 仍在内存中时写入 memory frame cache。
- `MemoryService`、resolver 和 embedding crop 只能读取 `MemoryFrameSnapshot` / interaction snapshot；不能从 public `visual_state` 反向重建 keypoints、tracks 或 scene context。public `visual_state` 继续保持精简，不向 agent-facing protocol 暴露 keypoints、track、bbox。
- 保存 request-arrival interaction snapshot：teach / resolve 到达时，从最近短窗口中选择新鲜稳定的 snapshot，记录 `observed_at_ms`、关联 frame timestamp、`active_interaction_target` ref、engagement_state、N-of-M stability window 结果。frame timestamp TTL、snapshot TTL、N-of-M 阈值必须可配置并写入 report；失败 reason 固定包含 `stale_interaction|no_active_interaction_target`。
- cache 有固定长度和 TTL，只服务当前解析窗口；不做长期审计、治理或平台化回放系统。

硬边界：

- server REST API 是示教入口。
- CLI 对 memory semantic event 子集只消费 server 确认的 3 类 memory events，做 allowlist、幂等、rate limit 和 Botified 字段投影。
- CLI 继续支持既有非 memory semantic events，例如 `waving`、`passing`、`left`；本计划不要求删除或收窄这些现有事件。
- CLI 不判断身份、不计算 familiar score、不解析 `.transcript`、不写 DB、不提供 merge/correction/link/summary 命令。
- embedding 慢或失败只能延迟或丢弃 memory event，不能阻塞 10Hz 主链路和 gaze。
- teach/write embedding 也必须走 bounded worker、线程池或等价非阻塞路径；async handler 不能同步运行 local embedding 阻塞事件循环。API 可以等待有上限的 worker 结果，但必须有 timeout/backpressure。
- fake/local backend 共用同一 API、store、event 和 CLI 投影路径。
- known-person recognition 不能只查 attention target。后续识别必须对稳定可见 person tracks 做 bounded multi-person retrieval：每帧/每 tick 限制 track 数、队列长度和频率，复用 worker、cooldown、rate limit，且不阻塞 10Hz/gaze。

## 6. Legacy 实现步骤记录

### 6.1 最小必要代码

1. 增加 public `target.kind` 到当前内部 `target.mode` / resolver target 的 adapter 或 schema migration：
   - 正式 agent-facing request 使用 `target.kind`。
   - adapter 负责转换为当前内部 target 结构，并统一处理 `person`、`scene`、实验性 `region` 和 `object` unsupported。
   - 完成前，`.transcript` 映射 payload 不能被视为当前代码可直接运行。
2. 补齐或确认 server REST API：
   - `POST /v1/memory/teach/person`
   - `POST /v1/memory/teach/scene`
   - `POST /v1/memory/person/{person_id}/conversation-summary`
   - `POST /v1/memory/link-external-user`
   - `GET /v1/memory/person/by-external-user/{external_user_ref}`
   - `POST /v1/memory/merge-anonymous-person`（maintenance/backfill only，不进入主 E2E）
   - `POST /v1/memory/correct-identity`
   - `POST /v1/memory/resolve-target`
3. 确保 teach API 是原子 `resolve + write`：
   - teach 内部重新调用 resolver；只有 `resolved` 且质量通过时写库；`resolve-target` dry-run 响应不得被当作后续写库凭证。
   - store 层必须新增 `create_person_with_embedding(...)`、`create_scene_with_embedding(...)` 或等价 transaction context；profile/scene + embedding + embedding provenance all-or-nothing，成功一起提交，失败一起回滚，不能留下 orphan row。
   - 新增 store/embedding provenance 字段或表，不能只依赖截图或 transient report；至少保存 `source_track_ref`、`source_frame_ref`、`crop_hash`、`crop_path_or_artifact_ref`、`resolver_target_ref`、`resolution_reason`、`embedding_type`、`embedding_model`、`embedding_version`、`embedding_dim`。这些字段随 person/scene + embedding 同事务持久提交。
   - 增加 orphan row injection / failure 测试：模拟 embedding 写入、provenance 写入或 profile/scene 写入失败，断言 transaction 回滚且没有孤儿 profile、scene、embedding 或 provenance row。
   - 所有 no-write case 必须在 response/report 中记录 `store_delta`；`store_delta` 来源于 DB/store 前后快照或 transaction observer，覆盖 store/migration 暴露的 memory-owned table universe。store/migration 层必须暴露或生成 memory-owned table universe 和 allowed diagnostic whitelist，runner/assertion 从该权威来源读取，不手写表名；除该白名单诊断/临时表外，no-write 的每个 memory-owned table delta 都必须为 0，report 列出 universe/whitelist 来源。
4. 确保 `MemoryService` 只从 server 内部 `MemoryFrameSnapshot` cache 和 request-arrival interaction snapshot 解析 target；无新鲜 stream、snapshot 过期、目标不稳定、歧义或质量不足时明确报错且不写入。
   - `MemoryFrameSnapshot` 由 stream processor/session 在 `TrackSnapshot` 仍包含 keypoints、event result / `scene_context` 仍在内存中时写入 memory frame cache，至少保存 frame bytes/ref、timestamp、image size、tracks、bbox、keypoints、attention target 输入、scene_context / engagement、track freshness / stability。
   - `MemoryService` 不能从 public `visual_state` 重建 keypoints、bbox、track refs；这些字段只能进入内部 resolver、report、visual evidence、runner-only envelope 或 gated internal-only debug/test route，不能进入 agent-facing REST request。
   - teach / resolve 必须按 request-arrival 绑定新鲜且稳定的 interaction snapshot；response/report 必须记录本次解析实际使用的 `request_snapshot_ref`、`source_frame_ref`、frame timestamp、snapshot `observed_at_ms`、frame timestamp TTL、snapshot TTL、N-of-M stability window，并能和固定 frame/window、`visual_evidence_index[]` join。不引入 token、`resolution_id` 或 dry-run 结果复用协议。
5. 确保 intent-to-target resolver 支持稳定输入 `target.kind=person|scene|object`；`region` 只在测试-only fixture 或后续能力路径中处理，不作为 stable agent-facing capability。正式 `status` 只输出 `resolved`、`ambiguous`、`not_found`；不支持的 target kind 返回 `status=not_found` 并带 `error_code=unsupported_target_kind` / reason。`resolve-target` 和 teach API 使用同一套 resolver，但 teach 不能复用 dry-run 结果。
6. 正式 REST schema 只允许 agent-facing 字段：`camera`、`target.kind`、`target.intent`、`target.referent_text`、profile/memory metadata。`track_id`、`bbox`、`point_uv`、`test_hint`、`source_scene`、`source_frame` 主路径只能在 runner-only envelope 或 report 出现；如确需 server debug/test channel，只能通过唯一 config gate `memory.test_debug_channel.enabled=true` 的 internal-only route，默认不注册 OpenAPI、不进入生产 handler。
   - Pydantic / strict schema 使用 `extra=forbid`。
   - 生产 handler 单元测试必须覆盖低层字段被拒绝或不可达，并断言默认 `debug_test_channel_enabled=false`。
7. 实现 `third_person_introduction` pose pointing resolver：
   - 复用现有 `YOLOv8n-pose` keypoints、tracker 和 attention target 输入，不新增模型、不训练模型、不引入手部模型。
   - 介绍人 A 由 `active_interaction_target` 决定；A 必须新鲜、可见、稳定，并处于可互动状态。没有 A 时返回 `ambiguous`，不写库。
   - `active_interaction_target` 由 server 从新鲜 attention target 输入、`engagement_state`、稳定可见 track 和最近互动窗口派生；不依赖 speaker track 或音频定位，不允许 raw attention target fallback 写库。
   - 只使用 shoulder/elbow/wrist 估算手臂方向；手臂向量、torso center、N-of-M 稳定窗口、归一化阈值、左右臂冲突策略和候选 margin 必须定义在 config defaults 和 report fields 中；不使用 finger keypoints，不声明手指方向。
   - torso center 使用可用 shoulder/hip keypoints 估算，缺失时按配置允许 bbox center fallback 或直接 `ambiguous`；该策略必须写入 report。
   - 每只手臂独立 scoring：检查 keypoint confidence、手臂伸展、方向前向性、射线到候选 bbox/torso center 的归一化距离、候选 freshness/stability 和 visibility。
   - N-of-M 窗口要求同一候选和 arm side 在短窗口内稳定；左右臂指向不同候选且 margin 不满足时返回 `ambiguous`。
   - 候选 B 只能来自当前其他 person tracks；排除 A 自己。
   - scoring 使用手臂射线到候选 bbox/torso center 的距离、候选可见性、track 稳定性、关键点置信度、短时间稳定性和领先 margin；阈值固定进 config/report。
   - 多人候选接近、关键点缺失、手臂没有明显伸出、A 不明确或 B 不可见时返回 `ambiguous`，不写库。
   - resolved evidence 必须包含 introducer/target 的内部 track ref、使用的手臂 side、keypoint confidences、指向几何分数、候选列表、margin、stability window 结果和 `resolution_reason=pose_pointing_to_person`；这些 evidence 不进入 agent-facing request body。
8. 保留 region path 时按实验能力处理：
   - 不作为 GA 主线或硬 gate。
   - 正式 request 不接收 bbox/point/test hint 等低层字段。
   - 没有可靠 region hint 或 runner-only fixture 输入时返回 `ambiguous` 或 `not_found`，不写库；debug channel 不是主路径，若启用必须遵守唯一 config gate 和 internal-only route 约束。
   - 如果实现 region crop/query path，report 可以记录 `region_query_path`、`crop_bbox`、`camera`、`embedding_source`、候选数和分数作为非 GA evidence。
9. 确保 SQLite/sqlite-vec 是唯一正式检索实现；不要添加第二套手写向量检索主路径。检索必须按 `embedding_type`、`embedding_model`、`embedding_version`、`embedding_dim` 过滤后再进入 sqlite-vec 相似度查询，不能混查不同 type/model/version/dim。person 和 scene 可以使用不同 `embedding_type` / `embedding_dim`，但每次查询必须按目标类型和维度过滤。
10. 确保 teach/write embedding 通过 bounded worker、线程池或等价非阻塞路径执行；async handler 不能同步跑 local embedding。worker queue、timeout 和 backpressure 必须可配置并写入 report。
11. 确保后续 recognition 走 bounded multi-person retrieval：对稳定可见 person tracks 做限流/限量检索，不只检查 attention target；复用 worker、cooldown、rate limit，不能阻塞 10Hz/gaze。
12. 确保 `known_person_present`、`scene_activated`、`familiar_unknown_present` 事件带稳定 evidence，且进入同一 cooldown/rate-limit 路径。
13. 确保 CLI allowlist 对 memory semantic event 子集只允许 3 类 memory events，并稳定投影 `memory_context`；CLI 不新增身份逻辑，也不删除既有非 memory 事件支持。
14. 更新 schema/API docs 和 CLI Botified projection docs，列出稳定字段、错误码和 evidence/report 字段边界。

核心逻辑 TDD 最小矩阵：`MemoryFrameSnapshotCache` TTL/N-of-M、`active_interaction_target` no-fallback、pose pointing scorer、bounded worker/backpressure、retrieval caps/cooldown、store_delta observer。

### 6.2 配置

新增或整理一份可交付配置示例，覆盖：

- `memory.enabled=true`
- `memory.db_path=runtime/memory/visual_memory.sqlite3`
- `memory.embedding.backend=fake|local`
- local person model path 和 scene model path 必须显式传入；server 不隐式下载模型。
- inference pose model 继续使用现有 `YOLOv8n-pose` backend 和显式 `model_path`；third-person introduction 不新增模型。
- interaction snapshot defaults 必须可配置并写入 report：frame timestamp TTL、snapshot TTL、N-of-M stability window、`observed_at_ms` clock source、`stale_interaction` / `no_active_interaction_target` 失败判定。
- third-person introduction defaults 必须可配置并写入 report：keypoint min confidence、arm vector definition、torso center strategy、arm extension ratio、normalized pointing distance threshold、candidate margin、N-of-M stability window、left/right arm conflict policy。
- bounded worker / retrieval defaults 必须可配置并写入 report：embedding worker queue size、timeout、max tracks per recognition tick、recognition cooldown、retrieval rate limit。
- artifact 输出默认在 `artifacts/memory-teaching-ga/`。

runtime DB、模型和 artifact 继续保持 gitignored。

### 6.3 测试工具

建议新增 `tools/run_memory_teaching_ga_e2e.py`，或扩展现有 `tools/run_memory_e2e.py` 增加 `--ga-val-data-suite` 模式。

要求：

- 不破坏 `run_memory_e2e` 现有 fake/synthetic 价值；原有命令继续用于快速确定性回归。
- GA runner 使用真实 `val-data/` 的实际 JPEG replay inventory，是本期新增交付；它不是现有 `run_memory_e2e` 的 synthetic/fake 快速回归。
- `run_memory_e2e` 只能继续作为 synthetic/fake 快速回归，不能等同于真实 `val-data` 全量 gate。
- 现有 `run_val_data_e2e` / manifest oracle 是旧 7 scene / 576 frame 视觉 oracle 口径，不能作为 memory/teaching 交互清单；可以复用工具代码，但验收必须以 GA runner 的实际发现报告为准。
- runner 负责驱动 server、发送 stream frame、调用 REST teach/resolve/summary/link/correct API、replay、采集 memory events、采集 CLI Botified stdout；`merge-anonymous-person` 只在 maintenance/backfill 单独测试中调用。
- runner 根据同 stem `.transcript` + `.jpeg/.jpg` 交互清单手工映射正式 REST request body，不调用 LLM 解析。
- runner-only envelope 与正式 REST request body 必须硬隔离：`source_scene`、`source_frame`、`track_id`、`bbox`、`point_uv`、`test_hint` 主路径只能存在于 envelope 或 report，不能进入稳定 REST contract 示例，也不能伪装成 agent 提供的字段；生产/普通 agent REST 路径不可达，低层 fixture 字段不得进入生产 handler 的 agent-facing request body。
- 如确需 server debug/test channel，只能通过唯一 config gate `memory.test_debug_channel.enabled=true` 打开 internal-only route；默认 `debug_test_channel_enabled=false`，默认 OpenAPI/生产 handler 不出现低层字段。
- runner 可以用 stream barrier 或等价方式等待预期固定 frame/window 已进入 `MemoryFrameSnapshot` cache，再发 REST；断言 teach/resolve response/report 使用的 `request_snapshot_ref`、`source_frame_ref`、frame timestamp 和 snapshot `observed_at_ms` 对应预期 snapshot/source frame，而不是其他帧。
- fake backend 覆盖完整稳定合同；local backend 只做固定样本 smoke，验证真实模型核心路径和同一 API/store/event/CLI 投影路径。
- local third-person 正例只能固定输入 frame/window；A/B/keypoints/tracks 必须来自真实 `YOLOv8n-pose` + tracker + `active_interaction_target` 路径，不能用 fixture 直接指定被介绍人 B。local core gate 必须同时记录并满足 `debug_test_channel_enabled=false`、`fixture_inputs_consumed=[]`（未消费 target fixture 输入）和 `debug_fixture_used_for_target_resolution=false`。
- fake backend 可以用构造 keypoints 覆盖 resolver 合同，但必须在 report 中标明 fake/constructed evidence，不能把它当作 local 真实模型通过。
- runner 输出机器可读 `report.json`、事件 timeline、teach payload 记录、API response 记录、Botified stdout/frames、失败样本列表和 visual evidence 路径；这些 report/timeline/evidence 是本期 GA runner 交付的一部分。
- runner 不更新 `val-data/`，不修改 manifest，不把 artifact 写进 Git。

### 6.4 Visual evidence

补充或复用 visual evidence 工具，生成：

- teach frame 缩略图和对应 payload 摘要。
- resolve-target 预览结果：`resolved`、`ambiguous`、`not_found`。
- replay timeline：帧号、场景名、memory event、confidence、cooldown 状态。
- memory event 到 Botified frame 的对应关系。
- third-person introduction：画出介绍人 A、被介绍人 B、手臂方向线、候选框、candidate score 和最终 `resolution_reason`。
- 每张 visual evidence 图必须叠加 `assertion_id`、track refs、stored crop hash/path、event_id 或 `memory_match_id`，确保人工截图能和机器 report 对齐。
- 生成 `visual_evidence_index[]` 并写入 report；每项至少包含 evidence file path、`assertion_id`、可选 `event_id`、`memory_match_id`、`request_snapshot_ref`、`source_frame_ref`、frame timestamp、snapshot `observed_at_ms`、`stored_crop_frame/path`、`crop_hash`、track refs，runner 必须断言文件存在且这些 key 能 join 到 assertion、teach/resolve response、event、stored crop、固定 frame/window 或 replay sample。
- 如果启用实验性 region path，记录 region crop 或样本索引作为非 GA evidence。
- negative cases 截图或样本索引。

visual evidence 是人工验收材料，不作为 CI gate。机器断言仍由 report 决定。

### 6.5 文档更新

实现完成后更新：

- server handoff：启动方式、memory 配置、local 模型路径、报告位置。
- development test plan：加入 stable memory/teaching 的手工/本地 gate。
- API 文档或 schema：列出正式 REST teach、resolve、summary、link、merge、correct 入口。
- CLI Botified 投影文档：列出 memory semantic event 子集只消费 3 类事件、既有非 memory 事件继续支持，以及 `memory_context` 的稳定字段。

不需要新增 governance/audit 文档主线。

## 7. Legacy 验证方案记录

### 7.1 数据集范围

使用 `val-data/` 的实际 JPEG replay inventory。当前本地发现为 15 个 scene 目录、2221 张 JPEG；该数字由 runner 目录发现写入 report，不从 manifest 推断，也不作为 identity/teaching 交互硬编码。

- `pci_stand`
- `pic_1_l_to_r`
- `pic_1_r_to_l`
- `pic_familiar_face`
- `pic_hello`
- `pic_leave`
- `pic_pace_back_and_forth`
- `pic_people_gathering`
- `pic_persone_walk_in`
- `pic_teach_item_phone`
- `pic_teach_me`
- `pic_teach_person`
- `pic_teach_scene_galbot`
- `pic_walk_away`
- `pic_walk_in_stop`

发现到的 JPEG scene 都要 replay 并进入报告。不是每个场景都必须触发 memory event；负例不触发同样是断言。report 还要列出 `transcript_cases[]` 和 manifest legacy mismatch；不要求 manifest 覆盖 identity cases。

third-person local smoke 的硬前置：

- `pic_teach_person` / `val-data` 必须提供一个固定 frame/window，其中介绍人 A、被介绍人 B 都稳定可见，A 的手臂姿态可由 `YOLOv8n-pose` 解析，并且 `active_interaction_target` 能稳定落在 A。
- 如果现有 `pic_teach_person` 没有稳定窗口，必须新增或选择专门的多人介绍场景作为 auxiliary local smoke scene；不能用 fixture 直接指定 B 来替代 local smoke。auxiliary scene 与发现到的 `val-data` JPEG replay inventory 并列报告，不改变基础发现口径。
- 如果 auxiliary scene 被正式加入 `val-data/`，报告中的发现数量必须自动变化，列出新增场景名、用途和是否纳入全量 replay；如果它只作为外部 auxiliary fixture，则报告必须单独列在 `auxiliary_local_smoke_scenes[]`，不把它伪装成基础 scene 之一。
- 如果使用现有 `pic_teach_person`，report 必须记录固定 frame/window、A/B 人工标注、期望 arm side、人工确认截图路径、`fixture_inputs_consumed=[]`（未消费 target fixture 输入）和 `debug_fixture_used_for_target_resolution=false`。

### 7.2 `.transcript` 使用规则

本节被 `docs/identity-overlay-product-development-plan.md` 的测试数据规则收敛。当前 `val-data/` 没有 `des.txt`；交互输入只来自参考图像旁边的同 stem `.transcript`，同 stem 图像可以是 `.jpeg` 或 `.jpg`，当前本地发现 4 个 transcript case。

使用方式：

- 不自动用 LLM 解析 `.transcript`；runner 手工映射固定 request 模板。
- 每个正式 REST request body 必须显式填写 `camera`、`target.kind`、`target.intent`、`target.referent_text`，以及必要的 `profile` 或 `memory` metadata；整图 scene 可使用 `target.intent=teach_scene`。
- `source_scene`、`source_frame`、`track_id`、`bbox`、`point_uv`、`test_hint` 主路径只能放在 runner-only envelope 或 report；如确需 server debug/test channel，只能走唯一 config gate 和 internal-only route。它们不得进入 agent-facing payload 示例，也不得进入生产 handler 的 agent-facing request body。
- report/evidence 必须记录 `source_text_path`、`source_image_path`、`source_frame_ref` 和 `request_snapshot_ref`。
- 不从 `.transcript` 派生 GA region payload。需要验证实验性 region path 时，只能走测试-only fixture；没有可靠 visual region hint 时预期返回 `ambiguous` 或 `not_found`，不写库。
- object 文本仍是 negative-only：预期 `status=not_found` 且带 `error_code=unsupported_target_kind` / reason；不写 object memory，也不降级为 scene/region memory。

### 7.3 场景组合

teach/replay/recognize：

- 从 `pic_teach_me` 可以先调用 `resolve-target` dry-run，得到 `resolved` 后再调用 teach person；teach API 必须重新 resolve 并在自身 response 中记录 resolve evidence。再 replay 同场景和选定相似人物场景，断言 `known_person_present` 至少一次，且 `memory_context.display_name` 正确。
- 从 `pic_teach_person` 可以先调用 `resolve-target` dry-run，得到 `resolved` 后再调用 teach person；teach API 仍必须重新 resolve。此路径必须使用 `target.intent=third_person_introduction` 和 `resolution_reason=pose_pointing_to_person`，并用机器字段证明写入的是被介绍人 B，不是介绍人 A：`stored_person_id`、`stored_embedding_source_track_ref`、`stored_crop_frame/path`、`stored_crop_hash`、`profile.display_name`、`resolver_target_ref`、`introducer_ref`。
- runner 必须硬断言 `stored_embedding_source_track_ref == resolver_target_ref` 且 `stored_embedding_source_track_ref != introducer_ref`；`stored_crop_hash` 必须能从 `stored_crop_frame/path` 重算，并等于 embedding provenance 中保存的 `crop_hash`。
- third-person 写库后必须做 B-positive replay 和 A-only negative replay：B 再出现时能命中新 person；只有 A 出现时不能命中新 person。B-positive / A-only replay 样本必须有可核对的 A/B 标注、frame/window、截图或 frame path，不能只用自然 replay 结果口头说明。
- 从 `pic_teach_scene_galbot` teach 整图 scene；再 replay 同场景，断言 `scene_activated`。
- 对已教学整图场景 replay，断言已教学场景可以触发 `scene_activated`，且无关场景不触发。

third-person introduction：

- 使用 `pic_teach_person` 或专门多人介绍场景，固定 teach frame/window，画面中至少包含介绍人 A 和被介绍人 B。
- `resolve-target` 和 teach response 必须记录 introducer A、target B、使用的 arm side、keypoint confidence、candidate scores、margin 和 `resolution_reason=pose_pointing_to_person`。
- fake backend 用构造 keypoints 证明 A 指向 B 时 resolved，A 指向不清或两个候选接近时 ambiguous。
- local backend 用真实 `YOLOv8n-pose` keypoints、tracker 和 `active_interaction_target` 在固定样本上证明至少一次 `third_person_introduction -> teach_person -> known_person_present`；local 正例不能用 fixture 直接指定 B，report 必须记录 `debug_test_channel_enabled=false`、`fixture_inputs_consumed=[]`（未消费 target fixture 输入），且 `debug_fixture_used_for_target_resolution=false`。
- 负例必须证明没有清晰手臂指向、介绍人 A 不明确、候选 B 不唯一、无 `active_interaction_target` 或 interaction snapshot stale 时不写库；response 至少包含 `retryable`、`ask_user_hint`、`ambiguity_type`，其中 `ambiguity_type` 覆盖 `introducer_unclear|pose_unclear|multiple_candidates|no_active_interaction_target|stale_interaction`。

experimental scene region：

- 本块是可选实验/fixture 验证，不作为 GA gate。
- 如果实现，使用 `pic_teach_scene_galbot` 或选定稳定场景，通过 runner-only envelope 提供固定 visual region hint；正式 request body 不携带 bbox/point/test hint。如确需 debug/test channel，必须遵守唯一 config gate 和 internal-only route 约束。
- 没有可靠 visual region hint 时，`target.kind=region` 必须返回 `ambiguous` 或 `not_found`，不写库。
- 如果写入并查询 region，report 记录最终解析 bbox/region、query crop/path、embedding source、candidate count、match score、threshold 和是否通过；这些作为非 GA evidence，不能只用 event 是否带 `region_id` 判断。
- 不实现区域编辑器，不把 phone 或其他物体 region 声称为 object memory。

familiar：

- 使用 `pic_familiar_face` 或多段同一未知人场景，在未命名状态下多次 replay。
- 达到阈值后断言最多触发一次 `familiar_unknown_present`，后续受 cooldown 控制。

merge：

- 主路径使用 `teach_person -> merged_anonymous_person` 自动转正 anonymous profile，详见 `docs/identity-overlay-product-development-plan.md`。
- replay 后断言不再输出同 anonymous 的 familiar event；如果身份路径足够清晰，可以输出正式 `known_person_present`。
- `/v1/memory/merge-anonymous-person` 只保留 maintenance/backfill 单独测试，不进入主 E2E、主 report 或主 timeline。

correction：

- 对一次错误或构造的 `memory_match_id` 调用 `correct-identity` API。
- replay 后断言不再高置信返回同一错误 person。
- correction 结果只进入 API response/report/evidence，不进入 memory semantic event 流。

ambiguity：

- 使用 `pic_people_gathering` 或多人接近场景调用 `resolve-target`。
- 目标不确定时返回 `ambiguous`；teach API 不写入 memory。
- `target_ambiguous` 不作为 Botified event 输出。
- `ambiguous` / `not_found` response 必须能被 agent 用来追问用户；不支持目标类型用 `status=not_found` 加 `error_code=unsupported_target_kind` / reason 表达。失败响应不写库、不进入 memory semantic event。
- no-write response/report 必须断言 `store_delta` 来自 DB/store 前后快照或 transaction observer，覆盖 store/migration 暴露的 memory-owned table universe；allowed diagnostic whitelist 也从同一权威来源读取，不手写表名。除白名单诊断/临时表外，每个 memory-owned table delta 都为 0。

negative cases：

- `pic_teach_item_phone` 不应触发 person identity teaching 成功；也不应因为“手机”文本产生 object memory。
- `target.kind=object` 必须返回 `status=not_found` 且带 `error_code=unsupported_target_kind` / reason，且不降级成 scene/region memory。
- `pic_leave`、`pic_walk_away` 等离开场景不应持续刷出已知人物事件。
- 与已教学人物/整图场景不相似的场景不应输出 confirmed memory event。
- 低相似度、低质量、无新鲜 frame、过期 runner-only target fixture 都必须返回明确错误或无事件。
- 所有 negative no-write case 必须断言 `store_delta` 来自 DB/store 前后快照或 transaction observer，覆盖 store/migration 暴露的 memory-owned table universe；allowed diagnostic whitelist 也从同一权威来源读取，不手写表名。除白名单诊断/临时表外，每个 memory-owned table delta 都为 0。

### 7.4 fake backend 和 local backend

fake backend 证明完整合同：

- REST API strict schema、intent-to-target resolver 三态、SQLite/sqlite-vec transaction、retriever、event generator、cooldown、CLI Botified 投影的确定性闭环正确。
- teach person、self introduction、third-person introduction pose pointing、known person、whole-scene teach/activation、familiar、teach auto merge anonymous、correction、summary、external user link、ambiguity、negative 和 object unsupported 合同正确；maintenance merge 单独测试。
- 只输出 3 类 memory semantic events；teach/merge/correction/ambiguity 等结果只在 API response/report/evidence 中出现。
- 适合 CI 或快速本地回归。
- fake backend 是完整合同 gate：API schema、状态流、store、event、cooldown、CLI projection、report 字段、no-write `store_delta`、DB transaction all-or-nothing 都必须覆盖；它可以使用构造 keypoints 和固定样本保证确定性。
- fake backend 必须覆盖 bounded worker / bounded retrieval 的合同字段和限流行为，即使 embedding 本身是确定性 fake。
- 如果保留实验性 region path，fake backend 可以覆盖其 resolver/write/query evidence，但该覆盖不属于稳定发布硬 gate。

local backend 证明：

- 在 PC 本地显式模型路径下，真实视觉 embedding 能完成固定样本 smoke；实际发现的 JPEG scenes 进入 replay/report，但 local 硬 gate 只验证核心真实模型路径。
- 固定样本至少证明 self introduction `teach_person -> known_person_present`。
- 固定样本至少证明 third-person introduction `pose_pointing_to_person -> teach_person -> known_person_present`，且被写入的是被介绍人，不是介绍人。
- 固定样本至少证明 whole-scene `teach_scene -> scene_activated`。
- supporting report 可用固定样本或构造输入证明 `familiar_unknown_present`，但它不作为 local core GA 发布能力宣称。
- 固定样本至少证明 known-person recognition 对稳定可见 person tracks 做 bounded multi-person retrieval，不只检查 attention target。
- teach auto merge/correct 可以用构造输入触发，但必须走同一 API、store、retriever/event 抑制路径；maintenance/backfill merge 如测试则单独列段，不要求等待真实模型自然产生每个错误分支。
- 固定样本至少证明 ambiguous 不写库。
- 固定样本至少证明 teach/write embedding 和 recognition retrieval 不阻塞 10Hz/gaze，且 worker backlog 有上限。
- local backend 不隐式下载模型，不写系统目录，不改变 fake backend 的测试价值。

local backend 可重复判定规则：

- local backend 是固定样本 smoke gate，不要求覆盖 fake backend 的每个构造分支，但必须使用同一 API、store、event 和 CLI projection 路径。
- teach frame 和 replay frame 必须固定：每个断言记录 `teach_frame_index`、`replay_frame_index` 或固定 frame 文件名；不能依赖“跑到哪里算哪里”的自然波动。
- target source 必须固定：使用稳定 request 输入和固定 recent frame；如需低层辅助，主路径只能通过 runner-only envelope 提供，并在 report 中标为非 agent-facing 输入；生产/普通 agent REST 路径不可达。
- thresholds/config 必须固定并写入 report：person match threshold、scene match threshold、familiar threshold、cooldown、model path、embedding backend、random seed 或 deterministic flag。
- resolver evidence 必须固定并写入 report：`target.kind`、`target.intent`、`referent_text`、candidates、confidence、resolution_reason、pose/attention-input/runner fixture 是否参与；third-person introduction 还必须记录 introducer/target 内部 refs、arm side、keypoint confidences、candidate scores、margin、N-of-M stability result、`debug_test_channel_enabled=false`、`fixture_inputs_consumed=[]`（未消费 target fixture 输入），以及 `debug_fixture_used_for_target_resolution=false`。
- correct/negative 可以用构造数据或固定样本触发，不要求等待真实模型自然错认；必须记录构造方式、输入 `memory_match_id` 或 negative pair、期望结果和实际结果。
- 实验性 scene region 如启用，必须使用固定 teach/replay frame 和固定 runner-only hint；如确需 debug/test channel，必须遵守唯一 config gate 和 internal-only route 约束。region crop/path evidence 不作为 local GA 硬 gate。
- 每个失败 assertion 必须带失败归因字段，例如 `failure_category=model_low_confidence|target_not_found|target_ambiguous|threshold_mismatch|event_missing|event_unexpected|cli_projection_missing|cooldown_failed|api_error`，以及最小证据路径。
- local gate 不依赖不可复现的自然波动；如果真实模型分数在阈值附近抖动，应调整固定样本或阈值配置，而不是把随机通过当作可交付结果。

只有 local backend 报告满足 `ok=true`、`uses_real_model_backend=true`、固定配置完整记录且关键 assertions 通过，才能声称“PC 本地 memory/teaching 可用”。

### 7.5 报告产物

输出目录建议：

```text
artifacts/memory-teaching-ga/
  report.json
  timeline.jsonl
  teach_payloads.json
  api_responses.jsonl
  botified_frames.jsonl
  visual-evidence/
    index.html
    frames/
```

`report.json` 至少包含：

- data dir、实际发现的 JPEG scene 列表、JPEG 数量、replay 到的 scene 数；如使用 auxiliary local smoke scene，单独列出 `auxiliary_local_smoke_scenes[]`。如果 auxiliary scene 加入 `val-data/`，发现数量必须自动变化并列出新增项。
- `transcript_cases[]`：每项包含 `source_text_path`、`source_image_path`、`source_frame_ref`、`request_snapshot_ref` 和映射出的 request 摘要。
- manifest legacy mismatch：记录 manifest 的旧 7 scene / 576 frame oracle 口径和实际 discovery 差异；不要求 manifest 覆盖 identity cases。
- backend 类型、inference pose model path、embedding model path、是否 real model、固定 thresholds/config。
- short window cache / interaction snapshot 摘要：`MemoryFrameSnapshot` count/source、`request_snapshot_ref`、`source_frame_ref`、snapshot timestamp、snapshot `observed_at_ms`、frame timestamp、frame timestamp TTL、snapshot TTL、freshness、N-of-M stability window、active_interaction_target ref、engagement_state、失败 reason `stale_interaction|no_active_interaction_target`；不包含 agent-facing request 禁止字段。
- third-person introduction 结果：introducer/target 内部 refs、arm side、pose keypoint confidence、candidate scores、margin、N-of-M stability result、`resolution_reason=pose_pointing_to_person`、`debug_test_channel_enabled=false`、`fixture_inputs_consumed=[]`（未消费 target fixture 输入）、`debug_fixture_used_for_target_resolution=false` 和是否写库。
- third-person 写库证明字段：`stored_person_id`、`stored_embedding_source_track_ref`、`stored_crop_frame/path`、`stored_crop_hash`、`profile.display_name`、`resolver_target_ref`、`introducer_ref`、B-positive replay result、A-only negative replay result。runner 必须断言 `stored_embedding_source_track_ref == resolver_target_ref` 且 `stored_embedding_source_track_ref != introducer_ref`，并从 `stored_crop_frame/path` 重算 `stored_crop_hash`。
- embedding provenance：store/DB 持久字段或表中的 `source_track_ref`、`source_frame_ref`、`crop_hash`、`crop_path_or_artifact_ref`、`resolver_target_ref`、`resolution_reason`、`embedding_type`、`embedding_model`、`embedding_version`、`embedding_dim`，以及与 person/scene + embedding 同事务提交的结果。report 必须列出本次检索使用的 type/model/version/dim filter，证明没有混查不同 embedding。
- B-positive / A-only replay 样本标注：每个样本的 A/B 人工标注、frame/window、截图或 frame path、期望结果和实际结果。
- bounded multi-person recognition 字段：`tracks_seen`、`tracks_eligible`、`tracks_queried`、`tracks_skipped_reason`、`attention_target_only=false`，证明不是 attention-only retrieval。
- runner-only envelope 摘要：`source_scene`、固定 teach frame、target fixture 输入、resolver evidence；这些不得混入正式 request body。
- teach/resolve/summary/link/correct API 调用结果；maintenance/backfill merge 如执行则单独列段，不混入主路径 timeline。
- 失败 response 字段：`retryable`、`ask_user_hint`、`ambiguity_type`。
- no-write cases 的 `store_delta` 和 `store_delta_source`：来源必须是 DB/store 前后快照或 transaction observer，覆盖 store/migration 暴露的 memory-owned table universe；allowed diagnostic whitelist 也从该权威来源读取。除白名单诊断/临时表外，每个 memory-owned table delta 都为 0，report 列出 universe/whitelist 的来源。
- DB transaction 结果：`create_person_with_embedding(...)`、`create_scene_with_embedding(...)` 或等价 transaction context 是否覆盖 profile/scene + embedding + provenance all-or-nothing，并包含 orphan row injection / failure 测试结果。
- replay assertions。
- memory event 计数、cooldown/drop 计数，并按 3 类事件分组。
- Botified stdout/frame 计数和 rate。
- main loop latency / 10Hz / gaze 不阻塞指标。
- bounded worker / retrieval 指标：embedding queue size、timeout/backpressure、max tracks per recognition tick、retrieval rate limit、worker backlog high-water mark。
- strict schema / debug channel gate 结果：生产 handler 禁止字段测试、默认 `debug_test_channel_enabled=false` 测试；如启用 debug/test channel，记录唯一 config gate、internal-only route、OpenAPI/生产 handler 不出现低层字段的断言。
- negative cases 结果。
- 如果启用实验性 scene region，记录正/负例结果以及 region query path/crop evidence，并标记为非 GA evidence。
- 每个失败 assertion 的 `failure_category` 和最小证据路径。
- `visual_evidence_index[]`：每个文件路径存在，且 key 能 join 到 assertion、teach/resolve response、event、stored crop、固定 frame/window 或 replay sample；每张图可通过 `assertion_id`、`request_snapshot_ref`、`source_frame_ref`、frame timestamp、snapshot `observed_at_ms`、track refs、stored crop hash/path、event_id 或 `memory_match_id` 对齐机器 report。

## 8. Legacy 验收记录（非当前 acceptance）

以下机器断言是历史验收记录，不能作为当前 active acceptance 或 handoff checklist。当前验收入口统一看 `docs/identity-overlay-product-development-plan.md`。

legacy core GA gate record：

- 实际发现的 `val-data` JPEG scenes 都被 replay，报告列出每个 scene 结果、JPEG 数量、`transcript_cases[]` 和 manifest legacy mismatch；如 auxiliary scene 加入 `val-data/`，发现数量自动变化并列出新增项，如仅作为外部 auxiliary local smoke scene 则单独列出且不改变 discovery 口径。
- 已完成 public `target.kind` 到内部 target 结构的 adapter / schema migration，`.transcript` 映射出的正式 request 可被当前服务执行；`region` 只作为测试-only fixture 或后续能力，不作为 stable agent-facing GA capability。
- 正式 REST contract 和示例只包含 `camera`、`target.kind`、`target.intent`、`target.referent_text`、profile/memory metadata；Pydantic / strict schema `extra=forbid`；生产 handler 单元测试禁止 `track_id`、`bbox`、`point_uv`、`test_hint`、`source_scene`、`source_frame`；debug/test channel 默认关闭且只由唯一 config gate 打开 internal-only route，默认 OpenAPI/生产 handler 不出现低层字段。
- server 内部 `MemoryFrameSnapshot` cache 包含 resolver 所需的 frame bytes/ref、timestamp、image size、tracks、bbox、keypoints、attention target 输入、event result、scene_context / engagement、track freshness / stability；它由 stream processor/session 在 `TrackSnapshot` 和 scene_context 仍在内存中时写入，不能由 `MemoryService` 从 public `visual_state` 重建；这些字段不进入 agent-facing request。
- teach / resolve 到达时绑定新鲜稳定的 request-arrival interaction snapshot；snapshot 过期、不稳定或无 `active_interaction_target` 时返回失败，不写库；teach/resolve response 和 report 包含本次实际使用的 `request_snapshot_ref`、`source_frame_ref`、frame timestamp、snapshot `observed_at_ms`、frame timestamp TTL、snapshot TTL、N-of-M stability window 和失败 reason `stale_interaction|no_active_interaction_target`，并可 join 到固定 frame/window 和 `visual_evidence_index[]`；不引入 `resolution_id` 或 token。
- teach API 是原子 `resolve + write`：teach 内部重新 resolve；store 层通过 `create_person_with_embedding(...)`、`create_scene_with_embedding(...)` 或等价 transaction context 让 profile/scene + embedding + provenance all-or-nothing；orphan row injection / failure 测试通过。
- fake backend 下 core 合同通过：self introduction、third-person introduction resolved/ambiguous、known replay、whole-scene teach/activation、ambiguous/no-write、object unsupported、bounded retrieval、nonblocking、Botified projection。
- local backend 下至少通过一次来自 `pic_teach_me` 的典型线下交互场景 self introduction `teach_person -> known_person_present`；self introduction 只接受新鲜、可互动、可见的 `active_interaction_target`，attention target 只作为派生输入，不走 raw attention target、单人、最近或最大候选 fallback；这不是合成治理用例。
- self introduction 负例必须覆盖用户说“请记住我/我是 xxx”但 `active_interaction_target` 不稳定或多人无法确定时返回 `ambiguous`，不写库，并带 `retryable=true`、`ask_user_hint=true`、`ambiguity_type=target_unclear`，response/report 记录可供 agent 追问重新说明、靠近或单独站位的失败事件。
- local backend 下至少通过一次真实模型 third-person introduction `pose_pointing_to_person -> teach_person -> known_person_present`；local 正例只能固定 frame/window，不能指定 B，且必须记录 `debug_test_channel_enabled=false`、`fixture_inputs_consumed=[]`（未消费 target fixture 输入）、`debug_fixture_used_for_target_resolution=false`。
- third-person 写库必须用 `stored_person_id`、`stored_embedding_source_track_ref`、`stored_crop_frame/path`、`stored_crop_hash`、`profile.display_name`、`resolver_target_ref`、`introducer_ref` 和持久 embedding provenance 证明写入的是 B 不是 A；provenance 必须包含 `embedding_type`、`embedding_model`、`embedding_version`、`embedding_dim`。runner 必须断言 `stored_embedding_source_track_ref == resolver_target_ref` 且 `stored_embedding_source_track_ref != introducer_ref`，并从 `stored_crop_frame/path` 重算 crop hash。
- third-person 必须通过有 A/B 标注的 B-positive replay 和 A-only negative replay；A 指向不清、候选接近、没有 `active_interaction_target`、手臂关键点不足或 snapshot stale 时返回 `ambiguous`，不写库，并返回 `retryable`、`ask_user_hint`、`ambiguity_type`。
- local backend 下至少通过一次真实模型 whole-scene `teach_scene -> scene_activated`。
- known-person recognition 对稳定可见 person tracks 做 bounded multi-person retrieval，不只查 attention target；检索必须按 `embedding_type`、`embedding_model`、`embedding_version`、`embedding_dim` 过滤，不混查不同 embedding，person/scene 可有不同 type/dim；report 必须包含 filter、`tracks_seen`、`tracks_eligible`、`tracks_queried`、`tracks_skipped_reason`、`attention_target_only=false`、max tracks、rate limit、worker backlog 和 cooldown。
- local/fake backend 下 ambiguous、not_found、conflict、object unsupported 等 no-write cases 都断言 `store_delta` 来源于 DB/store 前后快照或 transaction observer，覆盖 store/migration 暴露的 memory-owned table universe；allowed diagnostic whitelist 从同一权威来源读取，除白名单诊断/临时表外，每个 memory-owned table delta 都为 0，report 列出 universe/whitelist 来源。
- local/fake backend 下 `target.kind=object` 是 negative-only：返回 `status=not_found` 且带 `error_code=unsupported_target_kind` / reason；`pic_teach_item_phone` 不被当作 person/object memory 主线成功，不声称手机记忆可用。
- memory semantic event 子集只包含 `known_person_present`、`familiar_unknown_present`、`scene_activated`；teach/merge/correction/ambiguity 等结果只在 API response/report/evidence 中出现，既有非 memory semantic events 不在本条约束内。
- memory event 不误触发、不刷屏：无关场景不输出 confirmed memory event；同 person/scene/anonymous 在 cooldown 内不会重复输出 Botified frame。
- memory 慢不阻塞 10Hz/gaze：teach/write embedding 和 recognition retrieval 都走 bounded worker / 线程池或等价非阻塞路径；主链路 p95 latency 和 gaze stale 行为满足现有 PC gate；stdout/Botified 慢不能导致无界排队。
- local backend report 必须记录固定 teach/replay frame、固定 target source、固定 thresholds/config、pose pointing scoring 字段、失败归因字段和 `visual_evidence_index[]` 对齐字段。

legacy supporting contract/report gate record：

- `resolve-target` dry-run 返回 `resolved`、`ambiguous`、`not_found`，失败 response 包含 `retryable`、`ask_user_hint`、`ambiguity_type`，枚举值限定为 `introducer_unclear|target_unclear|pose_unclear|multiple_candidates|stale_interaction|no_active_interaction_target|unsupported_target_kind|quality_too_low`。self-introduction 指向不清时固定使用 `target_unclear`。
- duplicate teach strategy 通过 API/report 断言：同名/同 external ref 高置信已有 person 只更新 metadata；高置信同脸不同名在 teach 写入阶段返回 `conflict` outcome / `error_code` / HTTP 409，不新建；anonymous match 走 `teach_person -> merged_anonymous_person`；不自动创建 duplicate。
- `familiar_unknown_present`、teach auto merge anonymous、maintenance/backfill merge 单独测试、correct identity、conversation summary / compact background、external user link、`memory_context` 字段在 fake backend 下走同一 API/store/retriever/event/CLI projection 路径；local backend 可用固定样本或构造输入证明路径，不要求每个分支都成为重型真实模型 gate。`familiar_unknown_present` 是 supporting contract gate，不作为 core GA 发布能力宣称。
- `memory_context` 可解析、短小，不含图片、embedding、完整 tracks 或大段聊天原文。
- 如果实验性 scene region 被纳入报告，其断言必须证明 query 走 region crop/region path，不能只检查 event 带 `region_id`；该结果标为非 GA evidence。没有可靠 visual region hint 时必须返回 `ambiguous` 或 `not_found`，不写库。
- schema/API docs 和 CLI Botified 投影文档与实际 report 字段一致。

legacy 人工 visual evidence 关注点：

- 能看到 teach frame、payload 摘要、resolve-target 预览、replay event 和 Botified frame 的对应关系。
- 每张图叠加 `assertion_id`、`request_snapshot_ref`、`source_frame_ref`、frame timestamp、track refs、stored crop hash/path、event_id 或 `memory_match_id`，能和 `report.json` 中的同名字段对齐。
- `visual_evidence_index[]` 中每个文件都存在，且 key 能 join 到 assertion、teach/resolve response、event、stored crop、固定 frame/window 或 replay sample。
- 能快速检查至少一个 self introduction person 正例、一个 third-person introduction person 正例、一个 scene 整图正例、一个 familiar / teach auto merge 样例、一个 correction 样例、一个 ambiguity 样例、一个 negative 样例；third-person introduction 证据需要画出介绍人、被介绍人、手臂方向和候选分数，实验性 region 如启用则额外展示其正/负例 evidence。
- visual evidence 不作为硬 gate；它不能替代 report assertions。

交付表述限制：

- 可以说：PC 本地、指定 fake/local backend、指定 `val-data` 报告通过。
- 不可以说：真机通过、RK3588 通过、现场通过、物体记忆已可用、手机记忆已可用、手指级指向已可用、物体指向已可用、通用手势理解已可用。

## 9. Legacy Handoff Checklist（非当前 checklist）

以下 checklist 仅保留为历史实现参考；当前 handoff / QA checklist 以 `docs/identity-overlay-product-development-plan.md` 为准。

1. 确认当前 memory REST API 路由和 schema；缺失则补齐，不增加第二套入口。
2. 实现并验证 public `target.kind` 到内部 target / `target.mode` 的 adapter 或 schema migration。
3. 确认正式 REST contract 与示例不包含 `track_id`、`bbox`、`point_uv`、`test_hint`、`source_scene`、`source_frame`；Pydantic / strict schema `extra=forbid`，生产 handler 单元测试覆盖这些字段被拒绝或不可达。
4. 确认 runner-only envelope 不混入 agent-facing request body；如确需 debug/test channel，只能由唯一 config gate `memory.test_debug_channel.enabled=true` 打开 internal-only route，默认关闭、默认不进 OpenAPI/生产 handler。
5. 确认 stream processor/session 写入内部 `MemoryFrameSnapshot`，其中保存 frame bytes/ref、timestamp、image size、tracks、bbox、keypoints、attention target 输入、event result、scene_context / engagement、track freshness / stability；`MemoryService` 不从 public `visual_state` 重建这些字段，并只供内部 resolver 使用。
6. 确认 teach / resolve 使用 request-arrival interaction snapshot；response/report 记录实际使用的 `request_snapshot_ref`、`source_frame_ref`、frame timestamp、snapshot `observed_at_ms`、frame timestamp TTL、snapshot TTL、N-of-M stability window，并能 join 到固定 frame/window 和 `visual_evidence_index[]`；runner 用 stream barrier 或等价方式证明 REST 到达前预期 fixed frame/window 已进入 `MemoryFrameSnapshot` cache，没有 snapshot、snapshot stale 或无 `active_interaction_target` 时失败不写库，reason 为 `stale_interaction|no_active_interaction_target` 等固定枚举。
7. 确认 teach API 内部重新 resolve，并通过 `create_person_with_embedding(...)`、`create_scene_with_embedding(...)` 或等价 transaction context 原子写库；profile/scene + embedding + provenance all-or-nothing；`resolve-target` 只是 dry-run/debug/agent 追问辅助，不提供写库承诺。
8. 确认 intent-to-target resolver 接受 stable `target.kind=person|scene|object`，`region` 仅为测试-only fixture 或后续能力；正式 `status` 只输出 `resolved`、`ambiguous`、`not_found`。
9. 确认 `target.intent=self_introduction` 只使用新鲜、可互动、可见的 `active_interaction_target` 解析“我”；attention target 只作为派生输入，不走 raw attention target、单人、最近或最大候选 fallback。
10. 确认 `target.intent=third_person_introduction` 使用 `active_interaction_target` 作为介绍人 A，并用现有 `YOLOv8n-pose` keypoints + tracker + pose pointing resolver 解析 B；不新增模型，不依赖 speaker track 或音频定位，不允许 raw attention target fallback 写库。
11. 确认 pose pointing scoring 的手臂向量、torso center、N-of-M 稳定窗口、归一化阈值、左右臂冲突策略、候选 margin 都在 config defaults 和 report fields 中。
12. 确认 embedding provenance 持久保存 `source_track_ref`、`source_frame_ref`、`crop_hash`、`crop_path_or_artifact_ref`、`resolver_target_ref`、`resolution_reason`、`embedding_type`、`embedding_model`、`embedding_version`、`embedding_dim`；retriever/report 证明查询按 type/model/version/dim 过滤，不混查不同 embedding，person/scene 可有不同 type/dim。
13. 确认第三人称写库可机器证明写的是 B 不是 A：`stored_embedding_source_track_ref == resolver_target_ref` 且 `stored_embedding_source_track_ref != introducer_ref`，stored crop hash 可从 `stored_crop_frame/path` 重算，并要求有 A/B 标注的 B-positive replay 与 A-only negative replay。
14. 确认 teach/write embedding 和 bounded multi-person recognition 都走 bounded worker / 线程池或等价非阻塞路径；recognition 对稳定可见 person tracks 限流/限量 retrieval，不只查 attention target，report 包含 `tracks_seen`、`tracks_eligible`、`tracks_queried`、`tracks_skipped_reason`、`attention_target_only=false`。
15. 确认 duplicate teach strategy：同名/同 external ref 更新 metadata；同脸不同名在 teach 写入阶段返回 `conflict` outcome / `error_code` / HTTP 409，不新建；anonymous match 走 `teach_person -> merged_anonymous_person`；不自动 duplicate。
16. 确认 no-write cases 的 `store_delta` 来自 DB/store 前后快照或 transaction observer，覆盖 store/migration 暴露的 memory-owned table universe；allowed diagnostic whitelist 从同一权威来源读取，runner/assertion 不手写表名；除白名单诊断/临时表外 delta 全为 0，report 列出 universe/whitelist 来源。
17. 确认 object 是 negative-only；`target.kind=object` 返回 unsupported，不写库，不降级为 scene/region。确认 region 不作为 stable agent-facing GA capability。GA 示例和验收主线不得漂移到 region/object teaching；region/object 只作为 next/辅助测试素材，不作为 0.3/0.4 GA 交付承诺。
18. 确认 `familiar_unknown_present` 只作为 supporting contract/report gate，不作为 core GA 发布能力宣称。
19. 确认 CLI 对 memory semantic event 子集只消费 3 类 memory event，做 allowlist、幂等、rate limit 和 Botified projection，同时继续支持既有非 memory 事件。
20. 新增 GA runner 或扩展现有 runner，保留原 `run_memory_e2e` fake/synthetic 快速回归；不要把旧 `run_val_data_e2e` / manifest oracle 当作 memory/teaching interaction inventory。
21. 在 runner 中按实际目录发现列出并 replay `val-data/` JPEG scenes，采集 `.transcript` request、runner-only envelope、Botified stdout、timeline、report 和 visual evidence；如果 auxiliary scene 加入 `val-data/`，发现数量必须自动变化并列出。
22. 手工把同 stem `.transcript` 内容转换为正式 request；不要调用 LLM 自动解析。
23. 确认 `pic_teach_person` 或专门多人介绍场景满足 local third-person smoke；local 正例固定 frame/window，不能指定 B，report 记录 A/B 人工标注、期望 arm side、人工确认截图、`debug_test_channel_enabled=false`、`fixture_inputs_consumed=[]`（未消费 target fixture 输入）、`debug_fixture_used_for_target_resolution=false`。
24. 实现 core GA assertions：self introduction、third-person introduction、known replay、whole-scene、ambiguous/no-write、object negative、nonblocking；小单元矩阵覆盖 `MemoryFrameSnapshotCache` TTL/N-of-M、`active_interaction_target` no-fallback、pose pointing scorer、bounded worker/backpressure、retrieval caps/cooldown、store_delta observer。
25. 实现 supporting contract/report assertions：familiar、teach auto merge anonymous、maintenance/backfill merge 单独测试、correct、resolve-target response contract、summary、external link、duplicate teach strategy；region 只在启用实验 path 时作为额外 evidence。
26. 输出 `report.json`、`timeline.jsonl`、`teach_payloads.json`、`api_responses.jsonl`、`botified_frames.jsonl` 和 visual evidence；report 包含 `visual_evidence_index[]`，每张 visual evidence 图和 report 通过 `assertion_id`、`request_snapshot_ref`、`source_frame_ref`、frame timestamp、track refs、stored crop hash/path、event_id 或 `memory_match_id` 对齐。
27. 分别跑 fake backend 和 local backend；local backend 必须显式传模型路径，且只声称固定样本 smoke。
28. 检查 `val-data/`、runtime DB、模型、cache、artifacts 没有进入 Git。
29. 更新 server handoff、test plan、API/schema 文档和 CLI Botified 投影文档中的最小必要说明。
30. 最终汇报只声明 PC 本地结果，并附报告路径和未通过项；不要外推到真机或现场。
