# Stable Memory/Teaching 发布计划

日期：2026-06-29

## 1. 产品目标

本阶段目标是把“记忆和示教”作为稳定可发布能力交付：用户通过 agent 明确示教当前画面中的人、整图场景或最小场景区域，服务端保存记忆；后续 replay 或在线流再次看到同一人、熟悉未知人或已教学场景/区域时，服务端输出低频 memory semantic event，CLI 只把 memory 事件投影成 Botified frame。

交付定义：

- PC 本地可以启用 memory，完成 `val-data/` 全 15 个场景的端到端测试并生成报告。
- REST API 是唯一正式示教入口；CLI 只消费 server memory events，不做身份判断、不写 memory DB、不提供 memory 管理命令。
- `teach_person`、`known_person_present`、`teach_scene`、`scene_activated`、`memory_context`、conversation summary / compact background、external user link、`familiar_unknown_present`、`merge-anonymous-person`、`correct-identity`、`resolve-target` 三态预览都完整可用。
- memory semantic event 子集只保留 3 个：`known_person_present`、`familiar_unknown_present`、`scene_activated`。
- merge、correction、resolve-target、teach 结果、歧义和证据只通过 API response、report 或 evidence 表达，不新增 `teach_succeeded`、`anonymous_merged`、`identity_corrected`、`target_ambiguous` 等 Botified 事件。
- 产出机器可读报告和人工 visual evidence；visual evidence 用于人工确认，不作为硬 gate。

本阶段只能声称“PC 本地通过”。不声称真机、RK3588 或现场通过。

## 2. 原则

- KISS：只保留一个 memory 侧链、一个 REST 示教入口、一个 SQLite/sqlite-vec 存储路径。
- DRY：身份识别、场景检索、cooldown、Botified 投影只实现一次；CLI 不复刻 server 规则。
- YAGNI：不建设治理平台、审核后台、复杂 oracle、manifest/audit 主线或对象记忆系统。
- 一个功能一种做法：示教走 server REST API；memory event 走 `visual_state.semantic_events`；CLI 对 memory semantic event 子集只做 allowlist、rate limit 和 Botified frame 投影。
- 不测试测试工具本身：新增 runner 只作为执行器，验收断言针对 server/API/DB/event/CLI 行为。
- 内部私有场景不以隐私合规为本阶段约束；主要风险是错认、错绑、误触发、刷屏和实时链路阻塞。
- `val-data/`、runtime DB、模型、cache、artifacts 不进 Git。

manifest mismatch 可以记录为数据清单风险，但不作为本计划主线，也不要求更新 `val-data/manifest.json` 才能交付。

## 3. 当前状态和缺口

已有基础：

- memory API、SQLite/sqlite-vec store、fake/local embedding backend 已存在。
- fake backend 可以支撑确定性的 memory API 和 E2E 检查。
- local 模型已有 smoke 路径，但需要用真实 `val-data` teach/replay 串成可交付证据。
- CLI 已能消费 memory semantic events，并把 compact `memory_context` 投影到 Botified frame。

主要缺口：

- 还缺真实 `val-data/` 全 15 场景的 memory E2E runner 或现有工具扩展。
- 还缺用 `des.txt` 手工构造 teach payload 的明确规则。
- 还缺 teach/replay/recognize/familiar/merge/correction/resolve-target/scene region/negative cases 的统一报告。
- 还缺可交付配置文档：如何启用 fake/local backend、DB 路径、模型路径、输出 artifact 路径。
- 还缺 visual evidence 串联：从 teach payload、replay timeline、memory event、Botified frame 到截图/帧样本。
- REST teach 入口需要作为正式入口写清楚并在 E2E 中验证；CLI 不允许成为示教入口。
- schema/API docs 和 CLI Botified 投影需要稳定化，避免下游依赖临时字段。
- 还缺稳定的 intent-to-target 解析合同：agent 只提供用户意图和目标类型，server 负责把意图关联到画面目标；解析失败或不确定时返回三态结果，不写库。

## 4. 稳定功能范围

### 4.1 示教和检索 API

`teach_person`

- 通过 `POST /v1/memory/teach/person` 教学当前画面目标人。
- teach 前必须支持 `resolve-target` 预览，返回 `resolved`、`ambiguous` 或 `not_found`。
- payload 中的用户指示、目标类型和元信息由测试或 agent 提供；agent 不需要也不应依赖 `track_id`、`bbox` 这类视觉内部状态。
- local backend 下可用脸部身份路径；没有可用脸、目标歧义、目标过期或质量不足时返回明确错误，不写入可识别身份。

`teach_scene`

- 通过 `POST /v1/memory/teach/scene` 教学当前整图场景或最小场景区域。
- 整图教学使用 `target.kind=scene`。
- 区域教学只做最小实现：支持 `target.kind=region`、`region_id`、`bbox` 或 normalized bbox、`camera`；bbox/point 可以作为 server 解析证据或测试 fixture，不代表 agent 需要提供这些视觉内部字段。
- 不做场景区域编辑器，不做对象记忆，不把局部物体教学声明为可用。

`resolve-target`

- 通过 `POST /v1/memory/resolve-target` 作为示教前安全预览。
- 返回三态：`resolved`、`ambiguous`、`not_found`。
- 稳定输入是 `target.kind=person|scene|region|object` 和可选 `referent_text`，例如“我”“这个人”“这里”“手机”。
- server 内部可以使用 attention target、scene context、tracks、bbox、pose/keypoints、region hints 和测试 fixture 解析候选；这些不是 agent 必须提供的字段。
- 手臂/姿态指向可以作为候选排序或 tie-break 证据，但不能单独在低置信情况下强行 `resolved`；不承诺 finger pointing、精确射线、跨帧手势动作理解或物体指向理解。
- response 必须返回 candidates、confidence、resolution_reason 和 evidence，便于 agent 在失败时追问用户。
- `ambiguous` 或 `not_found` 时 teach API 必须拒绝或要求明确目标，且不写库。

`merge-anonymous-person`

- 通过 `POST /v1/memory/merge-anonymous-person` 把已观察到的 anonymous profile 显式合并到正式 person。
- 不自动把 anonymous 命名为 person。
- merge 后同一 anonymous 不应再作为 anonymous familiar event 输出；如果身份路径足够清晰，可以输出正式 `known_person_present`。

`correct-identity`

- 通过 `POST /v1/memory/correct-identity` 接收 `memory_match_id` 和正确/否定信息。
- 记录 negative match 或 correction evidence，影响后续同错误匹配。
- 不做在线训练，不重写历史 embedding。
- correction 成功与否只在 API response/report/evidence 中体现，不新增 Botified 事件。

### 4.2 Intent-To-Target 解析合同

agent 只负责把用户语言整理成目标意图，visual-events 服务负责把目标意图和当前画面关联起来。系统可以使用 attention target、track、bbox、pose/keypoints、场景状态、region hints 和测试 fixture 做隐式解析，但不能静默猜错并写库。

稳定 agent 输入：

- `target.kind=person`：用户要教某个人，例如“我”“这个人”“这是彭刚”。
- `target.kind=scene`：用户要教当前整图场景，例如“这是银河通用的办公室”。
- `target.kind=region`：用户要教某个局部场景区域，例如“这里是货柜/门口/展示区”。
- `target.kind=object`：用户要教某个物体。本发布可以识别出该意图，但必须返回 `unsupported_target_kind` 或 `not_found`，不写 memory。

server 解析规则：

- `person` 主路径是 attention target：用于“请记住我”或当前正在与机器人互动的单一稳定目标。只有 `scene_context.engagement_state` 表示可互动、attention target 新鲜且目标可见时才能写入；否则进入候选解析。
- `person` 候选解析可以使用单人可见、track 稳定性、中心/近距关系、pose/手臂方向 tie-break 和测试 fixture。多人候选分数接近时必须返回 `ambiguous`，不能选择最大的人或最近的人强行写库。
- `scene` 直接解析为整图。
- `region` 只支持最小矩形区域：由 region hint、测试 fixture、显式 `point_uv` fallback 或 bbox 解析出来。没有可解析区域时返回 `ambiguous` / `not_found`。
- `object` 在本发布中明确拒绝，不降级写成 `region` 或 `scene`，除非 agent 重新发起 `target.kind=region|scene` 的请求。
- 所有 `resolved` 结果必须带 `resolution_reason`，例如 `attention_target`、`single_visible_person`、`pose_tiebreak`、`manual_bbox_fixture`、`scene_full_frame`、`region_fixture`。
- 所有 `ambiguous` / `not_found` / `unsupported_target_kind` 结果必须通过 API response 返回给 agent；agent 可以据此告诉用户“我没看清你指的是谁/哪里，请重新指定”。这些失败结果不进入 memory semantic event 流，也不写库。

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

- 同一未知人多次稳定出现、但还没有被命名或绑定正式 person 时输出。
- fake backend 和 local backend 至少要在 `pic_familiar_face` 或选定重复人物场景中证明只触发一次，后续受 cooldown 控制。
- merge 后不再输出同 anonymous 的 familiar event。

`scene_activated`

- replay 或在线流再次看到已教学整图场景或已教学 region 场景时输出。
- 整图事件必须覆盖 `teach_scene target.kind=scene`。
- region 事件必须带 `region_id` 和 `source_target_mode`，并有 query region crop 路径。
- 本期不新增 anonymous/familiar scene profile，也不做“重复出现后自动熟悉化”的整图场景能力；`scene_activated` 只表示已教学 scene/region 被激活。
- 不把局部物体或手势强行映射成 scene memory。

### 4.4 memory context 和用户链接

`memory_context`

- Botified frame 的 `visual_context.memory_context` 包含 agent 需要的短背景：人物 display name、description、summary、scene label、region label 等。
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
- 不承诺自动手势示教、自动理解手指方向、跨摄像头 ReID、管理后台、云同步、多租户或正式隐私治理。
- 不让 server/CLI 生成最终话术或决定机器人业务动作。
- CLI 不做身份判断、不写 DB、不做 memory 管理命令。

## 5. 架构和入口

实时主链路保持不变：

```text
DDS JPEG -> CLI -> server /v1/stream
  -> inference / tracking / attention / semantic events
  -> visual_state @10Hz -> CLI
  -> DDS gaze target + low-frequency Botified frame
```

memory 侧链：

```text
server recent frame cache
  -> REST teach / query / link / summary API
  -> target resolver
  -> embedding backend
  -> SQLite + sqlite-vec
  -> retriever / memory event generator
  -> visual_state.semantic_events
  -> CLI Botified projection
```

硬边界：

- server REST API 是示教入口。
- CLI 对 memory semantic event 子集只消费 server 确认的 3 类 memory events，做 allowlist、幂等、rate limit 和 Botified 字段投影。
- CLI 继续支持既有非 memory semantic events，例如 `waving`、`passing`、`left`；本计划不要求删除或收窄这些现有事件。
- CLI 不判断身份、不计算 familiar score、不解析 `des.txt`、不写 DB、不提供 merge/correction/link/summary 命令。
- embedding 慢或失败只能延迟或丢弃 memory event，不能阻塞 10Hz 主链路和 gaze。
- fake/local backend 共用同一 API、store、event 和 CLI 投影路径。

## 6. 实现步骤

### 6.1 最小必要代码

1. 补齐或确认 server REST API：
   - `POST /v1/memory/teach/person`
   - `POST /v1/memory/teach/scene`
   - `POST /v1/memory/person/{person_id}/conversation-summary`
   - `POST /v1/memory/link-external-user`
   - `GET /v1/memory/person/by-external-user/{external_user_ref}`
   - `POST /v1/memory/merge-anonymous-person`
   - `POST /v1/memory/correct-identity`
   - `POST /v1/memory/resolve-target`
2. 确保 `MemoryService` 只从 app-level recent frame cache 解析 target；无新鲜 stream、目标过期、歧义或质量不足时明确报错且不写入。
3. 确保 intent-to-target resolver 支持稳定输入 `target.kind=person|scene|region|object`，输出 `resolved`、`ambiguous`、`not_found` 或 `unsupported_target_kind`；teach 前预览和 teach API 使用同一套 resolver。
4. 确保 scene region 最小实现完整：
   - 这是新增实现任务，不只是 schema 字段声明。
   - teach 时必须保存 region `bbox` 或 normalized bbox、`camera` 和 `region_id`；bbox 坐标系必须在 API/schema 中固定。
   - `teach_scene` 支持 `target.kind=region` 写入，region 可以由 bbox、point fallback、region hint 或测试 fixture 解析；不实现区域编辑器。
   - query 时必须对已教学 region 逐个按保存的 bbox/camera crop 当前帧并生成 scene embedding，或实现等价且可验证的 region path；不能只用整图 embedding 后按 `region_id` 标注。
   - `scene_activated` 携带 `region_id`、`source_target_mode` 和最小 evidence。
   - report 必须证明 query 真的走了 region crop/region path，例如记录 `region_query_path=crop`、`crop_bbox`、`camera`、`embedding_source=region_crop`、候选 region 数和匹配分数；不能只检查 event 带 `region_id`。
   - 正例和负例都进入 fake/local 报告。
5. 确保 SQLite/sqlite-vec 是唯一正式检索实现；不要添加第二套手写向量检索主路径。
6. 确保 `known_person_present`、`scene_activated`、`familiar_unknown_present` 事件带稳定 evidence，且进入同一 cooldown/rate-limit 路径。
7. 确保 CLI allowlist 对 memory semantic event 子集只允许 3 类 memory events，并稳定投影 `memory_context`；CLI 不新增身份逻辑，也不删除既有非 memory 事件支持。
8. 更新 schema/API docs 和 CLI Botified projection docs，列出稳定字段、错误码和 evidence/report 字段边界。

### 6.2 配置

新增或整理一份可交付配置示例，覆盖：

- `memory.enabled=true`
- `memory.db_path=runtime/memory/visual_memory.sqlite3`
- `memory.embedding.backend=fake|local`
- local person model path 和 scene model path 必须显式传入；server 不隐式下载模型。
- artifact 输出默认在 `artifacts/memory-teaching-ga/`。

runtime DB、模型和 artifact 继续保持 gitignored。

### 6.3 测试工具

建议新增 `tools/run_memory_teaching_ga_e2e.py`，或扩展现有 `tools/run_memory_e2e.py` 增加 `--ga-val-data-suite` 模式。

要求：

- 不破坏 `run_memory_e2e` 现有 fake/synthetic 价值；原有命令继续用于快速确定性回归。
- GA runner 使用真实 `val-data/` 全 15 场景，是本期新增交付；它不是现有 `run_memory_e2e` 的 synthetic/fake 快速回归。
- `run_memory_e2e` 只能继续作为 synthetic/fake 快速回归，不能等同于真实 `val-data` 全量 gate。
- 现有 `run_val_data_e2e` 旧 7 场景覆盖不足，不能作为 memory/teaching 全量 gate；可以复用工具代码，但验收必须以全 15 场景 GA runner 报告为准。
- runner 负责驱动 server、发送 stream frame、调用 REST teach/resolve/summary/link/merge/correct API、replay、采集 memory events、采集 CLI Botified stdout。
- runner 根据每个场景的 `des.txt` 手工映射 payload 驱动 teach，不调用 LLM 解析；测试 fixture 可提供固定 bbox/point/track 作为 server 解析证据，但 report 必须标记为 `test_hint`，不能伪装成 agent 提供的字段。
- runner 输出机器可读 `report.json`、事件 timeline、teach payload 记录、API response 记录、Botified stdout/frames、失败样本列表和 visual evidence 路径；这些 report/timeline/evidence 是本期 GA runner 交付的一部分。
- runner 不更新 `val-data/`，不修改 manifest，不把 artifact 写进 Git。

### 6.4 Visual evidence

补充或复用 visual evidence 工具，生成：

- teach frame 缩略图和对应 payload 摘要。
- resolve-target 预览结果：`resolved`、`ambiguous`、`not_found`。
- replay timeline：帧号、场景名、memory event、confidence、cooldown 状态。
- memory event 到 Botified frame 的对应关系。
- region 正例/负例 crop 或样本索引。
- negative cases 截图或样本索引。

visual evidence 是人工验收材料，不作为 CI gate。机器断言仍由 report 决定。

### 6.5 文档更新

实现完成后更新：

- server handoff：启动方式、memory 配置、local 模型路径、报告位置。
- development test plan：加入 stable memory/teaching 的手工/本地 gate。
- API 文档或 schema：列出正式 REST teach、resolve、summary、link、merge、correct 入口。
- CLI Botified 投影文档：列出 memory semantic event 子集只消费 3 类事件、既有非 memory 事件继续支持，以及 `memory_context` 的稳定字段。

不需要新增 governance/audit 文档主线。

## 7. 验证方案

### 7.1 数据集范围

必须使用 `val-data/` 全 15 个场景：

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

全场景都要 replay 并进入报告。不是每个场景都必须触发 memory event；负例不触发同样是断言。

### 7.2 `des.txt` 使用规则

当前有 4 个交互场景带 `des.txt`：

- `pic_teach_me/des.txt`：`请你记住我，我是小李飞刀`
- `pic_teach_person/des.txt`：`这是彭刚，请你记住`
- `pic_teach_scene_galbot/des.txt`：`这是银河通用的办公室，请你记住`
- `pic_teach_item_phone/des.txt`：`这是手机，请你记住`

使用方式：

- 测试中可以把 `des.txt` 手工编写成 REST API payload 的用户指示或元信息。
- 不自动用 LLM 解析 `des.txt`。
- 每个 payload 必须显式填写 `camera`、`target.kind`、`target.referent_text` 和 `source_scene`；agent-facing payload 不要求 `track_id`、`bbox` 或 `point_uv`。
- `pic_teach_me` 映射为 `target.kind=person`，`target.referent_text=我`，`profile.display_name=小李飞刀`，`profile.description` 保留原句或测试备注。server 优先用 attention target 解析。
- `pic_teach_person` 映射为 `target.kind=person`，`target.referent_text=彭刚/这个人`，`profile.display_name=彭刚`，`profile.description` 保留原句或测试备注。server 可以用 attention、单人/多人候选、pose tie-break 或测试 fixture 解析；不确定时返回 `ambiguous`。
- `pic_teach_scene_galbot` 映射为 `target.kind=scene`，`memory.title=银河通用办公室`，`memory.description` 保留原句。
- region payload 使用 `target.kind=region`，并填写 `region_id`、`referent_text`、`camera`、`memory.title`、`memory.description`；测试 fixture 可以提供 `test_hint.bbox_norm` 帮助 server 在固定数据上解析 region。
- `pic_teach_item_phone` 映射为 `target.kind=object`，`target.referent_text=手机`。预期结果是 `unsupported_target_kind` 或 `not_found`，不写 object memory，也不降级为 scene/region memory；report 中记录 `expected_negative=true` 和拒绝原因。

payload 示例：

```json
{
  "camera": "front",
  "target": {
    "kind": "person",
    "referent_text": "我",
    "source": "des_txt_manual_mapping",
    "source_scene": "pic_teach_me",
    "source_frame": 42
  },
  "profile": {
    "display_name": "小李飞刀",
    "description": "请你记住我，我是小李飞刀"
  }
}
```

```json
{
  "camera": "front",
  "target": {
    "kind": "region",
    "referent_text": "这里",
    "source": "des_txt_manual_mapping",
    "source_scene": "pic_teach_scene_galbot",
    "source_frame": 30,
    "region_id": "galbot_office_region_1",
    "test_hint": {
      "bbox_norm": [0.10, 0.20, 0.72, 0.82]
    }
  },
  "memory": {
    "title": "银河通用办公室",
    "description": "这是银河通用的办公室，请你记住"
  }
}
```

### 7.3 场景组合

teach/replay/recognize：

- 从 `pic_teach_me` 先调用 `resolve-target`，得到 `resolved` 后 teach person；再 replay 同场景和选定相似人物场景，断言 `known_person_present` 至少一次，且 `memory_context.display_name` 正确。
- 从 `pic_teach_person` 先调用 `resolve-target`，得到 `resolved` 后 teach person；再 replay 同场景，断言 `known_person_present`。
- 从 `pic_teach_scene_galbot` teach 整图 scene；再 replay 同场景，断言 `scene_activated`。
- 对已教学整图场景 replay，断言已教学场景可以触发 `scene_activated`，且无关场景不触发。

scene region：

- 使用 `pic_teach_scene_galbot` 或选定稳定场景，写入一个最小 region：`region_id`、normalized bbox、`camera`。
- `teach_scene` 支持 `target.kind=region` 写入；如果 region 由 point/bbox/test fixture 解析，report 必须记录最终解析到的 bbox/region。
- query 必须逐个 taught region crop 当前帧并走 region embedding/path；正例 replay 触发 `scene_activated`，事件包含 `region_id` 和 `source_target_mode`。
- report 必须记录 region query evidence：teach bbox/camera、query crop bbox/camera、region crop 文件或帧索引、embedding source、candidate count、match score、threshold 和是否通过；不能只用 event 是否带 `region_id` 作为断言。
- 负例 replay 不触发该 region 的 `scene_activated`。
- 不实现区域编辑器，不把 phone 或其他物体 region 声称为 object memory。

familiar：

- 使用 `pic_familiar_face` 或多段同一未知人场景，在未命名状态下多次 replay。
- 达到阈值后断言最多触发一次 `familiar_unknown_present`，后续受 cooldown 控制。

merge：

- 对已触发的 anonymous profile 调用 `merge-anonymous-person` API，创建或合并到正式 person。
- replay 后断言不再输出同 anonymous 的 familiar event；如果身份路径足够清晰，可以输出正式 `known_person_present`。
- merge 结果只进入 API response/report/evidence，不进入 memory semantic event 流。

correction：

- 对一次错误或构造的 `memory_match_id` 调用 `correct-identity` API。
- replay 后断言不再高置信返回同一错误 person。
- correction 结果只进入 API response/report/evidence，不进入 memory semantic event 流。

ambiguity：

- 使用 `pic_people_gathering` 或多人接近场景调用 `resolve-target`。
- 目标不确定时返回 `ambiguous`；teach API 不写入 memory。
- `target_ambiguous` 不作为 Botified event 输出。
- `ambiguous` / `not_found` / `unsupported_target_kind` response 必须能被 agent 用来追问用户，但不写库、不进入 memory semantic event。

negative cases：

- `pic_teach_item_phone` 不应触发 person identity teaching 成功；也不应因为“手机”文本产生 object memory。
- `target.kind=object` 必须返回 `unsupported_target_kind` 或 `not_found`，且不降级成 scene/region memory。
- `pic_leave`、`pic_walk_away` 等离开场景不应持续刷出已知人物事件。
- 与已教学人物/场景/region 不相似的场景不应输出 confirmed memory event。
- 低相似度、低质量、无新鲜 frame、过期 `track_id` 都必须返回明确错误或无事件。

### 7.4 fake backend 和 local backend

fake backend 证明完整合同：

- REST API、intent-to-target resolver 三态、SQLite/sqlite-vec、retriever、event generator、cooldown、CLI Botified 投影的确定性闭环正确。
- teach person、known person、teach scene、scene activated、familiar、merge、correction、summary、external user link、ambiguity、negative 和 scene region 合同正确。
- 只输出 3 类 memory semantic events；teach/merge/correction/ambiguity 等结果只在 API response/report/evidence 中出现。
- 适合 CI 或快速本地回归。
- fake backend 是完整合同 gate：API schema、状态流、store、event、cooldown、CLI projection、report 字段都必须覆盖；它可以使用构造数据和固定样本保证确定性。

local backend 证明：

- 在 PC 本地显式模型路径下，真实视觉 embedding 能完成 `val-data` teach/replay smoke。
- 至少证明 `teach_person -> known_person_present`。
- 至少证明 `teach_scene -> scene_activated`。
- 至少证明 `familiar_unknown_present`。
- 至少证明 merge 后不再输出同 anonymous familiar event。
- 至少证明 correct 后不再高置信错认同一 person。
- 至少证明 ambiguous 不写库。
- 至少证明一条 scene region 正例和一条 scene region 负例。
- local backend 不隐式下载模型，不写系统目录，不改变 fake backend 的测试价值。

local backend 可重复判定规则：

- local backend 是 smoke gate，不要求覆盖 fake backend 的每个构造分支，但必须使用同一 API、store、event 和 CLI projection 路径。
- teach frame 和 replay frame 必须固定：每个断言记录 `teach_frame_index`、`replay_frame_index` 或固定 frame 文件名；不能依赖“跑到哪里算哪里”的自然波动。
- target source 必须固定：使用固定 bbox/bbox_norm、track id fixture 或固定 `resolve-target` 输入；report 记录 `target_source` 和 `target_source_frame`。
- thresholds/config 必须固定并写入 report：person match threshold、scene match threshold、region match threshold、familiar threshold、cooldown、model path、embedding backend、random seed 或 deterministic flag。
- resolver evidence 必须固定并写入 report：`target.kind`、`referent_text`、candidates、confidence、resolution_reason、pose/attention/test_hint 是否参与。
- correct/negative 可以用构造数据或固定样本触发，不要求等待真实模型自然错认；必须记录构造方式、输入 `memory_match_id` 或 negative pair、期望结果和实际结果。
- scene region 正负例必须使用固定 teach/replay frame 和固定 bbox；report 记录 crop evidence。
- 每个失败 assertion 必须带失败归因字段，例如 `failure_category=model_low_confidence|target_not_found|target_ambiguous|threshold_mismatch|event_missing|event_unexpected|cli_projection_missing|region_path_not_used|cooldown_failed|api_error`，以及最小证据路径。
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

- data dir、场景列表、实际跑到的场景数，必须覆盖全 15 个 `val-data` 场景。
- backend 类型、模型路径是否显式配置、是否 real model、固定 thresholds/config。
- 每个 des.txt 场景的手工 payload 摘要：`camera`、`target.kind`、`referent_text`、target 来源、固定 teach frame、resolver evidence、`profile.display_name` 或 `memory.title/description`。
- teach/resolve/summary/link/merge/correct API 调用结果。
- replay assertions。
- memory event 计数、cooldown/drop 计数，并按 3 类事件分组。
- Botified stdout/frame 计数和 rate。
- main loop latency / 10Hz / gaze 不阻塞指标。
- negative cases 结果。
- scene region 正例和负例结果，以及 region query path/crop evidence。
- 每个失败 assertion 的 `failure_category` 和最小证据路径。
- visual evidence 路径。

## 8. 验收标准

机器断言必须满足：

- 全 15 个 `val-data` 场景都被 replay，报告列出每个场景结果。
- fake backend 下完整稳定合同全部通过：teach person、known person、teach scene、scene activated、familiar unknown、merge anonymous、correct identity、intent-to-target 三态、summary、external link、scene region、Botified projection。
- local backend 下至少通过一次真实模型 `teach_person -> known_person_present`。
- local backend 下至少通过一次真实模型 `teach_scene -> scene_activated`。
- local backend 下至少通过一次 `familiar_unknown_present`。
- local backend 下 merge 后不再输出同 anonymous familiar event。
- local backend 下 correct 后不再高置信错认同一 person。
- local backend 下 ambiguous 不写库。
- local/fake backend 下 `target.kind=object` 不写库，并返回 `unsupported_target_kind` 或 `not_found`。
- local/fake backend 至少各有一条 scene region 正例和一条 scene region 负例。
- `pic_teach_item_phone` 不被当作 person/object memory 主线成功，且不声称手机记忆可用。
- memory semantic event 子集只包含 `known_person_present`、`familiar_unknown_present`、`scene_activated`；既有非 memory semantic events 不在本条约束内。
- local backend report 必须记录固定 teach/replay frame、固定 target source、固定 thresholds/config 和失败归因字段。
- scene region 断言必须证明 query 走 region crop/region path，不能只检查 event 带 `region_id`。
- API response/report/evidence 可以记录 teach/merge/correction/ambiguity 结果，但这些不进入低频 Botified event 流。
- target failure response 可以驱动 agent 追问用户，但不作为 Botified memory event 输出。
- memory event 不误触发：无关场景不输出 confirmed `known_person_present`、`familiar_unknown_present` 或 `scene_activated`。
- memory event 不刷屏：同 person/scene/region/anonymous 在 cooldown 内不会重复输出 Botified frame。
- `memory_context` 可解析、短小，不含图片、embedding 或完整 tracks。
- memory 慢不阻塞 10Hz/gaze：embedding worker backlog 有上限；主链路 p95 latency 和 gaze stale 行为满足现有 PC gate；stdout/Botified 慢不能导致无界排队。
- schema/API docs 和 CLI Botified 投影文档与实际 report 字段一致。

人工 visual evidence 必须满足：

- 能看到 teach frame、payload 摘要、resolve-target 预览、replay event 和 Botified frame 的对应关系。
- 能快速检查至少一个 person 正例、一个 scene 整图正例、一个 scene region 正例和负例、一个 familiar/merge 样例、一个 correction 样例、一个 ambiguity 样例、一个 negative 样例。
- visual evidence 不作为硬 gate；它不能替代 report assertions。

交付表述限制：

- 可以说：PC 本地、指定 fake/local backend、指定 `val-data` 报告通过。
- 不可以说：真机通过、RK3588 通过、现场通过、物体记忆已可用、手机记忆已可用、自动手势/姿态指向已可用。

## 9. Handoff Checklist

1. 确认当前 memory REST API 路由和 schema；缺失则补齐，不增加第二套入口。
2. 确认 `MemoryService` recent frame cache、target resolver、SQLite/sqlite-vec、retriever 和 event generator 共用同一条路径。
3. 确认 intent-to-target resolver 接受 `target.kind=person|scene|region|object`，输出 `resolved`、`ambiguous`、`not_found` 或 `unsupported_target_kind`，且失败状态不写库。
4. 确认 scene region 最小实现完成：`target.kind=region`、`region_id`、bbox/normalized bbox、`camera`、point/bbox/test fixture 到 region 的解析、region crop query、`scene_activated.region_id`、`source_target_mode`、正负例验证。
5. 确认 CLI 对 memory semantic event 子集只消费 3 类 memory event，做 allowlist、幂等、rate limit 和 Botified projection，同时继续支持既有非 memory 事件。
6. 新增 GA runner 或扩展现有 runner，保留原 `run_memory_e2e` fake/synthetic 快速回归；不要把旧 `run_val_data_e2e` 7 场景当作 memory/teaching 全量 gate。
7. 在 runner 中列出并 replay `val-data/` 全 15 场景，采集 des.txt payload、Botified stdout、timeline、report 和 visual evidence。
8. 手工把 4 个 `des.txt` 内容转换为 teach payload；不要调用 LLM 自动解析。
9. 实现 person teach/replay、scene whole-image teach/replay、scene region teach/replay、familiar、merge、correct、resolve-target、ambiguity 和 negative assertions。
10. 输出 `report.json`、`timeline.jsonl`、`teach_payloads.json`、`api_responses.jsonl`、`botified_frames.jsonl` 和 visual evidence。
11. 分别跑 fake backend 和 local backend；local backend 必须显式传模型路径。
12. 检查 `val-data/`、runtime DB、模型、cache、artifacts 没有进入 Git。
13. 更新 server handoff、test plan、API/schema 文档和 CLI Botified 投影文档中的最小必要说明。
14. 最终汇报只声明 PC 本地结果，并附报告路径和未通过项；不要外推到真机或现场。
