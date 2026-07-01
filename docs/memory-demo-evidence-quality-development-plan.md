# Memory Demo Evidence Quality Development Plan

日期：2026-07-01

## 1. 背景与问题

当前 `artifacts/demo/memory` 的公开 memory demo 能跑出结果，但用户打开页面后很难建立信任：

- 页面信息太少。用户看不到自己说了什么、对应的是哪张 transcript 图片、系统实际用哪一帧做 evidence、overlay 又画在哪张图上。
- `visual-memory` / memory teaching evidence 中的 bbox、face bbox、pose skeleton 和 pointing ray 有时看起来画错。核心原因是 renderer 目前会优先使用 `result.source_image_path` 作为 overlay 底图，但很多 bbox / pose / face evidence 来自 `selected_window.frame` 或实际运行选中的帧，于是出现“在 transcript 图片上画了另一帧结果”的错位。
- memory local runner 对 self / third-person 等 case 会在场景里扫多帧，甚至扫整场景来找能通过的帧，而不是严格围绕 `.transcript` 同 stem 图片附近模拟用户消息。公开 demo 可能展示“用户在这张图说话”，但算法证据来自另一张图。用户肉眼会觉得不可信，也可能掩盖真实模型在对应时刻的失败。
- `run_memory_demo` 的公开 `report.json` 和 `index.html` 丢掉了很多可解释信息：用户消息、transcript 图片、实际 evidence frame、overlay 底图、frame delta、selected target、identity / familiar / scene / event 结果。

这不是单纯的绘图 bug。对用户来说，memory demo 的价值是证明“机器人看到用户正在指的那个人/当前场景，并据此记住或召回”。如果证据帧和用户消息图片对不上，即使算法内部偶然成功，demo 也会变成不可信的黑盒。

当前已定位到的实现根因：

- `tools/memory_teaching_evidence.py` 的 overlay 底图选择会优先使用 `source_image_path`，而不是优先使用 `selected_window.frame`。
- `tools/run_memory_teaching_ga_e2e.py` 的 self case 当前会扫描前若干帧，third-person case 会扫描整个 scene，scene / familiar case 当前偏向取 scene 第一帧。公开 demo 不应把这些宽搜索结果当作用户消息附近的证据。
- `tools/run_memory_demo.py` 生成的公开报告没有把 visual evidence、用户消息参考图、实际 evidence frame 和 overlay source frame 连接起来，并且公开 case 摘要倾向只保留 passed 项。

这些问题优先按 demo / evidence 链路修正。除非实现过程中确认生产 API 也会使用错误帧，否则不改变 server memory / identity 的业务逻辑。

## 2. 产品目标

让公开 memory demo 成为一个低心智负担、可肉眼核对的证据页：

- 用户能看到每个 case 的用户消息、用户消息对应图片、系统实际看的 evidence frame、两者关系和运行结果。
- 用户能看到每个 case 的期望结果和实际结果：系统本应选中谁 / 什么场景，本次实际选中了什么，是否真的完成了记忆写入和后续召回。
- bbox、face bbox、pose skeleton、pointing ray、identity / event label 只画在产生这些证据的同一张图或同一帧上。
- 当系统无法在用户消息对应图片或小范围邻近帧中解析目标时，demo 明确显示 `failed` / `insufficient visual evidence`，不悄悄换到远处帧让结果看起来通过。
- `report.json` 能解释每张图和每个结论来自哪一帧，方便研发复查。
- 公开入口继续只有两个：`tools/run_visual_demo.py` 和 `tools/run_memory_demo.py`。本计划只改 memory demo / evidence 质量，不新增产品入口；visual demo 不在本计划修改。

## 3. 非目标

- 不新增产品入口、Web 管理页、标注工具或排障平台。
- 不修改 visual demo。
- 不改 server identity / memory 业务逻辑、阈值、模型或 API 合同，除非团队确认发现的确是生产 bug。
- 不把 raw embedding、crop path、track id、bbox、keypoints 暴露给 agent-facing payload。`report.json` 和 demo 页面可以展示 bbox / overlay 信息，供人类检查。
- 不做像素黄金图，不做页面美术回归，不跑无关大套件。
- 不把历史 runner 或 evidence 工具重新包装成更多公开命令。

## 4. 设计原则

- KISS：一个公开 memory demo，一个 evidence frame 语义，一套页面解释方式。
- DRY：runner 负责产出事实字段，renderer 只消费事实字段并画图；不要在 renderer 里重新猜“哪一帧才对”。
- YAGNI：只补齐 demo 可信度所需的最小字段和页面，不做完整可观测系统。
- 一个功能一种做法：transcript 图片用于表达用户消息上下文，selected / evidence frame 用于画算法证据，overlay source frame 必须与证据同源。
- 低心智负担：页面直接说清楚 same / nearby / failed，不要求用户理解内部窗口、缓存或重放细节。
- 失败可见：对应窗口无法解析时明确失败，比扫远处帧伪造成功更有产品价值。
- 分层展示：主卡片先讲人话结论，技术细节放在 evidence detail 和 `report.json`，避免把 bbox、pose、frame 字段堆给非开发者。

## 5. 核心原则

同一张图上只能画来自同一张图或同一帧的以下证据：

- person bbox
- face bbox
- pose skeleton
- pointing ray
- selected target label
- identity label
- 已熟悉的未命名人物 label
- scene label
- event label

证据帧必须可追溯到用户消息对应图片或小范围邻近帧：

```text
transcript/source image  表达用户当时说话对应的参考图片
selected/evidence frame  表达系统实际用于解析目标、身份、场景或事件的帧
overlay source frame     表达 renderer 画 bbox/pose/face/event 的底图
```

规则：

1. `overlay source frame` 默认等于 `selected/evidence frame`。
2. `source image` 只作为用户消息参考图，不用于承载非本帧证据。
3. 如果 `selected/evidence frame` 与 `source image` 是同一帧，页面标记 `same`。
4. 如果两者在允许的小窗口内，页面标记 `nearby`，并展示 frame delta。
5. 如果该 case 没有用户输入，例如“见过但还不知道名字的人”，页面标记 `no_user_input`，并展示代表性 evidence frame。
6. 如果小窗口内无法解析，页面标记 `failed`，并展示失败原因，不继续扫整场景。

公开 demo 的 transcript 窗口固定为同 stem 图片前后最多 2 帧：

```text
candidate frames = anchor index - 2 到 anchor index + 2
```

页面必须显示实际使用的 `frame_delta`，例如 `same`、`+1 frame`、`-2 frames`。超过这个窗口的帧不能让公开 demo 通过。

## 6. Demo 必须展示的信息

memory demo 首页按四类主线组织，每张卡片都要让用户能肉眼判断“这件事是否可信”：

- self intro：用户介绍自己，系统应能看到当前交互人并写入 / 召回身份。
- third-person pointing teach：用户指向第三个人介绍身份，系统应能用 pose pointing 选择目标。
- scene teach：用户示教场景，系统应能展示场景参考图、evidence frame 和 scene / event 结果。
- 见过但还不知道名字的人：系统看到代表性未命名熟人图片，并展示结果摘要，避免只有文本状态。内部仍可使用 `familiar_unknown` 事件名，但页面主文案使用用户能懂的表达。

主卡片展示：

- 用户消息；没有用户消息的 passive observation case 显示“无用户输入，来自重复观察”。
- 用户消息对应图片，或 passive observation 的代表性 evidence frame。
- 系统实际看的 evidence frame。
- 两者关系：`same` / `nearby` / `no_user_input` / `failed`。
- 一句话结论：本次是否可信、为什么。
- 期望结果：本应选中的人 / 场景，以及本应写入或召回的内容。
- 实际结果：系统实际选中的目标、写入结果、后续召回或事件结果。
- overlay 图片，底图必须是 evidence frame。
- 失败原因或 `insufficient visual evidence`，如果该 case 没有足够证据。

evidence detail 和 `report.json` 展示：

- person bbox。
- face bbox。
- pose skeleton。
- pointing ray。
- selected target。
- identity / familiar / scene / event 结果。
- candidate frame 摘要和 frame delta。

并非每个 case 都一定有全部视觉元素。例如 scene teach 可能没有 pointing ray，“见过但还不知道名字的人”可能没有用户消息中的指向动作。页面应使用 `not applicable` / `not present` 清楚表达，而不是留空或伪造。

失败 case 不应从首页静默移除。公开 demo 的价值是展示真实能力边界，数据不足或模型失败也需要可见。

self intro 和 third-person pointing teach 必须展示 teach -> recall 链路：不只显示写入成功，还要显示后续 `known_person_present` 或等价召回结果。scene teach 必须展示 scene write -> `scene_activated` 链路。见过但还不知道名字的人必须展示重复 observation -> `familiar_unknown_present` 链路。

## 7. 业务逻辑边界

默认只改以下文件及对应小范围单元测试：

- `tools/run_memory_demo.py`
- `tools/run_memory_teaching_ga_e2e.py`
- `tools/memory_teaching_evidence.py`

允许新增或调整测试 fixture，但范围必须围绕本计划的帧语义、候选帧选择、report 字段和 HTML 展示。

不默认修改：

- server identity / memory 业务逻辑。
- 生产阈值、模型路径和模型行为。
- agent-facing REST / WebSocket 合同。
- `tools/run_visual_demo.py` 和 visual demo 页面。
- README、artifact、模型、runtime cache 或 val-data。

如果实现过程中发现现有 server 行为确实会在生产路径中错绑人、错选帧或输出错误事件，先把问题写清楚并让团队确认，再把 server 修复从本 demo 质量计划中拆出。

对 `tools/run_memory_teaching_ga_e2e.py` 的改动只收敛公开 memory demo 语义。这个文件里仍可能有开发用验证路径；除非它直接服务 `tools/run_memory_demo.py` 的公开输出，否则不要因为本计划顺手改变所有开发验证行为。

## 8. 实施步骤

### Step 1：统一帧语义字段

在 runner 和 renderer 之间明确区分三类图片 / 帧：

- `source_image_path`：继续表示用户消息对应图片，用于解释“用户当时在这张图说话”。公开报告可以把它展示为 `transcript_source_image`。
- `selected_window.frame`：继续表示系统实际用于目标选择、身份识别、场景匹配或事件判断的帧。公开报告可以把它展示为 `selected_evidence_frame`。
- `overlay_source_frame`：公开报告和 renderer 使用的显式底图字段，必须等于 `selected_window.frame` 或同一证据帧。

实现时不要保留两套互相竞争的字段语义。内部可以继续沿用现有字段名，公开 `report.json` 做一层清晰映射；renderer 不再猜测 `source_image_path` 是否可以当 overlay 底图。

`report.json` 中每个 case 记录最小可解释字段：

- `case_id`
- `case_type`
- `user_message`
- `transcript_source_image`
- `selected_evidence_frame`
- `overlay_source_frame`
- `frame_relation`
- `frame_delta`
- `expected_target`
- `actual_target`
- `expected_outcome`
- `actual_outcome`
- `verdict`
- `result_status`
- `failure_reason`
- `selected_target`
- `identity_result`
- `familiar_result`
- `scene_result`
- `event_result`
- `overlay_image`

`report.json` 可以记录 demo / report 所需的 bbox 和 overlay 信息；agent-facing payload 不新增 raw bbox、track id、crop path 或 embedding。

### Step 2：修正 overlay source 选择

调整 `tools/memory_teaching_evidence.py` 的底图选择：

1. 如果 case 有 `selected_evidence_frame`，优先使用它作为 overlay source。
2. 如果 case 已显式提供 `overlay_source_frame`，要求它与证据字段同源；不一致时在 report 中标记失败或降级为无 overlay。
3. `source_image_path` / transcript 图片只显示在“用户消息参考图”位置，不再用于画 bbox、face bbox、pose skeleton、pointing ray 或 event label。
4. 缺少 evidence frame 时，页面显示 `failed` / `insufficient visual evidence`，不把 transcript 图片拿来兜底画错位证据。

已有单元测试中如果断言“renderer 优先使用 `source_image_path` 作为 overlay 底图”，应改成断言“renderer 优先使用 `selected_window.frame` / evidence frame 作为 overlay 底图，`source_image_path` 只作为 reference image”。

### Step 3：修正 public memory demo 运行策略

调整 `tools/run_memory_demo.py` 和必要的 `tools/run_memory_teaching_ga_e2e.py` 复用逻辑：

1. 每个 transcript 从同 stem 图片开始找候选帧。
2. 只允许固定小邻近窗口用于稳定：同 stem 图片前后最多 2 帧。实现用一个清晰常量表达，例如 `PUBLIC_DEMO_FRAME_WINDOW_RADIUS = 2`。
3. 不允许为了公开 demo 成功而扫整场景或跨很远帧。
4. 如果对应窗口无法解析 self target、third-person target 或 scene evidence，case 结果为 `failed`，页面明确展示原因。
5. “见过但还不知道名字的人”不走 transcript window。它使用重复 observation 中的代表性 evidence frame，`frame_relation=no_user_input`。
6. 候选帧列表、anchor index、最终 selected frame、frame delta 和失败原因写入 `report.json`。

### Step 4：补全 public memory demo 页面

`artifacts/demo/memory/index.html` 仍是唯一 memory demo 人工入口。页面按四类主线展示：

- self intro。
- third-person pointing teach。
- scene teach。
- 见过但还不知道名字的人。

每张卡片至少包含：

- 用户消息文本。
- reference frame：用户消息对应图片。
- evidence frame：系统实际看的帧。
- frame relation：`same` / `nearby` / `no_user_input` / `failed`。
- 期望结果和实际结果。
- 一句话 verdict，例如“通过：选中右侧被介绍的人，并在后续画面召回为张三”。
- overlay：画有本帧 bbox / face / pose / ray / label 的图。
- selected target 摘要。
- identity / 已熟悉的未命名人物 / scene / event 结果摘要。
- failure reason，如果失败。

页面不要求用户理解内部对象 id。需要展示技术字段时，使用“reference frame”“evidence frame”“frame delta”“selected target”这类可解释名称。

HTML 中的 reference / evidence / overlay 图片必须是可打开的相对链接，不能要求用户理解本机绝对路径。坏图、缺图或不可打开图片都应计入 demo 失败原因。

### Step 5：补全 report.json 的最小可解释字段

公开 `report.json` 保持小而清楚：

- 顶层保留本次 demo 的 data dir、输出路径、case count、error count、模型路径和 runtime label。
- `cases[]` 是唯一 case 列表；首页卡片从它派生，避免 `demo_items` 和 `cases[]` 两套摘要漂移。
- `cases[]` 记录每个 case 的用户消息、reference frame、evidence frame、overlay source、frame relation、frame delta、候选帧摘要、expected target、actual target、expected outcome、actual outcome、verdict 和四类结果。
- demo / report 可以记录 bbox、face bbox、pose keypoints 的可视化摘要，用于复查 overlay 是否同源。
- agent-facing payload 不新增 raw embedding、crop path、track id、bbox 或 keypoints。

字段命名以可解释为先，避免把内部阶段性变量名直接暴露成产品页面文本。

路径规则：

- HTML 使用相对 artifact 路径，确保打开 `artifacts/demo/memory/index.html` 时图片可见。
- `report.json` 对 artifact 内文件使用相对路径，对 `val-data` 源图使用 data-dir 相对路径；必要时可保留绝对路径到内部 debug report，但不作为公开页面主信息。

### Step 6：补齐已熟悉的未命名人物图片证据

“见过但还不知道名字的人”demo 必须有代表性图片和结果摘要：

- evidence frame；没有用户消息 reference frame 时显示 `no_user_input`。
- overlay。
- 已熟悉的未命名人物状态；内部事件名仍可显示为 `familiar_unknown_present`。
- seen count / observed duration / familiar score，如果当前 report 已有。
- 如果没有足够图片或结果，显示 `failed` / `not present`，并解释缺失项。

不要只在页面上显示“familiar unknown passed”这类纯文本状态。

“见过但还不知道名字的人”没有用户消息参考图时，不要伪造 transcript。页面直接显示 `no_user_input`，并说明该 case 来自重复 observation。

## 9. 测试计划

采用核心逻辑 TDD，小范围覆盖高风险行为：

- renderer 使用 selected / evidence frame 作为 overlay 底图，而不是 transcript source image。
- runner 的候选帧只来自 transcript same-stem 和允许的小邻近窗口。
- 候选帧 helper 只返回 same-stem 前后 2 帧，窗口外帧即使可通过也不能让公开 demo passed。
- public `report.json` 包含用户消息、reference frame、evidence frame、overlay source、frame relation、frame delta、selected target 和结果摘要。
- public `report.json` 和 HTML 包含 expected target / actual target、expected outcome / actual outcome、verdict。
- failed / insufficient case 不会从 `cases[]` 或首页静默过滤。
- public HTML 包含关键解释字段：user message、reference frame、evidence frame、overlay、identity / 已熟悉的未命名人物 / scene / event result。
- 对应窗口无法解析时，结果展示 `failed` / `insufficient visual evidence`，不悄悄换远处帧。
- “见过但还不知道名字的人”case 至少有代表性图片和结果摘要，或明确失败原因。

不做：

- 像素黄金图。
- 测试测试工具本身。
- 页面美术快照。
- 无关 server / replay / model 大套件。

建议验证命令：

```bash
uv run --extra inference python tools/run_memory_demo.py --data-dir val-data
```

## 10. 验收标准

运行：

```bash
uv run --extra inference python tools/run_memory_demo.py --data-dir val-data
```

然后打开：

```text
artifacts/demo/memory/index.html
```

应能验收：

- 每个 case 都能看到 user message、reference frame、evidence frame、overlay、事件 / 身份结果。
- bbox、pose skeleton、face bbox、pointing ray 和 label 与 overlay 底图一致。
- `report.json` 能解释每张图来自哪一帧，以及 reference frame 与 evidence frame 的关系。
- 每个 case 能看到期望结果、实际结果和一句话 verdict，self intro / third-person pointing teach 能看到 teach -> recall，scene teach 能看到 scene write -> activation。
- self intro、third-person pointing teach、scene teach、“见过但还不知道名字的人”四类主线都有足够肉眼判断的信息。
- 失败 case 清楚显示失败原因，不以远处帧替代对应时刻。
- HTML 内所有 reference / evidence / overlay 图片链接可打开，不能出现坏图或要求用户理解本地绝对路径。
- 公开入口仍只有 `artifacts/demo/visual/index.html` 和 `artifacts/demo/memory/index.html`；本计划没有新增产品入口。

## 11. Review 结论

### 产品 review

- 通过标准：用户打开 memory demo 后，不需要读代码就能判断“用户消息图片”和“系统证据帧”是否一致或足够接近。
- 重点检查：页面是否直接暴露 same / nearby / no_user_input / failed，是否能看出期望结果和实际结果，失败是否可理解，“见过但还不知道名字的人”是否有图片证据。
- 收敛决定：公开 transcript 窗口固定为同 stem 前后 2 帧；超过窗口不算公开 demo 通过。

### 研发 review

- 通过标准：runner 不再为了公开 demo 扫整场景找成功帧；renderer 不再把非本帧证据画到 transcript 图片上。
- 重点检查：帧字段命名是否统一，report 字段是否足够解释问题，agent-facing payload 是否保持干净。
- 收敛决定：公开 report 以 `cases[]` 为唯一 case 列表；`source_image_path` 只作为 reference，`selected_window.frame` / explicit `overlay_source_frame` 才能作为 overlay 底图。

### 最终收敛决定

- memory demo 的公开可信度优先于表面通过率。
- 证据必须来自用户消息对应图片或小范围邻近帧。
- 画在图上的所有视觉证据必须与底图同源。
- 对应窗口无法解析时，公开显示失败，不换远处帧。
- demo 必须证明“视觉证据对齐”和“记忆链路结果”两件事：能看清系统选中了谁，也能看清是否写入并在后续召回。
