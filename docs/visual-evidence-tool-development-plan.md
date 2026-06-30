# Visual Evidence Tool Development Plan

> Note: 本文描述通用视觉 evidence 工具；Identity Overlay、memory teaching evidence 和示教证据的当前 active source of truth 是 [Identity Overlay 产品与开发计划](identity-overlay-product-development-plan.md)。

## 1. 产品目标和非目标

### 目标

新增一个独立的手工 demo/debug/review 命令，用于把已有 replay artifact 或 `visual_state.jsonl` 转成可由人眼快速检查的视觉证据包。默认不连接 server；只有显式传入 `--run-replay --server ...` 时才在线生成 replay artifact。

证据包必须帮助非实现者快速回答三个问题：

- 画面里检测到了谁？这里的“谁”指 bbox、track id、person slot、事件关联对象，不做身份识别承诺。
- 系统现在看谁，为什么不看？展示 attention target、attention point、engagement state、no-engage reasons。
- 为什么这一帧触发或没触发事件？展示事件 badge、底部事件条、scene state、关键 evidence 摘要，并把长 evidence 和 raw JSON 放到 HTML 中深挖。

工具只服务实时视觉链路，页面和图片展示范围限定为：

- `detection`：人框、置信度、person count。
- `tracking`：track id、lost/age 等当前 overlay 已支持或能从 `tracks` 中稳定读取的信息。
- `attention`：当前注意力目标、目标点位、目标 track id、attention 是否可用。
- `scene_context`：engagement state、no-engage reasons、reacquire 摘要等人眼 debug 需要的信息。
- `semantic_events`：事件类型、track id、事件 evidence 摘要、触发帧；图片底部画短摘要，HTML 中能定位到事件所在 scene/frame。

输出标准是“打开 HTML 后能解释服务端实时视觉行为”，不是机器判分。`summary.json` 只做导航和排查辅助，不作为默认 gate；不根据视觉统计判定 pass/fail。

### 非目标

- 不覆盖 memory teach、用户示教脚本、元信息写入、known-person 召回脚本。
- 不做 profile/embedding gallery。
- 不做身份纠错 UI。
- 不做标注平台。
- 不做 Web dashboard。
- 不引入新模型或新规则。
- 不替代或重塑 `tools/run_val_data_e2e.py` 的既有职责；本计划不调整它。
- 不实现第二套绘图逻辑；`tools/visualize_service_replay.py` 继续是单 scene 调试入口，overlay/HTML helper 只保留一份。
- 不追求视觉美术测试和像素级回归测试。

## 2. 推荐命令和输出目录结构

### 默认离线模式

默认从已有 replay artifact 或 `visual_state.jsonl` 生成证据：

```bash
uv run --extra inference python tools/generate_visual_evidence.py \
  --visual-state-jsonl artifacts/replay/visual_state.jsonl \
  --data-dir val-data \
  --out artifacts/visual-evidence
```

也可传 artifact 目录，由工具在目录内查找 `visual_state.jsonl`：

```bash
uv run --extra inference python tools/generate_visual_evidence.py \
  --replay-artifact artifacts/replay \
  --data-dir val-data \
  --out artifacts/visual-evidence
```

默认行为：

- 不连接 WebSocket，不启动 replay。
- 读取 `visual_state.jsonl` 中的 `scene`、`frame_id`、`latency_ms`、`response`。
- `--visual-state-jsonl` 只支持 `tools.replay_val_data` 产出的 wrapped JSONL 形状：每行包含 `scene`、`frame_id`、源帧信息或可映射源帧的信息、`latency_ms`、`response`。不兼容 `tools/visualize_service_replay.py` 产出的 raw visual_state-only JSONL，避免一条命令支持两种输入语义。
- 使用 `tools.replay_val_data.discover_scene_dirs()` 和 `iter_scene_frames()` 重建 scene/frame 顺序、source image path 和 header 语义。
- 输出目录可覆盖写入，但不得写入 `val-data` 内部。
- 如果 artifact 缺失或无法映射到源帧，直接失败并给出缺失项，不猜测路径。

### 显式在线模式

只有需要重新跑服务端时才使用在线模式：

```bash
uv run --extra inference python tools/generate_visual_evidence.py \
  --run-replay \
  --server ws://127.0.0.1:8765/v1/stream \
  --data-dir val-data \
  --out artifacts/visual-evidence \
  --camera front \
  --fps 10 \
  --head-motion stationary \
  --response-timeout-ms 10000
```

在线模式要求：

- `tools/generate_visual_evidence.py` 是薄 wrapper。
- 复用 `tools.replay_val_data.replay_scene()`、`discover_scene_dirs()`、`iter_scene_frames()` 和 JSONL 写入形状。
- 帧顺序、header 构造、WebSocket 发送/接收、超时记录都由 `tools.replay_val_data` 承担。
- 新工具不 import `websockets`，不直接调用 `encode_frame_message()`，不重写 WebSocket replay loop。
- 这里的“不重写 replay loop”只约束新工具：本计划不要求重构或删除现有单 scene `tools/visualize_service_replay.py` 的交互式 WebSocket loop，避免把 visual evidence 计划扩大成历史工具重构。
- 在线 replay 结束后，回到同一套离线证据生成路径处理刚产出的 `visual_state.jsonl`。

推荐输出结构：

```text
artifacts/visual-evidence/
  index.html
  summary.json
  visual_state.jsonl
  scenes/
    pci_stand/
      index.html
      summary.json
      visual_state.jsonl
      frames/
        000000_img_....jpg
      states/
        000000_img_....json
    pic_hello/
      ...
```

说明：

- 根 `index.html` 是入口页面，包含全量 summary、scene 列表、事件索引和关键帧入口。
- 每个 scene 的 `index.html` 展示该 scene 的逐帧证据卡片。
- 根 `visual_state.jsonl` 保留 replay JSONL 语义；scene 级 `visual_state.jsonl` 只包含当前 scene，便于局部排查。
- `summary.json` 是 HTML 的稳定数据来源，也是轻量测试的主要断言对象，但它不是质量 gate。实现时不要复用 `tools.replay_val_data` 中带 `passed` 或 gate 语义的 summary 字段；visual evidence summary 只从 JSONL / raw `visual_state` 聚合导航信息。

## 3. 页面和图片必须显示的信息

### 标注图片

每张输出图片必须基于当前 `tools/visualize_service_replay.py` overlay 能力绘制，并补齐全量工具需要的字段：

- bbox/track label：`track_id`、`confidence`、`lost_ms`，必要时显示 runtime person slot，保持已有颜色分配。
- head point：如 `head_uv` 可用，继续绘制。
- attention point：如 `attention.target_uv` 可用，绘制十字和 `target_track_id`；如 attention 不可用，在图片 header 或 HTML 卡片中显示 unavailable/null。
- event badge 和底部事件条：当前帧有 `semantic_events` 时，在图片底部画短摘要，最多展示前几条高信号事件。
- scene state：`frame_id`、`camera`、`scene`、`person_count`、`largest_person_stable`、`engagement_state`、`no_engage_reasons`、reacquire 摘要。

semantic events 底部展示建议：

```text
event=person_waving track=7 confidence=0.92 evidence=runtime_person_slot=2 wave_duration_ms=1200
event=attention_target_changed track=7 evidence=switch_reason=...
```

图片只放短摘要。如果 event evidence 很长，或 raw `visual_state` 很大，必须放到 HTML `<details>` 和 `states/*.json`，不要盖满图片、遮挡 bbox/attention。

### HTML

根 `index.html` 必须支持从 summary 定位到事件帧：

- 顶部展示全量统计 summary。
- scene 表格展示每个 scene 的帧数、person count 摘要、track 数、attention 可用比例、事件计数、关键帧数量。
- semantic event timeline：按事件类型和 scene/frame 列出事件，每项链接到对应 scene HTML 的 frame anchor。
- 关键帧列表：每个 scene 至少列出首帧、有 person 的首帧、attention 首次可用帧、每类 semantic event 首次触发帧、最后一帧。

scene `index.html` 必须支持：

- 每帧一个稳定 anchor，例如 `#scene-pic_hello-frame-000012`。
- 图片、source path、frame id、timestamp、latency、track ids、attention target、scene_context 摘要、semantic_events 摘要同屏展示。
- `scene_context` 摘要必须把 `engagement_state` 和 `no_engage_reasons` 转成人类可读文案，不能只裸露字段名。
- `semantic_events` 有独立可搜索文本区域，并链接到原始 `states/*.json`。
- 没有事件的帧也要明确显示 `events=none` 或等价文案，避免人眼误以为证据缺失。
- 保留 `<details><summary>visual_state</summary>` 形式的原始 JSON，便于人眼深挖。

## 4. 全量 val-data 统计 summary

根 `summary.json` 和 HTML 顶部必须包含全量统计，scene 级 `summary.json` 包含同构子集。

必须统计：

- 帧数：总帧数、成功响应帧数、错误/超时帧数，每 scene 帧数。
- person count：每帧 `scene_flags.person_count`，输出 `max_person_count`、`frames_with_person`、`person_frame_ratio`，以及可选的 person count 分布。
- track 数：全量唯一 track id 数、每 scene 唯一 track id 数、每帧 track count 的 max/avg、track id 列表。
- attention 可用比例：`attention` 存在且有有效 `target_track_id` 或 `target_uv` 的帧数 / 成功帧数；同时输出 null/unavailable 帧数、target switch 次数。
- 事件计数：总 `semantic_event_count`、按事件类型计数、按 scene 计数、每类事件首次触发 scene/frame。
- 关键帧：每 scene 的 `first_frame`、`first_person_frame`、`first_attention_frame`、`first_event_frame_by_type`、`last_frame`，并在 HTML 中链接到对应图片。

推荐 summary 形状：

```json
{
  "data_dir": "val-data",
  "source_jsonl": "artifacts/replay/visual_state.jsonl",
  "frames_total": 0,
  "frames_ok": 0,
  "errors": 0,
  "person": {
    "frames_with_person": 0,
    "person_frame_ratio": 0.0,
    "max_person_count": 0
  },
  "tracking": {
    "unique_track_count": 0,
    "track_ids": [],
    "max_tracks_per_frame": 0,
    "avg_tracks_per_frame": 0.0
  },
  "attention": {
    "available_frames": 0,
    "available_ratio": 0.0,
    "null_frames": 0,
    "target_switches": 0
  },
  "semantic_events": {
    "total": 0,
    "counts_by_type": {},
    "first_frame_by_type": {}
  },
  "keyframes": {},
  "scenes": []
}
```

统计从 `visual_state` 原始字段计算。若底层 replay 已有同名或等价统计字段，可复用；HTML 证据需要的关键帧链接由新工具自己的 evidence model 记录。

## 5. 实现步骤

### Step 1: 保持单 scene 工具可用

先不要删除或重写 `tools/visualize_service_replay.py`。它继续作为单 scene、少量帧快速调试入口，现有测试继续通过。

现有 overlay/summary/HTML 逻辑是唯一来源。第一步只抽出少量共享 helper，例如：

- `tools/visual_evidence_overlay.py`
- `draw_visual_state(image, state, *, scene=None) -> image`
- `event_summary(event) -> str`
- `scene_context_summary(scene_context) -> str`
- `render_frame_card(...) -> str` 或更小的 HTML 片段 helper

迁移原则：

- 旧单 scene 工具改为 import helper。
- 新 all-scenes 工具 import 同一 helper。
- 不在 `tools/generate_visual_evidence.py` 复制绘图逻辑或 HTML card 逻辑。
- helper 保持轻依赖，OpenCV import 仍可延迟到绘图函数内部。
- 只抽真正共享的 overlay/summary/HTML 小函数，不把 replay、CLI、统计都塞进 helper。

### Step 2: 新增 artifact-first 命令

新增 `tools/generate_visual_evidence.py`，默认只做三件事：

1. 读取 `--visual-state-jsonl` 或 `--replay-artifact`。
2. 用 `tools.replay_val_data.discover_scene_dirs()` 和 `iter_scene_frames()` 映射 `scene/frame_id` 到 source image。
3. 调用共享 helper 生成图片、scene HTML、根 HTML、summary。

内部使用简单 dataclass 或 dict 记录：

- `FrameEvidence`：scene、index、frame_id、timestamp、source path、image path、state path、latency、tracks、attention、scene_context、semantic_events、error。
- `SceneSummary`：scene 级帧数、person、tracking、attention、events、keyframes。
- `RunSummary`：全量聚合。

单帧错误写入 state、HTML 和 summary，继续处理后续帧；只有输入 artifact 缺失、JSONL 无法解析、源帧无法映射这类前置错误才提前退出。

### Step 3: 显式在线生成

当用户传入 `--run-replay --server ...` 时：

- 按 `discover_scene_dirs()` 顺序遍历 scene。
- 对每个 scene 调用 `tools.replay_val_data.replay_scene()`。
- 通过 `save_jsonl` / `append_jsonl` 写出根 `visual_state.jsonl`，必要时同步拆分 scene 级 JSONL。
- replay 完成后直接复用 Step 2 的 artifact-first 渲染路径。

禁止在新工具里新增 WebSocket replay loop。这样帧顺序、header、JSONL 形状和现有 replay 工具保持一致，后续维护只有一种做法。
本计划不要求把所有历史 replay WebSocket 代码收敛成全局唯一实现；只要求新工具不新增第三套 replay loop，并且在线模式复用 `tools.replay_val_data`。

### Step 4: 生成 HTML

先生成 scene 页面，再生成根页面：

- scene 页面沿用共享 card helper，增强 anchor、event area 和 summary 区。
- 根页面只做索引和 summary，不复制所有帧 raw JSON，避免过重。
- 所有链接用相对路径，保证移动整个 `artifacts/visual-evidence` 目录后仍可打开。

### Step 5: 收敛工具职责

完成后职责固定为：

- `tools/visualize_service_replay.py`：单 scene、少量帧、快速看 overlay。
- `tools/generate_visual_evidence.py`：全量或多 scene 视觉证据包；默认从 artifact 生成，显式 `--run-replay` 才在线跑服务端。
- 共享 helper：唯一 overlay/HTML 小逻辑来源。

## 6. 测试计划

只保留 3-4 个轻量测试，覆盖数据流和链接，不测试视觉美术细节。

建议新增/调整：

- artifact 输入测试：用 2 个 fake scene、每个 scene 2-3 帧的 `tools.replay_val_data` wrapped JSONL，断言目录结构、scene JSONL、state JSON、summary、HTML 被写出；绘图写图可 monkeypatch，避免依赖真实模型或服务端。
- summary 单元测试：用小型 fake `visual_state` 列表断言帧数、person count、track 数、attention 可用比例、事件计数、关键帧计算正确。
- HTML 单元测试：用 fake evidence 断言根 HTML 有 scene 链接、event timeline 链接、frame anchor、`states/*.json` 链接、`<details><summary>visual_state</summary>`。
- shared helper 单元测试：验证 event/scene/attention summary 文本稳定，长 evidence 不进入图片短摘要；迁移后更新 `tests/unit/test_visualize_service_replay.py` 的 import。

明确不测试：

- 不启动真实 server 做单元测试。
- 不跑全量 `val-data` 矩阵。
- 不引入 Playwright。
- 不做 golden image pixel diff。
- 不测试 bbox 颜色、字体、精确像素、图片美术细节。

## 7. 手工验收

如果已有 replay artifact，直接跑默认离线命令：

```bash
uv run --extra inference python tools/generate_visual_evidence.py \
  --visual-state-jsonl artifacts/replay/visual_state.jsonl \
  --data-dir val-data \
  --out artifacts/visual-evidence
```

如果需要现场生成 replay，先启动 server：

```bash
uv run visual-events-server --config configs/pc-ga-server.toml --port 8765
```

然后显式在线生成：

```bash
uv run --extra inference python tools/generate_visual_evidence.py \
  --run-replay \
  --server ws://127.0.0.1:8765/v1/stream \
  --data-dir val-data \
  --out artifacts/visual-evidence \
  --camera front \
  --fps 10 \
  --head-motion stationary \
  --response-timeout-ms 10000
```

打开 `artifacts/visual-evidence/index.html`，人眼至少抽查：

- `pci_stand`：person count、稳定 track、attention、`person_stopped_near_robot` 类事件展示。
- `pic_hello`：waving 事件在图片底部和 HTML event timeline 中都可定位。
- `pic_leave`：person left 事件和关键帧链接可定位。
- `pic_walk_in_stop`：approaching/stopped 事件顺序和关键帧可从 HTML 快速跳转。

抽查要求：

- 图片上 bbox/track label、attention point、event badge/底部事件条、scene state 都可读。
- 长 evidence 和 raw JSON 在 HTML 中可展开，不盖满图片。
- HTML 根 summary 数字与 scene 页面大体一致。
- 点击 event timeline 能跳到对应 frame anchor。
- 打开任意 frame 的 `states/*.json` 能看到完整 `visual_state`。

## 8. 开发原则

- KISS：默认从 artifact 生成证据；在线 replay 只有一个显式入口。
- DRY：overlay、event summary、scene_context summary、HTML frame card 的共享逻辑只保留一份。
- YAGNI：不做身份记忆链路、上传系统、标注平台、Web dashboard、新模型、新规则。
- One feature, one way：全量视觉证据只推荐 `tools/generate_visual_evidence.py`；`tools/visualize_service_replay.py` 保持单 scene 小工具定位。
- Human-first：输出优先支持人眼定位问题，summary 是导航和 debug 辅助。
- Failure-tolerant：单帧错误记录到 JSONL/summary，不轻易丢弃已生成证据。
- Stable artifacts：相对链接、稳定文件名、稳定 frame anchors，方便 review 时分享整个输出目录。
