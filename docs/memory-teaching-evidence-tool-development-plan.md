# Memory Teaching Evidence Tool Development Plan

日期：2026-06-30

> Renderer spec only：本文只定义 memory/teaching/identity evidence renderer 的输入和输出形态，不是当前产品范围、active acceptance source 或 handoff checklist。当前 active 产品/开发/QA 验收以 `docs/identity-overlay-product-development-plan.md` 为准。

## 1. 目标

新增一个独立的人眼证据生成命令，把一次 memory / teaching runner 产出的 artifacts 转成完整、可解释、可人工检查的 evidence 包。它面向开发、产品和 demo 检查人员，目标是打开一个 HTML 后能看懂：

- 用户示教“请记住我”时，服务端绑定的是谁，后续是否召回为 `known_person_present`。
- 用户介绍第三人时，服务端如何用介绍人的姿态指向解析被介绍人，后续是否召回为 `known_person_present`。
- 用户示教整图场景时，服务端是否能输出 `scene_activated`。
- Identity Overlay 是否进入 `visual_state.identity_context`，普通人物事件是否带 `identity_context`。
- `identify-current`、CLI current visual snapshot 和 `teach_person -> merged_anonymous_person` 是否有可读证据。
- object teaching 当前是 unsupported / no-write 负例，没有被误写成 person、scene 或 region memory。
- 每张图和机器报告能通过 `assertion_id`、track refs、`source_text_path`、`source_image_path`、`source_frame_ref`、`request_snapshot_ref`、crop hash/path、event id 或 `memory_match_id` 对齐。

这个工具不是新的测试框架，不是第二套 E2E runner，也不是发布审计层。它是 `generate_visual_evidence.py` 在 memory / teaching 领域的对应工具：离线优先读取既有 runner artifact，生成可人工查看的 evidence，而不是替代机器断言。

## 2. 原则

- KISS：一个 memory evidence 命令，一个底层 memory teaching runner，一个图片/HTML renderer。
- DRY：不复制 teach / resolve / replay / identify-current 业务流程；新工具只消费现有 runner artifact，或显式委托现有 runner 后再渲染。
- YAGNI：不做 Web dashboard、标注平台、profile 管理 UI、身份纠错 UI、模型训练、对象记忆系统或复杂审计报告。
- 一个功能一种做法：实时视觉流 evidence 继续用 `tools/generate_visual_evidence.py`；memory / teaching evidence 使用新命令；不要让一个命令同时承担两种输入语义。
- 低心智负担：默认从已有 runner artifact 生成 evidence；需要重新跑真实模型时，显式委托现有 runner，不在新工具里重写执行链路。
- identity evidence 复用同一个 renderer 和同一个命令，不新增第二套 identity evidence 工具。
- 不测试测试工具本身：只对核心 evidence contract、字段传播和高风险绘图 helper 做小范围测试；不做像素级回归和页面美术测试。
- `val-data/`、模型、runtime DB、cache、artifacts 不进 Git。

## 3. 非目标

- 不修改 agent-facing REST request 合同；`track_id`、`bbox`、`point_uv`、`keypoints`、`source_frame` 不进入 agent-facing request。
- 不把 internal debug evidence 投影给 Botified frame。
- 不把 `generate_visual_evidence.py` 改造成 memory teaching 入口。
- 不在 renderer 里重新计算“谁指向谁”、face bbox 或 embedding 结果；算法结论必须来自 runner / service / resolver 已记录的 evidence。
- 不要求真机、RK3588、现场或 DDS camera 实测；本阶段仍是 PC 本地 evidence。
- 不要求每一帧都生成图片。全量 `val-data` 结果可以通过 summary、JSONL 和关键帧/代表帧表达，避免 artifact 爆炸。

## 4. 推荐用户命令

默认离线模式：

```bash
uv run python tools/generate_memory_teaching_evidence.py \
  --artifact artifacts/memory-teaching-ga-local-smoke \
  --out artifacts/memory-teaching-evidence
```

默认行为：

- 不连接 server，不启动 runner，不加载模型。
- 读取 `--artifact` 下的 `report.json`、`timeline.jsonl`、`teach_payloads.json`、`api_responses.jsonl`、`botified_frames.jsonl`、`visual_states.jsonl`、stored crop artifacts 和 source JPEG path；如果存在 identity source records，也一起读取。
- 生成新的 HTML / 图片 evidence 包到 `--out`。
- 如果缺少关键 artifact、source frame 或 crop path，直接失败并说明缺失项，不猜测路径，不伪造图片。

### 4.1 最小 Source Artifact 输入合同

source artifact 仍以 runner 输出为唯一机器 truth，renderer 不重新跑业务逻辑。最小输入分为 required 和 optional：

- required：`report.json`、可 join 的 source frame/image/crop 引用、`source_frame_ref`、`request_snapshot_ref`。交互用例还必须有 `source_text_path` 和 `source_image_path`。
- required records：teach/resolve API records、event records、`visual_states.jsonl` 或等价 report section、Botified/CLI projection records。
- identity optional records：`identify-current` API records、`visual_states.identity_context`、event enrichment records、CLI current visual snapshot artifact、`teach_person -> merged_anonymous_person` response/evidence。
- optional identity evidence 缺失时，HTML 显示 `not_present`，不伪造结论，也不让 renderer 失败。
- source frame、source text、source image、stored crop 或 crop hash/path 缺失时失败；这些是图片和 report 双向对齐的必要证据。

identity overlay、event identity_context、identify-current、CLI current visual snapshot 和 teach auto merge 都通过同一 `visual_evidence_index[]`、同一 HTML renderer 和同一图片绘制 helper 展示。

可选真实模型便捷模式：

```bash
uv run python tools/generate_memory_teaching_evidence.py \
  --run-local-smoke \
  --data-dir val-data \
  --out artifacts/memory-teaching-evidence \
  --person-model-path runtime/models/face-buffalo-s \
  --scene-model-path runtime/models/scene-mobileclip2-s0 \
  --pose-model-path runtime/models/yolov8n-pose.pt
```

`--run-local-smoke` 只做一件事：调用现有 `tools/run_memory_teaching_ga_e2e.py` 的 local-smoke 执行函数，先产出 runner artifact，再走同一个离线渲染路径。它不直接实现 teach / resolve / replay。

通用要求：

- 输出目录默认 `artifacts/memory-teaching-evidence`。
- 运行前清空 `<out>/visual-evidence`，避免旧图片混淆。
- 如果使用 `--run-local-smoke`，runner 原始输出放在 `<out>/runner-artifact/` 或等价子目录中。
- 不隐式下载模型，不写系统目录或用户目录；Ultralytics cache 和 runtime cache 必须留在 artifact/runtime 目录下。
- 完成后打印 source `report.json`、source gate status 和根 `index.html` 的路径。
- 默认 exit code 只表示 renderer 是否成功。source `report.ok=false` 时仍应生成 failure evidence 并 exit 0；如需要把 source gate 失败变成命令失败，使用显式 `--strict-source-ok`。

保留现有验证命令：

```bash
uv run python tools/run_memory_teaching_ga_e2e.py --data-dir val-data --out artifacts/memory-teaching-ga
```

`run_memory_teaching_ga_e2e.py` 可以继续用于 legacy regression / source artifact 生成；新命令用于人工 evidence / demo。两者底层执行和 renderer 不能分叉；当前 active acceptance 不由本文定义。

## 5. 输出结构

```text
artifacts/memory-teaching-evidence/
  index.html                       # 唯一推荐人工入口
  source-artifact.json
  visual_evidence_index.json
  visual-evidence/
    index.html                     # 内部 evidence page，由根入口链接
    self-introduction-known-person.jpg
    third-person-pose-pointing.jpg
    teach-scene-scene-activated.jpg
    object-unsupported-no-write.jpg
    crops/
      self-person-crop.jpg
      third-person-target-crop.jpg
  runner-artifact/                 # 仅 --run-local-smoke 时生成
    report.json
    runtime/
    ...
```

说明：

- source artifact 中的 `report.json` 是机器 truth；HTML 和图片是人工查看入口。
- 根 `index.html` 是唯一推荐用户入口；`visual-evidence/index.html` 只是内部子页面。
- renderer 生成自己的 `visual_evidence_index.json` 或等价 index，并在 HTML 中展示 source report 路径。
- renderer 不直接修改 source artifact 的 `report.json`。如需要把 `visual_evidence_index[]` 写入 runner report，应由 runner 自己负责。
- `crops/` 只放和 evidence 直接相关的存储 crop 或 face crop 预览，不做 gallery。

## 6. 功能范围

### 6.1 Artifact-First Evidence Path

新命令提供一个正式 evidence path：离线渲染现有 memory teaching artifact。

输入 artifact 可以来自 fake/full contract 或 local-smoke/real model runner。默认推荐使用 local-smoke/real model artifact。renderer 必须展示 artifact 的 `mode`、`backend` 和 `real_model_evidence`，避免把 fake 证据误当真实模型证据。

对 local-smoke/real model artifact，证据包覆盖：

- self introduction：`pic_teach_me`，真实 person embedding，验证 `known_person_present`。
- third-person introduction：`pic_teach_person`，真实 YOLO pose + person embedding，验证 `pose_pointing_to_person` 和 `known_person_present`。
- scene teaching：`pic_teach_scene_galbot`，真实 scene embedding，验证 `scene_activated`。
- object negative：如果 source artifact 包含 object unsupported / no-write，则渲染；如果 local-smoke artifact 暂时没有 object negative，则明确显示 not present，不伪造结论。
- full scene replay summary：如果 source artifact 已包含 post-teach full scene replay，则渲染 per-scene compact summary；如果没有，则明确显示 not present。

是否补充真实 local full scene replay 或 object negative 属于 runner follow-up，不属于 renderer 前置条件。需要时只在现有 runner 中增加一次顺序 replay，不引入多轮 soak、并发矩阵或 release audit。renderer 的最低要求是正确展示 present / not present 状态。

### 6.2 图片必须补齐的算法证据

`self-introduction-known-person.jpg`：

- 源帧。
- 被示教 person bbox。
- stored crop 缩略图或 crop path。
- face detection bbox / face score：只有 local backend 在运行时记录了 face metadata 时才显示；否则只显示 target bbox 和 embedding crop，不把 crop 说成 face bbox。
- `person_id`、`event_id`、`memory_match_id`、match score、crop hash。

`third-person-pose-pointing.jpg`：

- 源帧。
- 介绍人 bbox，标记 `introducer_ref`。
- 被介绍人 bbox，标记 `resolver_target_ref`。
- 介绍人的 shoulder / elbow / wrist keypoints。
- 手臂指向线或 ray，显示 `arm_side`。
- `candidate_score`、`ray_intersects_bbox`、`perpendicular_distance`、`pose_stability_window.selected_count`。
- stored target crop 缩略图或 crop path。

`teach-scene-scene-activated.jpg`：

- 源帧。
- scene title / scene id。
- `event_id`、`memory_match_id`、scene match score、crop hash。

`object-unsupported-no-write.jpg`：

- 源帧。
- unsupported target kind / error code。
- no-write store delta summary。

图片只放短摘要，不把长 JSON 盖在图上。完整字段放到 HTML `<details>` 和 source `report.json`。

### 6.3 需要补齐的 report / evidence 字段

字段只进入 response evidence、runner report、runner-only artifact 或 visual evidence index，不进入 agent-facing request。renderer 不事后重新计算这些字段，只消费 runner/service 已记录的证据。

Person / face evidence：

```json
{
  "person_visual_evidence": {
    "source_frame_ref": "front:191:77000",
    "source_bbox_xyxy": [0, 0, 0, 0],
    "source_bbox_coordinate_space": "source_frame",
    "crop_box_xyxy": [0, 0, 0, 0],
    "crop_box_coordinate_space": "source_frame",
    "embedding_crop_path": "runtime/...",
    "face_detection": {
      "coordinate_space": "crop",
      "face_bbox_xyxy": [0, 0, 0, 0],
      "landmarks_5": [[0, 0]],
      "score": 0.0,
      "source": "local_embedding_scrfd"
    }
  }
}
```

Pose pointing evidence：

```json
{
  "pose_visual_evidence": {
    "introducer_ref": "front:track:1",
    "introducer_bbox_xyxy": [0, 0, 0, 0],
    "target_ref": "front:track:12",
    "target_bbox_xyxy": [0, 0, 0, 0],
    "arm_side": "left",
    "shoulder_xy": [0, 0],
    "elbow_xy": [0, 0],
    "wrist_xy": [0, 0],
    "ray_start_xy": [0, 0],
    "ray_end_xy": [0, 0],
    "candidate_scores": [],
    "pose_stability_window": {}
  }
}
```

实现时字段可以更小，但必须满足图片绘制和 HTML 对齐需要。不要为了未来 unknown 场景增加大而全 schema。每个 bbox 必须带明确坐标系或由字段名固定坐标系，尤其 face bbox 可能是 crop 坐标而不是 source frame 坐标。

## 7. 技术方案

### Step 1: 抽出 renderer，不复制流程

从 `tools/run_memory_teaching_ga_e2e.py` 中抽出 memory visual evidence 相关 helper 到一个小模块，例如：

- `tools/memory_teaching_evidence.py`

模块职责：

- 读取 source artifact。
- 构建 `visual_evidence_index[]`。
- 写 `visual-evidence/index.html`。
- 写 overlay 图片。
- 渲染 stored crop / face crop 小图。

`tools/run_memory_teaching_ga_e2e.py` 继续调用这个模块生成最小 runner evidence。新命令也调用同一个模块生成更完整 artifact-first evidence。不要保留两套 `_write_image_overlay` 和 HTML 生成逻辑。

### Step 2: 新增薄命令

新增：

- `tools/generate_memory_teaching_evidence.py`

它默认只做：

- 解析 `--artifact` 和 `--out`。
- 读取 source artifact。
- 调用 shared renderer。
- 打印产物路径。
- source `report.ok=false` 时仍可生成 failure evidence；默认 exit code 仍表示 renderer 是否成功。只有显式 `--strict-source-ok` 才把 source gate 失败转成非零退出。

可选 `--run-local-smoke` 只委托现有 runner，先生成 source artifact，再回到默认离线路径。它不直接实现 teach / resolve / replay，不 import DDS，不重写 server runner。

### Step 3: 可选 runner follow-up

这一步不是 renderer 的前置条件。只有当 demo 需要 local-smoke artifact 也包含 object negative 或 full scene replay 时，才收敛现有 local-smoke runner：

- 保留 self / scene / third-person 三条主线。
- 增加 object unsupported / no-write 负例。
- 如确有 demo 需要，增加 post-teach full scene replay compact summary。
- 确保所有输出仍写到 runner artifact 目录下。

如果为了避免影响现有 gate，需要新增内部参数，也只能在 runner 内部使用一个参数，例如 `include_full_replay=True`。不要新建第二套 local runner，也不要在 renderer 里补跑 replay。renderer 对缺失的 object/full-replay evidence 只显示 not present。

### Step 4: 补齐 face evidence

当前 local face backend 已经通过 SCRFD 检测 face，但只把 score 作为 embedding quality 返回。若要在图片里声明 face bbox，需要把最小 face evidence 返回给 service/report：

- 在 local embedding loader 输出中携带被选中 face 的 bbox、landmarks、score。
- service 在写 person memory evidence 时记录 crop box 和 face bbox，并标明坐标系。
- 如能可靠转换，记录 source-frame face bbox；否则 HTML 至少在 stored crop 上画 face bbox，并明确坐标系是 crop。

本计划最终增强目标包含真实 face bbox；完成该目标必须实现运行时 face metadata 传递和测试。如果开发团队选择先交付 artifact-first renderer，则可以先展示 embedding crop bbox / stored crop thumbnail / crop hash，但页面必须明确 `face_detection=not_recorded`，不得标注为 face bbox。

### Step 5: 补齐 pose pointing visual evidence

当前 `pose_pointing_scoring` 已有 `arm_side`、`arm_vector`、candidate scores，但图片缺少介绍人 bbox、关键点坐标和 ray。需要在 resolver evidence 或 service report 中增加绘图所需的最小几何信息：

- introducer bbox。
- shoulder / elbow / wrist 坐标。
- ray start / end。
- target bbox。
- candidate scores 可选带 bbox，便于画候选框和分数。

优先在 `TargetResolver.preview_pose_pointing_person()` 产生 evidence 时补齐这些字段，避免 renderer 回头从 public `visual_state` 重建 keypoints 或重新算 pose scoring。

字段 ownership：`pose_pointing_scoring` / `pose_stability_window` 仍是算法结论的唯一来源；`pose_visual_evidence` 只补绘图几何。重复字段必须从同一个 resolver/service evidence 对象派生，renderer 不组装第二份算法结论。

### Step 6: HTML 信息收敛

`visual-evidence/index.html` 增加一个简单表格：

- assertion。
- scene。
- status。
- image link。
- report section。
- key refs：`request_snapshot_ref`、`source_frame_ref`、`introducer_ref`、`resolver_target_ref`、`event_id`、`memory_match_id`、crop hash。

每项用 `<details>` 放完整 item JSON。不要做交互式 dashboard。

## 8. 测试计划

只测核心合同和高风险边界。

单元测试：

- `generate_memory_teaching_evidence.py` 参数解析和 preflight：`--artifact` 缺少 `report.json` 时失败，`--out` 不允许写入 source artifact 或 `val-data/` 内。
- `--run-local-smoke` 调用现有 runner 的参数正确；用 monkeypatch，不启动真实模型。
- renderer 用一张小 JPEG 和构造好的 source artifact 生成图片和 HTML；断言文件存在、PIL 可打开、index join 字段完整。
- face evidence 字段从 local embedding result 传到 teach response / report；使用 stub loader / stub backend 做 TDD，不在单元测试里跑真实 ONNX。fake backend 不需要 face evidence。
- pose visual evidence 字段包含 introducer bbox、target bbox、wrist/shoulder/ray，并且不出现在 agent-facing request。

集成 / 手工 smoke：

```bash
uv run python tools/generate_memory_teaching_evidence.py \
  --artifact artifacts/memory-teaching-ga-local-smoke \
  --out artifacts/memory-teaching-evidence
```

如需一条命令重跑真实模型：

```bash
uv run python tools/generate_memory_teaching_evidence.py \
  --run-local-smoke \
  --data-dir val-data \
  --out artifacts/memory-teaching-evidence \
  --person-model-path runtime/models/face-buffalo-s \
  --scene-model-path runtime/models/scene-mobileclip2-s0 \
  --pose-model-path runtime/models/yolov8n-pose.pt
```

Renderer smoke checks（非产品验收）：

- exit code 为 0。
- renderer 成功时 exit code 为 0；source report status 在 HTML 和 stdout 中明确展示。使用 `--strict-source-ok` 时，source `report.ok=false` 应返回非零。
- discovered scene count 等于 source artifact 对应的实际 scene 数。
- self、third-person、scene、artifact skeleton checks 在 source report 中通过；object negative 和 full replay 如果 source artifact 包含，也必须展示。
- visual evidence 图片存在且可打开。
- third-person 图上可见 introducer、target、关键点和指向线。
- self / third-person crop 图或 inset 可见 embedding crop；最终增强完成后应可见真实 face bbox / face score。若 source artifact 尚未记录 face metadata，页面必须明确 `face_detection=not_recorded`。

不做：

- 不做像素级截图回归。
- 不对 HTML CSS 做测试。
- 不写测试去验证测试报告文案。

## 9. Renderer Completion Checks（非当前 handoff）

renderer spec 完成条件：

- 新命令能从现有 memory teaching artifact 生成 `artifacts/memory-teaching-evidence`。
- 可选 `--run-local-smoke` 能委托现有 runner 用真实 local 模型先产出 source artifact，再生成 evidence。
- 新命令和 E2E runner 复用同一套 memory teaching execution / renderer，不存在两套 teach / resolve / replay 逻辑。
- 图片能解释关键算法证据，而不是只显示最终结论。
- report 和图片能双向对齐：从图片能找到 report section，从 report 能找到图片。
- object negative 仍然是 no-write，不被新 evidence 工具误包装成可用 object memory。
- renderer 成功和 source report 成功在 stdout / HTML 中分开展示，不把人眼 evidence 命令变成隐形产品 gate。
- agent-facing REST contract 没有新增低层字段。
- `val-data/`、模型、runtime、cache、artifacts 仍不进入 Git。

## 10. Review 结论

产品结论：

- 这个需求合理。memory / teaching 已经是可展示能力，继续只靠 E2E runner 顺手输出 evidence 会让使用者误以为它是测试内部产物。
- 独立命令能降低使用门槛，但必须保持它是 evidence 入口，不是第二套产品运行链路。
- evidence 的核心价值是证明“绑定了谁、为什么绑定、是否写错、是否召回”，不是多生成几张漂亮图片。

研发结论：

- 当前算法证据在 JSON 中已有一部分，图片表达不足。
- face bbox 缺失的根因是 local embedding backend 没把 SCRFD 检测结果作为 evidence 暴露；在补齐运行时 metadata 前，evidence 只能展示 embedding crop，不能声称是 face bbox。
- pose 指向缺失的根因是 renderer 只有 target bbox，缺少 introducer/keypoint/ray 几何字段。
- 推荐结构性修正 evidence 字段和 renderer，不用硬编码画假框，不从 public visual_state 反推内部 keypoints，也不在 renderer 中重新计算 pose scoring。

本文按 KISS / DRY / YAGNI 收敛，可作为 renderer 开发参考；实现时如发现字段无法可靠获得，应优先在内部 evidence 产生点补齐，而不是在 renderer 里猜测。当前 handoff 入口仍以 identity plan 为准。
