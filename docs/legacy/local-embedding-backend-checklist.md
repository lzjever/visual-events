# Local Embedding Backend 实现 Checklist

Legacy note: historical reference only. Current active entry points are `docs/ga-development-plan.md` and `docs/identity-overlay-product-development-plan.md`.

日期：2026-06-29

## 目标

补齐 `memory.embedding.backend = "local"`，让 v0.3 视觉记忆从 fake embedding 进入真实本地模型推理：

- `teach_person` 使用真实人脸 embedding 写入已知人物记忆。
- `known_person_present` 使用真实人脸 embedding 匹配已教学人物。
- `teach_scene` 使用真实图像 embedding 写入场景记忆。
- `scene_activated` 使用真实图像 embedding 匹配已教学场景。
- 保留现有 fake backend，用于单元测试和手工 PC memory E2E 流程验证；fake 结果不能作为真实 local 模型证据。

这不是新产品线，也不是治理项目。实现范围只覆盖真实 local embedding backend。

## 不做

- 不训练模型。
- 不自动下载模型。
- 不把模型、cache、`runtime/`、`artifacts/`、`val-data/` 加进 Git。
- 不新增第二套教学 API、streaming 协议或 CLI 控制链路。
- 不把 YOLO pose latent 当作人物身份 embedding。
- 不做 v0.4 的匿名熟人、身份纠错、自动聚类或泛化手势指向增强；`third_person_introduction` 的受限人指向人解析属于 target resolver，不属于 local embedding backend。
- 不承诺 RK3588 已验证通过；本阶段只保持模型/runtime 选择对后续 RK3588 迁移友好。

## 实现 Checklist

### 1. Backend 边界

- [ ] 在 `src/visual_events_server/memory/embedding.py` 增加 `LocalEmbeddingBackend`。
- [ ] 继续复用现有 `MemoryEmbeddingBackend` 协议，不新增第二套 embedding API。
- [ ] `embed_person(image_crop)` 返回真实 face embedding。
- [ ] `embed_scene(image_or_crop)` 返回真实 scene/image embedding。
- [ ] `image_crop` / `image_or_crop` 必须是真实可解码的 JPEG bytes：人物目标由 service / target resolver 从原帧解码裁剪后重新编码 crop JPEG，整图场景使用原始全图 JPEG。
- [ ] embedding backend 不解析 `prefix + 原 JPEG` 这类复合载荷，也不从原图中自行裁剪目标区域；目标解析和图像裁剪属于 service 边界。
- [ ] 所有输出向量继续使用现有 `EmbeddingResult` 和 `normalize_vector()`。
- [ ] 低质量、无脸、模型缺失、模型推理失败时抛出 `EmbeddingUnavailable(code, message)`，不写入错误记忆。

### 2. 模型与配置

- [ ] `person_model_path` 和 `scene_model_path` 必须显式配置。
- [ ] `person_model_path` 可以是 face detector + face recognizer 的本地 bundle 路径，不为每个子模型新增一组配置字段。
- [ ] face bundle metadata 必须至少声明 `model_name`、`version`、`dim`、`runtime`、`files.detector`、`files.recognizer`、`input_size.detector`、`input_size.recognizer`；两个 `input_size` 都必须是 `[width, height]` 两个正整数。
- [ ] `scene_model_path` 可以是 image embedding 模型和 preprocess metadata 的本地 bundle 路径；当前 scene image-only 主线不需要 tokenizer/config。
- [ ] scene bundle metadata 必须至少声明 `model_name`、`version`、`dim`、`runtime`、`files.model`、`input_size`、`input_name`、`output_name`、`preprocess`；`input_size` 必须是 `[width, height]` 两个正整数。
- [ ] scene bundle 的 `preprocess.resize_mode` 必须显式声明为 `resize_shorter_center_crop`，并与 `mean` / `std` 一起启动校验。
- [ ] bundle metadata 只用于启动校验和记录 embedding provenance（`embedding_type`、`embedding_model`、`embedding_version`、`embedding_dim`），不新增统一 manifest/schema、治理报告或 release gate。
- [ ] 模型文件只从 `runtime/models/` 或显式路径读取。
- [ ] server 启动时如果 `backend = "local"` 但模型路径缺失，必须明确失败。
- [ ] 不调用会隐式下载模型的 API。
- [ ] 记录实际 embedding 类型、模型名、版本、维度和 runtime，持久写入 `embedding_type`、`embedding_model`、`embedding_version`、`embedding_dim`。
- [ ] face 模型优先选择成熟 ONNX/InsightFace/ArcFace 类轻量方案。
- [ ] scene 模型优先选择成熟 CLIP/OpenCLIP/SigLIP/MobileCLIP 类轻量图像 embedding 方案。
- [ ] 只选一条 face 路径和一条 scene 路径作为主线，不并行维护多套实现。

### 3. 依赖管理

- [ ] 使用 `uv` 管理依赖。
- [ ] 把真实 embedding 依赖放在独立可选依赖组中，避免默认安装变重。
- [ ] 依赖必须能在 repo-local runtime 环境安装，不写系统目录或用户目录。
- [ ] PC x86_64 先跑通；aarch64/RK3588 兼容性作为选型约束记录，但不作为本阶段通过条件。

### 4. 人脸 embedding 行为

- [ ] `teach_person` 对 resolved person crop 做人脸检测。
- [ ] face 主线只接受带 5 点 landmarks 的 SCRFD 9-output/kps detector；detector 预处理使用等比例 resize + pad/letterbox，bbox/kps 用同一 scale/pad 反解回原图。
- [ ] 无可用人脸时返回 `no_usable_face`，不创建可用于身份识别的 person embedding。
- [ ] 多张脸时只接受目标 crop 内最可信的一张脸；歧义明显时返回错误，不猜。
- [ ] face embedding 只用于已教学人物匹配。
- [ ] 阈值和 top1/top2 margin 继续复用现有 retriever/matching 配置。

### 5. 场景 embedding 行为

- [ ] `teach_scene` v0.3 只接受 `target.kind=scene` 整图教学；bbox/point/region 场景教学必须明确拒绝且不写入，等查询侧支持同类 region query 后再开放。
- [ ] scene 图像预处理使用等比例 resize shorter side / cover resize 后 center crop 到 `input_size`，不直接压扁非方图。
- [ ] `scene_activated` 继续通过现有 retriever、threshold、cooldown 输出低频事件。
- [ ] 不引入文本生成、场景分类标签器或 LLM；scene 语义来自用户教学文本。
- [ ] 场景 embedding 推理放在低频 memory 侧链，不进入 10Hz gaze 主链路。

### 6. 存储与检索

- [ ] 继续使用现有 SQLite/sqlite-vec store。
- [ ] 不新增独立向量数据库。
- [ ] 如果真实模型维度不同，由 `LocalEmbeddingBackend` 初始化时给出 person/scene dim，再复用现有 store 初始化流程，不写硬编码分支。
- [ ] store / retriever 打开已有 SQLite DB 时，如果 SQLite schema、`sqlite-vec` virtual table 维度或 table shape 与当前 local backend 不匹配，必须 fail fast 并给出明确错误。
- [ ] `embedding_type`、`embedding_model`、`embedding_version`、`embedding_dim` 通过 retriever 查询条件过滤；同一次相似度检索不得混查不同 `embedding_type`、`embedding_model`、`embedding_version` 或 `embedding_dim` 的向量。
- [ ] 切换模型不会隐式迁移或重建旧向量；必要时人工新 DB 或显式离线迁移，不属于本 checklist。
- [ ] 已有 fake backend 测试继续通过。
- [ ] local backend 的测试只覆盖核心运行边界，不对测试工具再写测试。

### 7. PC 本地验证

- [ ] 用 `val-data/` 做真实 local backend 手工 smoke。
- [ ] `tools/run_memory_e2e.py` 是手工 memory E2E 工具，覆盖 v0.3 teach/replay 和当前已有 v0.4 memory checks；它不是默认发布 gate。
- [ ] fake 确定性侧链检查仍使用默认命令：

```bash
uv run python tools/run_memory_e2e.py --data-dir val-data --out artifacts/memory-e2e --scene pic_hello
```

- [ ] local 模型 smoke 必须显式选择 backend 和两条本地模型路径；这是手工 local embedding gate，不加入默认发布 gate：

```bash
uv run python tools/run_memory_e2e.py \
  --data-dir val-data \
  --out artifacts/memory-e2e-local \
  --scene pic_hello \
  --embedding-backend local \
  --person-model-path runtime/models/<face-model-bundle> \
  --scene-model-path runtime/models/<scene-model-bundle>
```

- [ ] local smoke 的硬通过条件：`report.ok=true`。
- [ ] local smoke 的硬通过条件：`report.embedding.uses_real_model_backend=true`。
- [ ] local smoke 的硬通过条件：报告中的 `v0.3 teach person replay known_person_present and scene_activated` check 通过，且其 assertions 里 `known_person_present=true`、`scene_activated=true`。
- [ ] 生成简单可视化 artifact，能肉眼查看检测框、track、教学目标和匹配结果。
- [ ] 记录每次 embedding 推理耗时，用于判断是否会影响低频 memory 侧链。
- [ ] 验证 10Hz 主推理、tracking、gaze 输出不因 local embedding 慢而阻塞。

### 8. 代码收口

- [ ] 删除或替换 `_embedding_backend_from_config()` 里 `local` 的 fail-fast 占位逻辑。
- [ ] 增加必要单元测试：配置错误、模型缺失、fake/local backend 分支、无脸错误、向量维度。
- [ ] 不新增 release report、manifest schema、审计层或额外治理 gate。
- [ ] 完成本 checklist 后再更新 v0.3 文档中的 backend 状态；完成前只能声明 fake 已可测、local 待完成。

## 验收标准

本任务完成时，应该能用一份本地配置启动 server：

```toml
[memory]
enabled = true

[memory.embedding]
backend = "local"
person_model_path = "runtime/models/<face-model>"
scene_model_path = "runtime/models/<scene-model>"
```

并在 PC 本地完成：

- 已教学人物再次出现时输出 `known_person_present`。
- 已教学场景再次出现时输出 `scene_activated`。
- local smoke 报告满足 `ok=true`、`uses_real_model_backend=true`、v0.3 `known_person_present` + `scene_activated` assertions 全部通过。
- fake backend 的既有单测和 E2E 不退化。
- 模型文件、cache、测试数据和 artifact 仍然不进入 Git。

## 自查结论

这个 checklist 只补齐 v0.3 当前缺口：真实 local embedding backend。它没有把 v0.4 的匿名熟人、纠错、自动聚类、指向增强提前塞进来，也没有新增数据库服务、治理报告或发布审计。实现时如果遇到模型选型分歧，只保留一条 face 主线和一条 scene 主线，避免同一功能多种做法。
