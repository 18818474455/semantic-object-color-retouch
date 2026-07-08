# Stage 0 Pipeline (语义物体调色 · 零大模型骨架)

对应 V3 方案的 Stage 0。目标：在**不接大模型**的前提下把完整链路跑通——

```
一张图 -> 区域指标 JSON -> 3-5 个 plan JSON -> 本地预览图 -> 人工评审行
```

现在就能跑，用来验证「识别 → 门控 → 方案 → 执行 → 评审」这条闭环是否成立，
之后再逐块把占位实现换成真模型。

## 环境

```bash
# 已创建专用 venv（numpy 2.5.1 + Pillow）
../.venv/bin/python scripts/run_stage0.py --help
```

## 运行

```bash
cd stage0_pipeline

# 单张
../.venv/bin/python scripts/run_stage0.py --image-id img_000578

# 前 N 张
../.venv/bin/python scripts/run_stage0.py --limit 10

# 全部 100 张 Stage 0 验证集
../.venv/bin/python scripts/run_stage0.py
```

产物写入 `stage0_pipeline/outputs/`：

```
regions/<id>.json          每张图的场景 + 全图/区域色彩指标 + 路由指标
masks/<id>/<region>.png    区域 mask
plans/<id>.json            3-5 个调色方案（含 action、strength、executor 路由）
previews/<id>/<plan>.jpg   本地渲染预览
sheets/<id>.jpg            原图 vs 各方案对比图（人工评审用）
stage0_pipeline_review.csv 评审表（自动列已填，人工列留空）
```

## 目录

```
config/
  actions.v1.json      冻结 action 词表：每个 action 绑定 local_params + gpt 提示词片段
  object_prompts.json  M2 Grounding DINO 检测提示词（现在未使用）
  thresholds.json      置信度门控 + 执行器路由阈值 + 默认方案排序
scripts/
  common.py            色彩空间/ mask / IO 工具（纯 numpy+PIL）
  region_provider.py   区域检测「插拔接口」+ Stage 0 启发式天空检测占位
  build_region_metrics.py  逐区域色彩指标 + 启发式场景分类
  generate_plans.py    规则化 action 选择（VLM 占位）+ 语义门控 + 路由 + 方案组装
  render_local_preview.py  mask 混合本地调色渲染器（= C++ 需对齐的参数契约）
  build_review_sheet.py    对比图 + 评审 CSV
  run_stage0.py        编排器
```

## 已验证行为（四类 bucket）

- **outdoor_sky**：暗淡日间天空 → 门控放行 → sky.natural_daylight_blue，本地执行，人物/横幅因 mask 保留。
- **stage_led_mixed**：场景判为 stage → **不强制蓝天**，出 restore_original_intent，全部路由到 GPT（混合光 + 低清晰度）。
- **difficult（夜/暗）**：暗部裁切 71% → 自动追加 latitude_recovery（GPT）。
- **person_event**：本地执行；启发式天空可能误检（见下）。

## Stage 0 占位与下一步替换点

| 模块 | 当前占位 | 替换里程碑 |
|---|---|---|
| 区域检测 | `HeuristicRegionProvider` 只按颜色+垂直先验找天空，人物图会误检天空 | **M2**：新增 `grounding_dino_sam2` provider，实现同一 `detect()` 接口即可 |
| 皮肤/人脸 | 暂无（词表已就绪） | **M2**：接现有人脸检测 + 阿里云 face manifest + SAM2 |
| 场景/语义门控 | `classify_scene` 用 bucket+色彩启发 | **M3**：VLM 输出 scene + 在冻结词表内选 action，替换 `select_region_actions` |
| 本地执行 | Python `render_local_preview` | **M4**：`local_params` 契约 1:1 移植到 BeautySDK `pe_process_image` + 区域混合 |
| GPT 执行 | 只生成三段式锁定 prompt，不实际调用 | **M5**：接 APIYi gpt-image-2-all，先 10 张冒烟 |
| QA | 评审 CSV 人工列 | **M6**：加自动 QA（人脸相似度/OCR/保护区 ΔE） |

接口稳定：换真模型时下游 metrics / plan / render / review 代码都不用动。

## 关键约定

- action 词表是**封闭枚举**。VLM/学生模型只能选 action id + strength，不写自由参数或 prompt。
- 所有 GPT 请求走 `generate_plans.LOCK_BLOCK` 三段式强锁定，规划器只填中间操作段。
- 低置信度默认 keep / no-edit（`thresholds.json > confidence_gates`）。
- 默认方案排序 = 自然交付优先（`thresholds.json > plan_ranking_default`），确认房子风格后可改。
