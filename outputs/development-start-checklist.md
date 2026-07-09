# Development Start Checklist

## Current Status

Development can start now.

Completed:

- Audited source dataset.
- Confirmed 20,562 usable JPG images.
- Confirmed `._*` macOS resource files must be excluded.
- Created Stage 0 validation dataset with 100 images.
- Created clean manifest and review CSV.
- Created Stage 0 contact sheet.
- Wrote V3 execution development plan.

Generated files:

- `outputs/semantic-object-color-retouch-dev-plan-v3.md`
- `outputs/stage0/manifest_clean.jsonl`
- `outputs/stage0/image_metrics.jsonl`
- `outputs/stage0/stage0_selection.jsonl`
- `outputs/stage0/stage0_review.csv`
- `outputs/stage0/stage0_contact_sheet.png`

## Development Order

### Phase 1: Stage 0 Baseline

Goal:

Prove the object-aware color retouching pipeline works on 100 images before training.

Tasks:

1. Build region perception pipeline.
   - Detect sky, person, skin, clothing, grass/tree, building, LED/stage, text/logo.
   - Generate masks.
   - Save region JSON.

2. Compute per-region color metrics.
   - LAB / HSV / brightness percentiles.
   - clip_high_pct / clip_low_pct.
   - saturation / colorfulness.
   - sharpness proxy.

3. Freeze action vocabulary V1.0.
   - Each action maps to local renderer params.
   - Each action maps to GPT Image 2 prompt fragment.

4. Build plan generator.
   - Generate 3-5 plans per image.
   - Include executor route.
   - Include no-edit fallback.

5. Build local preview renderer.
   - First use Python mask blending.
   - Later port to C++ BeautySDK region blending.

6. Run GPT Image 2 smoke test.
   - 10 images first.
   - Then 30-40 selected Stage 0 images.
   - Always use three-part lock prompt.

7. Build QA and review loop.
   - Save QA JSON.
   - Fill review CSV.
   - Track failure cases.

### Phase 2: Pseudo-Label Dataset

Goal:

Create 2,000-5,000 pseudo-labeled samples after Stage 0 passes.

Tasks:

- Expand sampling.
- Run teacher perception pipeline.
- Run teacher planner.
- Save masks, metrics, plans, QA, preferences.

### Phase 3: Student Planner Model

Goal:

Distill teacher pipeline into low-cost planner.

Recommended:

```text
Grounding DINO + SAM2
+ SigLIP / DINOv2 / ConvNeXt encoder
+ structured heads for action / strength / route / plan rank
+ deterministic template engine
```

Do not train pixel generation in V1.

---

## Phase C 双轨（V3.1，2026-07-09）

| 轨道 | 文档 | 状态 |
|------|------|------|
| **C2 Reference 自蒸馏（主路径）** | `outputs/phase-c2-reference-self-distill-design.md` | ✅ C2.1/C2.2/C2.3 扩样跑通（97 样本/208 class-rows），held-out MAE=4.20 < 基线 6.31 |
| **C1 GPT teacher 量化（辅助）** | `semantic-object-color-retouch-dev-plan-v3.1-c2-addendum.md` | API 已切 API易；待双图冒烟 |

C2 teacher v0 = `color_reference_transfer.py` medium 伪标签 → RegionalParamHead → Smart Color v2。

**2026-07-09 扩样已跑通**（C2.1 新增读取 Stage 0 100 张验证集，20 图回归集 + Stage 0 补充 = 97 样本）：

```bash
cd stage0_pipeline
../.venv-m2/bin/python scripts_c2/export_bootstrap_dataset.py   # 97 样本
../.venv-m2/bin/python scripts_c2/fit_region_params.py          # 208 class-rows
../.venv-m2/bin/python scripts_c2/train_per_class_head.py       # held-out MAE=4.20
```

样本量从 41 扩到 208 后，held-out MAE / 基线 的降幅比例几乎不变（34.3% → 33.4%），说明规则教师信号稳定可泛化。过程中顺带修复了 `fit_region_params.py` 里一个真 bug：一张 person_event 照片假天空检测（原图近乎纯色）导致 Lab-affine scale 除以近零方差爆炸到 68 倍，改为方差过低时 scale 退化为 1.0 + 只用均值差算 shift（详见设计稿 §7）。

---

## What The User Needs To Prepare

Required:

1. Decide whether we can install/download open-source models on this machine or server.
   - Grounding DINO
   - SAM2
   - Qwen-VL / InternVL

2. Provide API access for GPT Image 2 smoke tests.
   - API key / proxy details if needed.
   - Budget approval for roughly 10 first tests, then 30-40 Stage 0 tests.

3. Confirm BeautySDK access.
   - Path to `pe_process_image`.
   - Existing `PARAM_PROTOCOL.md`.
   - Smart Color v2 docs/code location.
   - Whether C++ region mask blending can be developed in this repo.

4. Confirm the target retouching style.
   - Event delivery natural color.
   - Commercial clean color.
   - Cinematic portrait.
   - Or a ranked default order.

Useful but optional:

- A small set of photographer-edited reference images.
- Original/edited pairs.
- Examples of "good blue sky" and "bad fake blue sky".
- Examples of accepted/rejected skin tone.

## Immediate Next Engineering Step

Implement `stage0_pipeline/`:

```text
stage0_pipeline/
  config/actions.v1.json
  config/object_prompts.json
  scripts/build_region_metrics.py
  scripts/generate_plans.py
  scripts/render_local_preview.py
  scripts/build_review_sheet.py
```

The first runnable milestone should produce:

```text
one image
-> region metrics JSON
-> 3 plan JSON entries
-> local preview images
-> review row
```
