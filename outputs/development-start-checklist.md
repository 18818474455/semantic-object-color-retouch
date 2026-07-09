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

## Phase C 三轨（V3.1，2026-07-09 新增 C1c，同日完成实验并下调优先级）

| 轨道 | 文档 | 状态 |
|------|------|------|
| **C2 Reference 自蒸馏（主路径）** | `outputs/phase-c2-reference-self-distill-design.md` | ✅ C2.1/C2.2/C2.3 扩样跑通（97 样本/208 class-rows），修复过冲光晕后重跑 held-out MAE=4.02 < 基线 6.08 |
| **过冲光晕瑕疵：根因+修复+重跑** | `outputs/phase-teacher-overshoot-halo-fix.md` | ✅ Web Demo 肉眼验证发现，跟分割精度/BSHM无关，已修复并重跑 C2 全流程 |
| **neutral 加法mood-cast（人群/前景多样性保留）+重跑** | `outputs/phase-teacher-neutral-mood-cast.md` | ✅ 密集人群没被检测器识别落进neutral兜底，重缩放式分级洗掉衣服色彩多样性，改用固定加法偏移后修复，重跑 C2 全流程 |
| **同标签类别外观差过大压制强度（背景/前景脱节）+重跑（新增）** | `outputs/phase-teacher-class-mismatch-fix.md` | ✅ 开放词汇标签把两种物理不同的东西都标成同一类别强行统计匹配（商场白吊顶 vs 钢架顶棚"building"），新增类别配对置信度修复，重跑 C2 全流程，MAE 4.07→3.74（真实改善） |
| **C1c 本地/托管 VLM 语义门控（实验完成，优先级下调）** | `outputs/phase-c1c-vlm-sky-gate-results.md` | ✅ `qwen3-vl-plus` 对比 30 个 sky 样本，100% 认同启发式规则（0 语义假阳性）；"替代规则"动机不成立 |
| **C1 GPT teacher 量化（辅助）** | `semantic-object-color-retouch-dev-plan-v3.1-c2-addendum.md` | API 已切 API易；待双图冒烟 |

C2 teacher v0 = `color_reference_transfer.py` medium 伪标签 → RegionalParamHead → Smart Color v2。

**2026-07-09 扩样已跑通**（C2.1 新增读取 Stage 0 100 张验证集，20 图回归集 + Stage 0 补充 = 97 样本）：

```bash
cd stage0_pipeline
../.venv-m2/bin/python scripts_c2/export_bootstrap_dataset.py   # 97 样本
../.venv-m2/bin/python scripts_c2/fit_region_params.py          # 208 class-rows
../.venv-m2/bin/python scripts_c2/train_per_class_head.py       # held-out MAE=4.02（修复后重跑）
```

样本量从 41 扩到 208 后，held-out MAE / 基线 的降幅比例几乎不变（34.3% → 33.4%），说明规则教师信号稳定可泛化。过程中顺带修复了 `fit_region_params.py` 里一个真 bug：一张 person_event 照片的天空区域原图近乎纯色，导致 Lab-affine scale 除以近零方差爆炸到 68 倍，改为方差过低时 scale 退化为 1.0 + 只用均值差算 shift（详见设计稿 §7）。**这个 bug 最初被误判为"假天空检测"（语义问题），C1c 实验用 Qwen3-VL 复核 + 人工看原图后确认那其实是真实过曝天空——bug 纯粹是数值拟合问题，见 `outputs/phase-c1c-vlm-sky-gate-results.md`。**

**ridge → MLP 升级尝试（`train_per_class_head_mlp.py`）**：用 5-fold CV（仅在训练集内部）选出最优 MLP 配置后，在同一个 held-out 测试集上评估，结果 ridge (4.20) 明显优于 MLP (6.13)——训练集内 CV 分数两者几乎一样（3.26 vs 3.27），但 MLP 迁移到新样本上的能力更弱，说明 n=208 还没到能撑起非线性模型的规模。**继续用 ridge (v0) 作生产头**，详见 `outputs/phase-c2.3b-mlp-head-experiment.md`。

**过冲光晕瑕疵修复 + C2 重跑（2026-07-10）**：Web Demo 肉眼验证真实回归图时发现天空/树冠边界有不自然的光晕，排查后确认跟分割精度（BSHM 之类的抠图模型）无关，根因是 `STRENGTH_PRESETS` 里 `cs>1` 的"过冲"设计在羽化边界/材质混杂区域上失控。已实现按局部方差自适应抑制过冲的修复（`_class_outlier_confidence`），20 图回归验证门槛决策不变、视觉瑕疵明显改善；因为 C2 伪标签的老师就是这套 medium 档，重跑了 C2.1→C2.3 全流程，新 held-out MAE=4.02（略优于修复前 4.20）。详见 `outputs/phase-teacher-overshoot-halo-fix.md`。

**neutral 改用加法 mood-cast + C2 再次重跑（同日）**：用户拿一张密集人群商超照片测试，反馈"只模仿了背景，前景没反应"。排查发现人群没被 Grounding DINO 识别成 `clothing`，掉进保守的 `neutral` 兜底类；试着单独给"人群"开类别用重缩放式分级，结果把所有人的不同颜色衣服洗成同一种色调，比不处理还差。改成 `_grade_neutral_additive()`——对 neutral 区域做固定加法偏移而不是重缩放，保留每个像素跟邻居的相对色差（衣服多样性），只把整片区域的重心朝参考图挪一点。20 图回归验证通过，重跑 C2.1→C2.3，新 held-out MAE=4.07（跟上一版 4.02 基本持平，噪声级波动）。详见 `outputs/phase-teacher-neutral-mood-cast.md`。

**同标签类别外观差过大压制强度 + C2 第三次重跑（同日）**：用户拿商场/机场密集人群照片测试，反馈"背景跟前面人物严重脱节"，质疑要不要退回整图仿色，并要求回顾昨天 Polarr 仿色分析的结论。查文档确认整图仿色是早就踩坑退回来的旧方案（会串色）。实测发现根因是开放词汇标签"building"同时框住了参考图"商场白色吊顶"（L=78）和目标图"钢架顶棚"（L=49）这两种物理上完全不同的东西，ΔL≈29.4 还叠加过冲系数，硬拉出了不自然的青蓝色块。新增 `_class_pair_confidence()`——按类别整体统计量的绝对 Lab 差距（不做内部方差归一化）压制过冲，跟已有的像素级 outlier confidence 相乘一起生效。20 图回归验证通过，顺带修好另一张过曝背景模糊的图。重跑 C2.1→C2.3，新 held-out MAE=3.74（上一版 4.07，这次是真实改善而非噪声）。详见 `outputs/phase-teacher-class-mismatch-fix.md`。

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
