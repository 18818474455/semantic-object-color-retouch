---
name: semantic-color-retouch
description: Acts as the entry point for the "语义物体调色专家" (semantic object color retouch) project — a standalone predev for photo color-matching (仿色) that recognizes semantic regions (sky, skin, clothing, buildings, grass...) and applies per-region color grading from a reference photo. Use when the user mentions 仿色/调色/color reference transfer/Chroma Engine student model/C1/C2/C1c phases, or asks to resume work on this project from the 整体代码1.0/仿色模型 baseline.
---

# 语义物体调色专家 (Semantic Object Color Retouch)

## What this project is

Standalone R&D codebase (independent of the 云享传 App repo) that takes a *reference* photo + a *target* photo, detects semantic regions in both (sky/skin/clothing/building/grass/...) via Grounding DINO + SAM, and grades each region of the target towards the reference's Lab color statistics — a smarter, region-aware version of "仿色" than naive global histogram matching. End goal: distill this into a lightweight per-region MLP head that plugs into the company's existing Smart Color v2 / Chroma differentiable renderer (no dependency on unstable external APIs).

**This folder (`整体代码1.0/仿色模型/`) is the canonical and only local base.** Save all future code/doc changes here. It is a full git repo (`git remote -v` → `github.com/18818474455/semantic-object-color-retouch`). The former `/Users/mac/Documents/Codex/2026-07-05/gpt-image-2/` copy was deleted after migration.

## Directory map

```
outputs/                    开发方案 v1→v2→v3(执行版)→v3.1 增补 + 各阶段实验/修复文档
stage0_pipeline/
  scripts/                  Stage 0 主链路（无需大模型）
  scripts_m2/               仿色正式版 color_reference_transfer.py（Grounding DINO+SAM，正式技术栈）
  scripts_c1c/               C1c：Qwen3-VL 语义门控实验
  scripts_c2/                C2：Reference 自蒸馏 bootstrap 导出/拟合/训练脚本
  webdemo/                    仿色 Web Demo（Flask，参考图+目标图+强度滑杆）
  secrets/api.local.json      GPT Image 2 / API易 密钥（gitignored，从 .example 复制）
  requirements-venv.txt       .venv (py3.14) 依赖快照 — Stage 0 主链路
  requirements-venv-m2.txt    .venv-m2 (py3.13) 依赖快照 — Grounding DINO/SAM/torch/Qwen-VL
```

Read `README.md` at project root first — it has current status, quick-start commands, and a live "建议下一步" priority list (kept up to date after every milestone).

## Resuming environment (venvs were NOT copied — too large, 1GB+)

```bash
cd /Users/mac/Desktop/整体代码1.0/仿色模型
python3.14 -m venv .venv && .venv/bin/pip install -r stage0_pipeline/requirements-venv.txt
python3.13 -m venv .venv-m2 && .venv-m2/bin/pip install -r stage0_pipeline/requirements-venv-m2.txt
```
If the exact python minor version isn't available, install the closest 3.13/3.14 and re-freeze if there are conflicts.

## Phase status (as of 2026-07-10, see README.md for latest)

- **Phase 0 / 仿色产品化**: done. Style-profile caching, strength presets, content-match gating, 20-image cross-scene regression all pass.
- **Phase C split into three tracks** (do not conflate):
  - **C1** — GPT Image 2 (via API易) as a quantitative teacher. Auxiliary, not blocking.
  - **C2** — Reference self-distillation (**main line**): use the rule-based `color_reference_transfer.py` itself as a pseudo-teacher to bootstrap a dataset, fit per-class Lab-affine params, train a `PerClassHead` (ridge regression beat an MLP upgrade at n=208 rows — see `outputs/phase-c2.3b-mlp-head-experiment.md`).
  - **C1c** — Local VLM (Qwen3-VL) semantic gating experiment. De-prioritized: 100% agreement with existing heuristic, motivating bug turned out to be numerical not semantic.
- **Current held-out MAE**: 3.74 (ridge, n=208 rows), after three teacher bug fixes discovered via the Web Demo (see below).
- **Current main line**: C3 Teacher v1 coherence upgrade — global mood base (C3-1, done) + trust-controlled regional residuals (C3-2, in progress). The old formula `1 + (cs_base - 1) * confidence` left a full statistical match at confidence=0, so the previous patch-based teacher is frozen as `legacy_v0`. M3.7 is paused until `FG-BG-Coord-v1` passes visual acceptance.
- **C3-0 baseline**: `stage0_pipeline/baselines/c3-0/legacy_v0/manifest.json` binds teacher code, 97-sample/208-row C2 data and ridge head to commit `85edb68` and SHA-256 hashes. `pipeline=legacy` is the default.
- **C3-1 global mood base (done)**: `coherence_controller.py` — a hard-capped whole-image additive Lab shift (ΔL≤10, Δab≤9), scaled by content-match confidence and preset tier, skin halved. Verified on the two original bug photos: the material-mismatch disconnect (`person_event_DSC04819_r2`) and the overshoot halo (`outdoor_sky_DSC04085(1)`) are both visibly gone. See `outputs/phase-c3-1-global-mood-base.md`.
- **C3-2 (done)**: `_render_coherence_from_analysis` now grades each class FROM `base_lab` (not the raw target) to get a residual (`delta = graded - base_lab`), then scales it by `region_strength = region_cap * trust` where `trust = scene_confidence * pair_confidence * homogeneity_confidence * pixel_confidence`. `pair_confidence`/`pixel_confidence` reuse the existing `_class_pair_confidence`/`_class_outlier_confidence` (pair_confidence now recomputed against `base_lab`); `_region_homogeneity_confidence` is new (Lab-variance-only proxy for "is this region one material"). New `region_default/region_skin/region_neutral` preset caps (1.05–1.25, well under legacy's 1.6–2.0) live in the same `STRENGTH_PRESETS` dicts so webdemo slider interpolation covers them for free. `trust→0` now converges to "stay at the global base", fixing the exact bug the plan doc flagged (`cs = 1 + (cs_base-1)*confidence` never dropped below 1.0). 20-image regression clean, both original bug cases still safe, sky-type regions regained some punch vs C3-1-only. See `outputs/phase-c3-2-region-residual.md`.
- **C3-3 (done)**: `coherence_controller.edge_aware_weights`/`guided_filter` — box-filter guided filter (He et al.) using the target's own L channel as guide, replacing `analyze_target`'s Gaussian-feathered `weights` for the coherence pipeline only (legacy untouched). `_region_homogeneity_confidence` now multiplies its existing Lab-variance term by a new `_region_texture_confidence` (Sobel edge density, region vs whole-photo average). 20-image regression clean, both original bug cases still safe. See `outputs/phase-c3-3-edge-aware-fusion.md`.
- **C3-4 (in progress)**: `eval_harmony.py` (done) — automated metrics (fg/bg luma diff, boundary ΔE, skin hue drift, highlight/shadow clip, region pair confidence) computed from the SAME `analyze_target()` output for both pipelines so comparisons aren't confounded by re-segmenting. `FG-BG-Coord-v1` grown from 20→30 samples (10 new `mall_event_*` from user-pointed `/Volumes/T7/松雅湖吾悦广场/20250501松雅湖吾悦广场/原图`: dense crowd+ceiling, white coats, arcade LED, warm dim restaurant, weak ref-match). Ran on all 30: coherence vs legacy — flags -65%, fg/bg luma diff -85%, boundary ΔE p95/p99 -51%/-47%, skin hue drift -62%, highlight/shadow clip flat-to-better (harder new samples made the gap BIGGER, not smaller — the improvement isn't an artifact of easy old samples). `--summarize --out-root <dir>` aggregates existing metrics.json; `--score-summary --out-root <dir>` aggregates filled-in `review.json` against §四 acceptance thresholds once scoring is done. Each sample dir also has a `review_sheet.jpg` 4-panel composite (reference/target/legacy/coherence) for convenient scoring. See `outputs/phase-c3-4-eval-harmony.md`. **NOT done**: nobody has scored anything yet — `review.json` per sample is still an empty template, this is the one remaining blocker. **Also unresolved**: the specific "商场钢架桁架顶棚(exposed steel roof truss)+dense crowd" original bug photo the plan doc names — searched ~120/712 photos across the user-pointed T7 folder (full contact sheets of start/middle/end), found NONE with an exposed truss ceiling (all standard suspended/vaulted ceilings); substituted "dense crowd + standard mall ceiling" from the same shoot to hit the 30-sample count, but flagged in all docs that this is NOT the original bug photo — ask the user again if it turns out to matter for acceptance. Do not claim C3-4 complete or start C3-5/C3-6 until human scoring is done and the plan doc §四 acceptance thresholds are actually evaluated against real human scores.
- **C3 visual gate**: `stage0_pipeline/eval/fg_bg_coord_v1/` has the inherited 20 regression pairs; add at least 10 real problem pairs before acceptance.
- **Rebuilding `.venv-m2`**: was deleted with the old dev path; rebuilt via `python3.13 -m venv .venv-m2 && .venv-m2/bin/pip install -r stage0_pipeline/requirements-venv-m2.txt` (torch download, a couple minutes).

## Established workflow pattern (repeat this loop for any teacher/grading bug)

1. **Discover visually** via the Web Demo (`stage0_pipeline/webdemo/`, run with `.venv-m2/bin/python webdemo/app.py`, http://127.0.0.1:5057) — numeric Lab MAE alone hides artifacts the naked eye catches (halos, disconnects, washed-out diversity).
2. **Root-cause in `stage0_pipeline/scripts_m2/color_reference_transfer.py`** — check `analyze_target()` (expensive, strength-independent: segmentation + per-class Lab stats/confidence) vs `render_from_analysis()` (cheap, strength-dependent: blending using cached analysis). Most bugs live in the grading math (`cs` overshoot factor, per-class rescale vs additive shift).
3. **Fix + write a doc** in `outputs/phase-teacher-<bug-name>-fix.md` explaining root cause, fix, and validation.
4. **Validate**: run the 20-image cross-scene regression before/after.
5. **Rerun the full C2 pipeline** (teacher changed → training data is stale):
   ```bash
   .venv-m2/bin/python stage0_pipeline/scripts_c2/export_bootstrap_dataset.py   # C2.1
   .venv-m2/bin/python stage0_pipeline/scripts_c2/fit_region_params.py          # C2.2
   .venv-m2/bin/python stage0_pipeline/scripts_c2/train_per_class_head.py       # C2.3, needs required_permissions:["all"] to write model json
   ```
6. **Update docs**: `README.md`'s "建议下一步", `outputs/phase-c2-reference-self-distill-design.md` §10, `outputs/semantic-object-color-retouch-dev-plan-v3.1-c2-addendum.md`, `outputs/development-start-checklist.md`, and the Obsidian handover doc at `/Users/mac/Documents/云享传公司/02-需求与规划/语义物体调色专家-项目现状与接手指南.md`.
7. Commit + push (git remote already configured).

## Known unsolved issues (don't re-litigate, just pick up)

- Broad cyan/cool cast on building facades (whole-region, not a boundary halo) — separate mechanism from the fixed overshoot-halo bug, still open.
- "White ghost" residual artifact — flagged as pending in the Obsidian handover doc, not yet root-caused.

## Related knowledge base docs (read for context, don't duplicate)

- `/Users/mac/Documents/云享传公司/02-需求与规划/语义物体调色专家-项目现状与接手指南.md` — human-facing handover guide, most current narrative.
- `/Users/mac/Documents/云享传公司/09-行业方案与知识库/App/2026-07-09-ModelScope图像处理类模型盘点.md` and `...图文多模态VLM模型盘点.md` — surveyed alternative models (BSHM matting, Qwen-VL family) considered and why they were/weren't adopted.
