---
name: semantic-color-retouch
description: Acts as the entry point for the "语义物体调色专家" (semantic object color retouch) project — a standalone predev for photo color-matching (仿色) that recognizes semantic regions (sky, skin, clothing, buildings, grass...) and applies per-region color grading from a reference photo. Use when the user mentions 仿色/调色/color reference transfer/Chroma Engine student model/C1/C2/C1c phases, or asks to resume work on this project from the 整体代码1.0/仿色模型 baseline.
---

# 语义物体调色专家 (Semantic Object Color Retouch)

## What this project is

Standalone R&D codebase (independent of the 云享传 App repo) that takes a *reference* photo + a *target* photo, detects semantic regions in both (sky/skin/clothing/building/grass/...) via Grounding DINO + SAM, and grades each region of the target towards the reference's Lab color statistics — a smarter, region-aware version of "仿色" than naive global histogram matching. End goal: distill this into a lightweight per-region MLP head that plugs into the company's existing Smart Color v2 / Chroma differentiable renderer (no dependency on unstable external APIs).

**This folder (`整体代码1.0/仿色模型/`) is the canonical base.** Save all future code/doc changes here. It is a full git repo (`git remote -v` → `github.com/18818474455/semantic-object-color-retouch`), synced from the original dev path `/Users/mac/Documents/Codex/2026-07-05/gpt-image-2/`.

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
- **Current main line**: M3.7 — graft the trained `PerClassHead` into Smart Color v2 / Chroma Engine's differentiable renderer (`feature/regional-smart-color-head` branch in the Chroma repo).

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
