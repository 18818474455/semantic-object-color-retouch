# Phase C3-4：自动化协调性指标（eval_harmony.py）

日期：2026-07-10
状态：自动指标脚本已实现并跑通现有20组；**30组人工评审集尚未补齐，人工打分尚未开始**——这两项是本阶段唯一还没做完的工作，不能跳过就宣布C3-4完成。

## 做了什么

### 新增 `stage0_pipeline/scripts_m2/eval_harmony.py`

按方案文档 §四实现的自动指标，都从同一次 `analyze_target()` 输出算，保证 legacy/coherence 对比时不会因为两次分析用了不同的分割/权重而失去可比性：

| 指标 | 含义 |
|---|---|
| `fg_bg_luma_change_diff` | 前景（skin/clothing/neutral）平均ΔL vs 背景（sky/building/...）平均ΔL 的绝对差 |
| `fg_bg_tone_direction_cos` | 前景色调变化向量 vs 背景色调变化向量的余弦相似度（只在两者变化幅度都>0.3时才有意义，否则为null） |
| `neutral_vs_bg_change_ratio` | neutral区域变化幅度 / 背景区域变化幅度 |
| `boundary_delta_e_p95`/`p99` | 蒙版边界像素（任一类别混合权重梯度落在前10%）的输出-原图ΔE分位数 |
| `skin_hue_drift_deg` | 皮肤区域输出前后的平均色相角偏移 |
| `highlight_clip_frac`/`shadow_clip_frac` | 输出RGB任一通道触顶(≥0.999)/触底(≤0.001)的像素占比 |
| `region_max_delta_e_p95` | 每个已匹配类别自己的ΔE p95（逐类别） |
| `region_pair_confidence` | 每个已匹配类别的 `_class_pair_confidence`（复用现有函数） |

`compute_harmony_metrics(analysis, out_rgb)` 是核心函数；`run_pair()` 对同一对(ref,tgt)跑 legacy 和 coherence 两次渲染，产出可直接对比的两组指标；`run_manifest()` 批量跑 `FG-BG-Coord-v1` 的 `manifest.jsonl`，按README约定的目录结构写 `reference.jpg`/`target.jpg`/`legacy_v0.jpg`/`coherence_v1.jpg`/`metrics.json`，并为没有 `review.json` 的样本创建空评分模板。`_flag_summary()` 是保守的"值得人看一眼"提示，不是通过/不通过的硬门槛。

新增 `--summarize` 模式：聚合已有 `metrics.json`，产出 legacy vs coherence 的均值对比表和flag计数——用于快速看"整体上是不是变好了"，不能替代人工评审（方案文档 §四明确写了这条）。

### 在现有20组样本上跑通

```bash
.venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py --manifest stage0_pipeline/eval/fg_bg_coord_v1/manifest.jsonl
.venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py --summarize --out-root stage0_pipeline/eval/fg_bg_coord_v1/outputs
```

20组全部跑通，无报错。

## 结果：legacy_v0 vs coherence_v1 汇总对比（n=20）

| 指标 | legacy_v0 | coherence_v1 | 变化 |
|---|---|---|---|
| 自动flag总数 | 37 | 18 | **-51%** |
| `fg_bg_luma_change_diff`（越小越协调） | 9.70 | 1.83 | **-81%** |
| `fg_bg_tone_direction_cos`（越大越协调，>0才算方向一致） | 0.051 | 0.328 | **+0.28** |
| `neutral_vs_bg_change_ratio`（越接近1越协调） | 2.75 | 2.05 | -0.70（仍偏高） |
| `boundary_delta_e_p95`（越小越无光晕） | 26.3 | 14.8 | **-44%** |
| `boundary_delta_e_p99` | 32.7 | 19.7 | **-40%** |
| `skin_hue_drift_deg`（越小越自然） | 6.48 | 4.34 | -33% |
| `highlight_clip_frac` | 0.227 | 0.239 | +0.012（基本持平） |
| `shadow_clip_frac` | 0.292 | 0.322 | +0.030（基本持平） |

**解读**：跟C3-1/C3-2/C3-3每一步的视觉抽查结论完全一致——`coherence` 在"前后景明暗差异"、"边界ΔE"（光晕代理指标）上都有非常明显（40%~80%）的数值改善，肤色更稳定，前后景色调方向从"基本无关"（0.05）变成"有正相关"（0.33，虽然还没到强相关）。高光/暗部裁剪比例基本没变化，说明改善不是靠简单地把画面拉灰/降低对比度换来的。`neutral_vs_bg_change_ratio` 仍然偏高（2.05，理想值接近1），是目前唯一一个数值上还不够理想的指标，值得在C3-5之前再看一眼具体样本。

**重要限制**：n=20，且这20组是"继承自旧回归集"的样本，不是专门为"前后景协调性"设计和挑选的（很多是天空/舞台LED这类跟前后景协调关系不大的场景）。这组数字是有意义的方向性证据，但**不能替代**方案文档要求的30组专项人工评审——数值改善不等于"肉眼看起来更协调、更愿意交付"，尤其像 `fg_bg_tone_direction_cos=0.33` 这种"有正相关但不强"的结果，到底算"过关"还是"还不够"，需要人来判断。

## 还没做完的工作（不能跳过）

1. **`FG-BG-Coord-v1` 还停留在20组，没有补齐到方案文档要求的最低30组**。`stage0_pipeline/eval/fg_bg_coord_v1/README.md` 里点名要求的"商场钢架顶棚+密集人群"原始问题图，**目前仍未确认具体文件路径**——不能用相似照片（比如已经在用的商场吊顶vs户外亭子那组）冒充，需要用户指认或授权在外置硬盘里按内容搜索。
2. **人工1~5分评审一条都没打**：`review.json` 目前只是空模板（`preferred`/`scores`/`severe`全是null），验收标准里的"严重问题0例"、"愿意交付≥80%"、"coherence相对legacy胜率≥70%"三条都需要真人来看图打分，我不能代替签字。
3. 在完成以上两条之前，C3-4 不算真正完成，C3-5（重建C2数据）和C3-6（恢复M3.7嫁接）都不能启动——跟方案文档§五的顺序要求一致。

## 涉及文件

- 新增 `stage0_pipeline/scripts_m2/eval_harmony.py`
- 新增 `stage0_pipeline/eval/fg_bg_coord_v1/outputs/<id>/{reference,target,legacy_v0,coherence_v1}.jpg`（gitignore排除，不进git）+ `metrics.json`（进git）+ `review.json`（进git，等待人工填写）
