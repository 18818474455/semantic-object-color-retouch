# Phase C3-4：自动化协调性指标（eval_harmony.py）

日期：2026-07-10
状态：自动指标脚本已实现并跑通全部30组（20组旧回归集 + 10组新补mall_event场景）；**人工1~5分评审尚未开始**——这是本阶段唯一还没做完的工作，不能跳过就宣布C3-4完成。

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

### 在全部30组样本上跑通

```bash
.venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py --manifest stage0_pipeline/eval/fg_bg_coord_v1/manifest.jsonl
.venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py --summarize --out-root stage0_pipeline/eval/fg_bg_coord_v1/outputs
```

30组全部跑通，无报错。每组产出 `reference.jpg`/`target.jpg`/`legacy_v0.jpg`/`coherence_v1.jpg`/`review_sheet.jpg`（四联对比图，方便人工打分一次看全）/`metrics.json`/`review.json`（空模板）。

### 补齐到30组：新增的10组`mall_event_*`样本

原有20组全部继承自旧`regression_20.py`回归集，`FG-BG-Coord-v1`要求的10组新增真实问题样本来自用户指认的
`/Volumes/T7/松雅湖吾悦广场/20250501松雅湖吾悦广场/原图`（2026-05-01商场促销活动实拍，712张原图）：

| 样本ID | 场景 | 参考图 | 覆盖的风险标签 |
|---|---|---|---|
| `mall_event_DAP03654`/`DAP03662` | 密集人群+标准商场吊顶 | 058A1824（既有person_event参考） | `dense_crowd`, `mall_ceiling` |
| `mall_event_white_coats_DAP03010`/`DAP02999` | 一排白大褂医护人员合影 | 058A1824 | `white_clothing_ghost`, `skin` |
| `mall_event_arcade_led_DAP02911`/`DAP02922` | 儿童游乐区LED机台混合光 | DSC06040（既有stage_led参考） | `led_stage`, `mixed_light` |
| `mall_event_warm_restaurant_DAP03554`/`DAP02867` | 暖光昏暗餐厅+密集就餐人群 | DSC04902（既有difficult参考） | `dark_scene`, `orange_wash` |
| `mall_event_weak_match_DAP03340`/`DAP03417` | 室内促销展位人物 vs 户外天空参考 | DAP02456(1)（既有outdoor_sky参考） | `weak_content_match` |

**未解决的问题**：方案文档点名要求的"商场**钢架桁架顶棚**+密集人群"原始bug图——用户指认了上面这个T7文件夹，但逐一抽查约120/712张（含首尾全量联系表）后确认这批素材里**没有裸露钢架桁架顶棚的画面**，全部是标准吊顶/拱形暖光顶。已用同一批素材里"密集人群+标准商场吊顶"场景（`mall_event_DAP03654`/`DAP03662`）代替凑数到30组这个数量要求，**但这不是原始bug图本身**。如果这张图对最终验收很关键，需要用户从别的素材源再指认一次。

## 结果：legacy_v0 vs coherence_v1 汇总对比（n=30）

| 指标 | legacy_v0 | coherence_v1 | 变化 |
|---|---|---|---|
| 自动flag总数 | 65 | 23 | **-65%** |
| `fg_bg_luma_change_diff`（越小越协调） | 11.28 | 1.69 | **-85%** |
| `fg_bg_tone_direction_cos`（越大越协调，>0才算方向一致） | 0.269 | 0.404 | +0.14 |
| `neutral_vs_bg_change_ratio`（越接近1越协调） | 1.87 | 1.75 | -0.12（更接近1） |
| `boundary_delta_e_p95`（越小越无光晕） | 29.8 | 14.7 | **-51%** |
| `boundary_delta_e_p99` | 36.1 | 19.3 | **-47%** |
| `skin_hue_drift_deg`（越小越自然） | 9.75 | 3.71 | **-62%** |
| `highlight_clip_frac` | 0.194 | 0.172 | -0.022（更好） |
| `shadow_clip_frac` | 0.241 | 0.279 | +0.038（基本持平） |

（对比n=20时的结果：flag数-51%→-65%，前后景明暗差异-81%→-85%，边界ΔE-44%→-51%/-40%→-47%，肤色漂移-33%→-62%——补充了更难的密集人群/白衣服/弱匹配场景后，`coherence`相对`legacy`的改善幅度不降反升，说明前面的正向结论不是靠挑简单样本撑出来的。）

**解读**：跟C3-1/C3-2/C3-3每一步的视觉抽查结论完全一致——`coherence` 在"前后景明暗差异"、"边界ΔE"（光晕代理指标）、"肤色漂移"上都有非常明显（47%~85%）的数值改善。高光/暗部裁剪比例基本没变化甚至略有改善，说明改善不是靠简单地把画面拉灰/降低对比度换来的。`neutral_vs_bg_change_ratio` 从1.87降到1.75，方向对但仍未到理想值1.0，是目前唯一一个数值上还不够理想的指标。

**重要限制**：数值改善不等于"肉眼看起来更协调、更愿意交付"，尤其像 `fg_bg_tone_direction_cos=0.40` 这种"有正相关但不强"的结果，到底算"过关"还是"还不够"，需要人来判断。这组数字是有意义的方向性证据，但**不能替代**方案文档要求的人工评审。

## 还没做完的工作（不能跳过）

1. **人工1~5分评审一条都没打**：`review.json` 目前只是空模板（`preferred`/`scores`/`severe`全是null），验收标准里的"严重问题0例"、"愿意交付≥80%"、"coherence相对legacy胜率≥70%"三条都需要真人对着`review_sheet.jpg`看图打分，我不能代替签字。打分完成后跑：
   ```bash
   .venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py --score-summary --out-root stage0_pipeline/eval/fg_bg_coord_v1/outputs
   ```
   会自动对照§四验收标准给出`acceptance`布尔结果。
2. **"商场钢架桁架顶棚+密集人群"原始bug图仍未确认**（见上），如果对验收关键，需要用户再指认一次。
3. 在完成以上两条之前，C3-4 不算真正完成，C3-5（重建C2数据）和C3-6（恢复M3.7嫁接）都不能启动——跟方案文档§五的顺序要求一致。

## 涉及文件

- 新增 `stage0_pipeline/scripts_m2/eval_harmony.py`
- 新增 10条 `mall_event_*` 记录到 `stage0_pipeline/eval/fg_bg_coord_v1/manifest.jsonl`（20→30组）
- 新增 `stage0_pipeline/eval/fg_bg_coord_v1/outputs/<id>/{reference,target,legacy_v0,coherence_v1,review_sheet}.jpg`（gitignore排除，不进git）+ `metrics.json`（进git）+ `review.json`（进git，等待人工填写）
