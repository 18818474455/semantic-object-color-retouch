# FG-BG-Coord-v1

前景/背景协调性专项人工评审集。C3-0 先冻结已有 20 组回归样本；C3-1 前补齐至少 10 组真实问题样本，使总数达到最低 30 组。

## 当前覆盖

- **已登记：30 组，达到方案文档最低要求**（20组继承自`regression_20.py`旧回归集 + 10组`mall_event_*`新补充，均已用`eval_harmony.py`跑通并生成`metrics.json`/`review_sheet.jpg`/空`review.json`模板）
- 新补充的10组来自 `/Volumes/T7/松雅湖吾悦广场/20250501松雅湖吾悦广场/原图`（2026-05-01商场促销活动实拍），覆盖：密集人群+商场吊顶(`mall_event_DAP03654`/`DAP03662`)、白/浅色衣服(`mall_event_white_coats_*`)、LED游乐设施混合光(`mall_event_arcade_led_*`)、暖光昏暗室内+密集人群(`mall_event_warm_restaurant_*`)、参考图与目标图弱匹配(`mall_event_weak_match_*`)
- **未解决的特别问题样本**：方案文档点名要的"商场**钢架桁架顶棚**+密集人群"原始bug图——在T7这个文件夹里逐一抽查约120/712张（含首尾全量联系表）后确认**没有裸露钢架桁架顶棚的画面**，全部是标准吊顶/拱形暖光顶。已用同一批素材里的"密集人群+标准商场吊顶"场景（`mall_event_DAP03654`/`DAP03662`）代替凑数，**但这不是原始bug图本身**，如果这张图对验收很关键，需要用户从别的素材源再指认一次
- **2026-07-10**：`eval_harmony.py` 已实现并跑通全部30组，`outputs/<id>/metrics.json`（legacy_v0/coherence_v1两套自动指标+flag）+ `review_sheet.jpg`（4联对比图，方便打分时一次看全）已生成；`review.json` 是空模板，**人工打分尚未开始**。汇总对比见 `outputs/phase-c3-4-eval-harmony.md`。

## 每组输出

```text
outputs/<id>/
  reference.jpg
  target.jpg
  legacy_v0.jpg
  coherence_v1.jpg
  metrics.json
  review.json
```

原图不复制进 Git；`manifest.jsonl` 只记录路径与风险标签。输出图片保持在本地并由 `.gitignore` 排除，评分和指标 JSON 可提交。

## 人工评分

每项 1~5 分：

- `foreground_change_visible`
- `background_strength_natural`
- `fg_bg_same_tone`
- `skin_natural`
- `halo_free`
- `local_dirty_color_free`
- `delivery_willingness`

另设严重问题布尔字段：

- `severe_fg_bg_disconnect`
- `severe_halo`
- `severe_skin_error`

`preferred` 取 `legacy_v0`、`coherence_v1` 或 `tie`。

## 验收

- 严重前后景割裂、严重光晕、严重肤色异常均为 0 例
- `delivery_willingness >= 4` 的比例不低于 80%
- `coherence_v1` 相对 `legacy_v0` 的人工胜率不低于 70%
- 自动指标仅作异常门禁，不能代替人工验收

## C3-0 状态

`legacy_v0` 的源码、C2 数据和 ridge head 由
`stage0_pipeline/baselines/c3-0/legacy_v0/manifest.json` 绑定到 Git commit
`85edb68` 及 SHA-256。运行基线：

```bash
.venv-m2/bin/python stage0_pipeline/scripts_m2/regression_20.py --pipeline legacy
```

`coherence` 现已实现（C3-1~C3-3：全局氛围基底+区域受信任度残差+边缘感知融合）。

## 如何跑自动指标

```bash
# 对整份清单跑一遍（重新分割+渲染+写metrics.json/review.json模板）
.venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py \
  --manifest stage0_pipeline/eval/fg_bg_coord_v1/manifest.jsonl

# 只汇总已有 metrics.json，不重新渲染
.venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py \
  --summarize --out-root stage0_pipeline/eval/fg_bg_coord_v1/outputs
```

自动指标只是异常门禁/参考证据，**不能代替下面的人工评分**——`review.json` 里的字段要由人对着 `outputs/<id>/review_sheet.jpg`（reference/target/legacy_v0/coherence_v1 四联对比图，也可以分开看同目录下的4张单图）实际打分后回填。

## 打分网页（推荐用这个，不用手改JSON）

```bash
cd stage0_pipeline/eval/fg_bg_coord_v1
../../../.venv-m2/bin/python review_app.py
# 打开 http://127.0.0.1:5058
```

一次只跑纯Python/Flask，不加载任何模型，秒开。功能：

- 首页自动跳到第一个还没打分的样本
- 每页显示 `review_sheet.jpg`（4联对比图）+ 风险标签 + 内容匹配gate状态 + 自动指标flag提示
- 7项1~5分打分（点数字按钮）、3项严重问题勾选、更喜欢哪一版单选、备注
- "保存并下一组/上一组"、"仅保存"三个按钮，顶部有已打分/总数进度条
- `/summary` 路由直接看聚合结果（等价于命令行的`--score-summary`）

打完全部30组后，可以继续用网页的`/summary`，也可以用命令行：

```bash
.venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py --score-summary --out-root stage0_pipeline/eval/fg_bg_coord_v1/outputs
```

## 如何汇总人工评分（评分完成后）

```bash
.venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py \
  --score-summary --out-root stage0_pipeline/eval/fg_bg_coord_v1/outputs
```

会输出：已打分/未打分数量及ID列表、三类严重问题计数、`delivery_willingness>=4`比例、`coherence_v1`相对`legacy_v0`胜率，以及逐条对照§四验收标准的布尔结果（`acceptance`字段）。未打分样本会被排除在比例计算之外并单独列出，不会因为评审没做完而虚报100%通过。
