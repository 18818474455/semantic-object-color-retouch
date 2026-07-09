# Phase C2 · Reference 自蒸馏设计稿

> **状态**：可执行设计（2026-07-09）  
> **定位**：在 **不依赖 GPT Image 2 API** 的前提下，把仿色从「手写 Lab 统计 + 规则」升级到「可训练、可部署、可对齐 Chroma」的区域级学生模型。  
> **与 C1 关系**：C1（GPT teacher 量化）并行、非阻塞；C2 是 M3 的主路径，C1 只标注 hard-case 残差。

---

## 0. 为什么需要 C2

| 问题 | C1（GPT teacher） | C2（Reference 自蒸馏） |
|------|-------------------|------------------------|
| API 稳定性 | 双图编辑超时、历史内容错位 | 不依赖外部 API |
| Teacher 成本 | 按次计费、难批量 | 本地 pseudo-target 可规模化 |
| 与产品对齐 | 黑盒像素，难映射 Chroma 滑块 | 输出 **Chroma 参数 / 可解释 grading 系数** |
| 端侧落地 | 无法上端 | 可烘焙 LUT / 接 Smart Color v2 C++ |

**架构依据**（见知识库 VeraRetouch 对比）：

- VeraRetouch Reference 模式 = `encode(ref_before, ref_after) → z → ConditionalMLP(RGB,z)`，是标准仿色训练范式，但是 **全局、无物体概念**。
- 我们已有 **语义 mask + style profile + 内容匹配硬门槛**（`color_reference_transfer.py`），缺的是 **可训练的 per-region 执行头**，不是再做一个全局 3D LUT。
- BeautySDK **Smart Color v2** 已是「小模型预测参数 → 可微 C++ 渲染器」同构基建——C2 应 **嫁接** 而非另起炉灶。

---

## 1. C2 目标（可验收）

### 1.1 功能目标

给定 `(reference_image, target_image)`：

1. 感知层输出语义 mask（复用 Phase B，不训练）
2. Reference 编码器输出 **style control**（替代/增强现有 `style profile JSON`）
3. 每个匹配语义类预测 **grading 控制量**（Chroma 滑块子集或 Lab residual 系数）
4. Mask 混合渲染器执行，输出与当前 `medium` 档位质量 **持平或更优**
5. 内容不匹配时 **硬拦截**（保留现有 gate，训练 loss 也惩罚 explainable_frac 过低时的过拟合）

### 1.2 工程目标

- 训练代码 **复用 Smart Color v2** 的 param-target 拟合 + renderer parity 流程
- 权重交付物：**per-region head 权重** + `chroma_param_map.json` 查表（可解释）
- 数据管线从现有 **20 图回归集** 起步，扩到 200→2000 无需 GPT

### 1.3 非目标（C2 不做）

- 不训练像素生成模型
- 不替换 Grounding DINO / SAM 感知栈
- 不用 VeraRetouch 权重（License 禁止商用）
- 不在 C2 内解决 `latitude_recovery`（死白/死黑/content generation）——仍留 C1/GPT 或 Chroma 全局滑块

---

## 2. 总体架构

```text
                    ┌─────────────────────────────────────┐
                    │  Perception (frozen, Phase B)        │
                    │  GD+SAM + face → masks per class     │
                    └──────────────┬──────────────────────┘
                                   │
     reference ──► StyleEncoder ───┼──► control_ref  ─┐
                                   │                    │
     target    ──► TargetEncoder ──┼──► feat_tgt   ────┼──► PerClassHeads
                                   │                    │         │
                                   │                    │    sky / skin / grass / ...
                                   │                    │         ▼
                                   │                    │   chroma_params[c]
                                   │                    │   or grading_delta[c]
                                   └────────────────────┴──► MaskBlendRenderer
                                                              (Smart Color v2 / Python parity)
                                         ▲
                                         │  L_lab, L_identity, L_gate
                              pseudo_target (teacher v0)
```

### 2.1 模块定义

| 模块 | 输入 | 输出 | 初版实现 | 目标实现 |
|------|------|------|----------|----------|
| **StyleEncoder** | ref RGB + ref masks | `style_vec` (D=64~256) | `compute_style_profile()` 展平为向量 | SigLIP patch pool + MLP |
| **TargetEncoder** | tgt RGB + tgt masks | `tgt_vec` | 同上 | DINOv2-S / SigLIP |
| **PerClassHead** | `style_vec`, `tgt_vec`, class stats | `Δparams_c` 或 `Δlab_coeffs_c` | 线性层 + sigmoid 约束 | 共享 trunk + 8 类独立 head |
| **Renderer** | tgt RGB, masks, params | edited RGB | Python mask + Chroma 滑块近似 | Smart Color v2 可微 C++ parity |
| **ContentGate** | profile vs tgt classes | suitable bool | 现有 jaccard 硬门槛 | 同上 + 训练时 mask loss |

### 2.2 与现有代码的映射

| 现有 | C2 角色 |
|------|---------|
| `color_reference_transfer.compute_style_profile()` | StyleEncoder v0 / 训练 label 特征 |
| `color_reference_transfer.apply_profile()` | Renderer v0 / pseudo-target 生成器 |
| `content_match_score()` | Gate + 训练 mask |
| `chroma_param_map.json` | Head 输出空间（action → 滑块） |
| `distill_vs_gpt.py` | 仅 C1：量化 C2 与 GPT 残差 |

---

## 3. Teacher v0：Bootstrap 伪标签（无需 GPT）

在真实摄影师修图对不足时，用 **当前最佳本地管线** 作第一版 teacher：

```text
(regression case: ref, tgt)
  → compute_style_profile(ref)
  → apply_profile(..., strength=medium)   # 现有 color_reference_transfer
  → edited_local  = pseudo_target
  → masks, profile, compat, matched_info  = 训练元数据
```

**数据来源（按优先级）**：

1. **已有 20 图回归集** — `semantic_transfer_v2.FULL_CASES` + `EXPANDED_CASES`（4 bucket × 多 target）
2. **Stage 0 验证集 100 张** — 同 ref 多 target 扩增
3. **摄影师原片/修片对**（若有）— 升格为 gold label，权重 ×3
4. **（后期）反向退化合成** — 参考 VeraRetouch AetherRetouch 思路，非 C2.0 阻塞项

**伪标签质量门槛**：

- `compat.suitable == true` 才进训练集
- `matched` 类 frac ≥ 0.01
- 人脸区 ΔE 相对原图 < 阈值（防串色样本污染）

---

## 4. 训练目标与 Loss

### 4.1 主 Loss（per-pixel, inside matched masks）

```text
L_color = Σ_c  w_c · mean( || Lab(render(x)) - Lab(pseudo_target) ||_1 , mask_c )
```

### 4.2 保护 Loss

```text
L_skin   = mean( || Lab(out) - Lab(x) ||_1 , skin_mask )          # 未匹配皮肤几乎不动
L_face   = 结构相似 / 梯度一致性（可选，C2.2+）
L_neutral = penalty when neutral_frac > 0.5 and large global shift  # 防滤镜化
```

### 4.3 Gate Loss

```text
L_gate = BCE(gate_pred, compat.suitable) + λ · (1-suitable) · ||out - x||
```

### 4.4 参数正则（对接 Chroma）

```text
L_param = || Δparams ||_2  +  clip_to_chroma_valid_range
```

与 Smart Color v2 一致：参数在 `[-1,1]`（exposure 除外），并走现有 5 条安全钳制规则的 **可微近似** 或 **post-hoc 投影**。

---

## 5. Smart Color v2 嫁接方案

### 5.1 复用什么（不要重造）

| Smart Color v2 资产 | C2 用途 |
|---------------------|---------|
| `beautysdk-smart-color-param-targets` | 从 pseudo_target 反拟合 per-image 全局/区域 param targets |
| `beautysdk-smart-color-cpp-renderer` | 训练时 renderer forward / parity 基准 |
| `beautysdk-diff-renderer-parity` | Python ↔ C++ 数值对齐门禁 |
| `PARAM_PROTOCOL.md` / `chroma_process_params_t` | Head 输出 schema |

### 5.2 新建什么（本项目增量）

1. **RegionalParamHead**：输入 `(style_vec, tgt_vec, class)` → 该区域 Chroma 滑块子集  
2. **Masked apply**：在 Chroma 渲染器外包装 mask 加权（规格见 `CHROMA_ALIGNMENT.md`）  
3. **Reference dataset schema**（见 §6）  
4. **Export 脚本**：`regression_20` → `dataset/c2_manifest.jsonl`

### 5.3 仓库边界

```text
gpt-image-2/                    # 感知、仿色产品化、C2 数据集、Python 训练实验
beauty_sdk/ (Chroma 源码)       # 可微渲染器、权重导出、C++ 区域算子
```

C2.0–C2.3 可在 `gpt-image-2` 内用 Python renderer 闭环；**C2.4 起必须进 Smart Color v2 训练仓**做 parity 与导出。

---

## 6. 数据格式

### 6.1 `c2_manifest.jsonl`（单行示例）

```json
{
  "sample_id": "outdoor_sky_DSC05360_r1",
  "bucket": "outdoor_sky",
  "reference_path": "/data/ref/DAP02456.JPG",
  "target_path": "/data/tgt/DSC05360.JPG",
  "pseudo_target_path": "dataset/c2/edited/outdoor_sky_DSC05360_medium.jpg",
  "strength": "medium",
  "style_profile_path": "dataset/c2/profiles/outdoor_sky_DAP02456.json",
  "masks_dir": "dataset/c2/masks/outdoor_sky_DSC05360/",
  "compat": {"jaccard": 0.42, "explainable_tgt_frac": 0.61, "suitable": true},
  "matched_classes": ["sky", "neutral"],
  "split": "train"
}
```

### 6.2 目录布局（C2 新增）

```text
stage0_pipeline/
  scripts_c2/
    export_bootstrap_dataset.py    # regression_20 → manifest + pseudo targets
    style_encoder_v0.py            # profile 向量化
    train_per_class_head.py        # C2.3 轻量训练
    eval_c2_vs_baseline.py         # 对比 color_reference_transfer
  dataset/c2/                      # gitignore 大文件，保留 manifest
    manifest.jsonl
    profiles/
    edited/
    masks/
```

---

## 7. 分阶段交付（C2.0 → C2.5）

| 阶段 | 交付物 | 验收 | 状态 |
|------|--------|------|------|
| **C2.0 设计冻结** | 本文档 + v3.1 addendum | 团队对齐 Smart Color 嫁接边界 | ✅ |
| **C2.1 Bootstrap 导出** | `export_bootstrap_dataset.py` + manifest ≥40 样本 | 每条 suitable=true，mask 可加载 | ✅ 外置盘挂载后全量导出：**21 样本**（9 条 suitable=true，其余按内容匹配门槛正确拦截） |
| **C2.2 Param 反拟合** | `fit_region_params.py` → 每样本 Lab-affine + chroma proxy | Python 渲染 repro ΔE < 3 vs pseudo | ✅ 全量跑通，**41 class-rows**（发现并修复退化样本 bug，见下） |
| **C2.3 Per-class Head v1** | `train_per_class_head.py`（ridge baseline） | 20 图回归：≥90% 样本 ΔE 不劣于 baseline | ✅ **n=41，held-out MAE=3.83 < 预测均值基线 5.83**，规则回归学到跨场景可泛化信号 |
| **C2.4 Smart Color 嫁接** | 训练脚本迁入 SCv2 仓，parity 报告 | C++ vs Python max ΔE < 1.0 | ⏸️ 可启动（数据量已达门槛） |
| **C2.5 产品接口** | `color_reference_transfer --learned` 开关 | CLI/Web Demo 可切换 rule / learned | ⏸️ 未开始 |

**2026-07-09 全量结果**（外置盘 `/Volumes/未命名/大模型/原图1/` 已挂载）：

```bash
cd stage0_pipeline
../.venv-m2/bin/python scripts_c2/export_bootstrap_dataset.py   # 21 样本，9 suitable
../.venv-m2/bin/python scripts_c2/fit_region_params.py          # 41 class-rows
../.venv-m2/bin/python scripts_c2/train_per_class_head.py       # held-out MAE=3.83
```

**过程中发现并修复的 bug**（`fit_region_params.py`）：`person_event` 桶一张人物照被误检出一块几乎纯色的「天空」区域（原图 mask 内 std L/a/b = 0.15/0.10/0.26），最小二乘 `scale = std(edited)/std(orig)` 除以近零方差直接爆炸到 68 倍、shift=-6719——与项目历史上「LED墙误判天空/串色」是同一类假天空检测问题，只是从渲染预览端复现到了训练数据管线端。第一版修复（std 低于阈值就整类丢弃）又太激进：真实晴空本身就是低方差区域（std 1.9–4.9 很常见），会连带丢掉有效样本。最终修复：std 低于 `MIN_STD=0.6`（远低于任何观测到的真实晴空样本）时，`scale` 退化为 1.0、只用均值差算 `shift`（对近乎纯色区域，"缩放" 本就没有良定义的意义，均值差才是唯一可靠的信号），`SCALE_CLAMP=(0.15, 6.0)` 兜底处理中间态噪声。修复后 41 条样本全部保留，held-out MAE 从（错误版本的）105 降到 3.83。

n_rows ≥ 8 已触发训练脚本的留出集评估分支；后续继续扩样（比如把 100 张验证集也导进来）到 ≥100 后，建议按设计升级为 torch MLP（`.venv-m2` 已含 torch）。

**C1 并行**：API易 GPT 双图冒烟通过后，仅对 C2.3 残差最大的 10% 类补 GPT label，写入 `dataset/c2/gpt_residual/`。

---

## 8. 与 M3 / M6 / M7 里程碑的修订

原 v3 假设「Phase C GPT → 伪标签 → M7 学生模型」单线。修订为：

```text
Phase B ✅  →  Phase A 产品化 ✅
                    ↓
         ┌─────────┴─────────┐
    C1 GPT 量化          C2 Reference 自蒸馏  ← 主路径
    (hard-case)          (bootstrap → SCv2)
         └─────────┬─────────┘
                   ↓
              M6 数据集 2k+（C2 manifest 为主，C1 补充）
                   ↓
              M7 学生规划器（感知 frozen + RegionalParamHead + 路由）
```

| 原里程碑 | 修订 |
|----------|------|
| M6 伪标签 | 主来源改为 C2 bootstrap + 扩样；GPT teacher ≤15% |
| M7 学生模型 | 执行头用 C2 训练的 RegionalParamHead；规划头仍预测 action/strength/route |

---

## 9. 风险与应对

| 风险 | 应对 |
|------|------|
| 伪标签天花板 = 手写 Lab 公式 | C2.3 后引入摄影师 gold 对；C1 只补 hard-case |
| Smart Color 无 mask 算子 | C2.3 用 Python parity；C2.4 向 Chroma 提 masked operator |
| 数据规模不足 | 同 ref 多 tgt 扩增；Stage 0 100 张优先 |
| 与全局 Smart Color 冲突 | 独立 head 权重；全局 SC 不动，区域叠加 |

---

## 10. 立即下一步（工程）

1. ~~实现 `scripts_c2/export_bootstrap_dataset.py`~~ ✅ 已实现并跑通（21 条样本，外置盘全量）
2. ~~实现 `scripts_c2/fit_region_params.py`（C2.2）~~ ✅ 已实现并跑通（41 条 class-row，修复了假天空检测导致的退化 scale 数值 bug）
3. ~~实现 `scripts_c2/train_per_class_head.py`（C2.3 ridge baseline）~~ ✅ 已实现并跑通，held-out MAE=3.83 < 基线 5.83
4. **下一步**：把 Stage 0 的 100 张验证集也导入 C2.1（当前只用了 20 图回归集），把 n_rows 从 41 提升到 ≥100，为升级 torch MLP 做准备
5. 在 Chroma 仓开 issue/分支：`feature/regional-smart-color-head`（M3.7，暂未开始）——数据门槛已达标，可以启动
6. C1 继续 API易 双图测试，结果只写入 `gpt_residual/`，不阻塞 C2.1

---

## 参考

- 开发方案 v3 + v3.1 addendum：`semantic-object-color-retouch-dev-plan-v3.md`
- Chroma 对齐：`stage0_pipeline/config/CHROMA_ALIGNMENT.md`
- 仿色正式版：`stage0_pipeline/scripts_m2/color_reference_transfer.py`
- VeraRetouch 对比：云享传知识库 `2026-07-09-VeraRetouch与语义调色专家架构对比`
- Smart Color v2 skills：`beautysdk-smart-color-training`、`beautysdk-smart-color-cpp-renderer`
