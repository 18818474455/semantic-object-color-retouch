# 语义物体调色专家 (Semantic Object Color Retouch)

独立预研项目：识别照片中的语义区域（天空、人脸、皮肤、服装、草地、建筑等），判断各区域是否应调色及如何调色，并通过本地 Chroma 引擎或 GPT Image 2 执行。

**当前状态（2026-07-09）**：Stage 0 + 仿色产品化已完成；**C2 Reference 自蒸馏为主开发线**；C1 GPT 量化为辅助轨（API易已配置）。

## 目录结构

```
outputs/                    开发方案 v1/v2/v3 + v3.1 + C2 设计稿
stage0_pipeline/
  scripts_m2/               仿色正式版 color_reference_transfer.py
  scripts_c2/               C2 bootstrap 导出与后续训练脚本
```

## 快速开始

```bash
# 主链路（Stage 0，无需 GPU 大模型）
cd stage0_pipeline
../.venv/bin/python scripts/run_stage0.py --limit 10

# 仿色正式版（需 .venv-m2 + Grounding DINO/SAM）
../.venv-m2/bin/python scripts_m2/color_reference_transfer.py \
  --ref /path/to/reference.jpg \
  --tgt /path/to/target.jpg \
  --strength medium \
  --profile-out /tmp/style_profile.json
```

## 开发文档

| 文档 | 路径 |
|------|------|
| 最终执行方案（841 行 + V3.1 增补） | `outputs/semantic-object-color-retouch-dev-plan-v3.md` + `v3.1-c2-addendum.md` |
| C2 Reference 自蒸馏设计稿 | `outputs/phase-c2-reference-self-distill-design.md` |
| C2.1 Bootstrap 导出脚本 | `stage0_pipeline/scripts_c2/export_bootstrap_dataset.py` |
| 开发启动清单 | `outputs/development-start-checklist.md` |
| Stage 0 管线说明 | `stage0_pipeline/README.md` |
| 接手指南（Obsidian） | 云享传知识库 `02-需求与规划/语义物体调色专家-项目现状与接手指南.md` |

## GPT Image 2 API 配置（[API易](https://docs.apiyi.com/)）

复制模板并填入密钥（**不要提交**）：

```bash
cp stage0_pipeline/secrets/api.local.json.example stage0_pipeline/secrets/api.local.json
# 编辑 api.local.json：base_url=https://api.apiyi.com，model=gpt-image-2-all
```

## 建议下一步

1. ~~**C2.1/C2.2/C2.3** 全量导出 + 拟合 + 训练~~ ✅ —— 外置盘挂载后跑通：21 样本 / 41 class-rows，held-out MAE=3.83。过程中发现并修复了一个假天空检测导致 Lab-affine scale 数值爆炸的 bug（详见设计稿 §7）
2. ~~**扩样**：把 Stage 0 100 张验证集也导入 C2.1~~ ✅ —— n_rows 从 41 提到 **208**（97 样本），held-out MAE=4.20 < 基线 6.31，降幅比例（~34%）与扩样前基本一致，信号稳定可泛化
3. **M3.7**：Chroma 仓开 `feature/regional-smart-color-head` 分支，启动 C2.4 Smart Color v2 嫁接（数据门槛已达标）
4. **升级模型**：n=208 已足够，可把 `train_per_class_head.py` 从 ridge baseline 升级为 torch MLP
5. **C1** API易 双图冒烟（并行，不阻塞 C2）
6. Web Demo：参考图 + 目标图 + 强度滑杆
