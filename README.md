# 语义物体调色专家 (Semantic Object Color Retouch)

独立预研项目：识别照片中的语义区域（天空、人脸、皮肤、服装、草地、建筑等），判断各区域是否应调色及如何调色，并通过本地 Chroma 引擎或 GPT Image 2 执行。

**当前状态（2026-07-07）**：Stage 0 骨架与仿色功能（真语义分割 + 产品化）已完成；Phase C（GPT 老师蒸馏量化）因 API 不稳定被阻塞。

## 目录结构

```
outputs/                    开发方案 v1/v2/v3 + 启动清单 + Stage 0 数据集元数据
stage0_pipeline/            可运行管线（Stage 0 骨架 + M2 真模型 + 仿色正式版）
work/data_audit/            数据集审计脚本
.venv/                      主链路 Python 环境（numpy + Pillow）
.venv-m2/                   M2 真模型环境（torch + transformers + Grounding DINO/SAM）
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
| 最终执行方案（841 行） | `outputs/semantic-object-color-retouch-dev-plan-v3.md` |
| 开发启动清单 | `outputs/development-start-checklist.md` |
| Stage 0 管线说明 | `stage0_pipeline/README.md` |
| 接手指南（Obsidian） | 云享传知识库 `02-需求与规划/语义物体调色专家-项目现状与接手指南.md` |

## GPT Image 2 API 配置

复制模板并填入密钥（**不要提交**）：

```bash
cp stage0_pipeline/secrets/api.local.json.example stage0_pipeline/secrets/api.local.json
```

## 建议下一步

1. Phase C：重试 `scripts_m2/distill_vs_gpt.py`，积累 teacher 残差数据
2. 基于 `color_reference_transfer.py` 做轻量 Web Demo（参考图 + 目标图 + 强度滑杆）
3. Phase 3：蒸馏学生规划模型（见 development-start-checklist.md）
