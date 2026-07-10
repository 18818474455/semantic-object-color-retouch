# 语义物体调色专家 (Semantic Object Color Retouch)

独立预研项目：识别照片中的语义区域（天空、人脸、皮肤、服装、草地、建筑等），判断各区域是否应调色及如何调色，并通过本地 Chroma 引擎或 GPT Image 2 执行。

**当前状态（2026-07-10）**：C3 仿色一致性升级进行中——C3-0（基线冻结）、C3-1（全局氛围基底）、C3-2（区域受信任度残差）均已完成并验证。核心bug修复：低置信度区域现在收敛到"停在全局基底"而不是"仍执行100%统计匹配"（`region_strength = region_cap * trust`，`trust→0` 时残差→0）；两个原始bug case（背景/前景脱节、过冲光晕）在C3-2下依然肉眼正常，且比C3-1单独全局基底版本略有"抓眼感"回归。**当前主线是 C3-3：残差级边缘融合（Guided Filter）+ 材质混杂度特征细化**。M3.7 Smart Color v2 嫁接暂停，等待 30 组 `FG-BG-Coord-v1` 视觉验收。

**本目录即代码基地**：本项目已从 `/Users/mac/Documents/Codex/2026-07-05/gpt-image-2/` 迁移到 `/Users/mac/Desktop/整体代码1.0/仿色模型/` 作为唯一持续开发位置（`.git`/远程仓库随迁移保留）。虚拟环境 `.venv`/`.venv-m2` 太大（合计约1GB）未一起迁移，重建方式：

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r stage0_pipeline/requirements-venv.txt
python3.13 -m venv .venv-m2 && .venv-m2/bin/pip install -r stage0_pipeline/requirements-venv-m2.txt
```

项目级 Cursor skill：`.cursor/skills/semantic-color-retouch/SKILL.md`（新会话可直接读取快速接手）。

## 目录结构

```
outputs/                    开发方案 v1/v2/v3 + v3.1 + C2 设计稿
stage0_pipeline/
  scripts_m2/               仿色正式版 color_reference_transfer.py
  scripts_c2/               C2 bootstrap 导出与后续训练脚本
  webdemo/                  仿色 Web Demo（参考图+目标图+强度滑杆）
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

# 仿色 Web Demo（参考图 + 目标图 + 强度滑杆，本机浏览器体验）
../.venv-m2/bin/python webdemo/app.py   # 打开 http://127.0.0.1:5057
```

## 开发文档

| 文档 | 路径 |
|------|------|
| 最终执行方案（841 行 + V3.1 增补） | `outputs/semantic-object-color-retouch-dev-plan-v3.md` + `v3.1-c2-addendum.md` |
| C2 Reference 自蒸馏设计稿 | `outputs/phase-c2-reference-self-distill-design.md` |
| C2.1 Bootstrap 导出脚本 | `stage0_pipeline/scripts_c2/export_bootstrap_dataset.py` |
| C1c VLM 天空门控实验结果 | `outputs/phase-c1c-vlm-sky-gate-results.md` |
| C2.3b ridge→MLP 升级实验结果 | `outputs/phase-c2.3b-mlp-head-experiment.md` |
| 过冲光晕瑕疵：根因+修复+C2重跑 | `outputs/phase-teacher-overshoot-halo-fix.md` |
| neutral 加法mood-cast（前景/人群多样性保留）+C2重跑 | `outputs/phase-teacher-neutral-mood-cast.md` |
| 同标签类别外观差过大压制强度（背景/前景脱节）+C2重跑 | `outputs/phase-teacher-class-mismatch-fix.md` |
| C3 仿色一致性升级方案 | Obsidian `[[语义物体调色专家-仿色一致性升级方案]]` |
| C3-0 legacy 基线清单 | `stage0_pipeline/baselines/c3-0/legacy_v0/manifest.json` |
| C3-1 全局氛围基底（已实现+验证） | `outputs/phase-c3-1-global-mood-base.md`、`stage0_pipeline/scripts_m2/coherence_controller.py` |
| C3-2 区域受信任度残差（已实现+验证） | `outputs/phase-c3-2-region-residual.md`（`_render_coherence_from_analysis` 内） |
| FG-BG-Coord-v1 专项评审集 | `stage0_pipeline/eval/fg_bg_coord_v1/` |
| 仿色 Web Demo（参考图+目标图+强度滑杆） | `stage0_pipeline/webdemo/` |
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

1. ~~**C2.1/C2.2/C2.3** 全量导出 + 拟合 + 训练~~ ✅ —— 外置盘挂载后跑通：21 样本 / 41 class-rows，held-out MAE=3.83。过程中发现并修复了一个低方差区域导致 Lab-affine scale 数值爆炸的 bug（详见设计稿 §7）
2. ~~**扩样**：把 Stage 0 100 张验证集也导入 C2.1~~ ✅ —— n_rows 从 41 提到 **208**（97 样本），held-out MAE=4.20 < 基线 6.31，降幅比例（~34%）与扩样前基本一致，信号稳定可泛化
3. ~~**C1c**：Qwen3-VL 天空门控对比实验~~ ✅ —— `qwen3-vl-plus` 对 30 个样本 100% 认同启发式规则（0 语义假阳性），上面的 bug 案例人工复核后确认是真实过曝天空、非语义误检；C1c"替代规则"动机不成立，优先级下调（详见 `outputs/phase-c1c-vlm-sky-gate-results.md`）
4. ~~**升级模型**：ridge baseline 升级为 torch MLP~~ ✅ —— CV 选参 + 多 seed 评估后，n=208 时 MLP 泛化不如 ridge（held-out MAE 6.13 vs 4.20），**继续用 ridge (v0)**（详见 `outputs/phase-c2.3b-mlp-head-experiment.md`）
5. ~~**Web Demo**：参考图 + 目标图 + 强度滑杆~~ ✅ —— `stage0_pipeline/webdemo/`，本地 Flask，分析一次（跑分割）+ 滑杆秒级重渲染。用真实回归图肉眼验证时发现一个此前只看 Lab 数值指标没发现的瑕疵：`outdoor_sky` 桶天空过饱和 + 树冠边缘青色光晕
6. ~~**排查并修复过冲光晕瑕疵**~~ ✅ —— 根因是 `STRENGTH_PRESETS` 里 `cs>1` 过冲设计在羽化边界上失控（跟分割精度/BSHM无关），已按局部方差自适应抑制过冲修复，20图回归验证通过，**C2 训练数据用修复后 teacher 重新生成**（held-out MAE 4.20→4.02，小幅改善）。建筑立面整片过冲偏色是另一机制，留作后续（详见 `outputs/phase-teacher-overshoot-halo-fix.md`）
7. ~~**"前景没反应"问题：neutral 改用加法 mood-cast**~~ ✅ —— 密集人群没被检测器识别、掉进 neutral 兜底类，重缩放式分级会洗掉人群衣服的颜色多样性；改成固定加法偏移后人群色彩多样性保留、整体氛围仍有柔和偏移，20图回归验证通过，**C2 训练数据再次重新生成**（held-out MAE 4.02→4.07，噪声级波动，信号稳定）（详见 `outputs/phase-teacher-neutral-mood-cast.md`）
8. ~~**"背景跟前景脱节"问题：同标签类别外观差过大压制强度**~~ ✅ —— 开放词汇标签把两种物理上完全不同的东西（商场白吊顶 vs 钢架顶棚）都标成"building"强行统计匹配，新增类别配对置信度按绝对 Lab 差距压制过冲，20图回归验证通过（顺带修好一张过曝背景模糊的图），**C2 训练数据再次重新生成**（held-out MAE 4.07→3.74，真实改善）（详见 `outputs/phase-teacher-class-mismatch-fix.md`）
9. ~~**C3-0** 仿色一致性升级基线冻结~~ ✅ —— 已用 commit+SHA-256 锁定 `legacy_v0` teacher、97样本/208 class-rows C2 数据与 ridge head；核心接口新增 `pipeline=legacy/coherence`；`FG-BG-Coord-v1` 已登记旧20组
10. ~~**C3-1** 全局氛围基底~~ ✅ —— `coherence_controller.py`：整图 Lab 加法位移，`ΔL≤10`/`Δab≤9` 严格限幅 + 内容匹配度打折 + 皮肤减半，20图回归全通过；`person_event_DSC04819_r2`（材质错配脱节）和 `outdoor_sky_DSC04085(1)`（光晕）两个原始bug case 视觉验证：脱节/光晕都消失，效果比 legacy 更自然但更弱（预期内，区域残差还没加回来）（详见 `outputs/phase-c3-1-global-mood-base.md`）
11. ~~**C3-2** 区域受信任度残差~~ ✅ —— 区域分级改写成"相对全局基底的残差"，`trust=scene*pair*homogeneity*pixel` 控制整个残差而不是只控制`cs>1`过冲部分（`trust→0`时收敛到全局基底而不是100%匹配）；新增`_region_homogeneity_confidence`+`MAX_REGION_DELTA_E`预算；20图回归全通过，两个原始bug case复测依然正常，天空等高可靠区域的"抓眼感"比C3-1略有回归（详见 `outputs/phase-c3-2-region-residual.md`）
12. **C3-3（当前主线）**：残差级边缘融合（Guided Filter替代固定羽化半径）+ `homogeneity_confidence`补充纹理/边缘密度特征
13. **C3-4**：`eval_harmony.py` + 补齐 FG-BG-Coord-v1 到 30 组 + 人工A/B验收
14. **C3-5/C3-6**：验收通过后重建C2 teacher数据，再恢复M3.7 Smart Color v2嫁接
15. **C1** API易 双图冒烟（并行，不阻塞 C3）
