# 开发方案 V3.1 增补 · Phase C 双轨与 C2 自蒸馏

> **日期**：2026-07-09  
> **基线**：`semantic-object-color-retouch-dev-plan-v3.md`（841 行执行版）  
> **详细设计**：`phase-c2-reference-self-distill-design.md`

本增补 **不替换 V3**，只在 Phase C / 蒸馏 / 里程碑三处收束策略变更。未提及部分仍以 V3 为准。

---

## 1. 变更摘要

| 项 | V3 原表述 | V3.1 修订 |
|----|-----------|-----------|
| Phase C | 单一 GPT teacher 蒸馏量化 | **C1 ∥ C1c ∥ C2 三轨**：C1=GPT 残差量化；C1c=本地 VLM 语义门控/critic（新增）；C2=Reference 自蒸馏（主路径） |
| M6 伪标签来源 | 主要靠 teacher 栈 + GPT | **C2 bootstrap 为主**（本地 pseudo-target），GPT ≤15% hard-case |
| M7 学生模型 | 结构化规划头 + 模板引擎 | 规划头不变；**执行头改为 RegionalParamHead**（嫁接 Smart Color v2） |
| GPT 阻塞 | 阻塞 M3 | **不再阻塞**；C2 可独立推进到 C2.3 |
| API 提供商 | funai | 已切换 **API易**（`api.apiyi.com`，`gpt-image-2-all`） |
| §4.2 VLM 交叉验证门槛 | 方案要求但未落地，实现是启发式规则 | **新增 C1c**：Qwen3-VL-8B-Instruct（Apache-2.0）补齐这一格，见 §2 |

---

## 2. §12 蒸馏路线 · 修订版

在 V3 §12 之前插入本双轨定义：

### 12.0 Phase C 三轨（2026-07-09 新增 C1c）

```text
Phase C1 — GPT Teacher 量化（辅助轨）
  目的：标定哪些语义类 / 场景必须走 GPT（latitude、clip、结构级变化）
  输入：target + reference + 本地 medium 结果
  输出：per-class Lab 残差报告（distill_report.json）
  依赖：API易 gpt-image-2-all 双图编辑稳定
  状态：API 已切换；待冒烟验证

Phase C1c — 本地/托管 VLM 语义门控/critic（新增，实验已完成，优先级下调）
  目的：补齐 V3 §4.2 一直缺失的"VLM 交叉验证"格子，验证是否需要替代/校验启发式天空合理性规则
  依据：ModelScope VLM 盘点核实 Qwen3-VL 全系列 Apache-2.0，商用无门槛
        （09-行业方案与知识库/App/2026-07-09-ModelScope图文多模态VLM模型盘点.md）
  实现：本机 16GB 内存 + 无 MPS，跑不动自部署 8B 权重，改用 API易 代理的
        `qwen3-vl-plus`（Alibaba 托管商用 tier，非自部署 Apache-2.0 权重）
  验证：对 C2 数据集全部 30 个"启发式认定合理"的 sky 样本做对比
  结果：100% 一致，0 语义假阳性；原怀疑的"假天空"bug 案例人工复核后确认是真实
        过曝天空（数值拟合问题，非语义问题）——详见
        outputs/phase-c1c-vlm-sky-gate-results.md
  状态：✅ 实验已完成，结论是当前数据集下启发式规则可靠，"替代规则"动机不成立，
        优先级下调为次要备选

Phase C2 — Reference 自蒸馏（主轨）★
  目的：用可训练 per-region head 替代手写 Lab 统计，对接 Smart Color v2
  Teacher v0：color_reference_transfer.py medium 档 pseudo-target
  数据：20 图回归集 → 已扩至 97 样本/208 class-rows（无需 GPT）
  详见：phase-c2-reference-self-distill-design.md
```

### Stage 0 / 1 / 2 衔接调整

- **Stage 0**：已完成骨架 + 仿色产品化；C1 冒烟 **降级为可选验收项**（不达标不阻塞 C2）
- **Stage 1（M6）**：manifest 采用 `c2_manifest.jsonl` schema；优先导出 suitable 样本
- **Stage 2（M7）**：训练分两阶段  
  - **M7a 执行头**：C2 RegionalParamHead（仿色 / 区域 grading）  
  - **M7b 规划头**：action / strength / route（原 V3 Stage 2）

---

## 3. §13 Smart Color v2 · 嫁接结论

V3 §13 写「两个模型、两个交付物、可独立回滚」——**V3.1 明确执行方式**：

1. **全局 Smart Color v2**：不改训练链路、不改 v4.3.9 fallback  
2. **RegionalParamHead（C2 新建）**：  
   - 输入：style_vec + tgt_vec + 语义类  
   - 输出：`chroma_param_map.json` 定义的滑块子集  
   - 训练：复用 SCv2 param-target + renderer parity  
3. **Chroma 引擎增量**：masked base-adjust 算子（`CHROMA_ALIGNMENT.md` 已列规格）

---

## 4. §14 里程碑 · 新增行

| 里程碑 | 交付物 | 验收 |
|--------|--------|------|
| **M3.5 C2 Bootstrap** | `scripts_c2/export_bootstrap_dataset.py`、`dataset/c2/manifest.jsonl` | ≥40 suitable 样本，mask/profile/pseudo 三元组完整 |
| **M3.6 C2 Head v1** | `train_per_class_head.py`、eval 报告 | 20 图回归 ΔE 不劣于 rule baseline 的 ≥90% |
| **M3.7 SCv2 嫁接** | Chroma 仓 parity 报告 + 权重导出 | Python/C++ max ΔE < 1.0 |

原 M6/M7 保持不变，但 M6 输入改为 **C2 manifest 为主**。

---

## 5. §17 立即执行清单 · 更新（替换 V3 §17 第 8–10 条）

**已完成**（截至 2026-07-09）：

- [x] git init + GitHub 仓库  
- [x] API 切换 API易 + client 适配  
- [x] C2 设计稿 + V3.1 增补  
- [x] **C2.1 全量导出并扩样**：97 样本（20 图回归集 + Stage 0 100 张验证集补充）  
- [x] **C2.2 全量拟合**：208 class-rows（顺带修复了假天空检测导致的退化 scale bug，详见设计稿 §7）  
- [x] **C2.3 ridge baseline**：n=208，held-out MAE=4.20，明显低于预测均值基线 6.31，且降幅比例与 n=41 时几乎一致，信号稳定可泛化  
- [x] **ModelScope VLM 盘点**：核实 Qwen3-VL 全系列 Apache-2.0 商用无门槛，新增 C1c 轨（见 §2 12.0）  
- [x] **C1c 实验**：`qwen3-vl-plus` 对比 30 个 sky 样本，0 语义假阳性，结论「规则可靠，优先级下调」，见 `phase-c1c-vlm-sky-gate-results.md`  
- [x] **升级模型尝试**：`train_per_class_head_mlp.py`（CV 选参 + 多 seed），结论「n=208 时 ridge 泛化更好（held-out MAE 4.20 vs MLP 6.13），继续用 ridge」，见 `phase-c2.3b-mlp-head-experiment.md`  
- [x] **Web Demo**：参考图+目标图+强度滑杆，`stage0_pipeline/webdemo/`  
- [x] **排查并修复过冲光晕瑕疵**：Web Demo 肉眼验证发现 medium/strong 档在羽化边界上有不自然光晕，根因是 `cs>1` 过冲设计（跟分割精度/BSHM无关），已按局部方差自适应抑制过冲修复；因为 C2 伪标签老师就是这套 medium 档，**重跑了 C2.1→C2.3 全流程**，新 held-out MAE=4.02（略优于修复前 4.20），见 `phase-teacher-overshoot-halo-fix.md`  
- [x] **排查并修复 neutral 洗色问题**：用户拿密集人群图测试反馈"前景没反应"，排查发现人群没被识别成`clothing`掉进`neutral`兜底类；重缩放式分级会把人群衣服洗成单一色调（比不处理更差），改用固定加法偏移（`_grade_neutral_additive`）保留色彩多样性同时仍能整体偏移氛围；**再次重跑 C2.1→C2.3**，新 held-out MAE=4.07（与上一版基本持平），见 `phase-teacher-neutral-mood-cast.md`  

**当前优先级**：

1. **M3.7**：在 Chroma 仓开 `feature/regional-smart-color-head` 分支（数据门槛已达标，teacher 已两轮修复，可以启动 C2.4 嫁接）——**当前主线**  
2. **C1** API易 双图冒烟（并行，不阻塞 1）  
3. 与 Chroma 仓对齐 **M3.7 masked operator** 排期  
4. 数据量进一步扩大后（更多 Stage 0/新样本，或 C1 GPT hard-case 补充）重跑 `train_per_class_head_mlp.py` 复查 MLP 是否反超  
5. 建筑立面整片过冲偏色（跟边界光晕不是同一机制）留作后续单独排查  

---

## 6. §18 成功标准 · 补充

V3 Stage 0 成功标准保留。新增 **C2 成功标准**（可独立于 GPT 达成）：

```text
Bootstrap manifest ≥40 条 suitable 样本
Learned head 在 20 图回归集上 ≥90% 不劣于 rule baseline
至少 3 个 bucket（outdoor/person/stage）各有一条 learned ≤ rule 的 showcase
RegionalParamHead 输出可映射到 chroma_param_map（可解释、可手动微调）
```

达到 C2 成功标准即可进入 M6 扩样，**无需等待 GPT API 全量稳定**。
