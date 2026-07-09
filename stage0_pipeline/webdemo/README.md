# 仿色 Web Demo

参考图 + 目标图 + 强度滑杆，跑的是 C2 主路径的规则教师
`scripts_m2/color_reference_transfer.py`（真实语义分割 + 逐区域 Lab 统计匹配），不是 GPT。

目的：在决定要不要把这套逻辑嫁接进 Chroma/Smart Color v2（M3.7，更大的跨仓库工程投入）之前，
先用真实照片肉眼验证效果，而不是只看 Lab 数值指标。

## 运行

```bash
cd stage0_pipeline
../.venv-m2/bin/pip install flask   # 只需装一次
../.venv-m2/bin/python webdemo/app.py
# 打开 http://127.0.0.1:5057
```

只监听 `127.0.0.1`（本机），不对外网开放。单进程、内存态 session 缓存，是本地调试用的轻量 demo，
不是生产服务（Flask 开发服务器本身也会打印这条警告）。

## 交互设计 & 性能

- 上传参考图 + 目标图后点"开始分析"：这一步会跑真实的 Grounding DINO + SAM 分割 + 逐类 Lab 统计匹配，
  耗时数十秒（模型推理本身的成本，跟 demo 无关）。
- 分析完成后拖动强度滑杆：这一步复用分析阶段缓存的分割结果和逐类分级结果，只做"按强度混合 + 全局精修"，
  单次渲染 <150ms，滑杆可以流畅拖动。
- 实现上是把 `color_reference_transfer.apply_profile()` 拆成了两半：
  `analyze_target()`（跟强度无关，贵）+ `render_from_analysis()`（跟强度相关，便宜）。
  这个拆分是纯粹的 extract-method 重构，已经用新旧代码逐像素对比验证过（`max_abs_diff=0.0`），
  `apply_profile()` 的公开行为完全不变，CLI 用法不受影响。
- 强度滑杆 0-100 对应四个锚点线性插值：0=无效果（identity）、33=light、66=medium、100=strong，
  中间的取值是相邻两个已验证档位之间的线性插值，不会外推到"strong"之外的未知参数区间。

## 已发现并修复的瑕疵：过冲光晕（2026-07-10）

用 `outdoor_sky` 桶的真实回归测试图跑通 demo 后，在 medium/strong 档位能看到天空过饱和 + 树冠轮廓边缘
青色光晕的瑕疵。原版 CLI（不经过 demo）在同一对图片上复现了完全一样的输出，说明不是 demo 引入的问题，
是 pipeline 本身的视觉瑕疵——此前的验证只看了 Lab 空间的聚合统计指标（ΔE/MAE），没有人逐像素肉眼看过
渲染结果。

根因排查 + 修复 + C2 数据重跑详见 `outputs/phase-teacher-overshoot-halo-fix.md`。简要结论：不是分割精度
问题（跟 BSHM 人像抠图无关，出问题的边界没有人），是 `STRENGTH_PRESETS` 里 `cs>1` 的"过冲"设计在羽化
边界/材质混杂区域上失控；已用按局部方差自适应抑制过冲的方式修复，20 图回归验证通过，C2 训练数据已用
修复后的 teacher 重新生成（held-out MAE 4.20→4.02，小幅改善）。
