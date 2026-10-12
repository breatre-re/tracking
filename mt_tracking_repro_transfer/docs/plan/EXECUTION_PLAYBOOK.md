# 论文复现任务书：Position-only 多目标跟踪

> **用途：** 本文档是持续更新的执行计划和阶段验收记录。每次开始新 Phase、改变复现假设或生成正式结果后，更新“当前状态”“待办事项”和相应结果路径。面向理解、汇报的说明和总结放在根目录 `README.md` 与 `docs/reports/`。

## 1. 目标与边界

复现论文 *A robust multiple target tracking framework with Transformer-based data association and dynamic trajectory management in challenging environments* 的位置测量到航迹输出流程：数据生成、目标状态预测、CTDA 数据关联、一对一分配、BLUE 状态估计、轨迹评分/生命周期管理和论文指标评估。

- 输入：二维带噪目标位置测量点；当前不实现图像检测器。
- 输出：稳定航迹 ID、轨迹状态/生命周期、关联及跟踪指标。
- 真实目标 ID：只用于离线监督标签与评估；不得作为模型或在线 tracker 的特征。
- 论文未公开的训练数据细节、层数、假设和参数必须显式记录，不得把工程选择表述为作者设定。

## 2. 当前工作区结构

```text
README.md                         项目阅读入口与组会摘要
references/papers/                原文 PDF
src/simulation/                   数据生成器和 S0–S7 场景
src/tracking/                     baseline、BLUE/Kalman、生命周期
src/models/ctda/                  RDF、CTDA 结构、损失、tracker adapter
src/training/                     训练及 association-only 评估
src/evaluation/                   评估指标
src/experiments/                  Phase 2–4 运行入口
configs/                          Phase 1–4 配置
scripts/                          环境检查等工具
tests/                            单元/回归测试
data/datasets/                    按 Phase 和分布保存的 NPZ
data/results/                     各阶段逐用例 JSON 与结果
data/manifests/                  配置、seed、输入哈希和产物清单
models/checkpoints/               按 Phase 和运行版本存储 checkpoint/摘要
outputs/figures/                  轨迹图
outputs/smoke/                    环境和 smoke 产物
docs/plan/                        任务书、实现规格、review 计划
docs/reports/                     Phase 报告与复现假设
archive/phase4_handoff/           原始交接证据副本
```

工作目录为本文件所在项目目录。推荐激活 `mttrack-repro`；需要 GPU 的训练/评估从能访问主机 NVIDIA 设备的环境启动。不要因沙盒内看不到设备就改装 PyTorch 或系统驱动。

## 3. 阶段状态

### Phase 0：环境与可运行性 — 完成

- Conda 环境：Python 3.10、PyTorch 2.4.1、CUDA 12.1 runtime。
- 主机 NVIDIA driver 535.261.03；用户报告 GPU 3、4、5 当前空闲。训练启动前用 `nvidia-smi` 再核对并显式指定可用卡。
- 基线和 CTDA 接口 smoke 产物在 `outputs/smoke/`。

### Phase 1 / 1.5：可复现数据 — 完成

- Phase 1：初始 smoke/train/test 数据在 `data/datasets/phase1/`。
- Phase 1.5：baseline 与 `paper_assumption_v1` 数据在 `data/datasets/phase1_5/`；主要训练/测试序列数为 1,800/200。
- 目标数均匀采样、Poisson λ=20 的时间单位等属于已记录的复现假设；seed 与 SHA-256 在 `data/manifests/`。

### Phase 2：非学习型 tracker 场景验收 — 完成

- S0–S7 覆盖单目标、不交叉/交叉、漏检、虚警、生灭、重现、高目标数与高杂波。
- 运行器：`src/experiments/phase2_baseline_evaluation.py`。
- 产物：`data/results/phase2/legacy/`、`outputs/figures/phase2/`、`data/manifests/phase2_manifest.json`。

### Phase 3 / 3.1：动态分数与航迹管理 — 完成（实现/审计）

- BLUE/Kalman、候选/确认/丢失/删除生命周期和动态评分已实现。
- Phase 3.1 确认论文 Eq. (30) 文档化的漏检分数增量为正；其行为与一般直觉不一致，结果如实保留，不静默改符号。
- 入口：`src/experiments/phase3_dynamic_score_evaluation.py`、`src/experiments/phase3_1_score_direction_audit.py`。

### Phase 4：CTDA — 实现与评估管线完成；论文质量未通过

- RDF 样本、row-column、3-channel、focal loss、KM 解码、adapter 和 tracker 对比已实现。
- 已修复两个模型问题：3ch 通道注意力 mask 轴错误导致 NaN；原 3ch decoder 漏掉论文 Eq. (17) 的通道内 self-attention 和 FFN。
- 训练协议：1,620 train + 180 train-internal validation 序列，完整 80 帧，8 epochs / batch 256 / Adam / lr 1e-4 / seed 20261012；独立 test 为 200 序列。
- 同协议 association-only：row-column F1=0.7862；3ch Eq. (17) F1=0.0356。
- 同场景 tracker 均值：row-column 175.6 matches、296.3 FN、RMSE 5.16；3ch 75.8 matches、396.1 FN、RMSE 6.62。3ch 的 FP/IDSW/fragmentation 较低，但漏检与定位更差，整体质量不达标。
- 2D 下 `range_rdf` 代替论文 z 轴 RDF 是尚未验证的假设。不能将当前三通道结果称为论文质量复现。
- 最终 3ch 权重：`models/checkpoints/phase4/2026-10-12_eq17/row_column_3ch/`；同协议 row-column 权重：`models/checkpoints/phase4/2026-10-12_maskfix/row_column/`。
- 汇总见 `docs/reports/phase4_report.md`，各关联结果、逐案 tracker 指标和 manifest 在 `data/results/phase4/2026-10-12_eq17/`、`data/manifests/phase4_eq17_3ch_manifest.json` 与 `data/manifests/phase4_row_column_manifest.json`。

## 4. 下一步工作（优先级）

### P1：解释并改善 2D CTDA 关联质量

1. **审计 RDF 分布。** 对正匹配/负匹配分别输出 x/y/range 特征直方图、分位数和按目标数/杂波密度分桶的分布；确认公式尺度、饱和和通道一致性。
2. **做不泄漏的通道消融。** 至少比较 x/y 双通道、x/y/常数 z、当前 x/y/range；只用 train-internal validation 选择，test 只做最终一次报告。
3. **检查 3ch Eq. (17) 细节。** 对照论文图 3、式 (16)–(18)，核对每个 attention 的 token 轴、输入/输出、ElementMix 和 mask；将论文未确定的层数/FFN 记为配置假设。
4. **检查阈值与概率校准。** `lambda_a=0.1` 先按论文保留；可在 train-validation 上报告 PR 曲线与阈值敏感度，禁止依据 test 调参。
5. 保持当前 row-column checkpoint 作为有效对照；每次训练用新的 `models/checkpoints/phase4/<run_name>/`，不覆盖历史权重。

### P2：Phase 5 — CTDA 接入完整 tracker

CTDA adapter 已接入 Phase 3 tracker，但应在关联质量得到改善后开展正式验收：检查 unmatched rows/columns、出生/消亡、漏检重现和 ID 单调性；保持 gated-greedy 与 dynamic-score B 对照，确保所有 tracker 使用同一场景 seed 和输入哈希。

### P3：Phase 6 — 论文级评估

- 补充 OSPA distance/location/cardinality 和论文相关指标。
- 形成固定配置下 100 次 Monte Carlo 结果，报告均值、离散程度、运行时间和所有 seed。
- 先完成 2D，再开展 3D；3D 通道结构不应从二维 range 特征直接推断。
- 保存配置快照、checkpoint SHA-256、训练数据 hash、逐案结果及绘图源数据。

## 5. 可复现实验命令

所有命令从项目根目录运行；每次新结果指定独立目录，防止覆盖已归档产物。

```bash
conda run -n mttrack-repro python scripts/verify_environment.py
conda run -n mttrack-repro python -m src.simulation.generate_dataset \
  --config configs/phase1_2d.yaml --profile all --output-root data/datasets/phase1
conda run -n mttrack-repro python -m src.simulation.generate_dataset \
  --config configs/phase1_5_distributions.yaml --profile all --output-root data/datasets/phase1_5
conda run -n mttrack-repro python -m src.experiments.phase2_baseline_evaluation \
  --output-root data/results/phase2/new_run --plot-root outputs/figures/phase2/new_run \
  --manifest data/manifests/phase2_new_run.json
```

训练/关联评估示例：

```bash
conda run -n mttrack-repro python -m src.training.train_ctda \
  --variant row_column_3ch --output-root models/checkpoints/phase4/new_run
conda run -n mttrack-repro python -m src.training.evaluate_association \
  --checkpoint models/checkpoints/phase4/new_run/row_column_3ch/checkpoint.pt \
  --output data/results/phase4/new_run/association/test.json
```

## 6. 每阶段交付约定

每次完成或调整阶段后，更新本任务书和相应 `docs/reports/`：

1. 阶段目标、实现文件与论文依据；
2. 运行命令、配置快照、训练/测试切分、seed 和输入/checkpoint hash；
3. 指标结果、相同输入的 parity，以及失败案例；
4. 复现假设与未解决歧义；
5. 下一步唯一最重要的工作与运行产物位置。

不得删除 baseline、原始论文、生成数据、旧 checkpoint 或报告。对重跑结果使用新目录并保留来源；不得将假设性数据、截断运行或单次结果表述成论文官方结果。
