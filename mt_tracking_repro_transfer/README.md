# Position-only 多目标跟踪论文复现

本项目围绕论文 **A robust multiple target tracking framework with Transformer-based data association and dynamic trajectory management in challenging environments**，复现其从位置测量到多目标航迹维护的主要流程。输入是二维带噪测量点；当前不包含图像检测器。项目持续记录论文设定、实现假设、实验结果和未解决问题，供组会汇报及最终结果整理参考。

> **当前判断（2026-10-12）：** Phase 0–4 的工程流程已搭建，CTDA 训练中的 NaN 与 3ch decoder 漏项已经修复；但论文质量尚未复现。全量测试上 row-column 的关联 F1 为 0.786，3ch 模型 F1 为 0.036；两者 tracker 均仍有明显误差。请勿把工程流程跑通表述成论文结果复现成功。

## 论文方法与本项目流程

论文主要解决目标数未知、虚警、漏检、交叉和航迹生灭条件下的位置测量关联与航迹管理。复现流程可以概括为：

```text
二维真值轨迹 / 带噪位置测量
            ↓
RDF 相对距离特征矩阵
            ↓
CTDA：row-column cross Transformer
      + 3-channel cross Transformer（实验分支）
            ↓
关联概率 → 阈值 0.1 → Kuhn–Munkres 一对一分配
            ↓
BLUE/Kalman 状态预测与更新
            ↓
动态航迹分数与 candidate/confirmed/lost/deleted 管理
            ↓
位置误差、FP/FN、ID switch、fragmentation、MOTA-like 等评估
```

核心设计动机是：RDF 编码候选航迹与当前测量在各坐标轴上的相对位置；Transformer 学习数据关联；一对一分配约束避免重复匹配；动态分数和生命周期策略处理漏检、重现及目标生灭。本项目数据集中保留真值 ID 仅用于离线构造监督标签和评估，tracker 输入只含测量位置。

## 复现进度与当前结果

| 阶段 | 已完成内容 | 当前结论 |
|---|---|---|
| Phase 0 | Conda/PyTorch 环境、GPU 和 smoke 流程 | 主机环境运行通过；当前可用 GPU 3、4、5（开跑前仍以 `nvidia-smi` 为准） |
| Phase 1 / 1.5 | 可重复二维合成数据、train/test 切分、baseline 与 paper-assumption 分布 | 每个主要训练/测试 split 为 1,800/200 序列；未公开的生成参数均列为假设 |
| Phase 2 | S0–S7 场景覆盖 baseline | 包含交叉、漏检、虚警、生灭、短时重现和高杂波 |
| Phase 3 / 3.1 | BLUE/Kalman、动态评分航迹管理、漏检分数方向审计 | B 版按论文 Eq. (30) 文档化公式实现；符号影响单独报告 |
| Phase 4 | CTDA 特征、训练、关联解码、tracker 集成和 3ch 调试 | NaN 与 Eq. (17) decoder 缺项已修复；**模型质量仍未通过** |

最终同协议（完整 80 帧、8 epochs）关联-only 测试集结果：

| 模型 | Precision | Recall | F1 | 最佳验证 focal loss |
|---|---:|---:|---:|---:|
| row-column | 0.742 | 0.836 | **0.786** | 0.01091 |
| row-column + 3ch Eq. (17) | 0.034 | 0.038 | **0.036** | 0.02162 |

S0–S7 × 两分布的端到端对照中，3ch 平均匹配 75.8、FP 43.8、FN 396.1、ID switch 35.1、fragmentation 24.9、有限 RMSE 6.62；同协议 row-column 平均匹配 175.6、FP 146.8、FN 296.3、ID switch 51.1、fragmentation 39.4、有限 RMSE 5.16。3ch 减少了错误输出和 ID switch，但漏检及定位误差更大，不能据较低 FP 或正 MOTA-like 单独判断更好。

**主要待查点：** 论文的三通道是 x/y/z RDF；本项目二维数据把第三通道替换成 `range_rdf`，尚无证据表明这是等价替代。接下来应先做坐标通道消融和 RDF 数值分布审计，再决定 2D CTDA 应使用双通道还是不同的第三通道。

## 组会汇报建议

1. **问题背景：** 仅有位置测量时，多目标交叉、漏检、虚警和生灭使“当前点属于哪条航迹”难以判断。
2. **方法贡献：** RDF + Cross Transformer 生成关联概率；KM 保证一对一；BLUE 估计状态；动态分数与生命周期管理航迹。
3. **复现路线：** 先建立可重复数据和 baseline，再验证轨迹管理，最后接入 CTDA；对论文未公开参数显式记录假设。
4. **结果解释：** 当前 row-column 明显优于 3ch；3ch 数值稳定不代表质量通过。准确率受负样本比例影响，应同时汇报 Precision/Recall/F1 和端到端航迹指标。
5. **局限与下一步：** 2D 第三通道假设、候选行使用上一帧检测近似轨迹历史、生成参数和模型层数不完全确定；后续做消融、扩展 OSPA，并在测试集之外完成参数选择。

详细阶段报告、公式证据表和假设见 [`docs/reports/`](docs/reports/)。持续更新的任务书为 [`docs/plan/EXECUTION_PLAYBOOK.md`](docs/plan/EXECUTION_PLAYBOOK.md)。

## 项目结构

```text
README.md                    项目概览、组会说明和当前结论
references/papers/           论文 PDF
src/
  simulation/                 数据生成与场景
  tracking/                   baseline、BLUE/Kalman、航迹管理
  models/ctda/                 RDF、网络、损失、适配器
  training/                    CTDA 训练与关联评估
  evaluation/                  指标
  experiments/                 Phase 2–4 实验入口
configs/                       各 Phase 配置
scripts/                       环境检查等工具
tests/                         回归与接口检查
data/datasets/                输入数据与 train/test NPZ
data/results/                 逐用例评估结果
data/manifests/               数据、实验配置和哈希清单
models/checkpoints/             按阶段及运行版本存放权重和训练摘要
outputs/figures/                轨迹图
outputs/smoke/                  初始 smoke 结果和日志
docs/plan/                     可更新任务书、实现规格与 review 计划
docs/reports/                  阶段报告、假设和结果解读
archive/phase4_handoff/          原始 Phase 4 交接证据副本
```

## 常用命令

从项目根目录执行。环境为 `mttrack-repro`：

```bash
conda run -n mttrack-repro python scripts/verify_environment.py
conda run -n mttrack-repro python -m src.tracking.baseline_tracker
conda run -n mttrack-repro python -m src.simulation.generate_dataset \
  --config configs/phase1_2d.yaml --profile all --output-root data/datasets/phase1
conda run -n mttrack-repro python -m src.simulation.generate_dataset \
  --config configs/phase1_5_distributions.yaml --profile all --output-root data/datasets/phase1_5
conda run -n mttrack-repro python -m src.experiments.phase2_baseline_evaluation
conda run -n mttrack-repro python -m src.experiments.phase3_dynamic_score_evaluation
conda run -n mttrack-repro python -m src.experiments.phase3_1_score_direction_audit
```

CTDA 训练/评估入口：

```bash
conda run -n mttrack-repro python -m src.training.train_ctda \
  --variant row_column_3ch --output-root models/checkpoints/phase4/new_run
conda run -n mttrack-repro python -m src.training.evaluate_association \
  --checkpoint models/checkpoints/phase4/2026-10-12_eq17/row_column_3ch/checkpoint.pt \
  --output data/results/phase4/new_run/association/row_column_3ch_test.json
conda run -n mttrack-repro python -m src.experiments.phase4_ctda_evaluation \
  --checkpoint models/checkpoints/phase4/2026-10-12_eq17/row_column_3ch/checkpoint.pt \
  --output-root data/results/phase4/new_run/tracker \
  --plot-root outputs/figures/phase4/new_run \
  --manifest data/manifests/phase4_new_run_manifest.json
```

每次新实验使用独立 `new_run` 路径，保留已报告结果和模型；训练、验证、测试仍须按序列隔离。完整运行约定和后续工作清单见任务书。
