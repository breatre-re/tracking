# 多目标跟踪复现项目交付汇报（Phase 0–3）

**汇报对象：CG**  
**项目状态：Phase 0–3 已完成；CTDA Transformer 尚未实现或训练。**

## 1. 执行摘要

项目已从 baseline 环境验收推进到论文分布配置、固定场景回归及动态评分跟踪器对照：

- Phase 0：现有 Python/NumPy baseline 与预留 CTDA 接口 smoke test 均成功。
- Phase 1：独立可复现数据生成模块及 baseline 兼容入口完成；生成 4 条 smoke、1,800 条训练、200 条测试样本。
- Phase 1.5：保留旧数据并新增 baseline / `paper_assumption_v1` 两套对照数据；论文假设分布每条样本离散均匀采样 1–25 个目标，clutter 暂按每帧 Poisson λ=20。
- Phase 2：两套分布 × S0–S7 共 16 个 baseline 用例通过基础设施验收。
- Phase 3：baseline、文档公式版 dynamic-score、符号修正版诊断 tracker × 16 个用例，共 48 个用例通过基础设施验收；Phase 1/1.5/2 原始数据及 baseline 回归结果保留。

**整体判断：**数据管线、标签隔离、可重复性和评估产物已形成可复用基线；当前动态 tracker 的质量结果并未一致优于 baseline。最重要的未决问题是文档所列漏检分数增量符号会令漏检增加分数。高杂波性能、运行效率和恢复能力仍需进一步处理。以上结果不代表 CTDA 或论文完整复现已完成。

## 2. 各阶段完成情况

### Phase 0：环境和 baseline smoke 验收

- 使用已有 `mttrack-repro` 环境。
- Python 3.10.22，NumPy 1.26.4，PyTorch 2.4.1，PyTorch CUDA runtime 12.1。
- CUDA 可用，检测到 NVIDIA GeForce RTX 4090；GPU tensor test 通过。
- 默认 tracking smoke 与 `ctda-interface` smoke 均成功，输出 JSON 且指标有限。
- `ctda-interface` 当时仍是 baseline 占位实现，不是 Transformer。

### Phase 1：可复现数据集管线

独立生成模块、打包 NPZ 格式、manifest、baseline 数据读取和标签隔离机制已建立。基准配置为 80 帧、6 个目标、检测概率 0.9、测量噪声标准差 1.5、每帧 clutter Poisson 均值 1.5、过程加速度噪声标准差 0.08。

| Split | 样本数 | 测量点数 |
|---|---:|---:|
| Smoke | 4 | 1,861 |
| Train | 1,800 | 839,798 |
| Test | 200 | 93,371 |

相同 seed 重复生成文件逐字节相同；样本标签为目标 ID 或 `-1`（clutter）；tracker 输入只含位置，不含真值标签。

### Phase 1.5：论文分布配置化

新建 `baseline_distribution` 和 `paper_assumption_v1`，均固定 1,800 train / 200 test，使用独立 split seed 区间。旧 Phase 1 产物未覆盖。

| 分布 | 目标数 | Train 平均 clutter/帧 | Test 平均 clutter/帧 | Train / Test 测量点 |
|---|---|---:|---:|---:|
| baseline_distribution | 固定 6 | 1.5000 | 1.4997 | 839,798 / 93,371 |
| paper_assumption_v1 | 离散均匀 1–25；均值 12.77 / 12.76 | 20.0035 | 19.9832 | 3,885,985 / 431,886 |

λ=20 的时间单位若无法从论文确认，则“每个时间步的 clutter 数均值”是明确的 reproduction assumption；出生/死亡规则及其余未公开参数沿用项目 baseline 设置。

### Phase 2：固定场景矩阵

在两套分布分别验证 S0 单目标无噪声、S1 多目标不交叉、S2 交叉、S3 漏检、S4 虚警、S5 出生/消失、S6 短时漏检重现和 S7 高目标数高杂波。**16/16 用例通过基础设施验收，21 项测试通过。**

指标包括 matched detections、FP、FN、ID switch、fragmentation、position RMSE、MOTA-like 和 runtime。质量指标不设未经论文支持的阈值；低质量结果报告为 tracker 限制。

### Phase 3：BLUE 简化滤波及动态评分 tracker 对照

新增独立 4D Cartesian `[x,y,vx,vy]` BLUE/Kalman 风格滤波器、dynamic-score tracker、离线恢复率指标、配置和运行器。使用 Joseph covariance update、`solve` 与 `slogdet` 等数值稳定做法；关联采用 Mahalanobis gate + 物理距离 gate、一对一分配；生命周期包含 candidate、confirmed、lost、deleted，并支持 lost 状态在有效期内恢复原 ID。

比较的三个变体：

1. `baseline`：原始门限贪心 tracker。
2. `dynamic_score`：按实现规范中的漏检分数增量原式运行。
3. `dynamic_score_corrected`：将漏检增量符号改为惩罚的诊断变体；并非已确认的论文公式。

**48/48 用例通过基础设施验收，36 项测试通过。**生成 48 份 metrics JSON、48 张轨迹图和 Phase 3 manifest。确定性指标可复现（不含 wall-clock runtime）；runtime 48/48 有限且非负；标签隔离测试通过；baseline 结果与 Phase 2 逐位一致。Phase 1/1.5/2 manifest 哈希匹配。

## 3. Phase 3 关键对比结果

以下 MOTA-like 来自各 tracker 在两套分布的 S0–S7 矩阵结果。

| 场景 | baseline 分布：baseline | dynamic_score | corrected | paper 分布：baseline | dynamic_score | corrected |
|---|---:|---:|---:|---:|---:|---:|
| S0 单目标无噪声 | 1.000 | 0.988 | 0.988 | 1.000 | 0.988 | 0.988 |
| S1 多目标不交叉 | 0.512 | 0.110 | 0.705 | -9.240 | -12.637 | -1.070 |
| S2 交叉 | 0.241 | -0.316 | 0.731 | -11.556 | -15.325 | -1.788 |
| S3 漏检 | 0.292 | 0.142 | 0.408 | -14.671 | -20.279 | -2.233 |
| S4 虚警 | -2.179 | -4.450 | 0.375 | -15.279 | -20.950 | -2.025 |
| S5 出生/消失 | 0.442 | -0.358 | 0.728 | -10.549 | -14.797 | -1.827 |
| S6 短时丢失重现 | -0.044 | -0.994 | 0.169 | -23.081 | -30.581 | -4.975 |
| S7 高目标数/高杂波 | -1.278 | -2.104 | 0.170 | -1.273 | -2.192 | 0.044 |

### 结果解读

- 符号修正版在 baseline 分布的 S1–S7 均提高 MOTA-like；例如 S1 FP 从 baseline 的 176 降至 12，MOTA-like 从 0.512 提升至 0.705。
- 在 paper 高 clutter 分布中，符号修正版相较文档公式版明显减轻 FP/MOTA 损失，但 8 个场景的 MOTA-like 仍均为负数；尚不能称为解决高杂波关联。
- 文档公式版 S1 的 FP：baseline 分布 176、dynamic-score 279、corrected 12；paper 分布 4,037、5,377、660。文档公式版并未改善总体指标。
- S6 同 ID 恢复率：两种分布下，文档公式版 dynamic-score 均为 1.00；原 baseline 均为 0.00。符号修正版 baseline 分布为 1.00，但 paper 分布为 0.00。恢复成功不保证其余定位、FP/FN 或 fragmentation 指标也改善。
- S0 的 baseline MOTA-like 为 1.000、RMSE 约 0.075 m；两个新 tracker MOTA-like 约 0.988、RMSE 约 0.107 m，显示简化滤波器在当前无噪声基准下也存在少量估计偏差。

## 4. 重要公式问题与复现边界

项目 `docs/plan/IMPLEMENTATION_SPEC.md` 给出的漏检增量为：

`ΔS_miss = log((1 − P_FA) / (1 − P_D))`

当前配置使用 `P_D=0.90`、`P_FA=0.02`，所以该增量约为 **+2.28**。这意味着漏检会加分，可能促使 clutter 产生的候选轨迹被确认，造成 FP 激增、干扰真目标关联。实现忠实保留该文档公式作为 `dynamic_score` 默认版本，并另行提供 `likelihood_penalty` 符号修正版进行诊断。**需要 CG/论文来源进一步确认该项是否为原文符号或转写问题；在确认前，不应静默将修正版宣称为论文实现。**

Phase 3 当前实现的复现性质：

| 组成 | 性质 |
|---|---|
| 分数递推结构、候选/维持轨迹规则、文档中的关联/漏检/新生增量 | 按项目 `docs/plan/IMPLEMENTATION_SPEC.md` 转写实现 |
| 4D Cartesian Kalman/BLUE 风格模型 | 为保持 Phase 2 Cartesian `[x,y]` 输入而做的工程简化；不是论文 5D coordinated-turn + range/bearing 模型 |
| Mahalanobis gate、物理 gate、lost 扩展 gate、candidate/lost expiry | 工程扩展/复现假设 |
| `λ_s=0.95`、`λ_t=4`、`S_c=1.0`、`P_B=0.05`、`P_FA=0.02`、`P_D=0.90` 等数值 | 论文未给定或未验证的配置假设 |
| `dynamic_score_corrected` 漏检惩罚符号 | 诊断修正，不是已验证论文公式 |
| CTDA 关联 Transformer | 未实现、未训练 |

## 5. 当前问题清单与后续建议

1. **先确认漏检增量符号。**向论文公式、勘误、补充材料或作者核实；确认前同时保留文档公式版与修正版结果。
2. **优化高杂波的关联效率。**目前 dynamic tracker S7 paper runtime 约 8.5–8.7 秒，corrected 约 4.4 秒，原 baseline 约 0.09 秒；报告指出逐轨迹×逐测量计算是主要扩展性问题，可先做安全的候选预筛选并保持回归对照。
3. **分析 corrected 版 fragmentation / FN。**baseline 分布 S1/S3 fragmentation 可达 20；更强的 clutter 抑制伴随确认更严格、部分场景匹配下降。
4. **增强 paper clutter 场景下的关联。**corrected 版虽降低 FP，但 MOTA-like 仍为负，特别是 S6 恢复率为 0；需再评估数据密度假设、门控、生命周期参数和后续 CTDA。
5. **不要过度解读参数。**score、gate、生命周期参数的取值均是假设，当前矩阵未完成论文 Table 9/10 等价的 100 次 Monte Carlo + OSPA 验证。

## 6. 交付物索引

各阶段完整实验细节以原报告及 manifest 为准：

- 环境及 smoke 输出：`smoke_environment.stdout.txt`、`smoke_metrics.json`、`smoke_ctda_interface.json`
- Phase 1 数据集说明：[phase1_report.md](phase1_report.md)；配置：[phase1_2d.yaml](../../configs/phase1_2d.yaml)；manifest：[phase1_manifest.json](../../data/manifests/phase1_manifest.json)
- Phase 1.5 说明：[phase1_5_report.md](phase1_5_report.md)；配置：[phase1_5_distributions.yaml](../../configs/phase1_5_distributions.yaml)；manifest：[phase1_5_manifest.json](../../data/manifests/phase1_5_manifest.json)
- Phase 2 场景与 baseline 回归：[phase2_report.md](phase2_report.md)；配置：[phase2_scenarios.yaml](../../configs/phase2_scenarios.yaml)；manifest：[phase2_manifest.json](../../data/manifests/phase2_manifest.json)
- Phase 3 tracker 详细结果：[phase3_report.md](phase3_report.md)；配置：[phase3_dynamic_tracker.yaml](../../configs/phase3_dynamic_tracker.yaml)；manifest：[phase3_manifest.json](../../data/manifests/phase3_manifest.json)
- Phase 3 代码：[filter.py](../../src/tracking/filter.py)、[dynamic_score_tracker.py](../../src/tracking/dynamic_score_tracker.py)、[phase3_metrics.py](../../src/evaluation/phase3_metrics.py)、[src/experiments/phase3_dynamic_score_evaluation.py](../../src/experiments/phase3_dynamic_score_evaluation.py)

## 7. 给 CG 的最终结论

Phase 0–3 已建立并验证 baseline 到 dynamic-score 的可重复实验管线，且没有覆盖既有数据与回归结果。新 dynamic-score tracker 目前不能整体宣称优于 baseline：文档公式版暴露出漏检增量符号疑点，修正版在 baseline clutter 分布的若干场景有明显改善，但在 paper λ=20 高 clutter 下仍有较差 MOTA、恢复或运行时间限制。建议 CG 优先确认公式符号，再决定是否将修正版作为后续研究分支；CTDA 尚未进入实现阶段。
