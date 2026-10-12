# Phase 2：非学习型 tracking 场景验证报告

本报告面向后续接手的 AI / 工程师，说明 Phase 2 的执行方式、产物、指标定义、判定结论与当前 tracker 的问题清单。

- 范围：在固定 seed 的 8 个场景 × 2 个分布上验证现有**门限贪心最近邻 baseline**；不实现 CTDA，不修改 tracker 的关联/生命周期逻辑，不删除或覆盖 Phase 1 / Phase 1.5 产物。
- 结论：**16 个用例全部通过基础设施验收**；多数场景的 tracking 质量偏低，属于 **baseline 能力限制**（详见"问题清单"），不是 Phase 2 基础设施失败。
- 论文专属的 CTDA Transformer 仍未实现/训练；本 Phase 不对此作任何达标声明。

## 1. 交付物

新增：

| 文件 | 作用 |
|---|---|
| `configs/phase2_scenarios.yaml` | S0–S7 场景定义、两分布默认值、seed、评估门限 |
| `src/simulation/scenarios.py` | 确定性场景构建（`ScenarioSpec` / `generate_scenario` / `resolve_scenarios`） |
| `src/evaluation/phase2_metrics.py` | Phase 2 指标评估器（含贪心匹配、fragmentation） |
| `src/evaluation/__init__.py` | 包初始化 |
| `src/experiments/phase2_baseline_evaluation.py` | 矩阵运行器：生成/读取数据 → 去标签 → 计时 → 评估 → 写 JSON + 轨迹图 |
| `tests/test_phase2.py` | 矩阵、可重复性、指标完整性、标签隔离、Phase 1 兼容性测试 |
| `data/results/phase2/<distribution>/<scenario>/<scenario>_metrics.json` | 16 份逐用例指标 |
| `data/manifests/phase2_manifest.json` | 汇总 manifest（配置、种子、指标、SHA-256、图路径） |
| `outputs/figures/phase2/<distribution>/<scenario>_trajectory.png` | 16 张轨迹图 |

修改（最小、非破坏性）：

| 文件 | 修改 |
|---|---|
| `src/simulation/dataset.py` | `load_dataset_sample` 增加 schema 1 向后兼容：缺失 `target_counts` 时用 `truth_states.shape[2]` 推断目标数。仅影响读取旧 Phase 1 数据，不改变 schema 2 行为、不改任何 tracker 逻辑 |
| `README.md` | 增加 Phase 2 运行命令与产物布局 |

未改动：`src/tracking/baseline_tracker.py`（tracker 与关联逻辑）、`configs/phase1_*.yaml`、任何 Phase 1 / 1.5 数据文件。

## 2. 场景矩阵

S0–S7 的通用意图见 `configs/phase2_scenarios.yaml`；解析后每个用例的完整配置与 seed 都写入对应的 `*_metrics.json` 与 manifest。

| 场景 | 目标数 | 运动 | 显式覆盖 | 用途 |
|---|---:|---|---|---|
| S0 | 1 | solo | P_d=1.0, σ_meas=0, σ_proc=0, clutter=0 | 零噪声零杂波单目标 |
| S1 | 5 | parallel | 分布默认杂波/噪声 | 多条平行**不交叉**轨迹 |
| S2 | 4 | crossing | 分布默认杂波/噪声 | 轨迹**交叉** |
| S3 | 3 | parallel | P_d=0.45 | 受控漏检 |
| S4 | 3 | parallel | clutter: baseline 5.0 / paper 20.0 | 虚警/杂波 |
| S5 | 5 | parallel | 显式 lifetimes | 错峰出生/消失 |
| S6 | 2 | parallel | target 1 在帧 30–45 强制无检测 | 短时丢失后重现 |
| S7 | 25 | parallel | clutter=20.0（两分布） | 高目标数 + 高杂波压力 |

- 分布默认：`baseline_distribution` clutter=1.5、`paper_assumption_v1` clutter=20.0；其余参数（steps=80, area=100, P_d=0.9, σ_meas=1.5, σ_proc=0.08）一致。
- 除显式覆盖外，场景保留各自分布的杂波/目标假设。
- **S7 为指定的高杂波压力用例**，故两分布均显式设为 clutter=20.0；baseline 的"配置默认"为 1.5，此处按场景显式请求覆盖，报告中已明确记录。

### 解析后 seed

seed = 场景 seed + 分布 seed_offset（baseline 偏移 0，paper 偏移 10000）。

| 场景 | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2000 | 2011 | 2022 | 2033 | 2044 | 2055 | 2066 | 2077 |
| paper | 12000 | 12011 | 12022 | 12033 | 12044 | 12055 | 12066 | 12077 |

### 场景构建说明

现有 `simulate_scenario` 将前 4 个目标硬编码为交叉运动、生命周期固定，无法表达"不交叉 / 指定寿命 / 受控消失"。新增 `src/simulation/scenarios.py` 以相同数据语义（边界反射、出生后逐帧过程加速度、逐检测高斯测量噪声、区域内泊松杂波、逐帧洗牌）生成 `SimulationData`，但初始布局由 `motion` 决定：

- `solo`：单条直线轨迹；
- `parallel`：目标分布在平行车道、同向平移，**不交叉**；
- `crossing`：两组目标相向运动，路径在中部相交。

初始运动不消耗 RNG（有意为之），使布局与噪声流解耦、易于复现。`disappearance` 窗口内目标真值保持有限但强制不产生检测。

## 3. 指标定义

评估器 `src/evaluation/phase2_metrics.py` 仅消费真值 `SimulationData` 与 tracker 的 `TrackingResult`，做逐帧贪心"真值位置 ↔ 估计位置"匹配（评估门限 `match_gate=12.0 m`，与 tracker 自身门限 9.0 m 独立）。

- **truth object-frames**：每帧每个活跃真值目标计 1。
- **matched detections**：在门限内匹配到估计的真值目标-帧数量。
- **false negatives (FN)** = truth object-frames − matched detections。
- **false positives (FP)** = 输出估计数 − matched detections。
- **ID switches**：某真值目标在连续两个"被匹配"帧之间分配的 track ID 发生变化即计 1。
- **fragmentation**：对每个真值目标，取其被匹配帧排序后按连续段分组，累加 `段数 − 1`（即已建立的跟踪被打断的次数）。
- **position RMSE** = $\sqrt{\overline{d^2}}$，$d$ 为匹配对的真值-估计欧氏距离（米）。
- **MOTA-like** = $1 - (\mathrm{FN} + \mathrm{FP} + \mathrm{ID\,switches}) / \text{truth object-frames}$。

> 评估匹配用的是"真值位置 ↔ 估计位置"的位置匹配，区别于数据集侧车中的"测量 ↔ 真值 ID"来源匹配；后者只用于数据检查，不进入 tracker 输入。

## 4. 运行命令

```bash
# 全矩阵（8 场景 × 2 分布）
conda run -n mttrack-repro python -m src.experiments.phase2_baseline_evaluation

# 子集
conda run -n mttrack-repro python -m src.experiments.phase2_baseline_evaluation \
  --scenarios S0,S1,S7 --distributions paper_assumption_v1

# 全部测试
conda run -n mttrack-repro python -m unittest discover -s tests -v
```

产物默认写入 `data/results/phase2/`、`outputs/figures/phase2/`、`data/manifests/phase2_manifest.json`。

## 5. 结果

16 个用例的确定性指标如下。`runtime` 为 wall-clock，取自最终一次 `phase2_manifest.json` 运行，跨运行会有波动，不纳入可重复性判定，仅作参考。

### baseline_distribution

| 场景 | matched | FP | FN | IDS | frag | RMSE [m] | MOTA-like | runtime [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 80 | 0 | 0 | 0 | 0 | 0.075 | 1.000 | 0.0582 |
| S1 | 399 | 176 | 1 | 18 | 0 | 1.674 | 0.512 | 0.0085 |
| S2 | 318 | 232 | 2 | 9 | 1 | 1.647 | 0.241 | 0.0077 |
| S3 | 228 | 156 | 12 | 2 | 3 | 2.136 | 0.292 | 0.0054 |
| S4 | 240 | 755 | 0 | 8 | 0 | 1.765 | −2.179 | 0.0124 |
| S5 | 333 | 177 | 2 | 8 | 1 | 1.736 | 0.442 | 0.0081 |
| S6 | 150 | 156 | 10 | 1 | 1 | 1.820 | −0.044 | 0.0053 |
| S7 | 1995 | 4069 | 5 | 482 | 2 | 2.017 | −1.278 | 0.0894 |

### paper_assumption_v1

| 场景 | matched | FP | FN | IDS | frag | RMSE [m] | MOTA-like | runtime [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 80 | 0 | 0 | 0 | 0 | 0.075 | 1.000 | 0.0019 |
| S1 | 398 | 4037 | 2 | 57 | 2 | 1.908 | −9.240 | 0.0534 |
| S2 | 320 | 3944 | 0 | 74 | 0 | 2.149 | −11.556 | 0.0502 |
| S3 | 235 | 3703 | 5 | 53 | 4 | 3.861 | −14.671 | 0.0449 |
| S4 | 240 | 3885 | 0 | 22 | 0 | 2.087 | −15.279 | 0.0479 |
| S5 | 335 | 3832 | 0 | 37 | 0 | 1.918 | −10.549 | 0.0505 |
| S6 | 158 | 3830 | 2 | 21 | 1 | 2.917 | −23.081 | 0.0449 |
| S7 | 1997 | 4082 | 3 | 462 | 3 | 2.084 | −1.273 | 0.0885 |

说明：

- S0 两分布结果相同：该场景将测量/过程噪声与杂波全部置零，检测概率 1.0，轨迹与 seed 无关。RMSE 0.075 m 不为 0 是 tracker 内部更新增益（0.65）与过程协方差 $Q$ 造成的滤波滞后，属预期行为。
- 高杂波下 FP 数量级远超真值目标数：贪心关联把大量杂波测量初始化为新轨迹。MOTA-like 因此显著为负——这是**质量限制**，按计划不作为 Phase 2 基础设施失败，也不引入人为达标阈值。

## 6. 验收判定

工程验收规则（不含人为的质量阈值）：用例"通过基础设施验收"当且仅当**可复现（排除 wall-clock 运行时）、输入/输出可读、标签与 tracker 输入隔离、请求的指标与轨迹图全部产出**。tracking 质量只作报告，poor 结果记为 tracker 限制。

| 检查项 | 结果 |
|---|---|
| 16 用例全部解析并运行 | ✅ |
| 数据可复现（同 seed 逐位一致） | ✅（`test_generation_is_deterministic` + 双次整矩阵对比） |
| 确定性指标跨运行逐位一致 | ✅（双次 `run_matrix` 对比相等；仅 runtime 不同） |
| runtime 有限且非负 | ✅ 16/16 |
| 指标 JSON 完整（全部键）且有限 | ✅ |
| 轨迹图产出（16 张，非空） | ✅ |
| tracker 输入不含来源标签 | ✅（`test_tracker_input_carries_no_source_labels`） |
| 改写来源标签不改变输出 | ✅（`test_tracker_output_independent_of_source_labels`） |
| 单测 | ✅ 21 项全通过 |
| Phase 1 / 1.5 产物完整 | ✅ manifest SHA-256 全部匹配（3 + 4 个文件），未覆盖 |
| 场景符合意图 | ✅ S0 零噪声单目标、S1 不交叉、S5 寿命匹配、S6 有检测空洞且真值保留、S7 高杂波 |

## 7. 当前 tracker 问题清单（供下一阶段参考）

以下均为 **baseline（门限贪心最近邻 + 简单生命周期）** 在 Phase 2 暴露的限制，不属于 Phase 2 基础设施缺陷：

1. **高杂波下 FP 爆炸**：paper 分布（clutter=20/帧）FP 达 ~3.7k–4.1k，MOTA-like 低至 −23。贪心关联缺乏"测量应新建还是归属已有轨迹"的概率判据。
2. **交叉/近距离导致 ID switch**：S2、S7 出现大量 IDS（S7 两分布 IDS ≈ 462–482），交叉点附近轨迹易互换 ID。
3. **无轨迹置信分数与动态轨迹管理**：当前仅用 `min_hits` / `max_missed`，与论文 score 递推（λ_s、λ_t、S_c、P_B、P_FA、P_D）差距大；这些超参论文未给出数值，尚未配置化。
4. **关联门限固定**：gate=9.0 m 固定，不随杂波率/分布自适应，高杂波下过小、稀疏场景下偏大。
5. **短时丢失与重现**：S6 在检测空洞期出现 FP，重现后可能分配新 ID（表现为 FN/IDS）；缺少"维持未检测轨迹再重连"的机制。
6. **相关/密集目标下 fragmentation**：S3 低检测概率下出现 fragmentation（3–4）。
7. **Phase 1 原始数据集（schema 1）读取**：当前 loader 原只支持 schema 2，导致 `data/datasets/phase1/{smoke,train,test}` 无法加载。已做最小向后兼容修复（缺 `target_counts` 时取 `truth_states.shape[2]`）；建议后续统一 schema 或补写 schema 版本迁移。
8. **CTDA 未实现**：论文的 row-column / 3-channel cross Transformer、训练数据组织、训练损失均未实现；本 Phase 未训练任何模型。

## 8. Reproduction assumptions / 边界

1. S0–S7 的场景构造（运动布局、寿命、消失窗口）是本 Phase 为可复现验证**刻意设计**的工程设置，不是对论文未公开分布的断言。
2. 除显式覆盖外，场景沿用各分布的杂波/目标假设；S7 被显式指定为高杂波压力用例（两分布 clutter=20）。
3. 评估门限 `match_gate=12.0 m`、tracker 门限 9.0 m、`min_hits=2`、`max_missed=5` 为工程取值；tracker 参数保持 Phase 1 默认，未调参。
4. 运行时依赖硬件与实现，仅记录不做断言。
5. 未做 CTDA 训练；未引入任何论文 Table 9/10 的数值达标阈值。

## 9. 验证记录

- 单测：`python -m unittest discover -s tests -v` → **21 项通过**（Phase 1.5 5 项 + Phase 2 16 项）。
- 整矩阵双次运行：16 用例的 `(distribution, scenario, resolved_config, metrics)` 完全一致；runtime 均有限且 ≥0；16 张图与 16 份指标 JSON 均非空。
- 完整性：`phase1_manifest.json`（3 文件）与 `phase1_5_manifest.json`（4 文件）SHA-256 全部匹配，Phase 1/1.5 数据未被覆盖。
- 兼容性：schema 1（`data/datasets/phase1/smoke/smoke_dataset.npz`）与 schema 2（`data/datasets/phase1_5/baseline_distribution/test/test_dataset.npz`）均可由 loader 读取并通过 tracker 运行。

## 10. 下一步建议

1. 以本矩阵为回归基线，接入 CTDA `AssociationModel.associate(...)`，复用同一 runner/evaluator 对比 baseline 与 CTDA。
2. 将 tracker 生命周期迁移到论文 score 递推，并把 λ_s、λ_t、S_c、P_B、P_FA、P_D 配置化（论文未给数值，需标注假设）。
3. 补充 OSPA（c=100, r=1）与 100 次 Monte Carlo 评估，与 Table 9 对照（仅作比较目标）。
4. 统一数据集 schema，或在 loader 中保留版本迁移路径，避免新旧 split 读取不一致。
