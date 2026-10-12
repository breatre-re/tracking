# Phase 3：动态评分跟踪器对比报告

本报告面向后续接手的 AI / 工程师，说明新增**动态评分跟踪器**的设计、配置假设、与 baseline 的对比结果、验收判定与遗留问题。

- 范围：在**未修改** Phase 1/1.5/2 产物与 `src/tracking/baseline_tracker.py` baseline 行为的前提下，新增一个可单独选择的动态评分 tracker，并在**未改动的 Phase 2 S0–S7 矩阵（两分布）**上对 baseline 与动态评分 tracker 同时评估。
- 未实现 CTDA，未训练任何模型。
- 结论：**48 个用例全部通过基础设施验收**；质量差异与低指标按"tracker 质量限制"报告，不设人为达标阈值。同时发现一个 **score 增量符号问题**（见第 8 节），已显式报告而非默默修改。

## 1. 交付物

新增：

| 文件 | 作用 |
|---|---|
| `src/tracking/filter.py` | `FilterConfig` 与 4D Cartesian BLUE/Kalman 滤波器（Joseph 形式数值稳定更新） |
| `src/tracking/dynamic_score_tracker.py` | `ScoreTrackerConfig` 与 `DynamicScoreTracker`（Mahalanobis 关联、分数递推、生命周期、恢复） |
| `src/tracking/__init__.py` | 包初始化 |
| `src/evaluation/phase3_metrics.py` | 同 ID 恢复率评估（离线，针对配置的消失窗口） |
| `configs/phase3_dynamic_tracker.yaml` | 滤波器/门限/分数/生命周期参数与假设清单 |
| `src/experiments/phase3_dynamic_score_evaluation.py` | 运行未改动的 16 用例矩阵 × 3 个 tracker 变体，输出 JSON、轨迹图与 manifest |
| `tests/test_phase3_tracker.py` | 滤波器、生命周期、恢复、标签隔离、可重复性、产物测试（15 项） |
| `data/results/phase3/<tracker>/<distribution>/<scenario>/<scenario>_metrics.json` | 48 份逐用例指标 |
| `data/manifests/phase3_manifest.json` | 汇总 manifest |
| `outputs/figures/phase3/<tracker>/<distribution>/<scenario>_trajectory.png` | 48 张轨迹图 |

修改：仅 `README.md`（新增 Phase 3 用法）。`src/tracking/baseline_tracker.py`、Phase 2 runner/evaluator、Phase 1/1.5/2 数据与结果均未改动。

## 2. 实现说明

### 2.1 滤波器（4D Cartesian BLUE/Kalman）

计划指定的显式简化：状态顺序 `[x, y, vx, vy]`，测量为 Cartesian `[x, y]`，线性高斯恒速模型。

- `F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]`，`H = [[1,0,0,0],[0,1,0,0]]`。
- 过程协方差按加速度噪声 `q = σ_a²` 的块结构构造；测量噪声 `R = σ_r² I₂`。
- `predict` 后与 `update` 后都做对称化，`update` 使用 Joseph 形式 `P=(I-KH)P(I-KH)ᵀ+KRKᵀ`，保证协方差对称半正定。
- 暴露 `innovation`、`mahalanobis`、`log_likelihood`、`update`（返回创新对数似然）。线性求逆一律用 `np.linalg.solve`。

这不是论文的 5D 协调转弯/距离-方位模型，是明确的工程简化。

### 2.2 关联

- 自适应 Mahalanobis 门限（协方差归一化距离 ≤ `mahalanobis_gate`）**且** 物理距离 ≤ `physical_gate`。
- 门限内代价矩阵做**一对一分配**（优先 `scipy.optimize.linear_sum_assignment`，无 scipy 时回退贪心）。
- **丢失轨迹**使用更宽的重关联门限 `lost_gate_scale`（默认 2.5×），以支持跨越短时间隙的同 ID 恢复。

### 2.3 分数递推与事件增量

按 `docs/plan/IMPLEMENTATION_SPEC.md` 第 6 节实现：

$$
S_l(k) = \lambda_s\,S_l(k-1) + \Delta S_l(k) - \lambda_s^{\lambda_t}\,\Delta S_l(k-\lambda_t)
$$

事件增量：
- 未匹配测量 → 新建候选：$\Delta S = \log(P_B/P_{FA})$
- 未匹配轨迹（漏检）：$\Delta S = \log\!\big((1-P_{FA})/(1-P_D)\big)$
- 匹配轨迹：$\Delta S = \log p(z\mid \hat{x}) - \log P_{FA}$

每轨迹按帧保存增量，滞后项取 `k - λ_t` 帧的增量（不存在则取 0）。

### 2.4 生命周期

状态：`candidate` / `confirmed` / `lost` / `deleted`。

- 候选：匹配且 `S ≥ S_c` → confirmed；未匹配且（`S < 0` 或 `age > candidate_expiry`）→ 删除。
- 已确认：未匹配且（`S < 0` 或 `misses > max_missed`）→ 转为 **lost**。
- lost：仍参与关联（支持同 ID 恢复）；连续未匹配超过 `lost_expiry` → 删除；重新匹配 → 恢复为同一 ID 的 confirmed，使用文档化关联增量（可选 `reappearance_reward`，默认 0）。
- 报告：默认只报告 confirmed（`report_lost: false`），lost 轨迹保留关联资格但不输出（避免滑行轨迹抬高 FP）。

### 2.5 标签隔离

tracker 输入只含 `Detection(position)`，`source_target_id` 从不进入关联/更新；评估侧车才使用真值标签。同 ID 恢复按真值目标前后帧匹配到的估计 track ID 是否一致判定。

## 3. 配置与假设

配置文件 `configs/phase3_dynamic_tracker.yaml`。三个变体：`baseline`、`dynamic_score`（文档化增量）、`dynamic_score_corrected`（符号修正增量，诊断用）。

| 参数 | 值 | 状态 |
|---|---|---|
| 状态/测量 | `[x,y,vx,vy]` / Cartesian `[x,y]` | 工程简化（非论文 5D） |
| `process_accel_std` | 0.10 | 假设 |
| `measurement_noise_std` | 1.5 | 假设（与场景一致） |
| `initial_position_var` / `initial_velocity_var` | 0.25 / 1.0 | 假设 |
| `mahalanobis_gate` | 9.21（χ²₂ 99%） | 工程附加，假设 |
| `physical_gate` | 10.0 m | 工程附加，假设 |
| `lost_gate_scale` | 2.5× | 工程附加，假设（支持恢复） |
| `λ_s` (`decay`) | 0.95 | 假设 |
| `λ_t` (`history_lag`) | 4 | 假设 |
| `S_c` (`confirm_threshold`) | 1.0 | 假设 |
| `P_B` | 0.05 | 假设 |
| `P_FA` | 0.02 | 假设 |
| `P_D` | 0.90 | 假设 |
| `max_missed` | 5 | 假设 |
| `candidate_expiry` | 6 | 假设 |
| `lost_expiry` | 15 | 工程附加，假设 |
| `reappearance_reward` | 0.0 | 假设（恢复只用文档化增量） |

> 论文未公布 `λ_s、λ_t、S_c、P_B、P_FA、P_D`。以上数值均为复现假设，未从论文/作者验证。`dynamic_score_corrected` 的增量符号是工程修正，不是论文公式。

## 4. 运行与产物

```bash
# 全矩阵：3 变体 × 8 场景 × 2 分布 = 48 用例
conda run -n mttrack-repro python -m src.experiments.phase3_dynamic_score_evaluation

# 子集示例
conda run -n mttrack-repro python -m src.experiments.phase3_dynamic_score_evaluation \
  --trackers dynamic_score_corrected --scenarios S0,S1,S6 --distributions baseline_distribution

# 全部测试
conda run -n mttrack-repro python -m unittest discover -s tests -v
```

产物：`data/results/phase3/...`、`outputs/figures/phase3/...`、`data/manifests/phase3_manifest.json`。

## 5. 结果

指标来自未改动的 Phase 2 评估器（`match_gate=12 m`）。`recov` 为 S6 消失窗口的同 ID 恢复率（其余场景 `n/a`）。runtime 为 wall-clock，仅参考。

### baseline（与 Phase 2 逐位一致）

| 分布 | 场景 | match | FP | FN | IDS | frag | RMSE | MOTA | recov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | S0 | 80 | 0 | 0 | 0 | 0 | 0.075 | 1.000 | n/a |
| baseline | S1 | 399 | 176 | 1 | 18 | 0 | 1.674 | 0.512 | n/a |
| baseline | S2 | 318 | 232 | 2 | 9 | 1 | 1.647 | 0.241 | n/a |
| baseline | S3 | 228 | 156 | 12 | 2 | 3 | 2.136 | 0.292 | n/a |
| baseline | S4 | 240 | 755 | 0 | 8 | 0 | 1.765 | −2.179 | n/a |
| baseline | S5 | 333 | 177 | 2 | 8 | 1 | 1.736 | 0.442 | n/a |
| baseline | S6 | 150 | 156 | 10 | 1 | 1 | 1.820 | −0.044 | 0.00 |
| baseline | S7 | 1995 | 4069 | 5 | 482 | 2 | 2.017 | −1.278 | n/a |
| paper | S0 | 80 | 0 | 0 | 0 | 0 | 0.075 | 1.000 | n/a |
| paper | S1 | 398 | 4037 | 2 | 57 | 2 | 1.908 | −9.240 | n/a |
| paper | S2 | 320 | 3944 | 0 | 74 | 0 | 2.149 | −11.556 | n/a |
| paper | S3 | 235 | 3703 | 5 | 53 | 4 | 3.861 | −14.671 | n/a |
| paper | S4 | 240 | 3885 | 0 | 22 | 0 | 2.087 | −15.279 | n/a |
| paper | S5 | 335 | 3832 | 0 | 37 | 0 | 1.918 | −10.549 | n/a |
| paper | S6 | 158 | 3830 | 2 | 21 | 1 | 2.917 | −23.081 | 0.00 |
| paper | S7 | 1997 | 4082 | 3 | 462 | 3 | 2.084 | −1.273 | n/a |

### dynamic_score（文档化增量）

| 分布 | 场景 | match | FP | FN | IDS | frag | RMSE | MOTA | recov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | S0 | 79 | 0 | 1 | 0 | 0 | 0.107 | 0.988 | n/a |
| baseline | S1 | 360 | 279 | 40 | 37 | 2 | 1.749 | 0.110 | n/a |
| baseline | S2 | 298 | 364 | 22 | 35 | 1 | 1.571 | −0.316 | n/a |
| baseline | S3 | 222 | 183 | 18 | 5 | 1 | 2.003 | 0.142 | n/a |
| baseline | S4 | 235 | 1271 | 5 | 32 | 0 | 2.244 | −4.450 | n/a |
| baseline | S5 | 288 | 385 | 47 | 23 | 3 | 1.563 | −0.358 | n/a |
| baseline | S6 | 123 | 275 | 37 | 7 | 2 | 2.144 | −0.994 | **1.00** |
| baseline | S7 | 1929 | 5674 | 71 | 463 | 4 | 2.358 | −2.104 | n/a |
| paper | S0 | 79 | 0 | 1 | 0 | 0 | 0.107 | 0.988 | n/a |
| paper | S1 | 390 | 5377 | 10 | 68 | 1 | 2.169 | −12.637 | n/a |
| paper | S2 | 310 | 5150 | 10 | 64 | 1 | 2.362 | −15.325 | n/a |
| paper | S3 | 231 | 5050 | 9 | 48 | 2 | 2.804 | −20.279 | n/a |
| paper | S4 | 231 | 5226 | 9 | 33 | 0 | 1.883 | −20.950 | n/a |
| paper | S5 | 331 | 5230 | 4 | 58 | 0 | 2.473 | −14.797 | n/a |
| paper | S6 | 153 | 5016 | 7 | 30 | 0 | 2.710 | −30.581 | **1.00** |
| paper | S7 | 1955 | 5851 | 45 | 487 | 4 | 2.369 | −2.192 | n/a |

### dynamic_score_corrected（符号修正增量，诊断）

| 分布 | 场景 | match | FP | FN | IDS | frag | RMSE | MOTA | recov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | S0 | 79 | 0 | 1 | 0 | 0 | 0.107 | 0.988 | n/a |
| baseline | S1 | 305 | 12 | 95 | 11 | 20 | 1.340 | 0.705 | n/a |
| baseline | S2 | 270 | 21 | 50 | 15 | 16 | 1.862 | 0.731 | n/a |
| baseline | S3 | 110 | 7 | 130 | 5 | 20 | 2.858 | 0.408 | n/a |
| baseline | S4 | 216 | 117 | 24 | 9 | 18 | 2.000 | 0.375 | n/a |
| baseline | S5 | 257 | 9 | 78 | 4 | 19 | 1.218 | 0.728 | n/a |
| baseline | S6 | 54 | 26 | 106 | 1 | 4 | 2.512 | 0.169 | **1.00** |
| baseline | S7 | 1772 | 1011 | 228 | 420 | 74 | 2.750 | 0.170 | n/a |
| paper | S0 | 79 | 0 | 1 | 0 | 0 | 0.107 | 0.988 | n/a |
| paper | S1 | 285 | 660 | 115 | 53 | 15 | 3.329 | −1.070 | n/a |
| paper | S2 | 262 | 798 | 58 | 36 | 17 | 3.276 | −1.788 | n/a |
| paper | S3 | 163 | 674 | 77 | 25 | 27 | 5.796 | −2.233 | n/a |
| paper | S4 | 206 | 664 | 34 | 28 | 14 | 2.660 | −2.025 | n/a |
| paper | S5 | 287 | 870 | 48 | 29 | 13 | 2.152 | −1.827 | n/a |
| paper | S6 | 113 | 886 | 47 | 23 | 10 | 2.947 | −4.975 | 0.00 |
| paper | S7 | 1791 | 1212 | 209 | 491 | 94 | 2.867 | 0.044 | n/a |

### 生命周期诊断（仅动态 tracker，节选）

| tracker | 分布 | 场景 | confirm | recover | lost | deleted | births | max_conf | final_conf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| dynamic_score | baseline | S1 | 23 | 25 | 36 | 62 | 84 | 12 | 12 |
| dynamic_score | baseline | S7 | 182 | 422 | 500 | 94 | 244 | 115 | 104 |
| dynamic_score | paper | S1 | 167 | 406 | 493 | 105 | 228 | 94 | 80 |
| dynamic_score_corrected | baseline | S1 | 6 | 27 | 29 | 114 | 124 | 6 | 4 |
| dynamic_score_corrected | paper | S1 | 36 | 167 | 186 | 1093 | 1133 | 22 | 17 |
| dynamic_score_corrected | paper | S7 | 82 | 442 | 468 | 861 | 941 | 56 | 56 |

（完整 48 行见 `data/manifests/phase3_manifest.json` 的 `diagnostics`。）

## 6. 验收判定

基础设施 PASS 规则：两 tracker 变体在全部 16 矩阵用例上运行成功；所需指标/生命周期/恢复字段有限且非空；轨迹图非空；配置/种子/manifest 可读；同 seed 的确定性字段跨运行一致（排除 runtime）；真值标签变更不能改变 tracker 输出。

| 检查项 | 结果 |
|---|---|
| 48 用例全部运行（3 变体 × 16） | ✅ |
| 指标有限、生命周期/恢复字段齐全 | ✅ |
| 轨迹图产出（48 张，非空） | ✅ |
| 同 seed 确定性字段跨运行逐位一致 | ✅（`test_metrics_repeatable_runtime_finite` + 双次 `run_matrix` 对比） |
| runtime 有限且非负（48/48） | ✅ |
| 标签隔离（改写来源标签不改变输出） | ✅（`test_truth_labels_do_not_reach_tracker`） |
| baseline 结果与 Phase 2 逐位一致 | ✅（回归一致，未改 baseline） |
| Phase 1/1.5/2 产物完整 | ✅（manifest SHA-256 全部匹配，未被覆盖） |
| 单测 | ✅ 36 项全通过（Phase 1.5 5 + Phase 2 16 + Phase 3 15） |

tracking 质量无强制阈值；低指标/回归记为 tracker 质量限制。

## 7. 发现与问题清单

### 7.1 score 增量符号问题（重要）

`docs/plan/IMPLEMENTATION_SPEC.md` 规定漏检增量 $\Delta S = \log\!\big((1-P_{FA})/(1-P_D)\big)$。当 $P_D > P_{FA}$（本例 0.9 vs 0.02）时该值为 **正**（≈ +2.28），即**漏检会加分**。后果：

- 由杂波测量新建的候选在**一帧未匹配后即加分确认**，产生大量虚假 confirmed 轨迹，并抢走真值目标的关联 → FP 激增、FN 上升。
- 这正是 `dynamic_score` 在高杂波下 FP 巨大（paper S1 5377、S7 5851）的主因。

处理方式：**保留文档化公式为默认**（忠于规范，不静默修改），并通过 `missed_increment_mode: likelihood_penalty` 提供符号修正的惩罚 $\log\!\big((1-P_D)/(1-P_{FA})\big)$（`dynamic_score_corrected` 诊断变体）。修正后杂波被显著抑制（baseline S1 FP 279→12，MOTA 0.110→0.705），代价是确认更严格、fragmentation 上升、部分场景 match 下降。

**建议：后续与论文/作者核对漏检增量的正确方向**；这是规范中的一处疑似符号/转写问题。

### 7.2 其它 tracker 问题

1. **无关联预筛选**：逐轨迹×逐测量的 Mahalanobis 计算使复杂度随目标数/杂波上升；S7 与高杂波 paper 用例 runtime 达数秒（baseline ~0.09 s，dynamic ~8.5 s，corrected ~4.4 s）。
2. **确认延迟**：受滞后项 $\lambda_s^{\lambda_t}\Delta S(k-\lambda_t)$ 抑制，稳态分数有限，`S_c` 偏大时会长期候选、抬高 FN。
3. **恢复对门限敏感**：S6 间隙 16 帧内 lost 轨迹预测漂移超出 `physical_gate`，因此引入 `lost_gate_scale`；paper 分布下 corrected 变体 S6 恢复仍为 0（杂波干扰关联）。
4. **fragmentation 偏高**：corrected 变体在 baseline 分布 S1/S3 等 fragmentation 达 20，说明轨迹易被删除重建。
5. **baseline 无恢复能力**：S6 恢复率 0.00（`max_missed` 后直接删除），新 ID 重建。
6. **CTDA 仍未实现**：本 Phase 未训练任何模型。

## 8. Reproduction assumptions / 边界

1. 4D Cartesian 恒速 Kalman 是**明确工程简化**，非论文 5D CT/range-bearing 模型；保持 Phase 2 的 Cartesian 测量语义以不破坏固定回归矩阵。
2. Mahalanobis/物理门限、`lost_gate_scale`、`candidate_expiry`、`lost_expiry`、`report_lost=false` 均为工程附加，其数值为假设。
3. `λ_s、λ_t、S_c、P_B、P_FA、P_D` 论文未给出，表中数值为假设。
4. `dynamic_score` 使用文档化漏检增量；`dynamic_score_corrected` 是显式标注的符号修正诊断变体。
5. 真值/来源标签仅评估使用；在线 tracker 不可知测量是否为杂波，未匹配测量一律按 `P_B/P_FA` 先验处理。
6. 未改动 Phase 2 runner/evaluator 与所有既有产物；未实现 CTDA。

## 9. 验证记录

- 单测：`python -m unittest discover -s tests -v` → **36 项通过**（含 Phase 3 15 项：滤波器数值稳定性、候选/确认/丢失/删除生命周期、同 ID 恢复、clutter 抑制、标签隔离、可重复性、产物）。
- 确定性：对三个 tracker 变体在 S1 与 S6（baseline 分布）各运行两次，`(metrics, recovery opportunities/successes)` 逐位一致；`test_metrics_repeatable_runtime_finite` 覆盖 S1 重复运行。runtime 均有限且 ≥0（不参与等值判定）。
- 完整性：`phase1_manifest.json`(3)、`phase1_5_manifest.json`(4)、`phase2_manifest.json`(16) 的文件 SHA-256 全部匹配；baseline 变体结果与 Phase 2 逐位一致，说明未改动的部分保持稳定。
- 产物：48 份指标 JSON、48 张轨迹图、`phase3_manifest.json` 均已生成；无消失窗口的场景 `recovery_rate` 记为 `null`（JSON 标准），不再使用 NaN。

## 10. 下一步建议

1. 与作者核对漏检增量符号；若确认为笔误，把 `likelihood_penalty` 提升为默认并重跑矩阵。
2. 增加关联预筛选（距离/Mahalanobis 快速拒绝）以降低高杂波 runtime。
3. 以本矩阵为基线接入 CTDA `AssociationModel.associate(...)`，复用同一 runner/evaluator 对比。
4. 将分数/生命周期真正迁到论文 score manager，并把所有超参配置文件化（已在本 Phase 落地）。
