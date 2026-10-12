# Phase 3.1：score 方向审计报告

本报告面向后续接手的 AI / 工程师，审计 Phase 3 动态评分 tracker 的**分数符号、生命周期与阈值**，并在**完全相同的 Phase 2 S0–S7 矩阵（两分布）**上对比两个仅"漏检增量方向"不同的变体 A / B。所有新产物写入 `phase3_1` 命名空间，未改动任何 Phase 0–3 代码/配置/结果。

**核心结论（有原文证据）**：论文 Eq. (30) 明确给出漏检增量 $\Delta S_l(k)=\ln\frac{1-P_{FA}}{1-P_D}$。在配置的 $P_D=0.90>P_{FA}=0.02$ 下该值为 **+2.282（正）**。因此**变体 B（文档化公式）是论文忠实实现（source-confirmed）**；**变体 A（取负）不是论文公式**，仅作为诊断对照，用于量化"正增量"带来的实际后果。

## 1. 交付物（全部新增，未改动既有文件）

| 文件 | 作用 |
|---|---|
| `configs/phase3_1_score_audit.yaml` | A/B 变体定义、共享 tracker 参数、论文引用与假设 |
| `src/experiments/phase3_1_score_direction_audit.py` | 隔离运行器：对 A、B 跑同一 16 用例矩阵，强制并记录 parity |
| `src/evaluation/phase3_1_audit.py` | 离线关联结果诊断（真值 vs 杂波关联、候选来源） |
| `tests/test_phase3_1_audit.py` | 12 项审计测试（符号方向、parity、诊断、隔离、确定性） |
| `data/results/phase3_1/<variant>/<distribution>/<scenario>/<scenario>_metrics.json` | 32 份逐用例指标 |
| `data/manifests/phase3_1_manifest.json` | 汇总 manifest（含 parity 与逐用例输入哈希） |
| `outputs/figures/phase3_1/<variant>/<distribution>/<scenario>_trajectory.png` | 32 张轨迹图 |

复用（未修改）：`src/tracking/dynamic_score_tracker.py`（其 `missed_increment_mode` 已支持两种方向）、`src/evaluation/phase2_metrics.py`、`src/evaluation/phase3_metrics.py`、`src/simulation/scenarios.py`、`phase2_baseline_evaluation.plot_scenario`、`phase3_dynamic_score_evaluation._dynamic_config`。

## 2. 证据表：论文方程 vs 实现

论文原文位于工作区：`references/papers/Zhang 等 - 2026 - A robust multiple target tracking framework with Transformer-based data association and dynamic traj.pdf`（Aerospace Science and Technology 168 (2026) 111007），已用 `pdftotext` 提取核对。

| # | 论文出处 | 论文内容 | 实现 | 判定 |
|---|---|---|---|---|
| 1 | Eq. (26), p. 8 | $S_l(k)=\lambda_s S_l(k-1)+\Delta S_l(k)-(\lambda_s)^{\lambda_t}\Delta S_l(k-\lambda_t)$ | `_apply_score`：`decay*S + inc - decay**lag * inc(k-lag)` | **confirmed** |
| 2 | Eq. (27), p. 8 | 新候选：$\Delta S=\ln(P_B/P_{FA})$ | `birth_increment = ln(p_birth/p_false_alarm)` | **confirmed** |
| 3 | Eq. (30), p. 8 | 漏检：$\Delta S=\ln\frac{1-P_{FA}}{1-P_D}$ | `missed_increment_mode="documented"`（变体 B） | **confirmed（B 正确）** |
| 4 | Eq. (30) 数值 | $P_D=0.9,P_{FA}=0.02 \Rightarrow \ln\frac{0.98}{0.1}=+2.282$ | `computed_missed_increment_signs` | **confirmed**（B 为正） |
| 5 | Eq. (31), p. 8 | 成功关联：$\Delta S=\ln pr(\hat{x}(k|k)|\hat{x}(k-1|k-1))-\ln P_{FA}$ | `log_likelihood - ln(p_false_alarm)` | **consistent-with-spec** |
| 6 | Eq. (34), p. 8 | 闭式：$\Delta S=\ln\frac{D}{2\pi\lambda_c}-\ln|\Sigma_l^k|-\frac12 d_l^2$ | 实现用标准高斯对数密度 $\ln pr$ | **unresolved**：闭式中 $|\Sigma|$ 与 $1/2$ 的排版归属存在歧义（标准高斯应为 $-\frac12\ln|\Sigma|$）。本 Phase 不据此改动，保留 Eq. (31) 形式 |
| 7 | Section 4.4, p. 8 | 候选：$S\ge S_c$ 转确认；$0\le S<S_c$ 保留；$S<0$ 删除。维持：$S\ge0$ 继续；$S<0$ 终止并退休标签 | 候选/维持规则一致；另有工程扩展（见 #8） | **confirmed**（主体一致） |
| 8 | Algorithm 2, p. 9 | line 21-27：对 $\mathbb{L}_m\cup\mathbb{L}_c$ 计算 $S$，$S<0$ 即删除；line 28-31：$S>S_c$ 且候选 → 维持；line 33-35：仅输出 $\mathbb{L}_m$ | 输出仅 confirmed（`report_lost=false`）；但实现增加 `lost`（可恢复）状态与 `max_missed`/`lost_expiry` 上限 | **consistent-with-spec + 显式工程扩展**：论文在 $S<0$ 时立即终止，无"丢失待恢复"；扩展已在 Phase 3 报告中标注 |
| 9 | §4.3, p. 8 | "Assume $S_l(k)=0$ for $k\le t_b^l$"，出生帧施加 Eq. (27) | `_new_candidate` 在创建帧施加 birth increment | **confirmed** |

符号说明（论文 §4.3）：$P_B$ 目标出生概率；$P_{FA}$ 杂波点无法与真实目标区分（虚警率）；$P_D$ 传感器检测概率。论文 Remark 1 明确 $P_A,P_B,P_{FA}$ 属**场景相关**、可在合理范围内调整；本 Phase 采用的数值仍按假设标注。

## 3. A / B 定义与 parity

- **A（`A_missed_decay`）**：`likelihood_penalty`，$\Delta S=\ln\frac{1-P_D}{1-P_{FA}}=-2.282$（$P_D>P_{FA}$ 时为负）。
- **B（`B_documented`）**：`documented`，$\Delta S=\ln\frac{1-P_{FA}}{1-P_D}=+2.282$（论文 Eq. (30)）。

**唯一差异**就是 `missed_increment_mode`。运行器强制并记录 parity：

```
missed-increment sign values: {'A_missed_decay': -2.2823823856765264, 'B_documented': 2.2823823856765264}
parity ok: True   mismatches: []
```

parity 对每个 (distribution, scenario) 比较：生成输入哈希（truth / measurements / labels）、resolved scenario 配置、seed、`match_gate`、除符号外的 tracker 配置与分数参数。32 个用例的输入哈希全部一致（`data/manifests/phase3_1_manifest.json` 的 `parity` 与逐用例 `input_hashes`）。

> 说明：A、B 由同一 `generate_scenario(spec)`（同 seed）生成，输入逐字节相同；tracker 只接收坐标（`Detection(position)`），真值来源标签仅用于离线评估/诊断。

## 4. 运行与产物

```bash
conda run -n mttrack-repro python -m src.experiments.phase3_1_score_direction_audit                 # 全矩阵 A/B
conda run -n mttrack-repro python -m src.experiments.phase3_1_score_direction_audit --scenarios S0,S6 --distributions paper_assumption_v1
conda run -n mttrack-repro python -m unittest discover -s tests -v
```

## 5. 结果

### 5.1 变体 A（漏检衰减，非论文公式）

| 分布 | 场景 | match | FP | FN | IDS | frag | RMSE | MOTA | recov | clut_assoc | clut_birth |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | S0 | 79 | 0 | 1 | 0 | 0 | 0.107 | 0.988 | n/a | 0 | 0 |
| baseline | S1 | 305 | 12 | 95 | 11 | 20 | 1.340 | 0.705 | n/a | 6 | 107 |
| baseline | S2 | 270 | 21 | 50 | 15 | 16 | 1.862 | 0.731 | n/a | 7 | 117 |
| baseline | S3 | 110 | 7 | 130 | 5 | 20 | 2.858 | 0.408 | n/a | 7 | 115 |
| baseline | S4 | 216 | 117 | 24 | 9 | 18 | 2.000 | 0.375 | n/a | 62 | 319 |
| baseline | S5 | 257 | 9 | 78 | 4 | 19 | 1.218 | 0.728 | n/a | 6 | 122 |
| baseline | S6 | 54 | 26 | 106 | 1 | 4 | 2.512 | 0.169 | 1.00 | 13 | 118 |
| baseline | S7 | 1772 | 1011 | 228 | 420 | 74 | 2.750 | 0.170 | n/a | 661 | 888 |
| paper | S0 | 79 | 0 | 1 | 0 | 0 | 0.107 | 0.988 | n/a | 0 | 0 |
| paper | S1 | 285 | 660 | 115 | 53 | 15 | 3.329 | −1.070 | n/a | 520 | 1098 |
| paper | S2 | 262 | 798 | 58 | 36 | 17 | 3.276 | −1.788 | n/a | 562 | 1032 |
| paper | S3 | 163 | 674 | 77 | 25 | 27 | 5.796 | −2.233 | n/a | 491 | 1050 |
| paper | S4 | 206 | 664 | 34 | 28 | 14 | 2.660 | −2.025 | n/a | 494 | 1099 |
| paper | S5 | 287 | 870 | 48 | 29 | 13 | 2.152 | −1.827 | n/a | 526 | 1062 |
| paper | S6 | 113 | 886 | 47 | 23 | 10 | 2.947 | −4.975 | 0.00 | 554 | 1003 |
| paper | S7 | 1791 | 1212 | 209 | 491 | 94 | 2.867 | 0.044 | n/a | 703 | 851 |

### 5.2 变体 B（文档化公式，论文忠实）

| 分布 | 场景 | match | FP | FN | IDS | frag | RMSE | MOTA | recov | clut_assoc | clut_birth |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | S0 | 79 | 0 | 1 | 0 | 0 | 0.107 | 0.988 | n/a | 0 | 0 |
| baseline | S1 | 360 | 279 | 40 | 37 | 2 | 1.749 | 0.110 | n/a | 38 | 75 |
| baseline | S2 | 298 | 364 | 22 | 35 | 1 | 1.571 | −0.316 | n/a | 44 | 80 |
| baseline | S3 | 222 | 183 | 18 | 5 | 1 | 2.003 | 0.142 | n/a | 36 | 86 |
| baseline | S4 | 235 | 1271 | 5 | 32 | 0 | 2.244 | −4.450 | n/a | 259 | 122 |
| baseline | S5 | 288 | 385 | 47 | 23 | 3 | 1.563 | −0.358 | n/a | 53 | 75 |
| baseline | S6 | 123 | 275 | 37 | 7 | 2 | 2.144 | −0.994 | 1.00 | 44 | 87 |
| baseline | S7 | 1929 | 5674 | 71 | 463 | 4 | 2.358 | −2.104 | n/a | 1353 | 196 |
| paper | S0 | 79 | 0 | 1 | 0 | 0 | 0.107 | 0.988 | n/a | 0 | 0 |
| paper | S1 | 390 | 5377 | 10 | 68 | 1 | 2.169 | −12.637 | n/a | 1397 | 221 |
| paper | S2 | 310 | 5150 | 10 | 64 | 1 | 2.362 | −15.325 | n/a | 1372 | 222 |
| paper | S3 | 231 | 5050 | 9 | 48 | 2 | 2.804 | −20.279 | n/a | 1327 | 214 |
| paper | S4 | 231 | 5226 | 9 | 33 | 0 | 1.883 | −20.950 | n/a | 1378 | 215 |
| paper | S5 | 331 | 5230 | 4 | 58 | 0 | 2.473 | −14.797 | n/a | 1368 | 220 |
| paper | S6 | 153 | 5016 | 7 | 30 | 0 | 2.710 | −30.581 | 1.00 | 1330 | 227 |
| paper | S7 | 1955 | 5851 | 45 | 487 | 4 | 2.369 | −2.192 | n/a | 1355 | 199 |

`clut_assoc` = 关联到杂波测量的关联数；`clut_birth` = 由未匹配杂波测量新建的候选数（离线诊断，真值标签不进入在线 tracker）。

### 5.3 配对 A vs B

| 分布 | 场景 | A FP | B FP | A FN | B FN | A IDS | B IDS | A frag | B frag | A RMSE | B RMSE | A MOTA | B MOTA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | S0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0.107 | 0.107 | 0.988 | 0.988 |
| baseline | S1 | 12 | 279 | 95 | 40 | 11 | 37 | 20 | 2 | 1.340 | 1.749 | 0.705 | 0.110 |
| baseline | S2 | 21 | 364 | 50 | 22 | 15 | 35 | 16 | 1 | 1.862 | 1.571 | 0.731 | −0.316 |
| baseline | S3 | 7 | 183 | 130 | 18 | 5 | 5 | 20 | 1 | 2.858 | 2.003 | 0.408 | 0.142 |
| baseline | S4 | 117 | 1271 | 24 | 5 | 9 | 32 | 18 | 0 | 2.000 | 2.244 | 0.375 | −4.450 |
| baseline | S5 | 9 | 385 | 78 | 47 | 4 | 23 | 19 | 3 | 1.218 | 1.563 | 0.728 | −0.358 |
| baseline | S6 | 26 | 275 | 106 | 37 | 1 | 7 | 4 | 2 | 2.512 | 2.144 | 0.169 | −0.994 |
| baseline | S7 | 1011 | 5674 | 228 | 71 | 420 | 463 | 74 | 4 | 2.750 | 2.358 | 0.170 | −2.104 |
| paper | S0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0.107 | 0.107 | 0.988 | 0.988 |
| paper | S1 | 660 | 5377 | 115 | 10 | 53 | 68 | 15 | 1 | 3.329 | 2.169 | −1.070 | −12.637 |
| paper | S2 | 798 | 5150 | 58 | 10 | 36 | 64 | 17 | 1 | 3.276 | 2.362 | −1.788 | −15.325 |
| paper | S3 | 674 | 5050 | 77 | 9 | 25 | 48 | 27 | 2 | 5.796 | 2.804 | −2.233 | −20.279 |
| paper | S4 | 664 | 5226 | 34 | 9 | 28 | 33 | 14 | 0 | 2.660 | 1.883 | −2.025 | −20.950 |
| paper | S5 | 870 | 5230 | 48 | 4 | 29 | 58 | 13 | 0 | 2.152 | 2.473 | −1.827 | −14.797 |
| paper | S6 | 886 | 5016 | 47 | 7 | 23 | 30 | 10 | 0 | 2.947 | 2.710 | −4.975 | −30.581 |
| paper | S7 | 1212 | 5851 | 209 | 45 | 491 | 487 | 94 | 4 | 2.867 | 2.369 | 0.044 | −2.192 |

### 5.4 生命周期诊断（节选）

| 变体 | 分布 | 场景 | confirm | recover | lost | deleted | births | max_conf | final_conf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | baseline | S1 | 6 | 27 | 29 | 114 | 124 | 6 | 4 |
| A | baseline | S7 | 73 | 392 | 419 | 896 | 978 | 47 | 46 |
| B | paper | S1 | 167 | 406 | 493 | 105 | 228 | 94 | 80 |
| B | paper | S7 | 197 | 436 | 518 | 98 | 250 | 121 | 115 |

（完整 32 行见 `data/manifests/phase3_1_manifest.json`。）

### 5.5 结果解读（符号机制）

1. **正增量（B）把杂波"喂养"成轨迹**：B 在 paper S1 的 `clut_assoc`=1397（A 为 520），因为杂波候选每帧获得 $+2.28$ 而快速确认，随后不断"吸收"后续杂波测量 → FP 从 660 升到 5377，MOTA 由 −1.07 崩到 −12.6。
2. **负增量（A）抑制了杂波，但代价是确认更难**：A 的 `clut_birth` 更大（paper S1 1098 vs B 221）——因为杂波候选**永远不确认**，每条杂波测量始终未匹配而不断新建候选；但 A 的 FP 反而小得多（杂波轨迹不确认、不输出）。
3. **B 的匹配数更高、FN 更低**：B 把更多测量（含杂波）纳入已确认轨迹，`match` 更高、FN 更低，RMSE 略好；但这些"匹配"里包含大量杂波关联，MOTA/Frag 恶化。
4. **fragmentation**：A 明显更高（baseline S1 20 vs B 2），因为 A 的轨迹更难确认、更易删除重建。
5. **S6 同 ID 恢复**：B 在两分布均为 1.00；A 在 baseline 为 1.00、paper 为 0.00（高杂波干扰重关联）。
6. **S0**（零噪声零杂波）两变体完全一致（MOTA 0.988），说明差异**只**来自漏检增量方向，符合 parity 设计。

## 6. 验收判定

| 检查项 | 结果 |
|---|---|
| 证据表区分 confirmed / consistent / unresolved | ✅ 第 2 节（含论文 Eq. 编号与页码） |
| 论文原文可检索并用于判定符号 | ✅ 工作区 PDF 经 `pdftotext` 提取，Eq. (30) 明确 |
| 16 用例 × A/B 在新 `phase3_1` 路径产出 | ✅ 32/32 |
| 除符号外输入/配置/seed/评估设置完全一致 | ✅ `parity ok: True, mismatches: []` |
| 每用例指标有限且轨迹图非空 | ✅ 32 份 JSON + 32 张 PNG |
| 指标含 FP/FN/IDS/frag/RMSE/MOTA + 生命周期/恢复 | ✅ |
| 重复运行确定性字段一致 | ✅（A、B 各跑两次，`metrics`+`audit`+`input_hashes` 相等） |
| A/B 增量方向与标签隔离测试 | ✅ `test_phase3_1_audit.py` 12 项通过 |
| 既有 Phase 1/1.5/2/3 产物完整 | ✅ manifest SHA-256：3 + 4 + 16 + 48 全匹配 |
| 单测总数 | ✅ 48 项通过（Phase 1.5 5 + Phase 2 16 + Phase 3 15 + Phase 3.1 12） |

## 7. 结论与建议

1. **B 是论文忠实实现**：Eq. (30) 经原文确认就是 $\ln\frac{1-P_{FA}}{1-P_D}$，在 $P_D>P_{FA}$ 时为正。**不应**把 B 当作"笔误"；A 只是诊断对照。
2. **正增量在高杂波下会显著劣化跟踪**（FP 爆炸、MOTA 严重负），这是**论文评分机制在该场景的真实行为**，属质量限制而非实现缺陷。
3. **建议（符合论文原意地缓解）**：论文 Remark 1 明确 $P_{FA},P_B,P_D$ 为**场景相关**参数，且"近似足够"。若要抑制杂波，应**调整 $P_{FA}$/$\lambda_s$**（例如提高 $P_{FA}$ 使 Eq. (30) 的正增量变小、甚至变负），而**不是**把 Eq. (30) 取负。可在后续 phase 中以配置扫描的方式验证。
4. **不建议**把 A 设为默认：它脱离论文公式，且带来更高的 fragmentation 与更低匹配数。
5. 保持 Phase 3 结论：`lost` 状态、`max_missed`、`lost_expiry`、`lost_gate_scale`、以及仅输出 confirmed，都是**工程扩展**；论文在 $S<0$ 时直接终止且只输出 $\mathbb{L}_m$。

## 8. 遗留歧义（保留 A/B，不主张确定性修正）

1. **Eq. (34) 闭式**中 $|\Sigma_l^k|$ 与 $1/2$ 的排版归属存在歧义（标准高斯应为 $-\frac12\ln|\Sigma|$）。实现保留 Eq. (31) 的高斯对数似然形式，未按 (34) 改动。
2. **$P_{FA}$ 与 $\lambda_c$ 的关系**：(31) 用 $-\ln P_{FA}$，(34) 出现 $\ln\frac{D}{2\pi\lambda_c}$。两者关系未在原文显式给出，未强行统一。
3. **数值超参**：$\lambda_s,\lambda_t,S_c,P_B,P_{FA},P_D$ 论文未给出（称场景相关/经验取值），本 Phase 数值仍为假设。
4. **生命周期工程扩展**（lost 状态、missed-frame 上限、重关联门限放大）非论文内容，仅为支持同 ID 恢复。

> 依据验收准则：由于论文原文**已确认** Eq. (30) 的写法，本报告**明确**判定 B 为论文公式、A 为诊断对照；仅在 Eq. (34) 闭式与 $P_{FA}/\lambda_c$ 关系上保留歧义。

## 9. 验证记录

- 单测：`python -m unittest discover -s tests` → **48 项通过**。
- `phase3_1` 确定性：A、B 在 S1(baseline) 各运行两次，`metrics`/`audit`/`input_hashes` 相等；runtime 有限非负（不参与等值）。
- parity：32 用例输入哈希/配置/seed/评估设置一致，`mismatches: []`。
- 完整性：`phase1(3) + phase1_5(4) + phase2(16) + phase3(48)` 全部 SHA-256 匹配，未被覆盖。
- 符号：`A_missed_decay = −2.2824`，`B_documented = +2.2824`（由配置的 $P_D,P_{FA}$ 计算）。
