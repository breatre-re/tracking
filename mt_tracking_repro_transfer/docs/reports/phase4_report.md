# Phase 4：CTDA 实现与对比报告

本报告面向后续接手的 AI / 工程师，说明 CTDA（Cross Transformer-based Data Association）的实现、训练协议、关联-only 结果、与 baseline / Phase 3 动态评分 tracker 的对比，以及仍然存在的论文歧义。所有代码/配置/模型/结果均写在 **additive Phase 4 路径**，未修改任何 Phase 0–3.1 产物。

**原始执行结论（下文旧结果）：** Phase 4 管线已完成，但 `row_column_3ch` 训练失败，旧 tracker 对比使用 `row_column` 回退权重。2026-10-12 的修复与完整重跑结果见文末“后续调试复跑”；它们补充并取代旧结论中“尚未重跑”的状态。

**首次调试发现（2026-10-12）：** 对 3ch 融合路径复查发现 `channel_outputs` 转置为 `(batch, columns, rows)` 后，编码器与 cross-attention 仍使用 `row_mask`，把候选轨迹行 mask 错用于测量列 token，并可能使全 padding 序列进入全屏蔽注意力 softmax。代码已改为安全处理后的 `col_mask`。完整 GPU 重跑结果见文末。

- 依赖前置：Phase 3.1 score 方向审计已完成。**B（`documented`）作为论文忠实的动态评分版本**（Eq. (30) 已经原文确认），A 仅保留为诊断。
- 训练协议（已确认）：仅在 `paper_assumption_v1/train`（1,800 序列）训练；在其不重叠的 `test`（200 序列）做关联-only 评估；**不在 `baseline_distribution` 上训练**（该分布保留作 S0–S7 跨分布泛化）。
- 真值 ID 仅用于构造监督矩阵/评估，绝不进入模型输入或在线 tracker 特征。

## 1. 交付物（全部新增）

| 文件 | 作用 |
|---|---|
| `configs/phase4_ctda.yaml` | 数据集/特征/模型/训练/推理/评分配置与假设 |
| `src/models/ctda/features.py` | RDF 特征、padding/mask、关联样本构建、序列级切分 |
| `src/models/ctda/model.py` | 三个架构：`standard_transformer` / `row_column` / `row_column_3ch` |
| `src/models/ctda/objective.py` | Focal loss（Eq. 19）、阈值+KM 解码（Eq. 20 / Alg. 1）、关联指标 |
| `src/models/ctda/adapter.py` | `CTDAAssociation`（兼容既有 `AssociationModel` 契约） |
| `src/models/ctda/tracker.py` | `CTDADynamicScoreTracker`（仅替换关联步，复用 Phase 3 BLUE + B 评分与生命周期） |
| `src/training/train_ctda.py` | 训练入口（8 epochs / batch 256 / Adam / lr 1e-4） |
| `src/training/evaluate_association.py` | 关联-only 评估（accuracy / P / R / F1 / 密度分桶） |
| `src/experiments/phase4_ctda_evaluation.py` | 未改动 Phase 2 矩阵 × 3 tracker 对比运行器 |
| `tests/test_phase4_features.py`、`tests/test_phase4_model.py`、`tests/test_phase4_adapter.py` | 23 项 Phase 4 测试 |
| `models/checkpoints/phase4/legacy/<variant>/training_summary.json`；有效权重仅见 `standard_transformer`、`row_column` | 三个变体的训练摘要；`row_column_3ch` 未产出 checkpoint |
| `data/results/phase4/legacy/association/{standard_transformer,row_column}_test.json` | 两个有效训练变体的关联-only 测试结果；无 `row_column_3ch` 结果 |
| `data/results/phase4/legacy/tracker/...`、`data/manifests/phase4_manifest.json` | tracker 对比指标与汇总 manifest |
| `outputs/figures/phase4/legacy/...` | 48 张轨迹图（每个 tracker 用例一张） |
| `docs/reports/phase4_report.md`、`docs/reports/reproduction_assumptions.md` | 本报告与假设文档 |

## 2. 论文证据表（详见 `docs/reports/reproduction_assumptions.md`）

| 项 | 论文出处 | 判定 |
|---|---|---|
| RDF 矩阵（每轴一个通道），零填充到标准尺寸 | Section 3.1–3.2, Eq. (6)–(10), p. 4 | **confirmed** |
| Row-column cross Transformer：`h1e=ColEncoder(C)`、`h2e=RowEncoder(C^T)^T` | Section 3.3.1, Eq. (11) | **confirmed** |
| 双 MHA 交叉馈送解码器；element-wise / fully-connected 混合 | Eq. (12)–(15) | **confirmed** |
| 3-channel cross Transformer（跨通道 query + `W_{3×1}` 融合 + sigmoid） | Section 3.3.2, Eq. (16)–(18) | **confirmed** |
| 去除位置编码 | Section 3.3, p. 5 | **confirmed** |
| Focal loss，`alpha_t=0.75`，聚焦参数 `2` | Eq. (19) | **confirmed** |
| 阈值 `lambda_a` + Kuhn-Munkres 一对一 | Section 3.4, Eq. (20), Alg. 1 | **confirmed** |
| padded 维度 40、4 heads、dropout 0.3；8 epochs / batch 256 / Adam / lr 1e-4 | Section 5.1.2 / Table 3 | **confirmed** |
| 2D 关联阈值 `lambda_a = 0.1` | Section 5.2 | **confirmed** |
| RDF 元素式与"±1 表示相似"的文字描述**相互矛盾** | Eq. (6)–(7) 与其后文字 | **unresolved**（按公式实现） |
| Eq. (34) 闭式中 `\|Σ\|`、`1/2` 的归属；`P_FA` 与 `λ_c` 关系 | Eq. (31) vs (34) | **unresolved**（保留 Eq. (31) 高斯形式；两者分别配置，不等同） |

## 3. 数据 / 张量契约

- **行（候选轨迹）** = 前一帧的**真实检测**（合成候选历史表示；打包数据无轨迹历史）。**列（测量）** = 当前帧全部测量（含杂波）。
- **标签** `target[i,j]=1` 当且仅当两检测携带相同来源 ID；来源 ID 只用于构造监督矩阵（`test_labels_do_not_enter_features` 断言特征仅依赖位置）。
- 通道：2D 数据用 `x_rdf`、`y_rdf` + 一个无符号 `range_rdf`（论文第 3 通道为 z，2D 下的替代为假设）。
- Padding 到 40×40；有效行/列掩码；模型内部再把填充单元置零，保证填充不影响有效区域（测试断言）。
- 空帧 / 全填充 / 漏检 / 杂波 / 未匹配均有单元测试覆盖。
- 序列级切分（train 内部 10% 作验证用于**选 checkpoint**），test 仅用于最终报告。

## 4. 三个架构（共享 d_model=40、4 heads、dropout 0.3、pad=40）

1. `standard_transformer`：单个标准 Transformer 编码器处理 `2P` 个行/列 token + 点积关联头（无 row-column 交叉结构）。
2. `row_column`：按通道做 row-column cross Transformer（Eq. 11–14，element-wise 混合）。
3. `row_column_3ch`：在 (2) 基础上加 3-channel 跨通道融合（Eq. 16–18）。

## 5. 训练协议

- 8 epochs、batch 256、Adam、lr `1e-4`、focal（α=0.75, γ=2）、固定种子；`cudnn.deterministic=True`。
- 仅用 `paper_assumption_v1/train`；验证集为 train 内部 10% 序列（不触碰 test）。
- 保存 checkpoint、resolved config、逐 epoch 指标、训练集 SHA-256。

## 6. 结果

> **执行环境说明**：训练与评估在共享服务器上执行，机器被其他用户任务严重占用（load ≈ 11，多个 100% CPU 任务），因此训练/评估吞吐远低于预期。为在可用预算内完成，做了一处**显式工程适配**：训练时每个序列使用前 **20 帧**（仍为同 1,800 条 `paper_assumption_v1/train` 序列、同样 8 epochs / batch 256 / Adam / lr 1e-4 / 固定种子）；关联-only 评估在 test 划分的**前 1,500 个关联样本**上统计。这两点均在此明确标注，不视为论文设定。

### 6.1 训练（三变体，同一设置）

| 变体 | epochs | batch | lr | 种子 | 训练序列 | 每序列帧数 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| standard_transformer | 8 | 256 | 1e-4 | 20261012 | 1,620 (train 内部) | 20 | 训练摘要见 `models/checkpoints/phase4/legacy/<variant>/training_summary.json` |
| row_column | 8 | 256 | 1e-4 | 20261012 | 1,620 | 20 | 同上 |
| row_column_3ch | 8（运行完成） | 256 | 1e-4 | 20261012 | 1,620 | 20 | 损失非有限；`best_epoch=-1`、`best_validation_loss=inf`；无可用 checkpoint |

三个变体均有 `training_summary.json`（含逐 epoch 损失与训练集 SHA-256）。只有 `standard_transformer` 和 `row_column` 产出可用的 `checkpoint.pt`，并按 train 内部 10% 序列验证损失选点；`row_column_3ch` 没有有效权重。

### 6.2 关联-only（`paper_assumption_v1/test` 前 1,500 样本，λ_a = 0.1 + KM 一对一）

| 变体 | accuracy | precision | recall | F1 | TP | FP | FN | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| standard_transformer | 0.9407 | 0.0747 | 0.0836 | 0.0789 | 795 | 9,849 | 8,718 | 完成 |
| row_column | 0.9428 | 0.1051 | 0.1176 | 0.1110 | 1,119 | 9,525 | 8,394 | 完成 |
| row_column_3ch | — | — | — | — | — | — | — | **训练不稳定（NaN），未产出可用权重** |

观察：
- 论文式 **accuracy（Eq. 35）在极端类不平衡下意义有限**（正例占比 <0.4%，全零预测即可得 ~0.94），因此同时报告 precision/recall/F1。
- `row_column` 的 F1 明显优于 `standard_transformer`（0.111 vs 0.079），方向与论文"row-column 交叉结构更优"一致。
- **`row_column_3ch` 训练出现 NaN**（`training_summary.json` 中 `best_epoch=-1`、`best_validation_loss=inf`），未产出可用 checkpoint。现有证据不足以确认 NaN 的根因，不将其归因于某个特定通道。端到端 CTDA 对比因此使用回退权重 `models/checkpoints/phase4/legacy/row_column/checkpoint.pt`；manifest 的 `checkpoint` 和 SHA-256 记录了实际使用的权重。不能把端到端结果称为 `row_column_3ch` 的结果。

### 6.3 tracker 对比（未改动的 Phase 2 S0–S7 × 两分布，3 tracker × 16 = 48 用例，全部完成）

`data/manifests/phase4_manifest.json`：`parity.ok = true`（16 组输入哈希一致），`case_count = 48`。均值如下（B 分布与 paper 分布各 8 场景）：

| tracker | 用例 | 匹配数 | FP | FN | ID switch | fragmentation | 平均位置 RMSE | MOTA-like | 平均运行时间 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gated_greedy | 16 | 469.1 | 2,064.6 | 2.8 | 78.4 | 1.1 | 1.867 (16/16) | −5.354 | 0.037 |
| dynamic_score_B | 16 | 450.9 | 2,833.2 | 21.0 | 86.9 | 1.3 | 1.914 (16/16) | −7.672 | 2.585 |
| ctda_dynamic_score（回退 row_column 权重） | 16 | 87.2 | 52.4 | 384.7 | 28.9 | 27.2 | 5.679 (14/16) | +0.069 | 1.061 |

表中计数和运行时间为 manifest 16 个场景用例的算术平均；RMSE 仅对有限值求均值。CTDA+B 的另 2 个用例 RMSE 为 NaN（不是有效 RMSE）。平均匹配总量分别为 gated greedy **7,506**、dynamic-score B **7,214**、CTDA+B **1,395**。

解读：
- 平均 MOTA 被高杂波 paper 场景主导（FP 达数千），因此三者均值都为负或接近 0；**逐用例 JSON 在 `data/results/phase4/legacy/tracker/<tracker>/<distribution>/<scenario>/`**。
- CTDA+B 的匹配数只有 gated greedy 的约 **18.6%**，FN 和 fragmentation 较高、位置 RMSE 也更差。FP 降低与漏关联严重同时出现，不能将较低 FP 或正 MOTA-like 单独解释为跟踪质量改善。这是**质量失败**，不是基础设施通过。
- 结论：当前 CTDA 关联质量**远未达到可用于跟踪的水平**，与论文的 CTDA 结果不可比（论文用完整 1,800×80 序列训练、未公开的层数与大量调参）。需要更长训练与 `row_column_3ch` NaN 修复后再评估。



## 7. 验收判定

| 检查项 | 结果 |
|---|---|
| 3 个架构均运行了相同的 8-epoch/256/Adam/1e-4 训练协议 | ✅；但 `row_column_3ch` 损失为 NaN 且没有有效 checkpoint，模型训练质量不通过 |
| 测试 ID 不进入张量/统计/checkpoint 选择/调参 | ✅（序列级切分 + 特征仅依赖位置） |
| 输出遵守 padding mask；无非法行列关联；一对一；未匹配显式表示 | ✅（模型/解码测试） |
| test 与指标/配置/训练摘要可读、有哈希、已报告 | ✅（manifest + training_summary + 两个有效变体的 association JSON）；`row_column_3ch` 无结果/checkpoint |
| 固定种子重复运行的确定性字段一致 | ✅（数据/推理确定性；CUDA 严格 bitwise 见假设文档） |
| `CTDAAssociation` 满足既有接口契约，仅消费位置/状态特征 | ✅（`test_adapter_matches_baseline_interface_contract`） |
| tracker 输入不含真值 ID/标签 | ✅（`test_integrates_with_tracker_and_ignores_labels`） |
| 8 场景 × 2 分布对 greedy / dynB / CTDA+B 全部运行，记录相同 seed/场景/评估设置 | ✅（parity 见 manifest） |
| 输出 FP/FN/IDS/frag/RMSE/MOTA/生命周期/恢复/runtime/轨迹图 | ✅ |
| Phase 0–3.1 哈希/结果不变 | ✅（见验证记录） |

质量无强制阈值：CTDA 的跟踪质量按对比结果**如实报告**，不因基础设施通过而宣称质量达标。

## 8. 遗留歧义与风险

1. **RDF 语义矛盾**：公式与文字描述相反，按公式实现（学习模型可自行反转）；未擅自"修正"。
2. **Eq. (34) 归一化歧义**：保留 Eq. (31) 的高斯对数似然，未采用 (34) 闭式。
3. **`P_FA` vs `λ_c`**：分别配置，从不静默等同。
4. **候选历史为合成表示**：行来自上一帧检测，非论文的轨迹历史特征。
5. **2D 三通道**：第三通道为 `range_rdf` 替代论文的 z。
6. **CUDA 严格 bitwise 复现**：受内核非确定性影响，已在假设文档说明。
7. **论文未公开的层数/FFN 宽度**：为假设值。
8. **`row_column_3ch` 质量未达预期**：NaN mask 问题与 Eq. (17) decoder 缺项已在 2026-10-12 修复并完整重训，但 3ch 仍显著落后于 row-column；2D `range_rdf` 替代 z 轴的假设尚未消融。
9. **RMSE 非有限值序列化**：CTDA+B 有 2 个用例的 RMSE 为 NaN；当前产物需按非有限值处理，不能作为有效 RMSE。建议后续统一序列化为 JSON `null` 并加严格 JSON 校验。
10. **runner 默认 checkpoint 路径**：runner 默认 checkpoint 已更新为最终 Eq. (17) 权重；重跑时仍建议显式指定 checkpoint 与新输出目录。

## 9. 验证记录

- 测试：`conda run -n mttrack-repro python -m unittest discover -s tests -v` → **71 项通过**（本次复核，13.046 秒）。
- 复跑关联-only 的可用模型：`conda run -n mttrack-repro python -m src.training.evaluate_association --checkpoint models/checkpoints/phase4/legacy/row_column/checkpoint.pt --output data/results/phase4/legacy/association/row_column_test.json`。
- 复跑 tracker 矩阵：`conda run -n mttrack-repro python -m src.experiments.phase4_ctda_evaluation --checkpoint models/checkpoints/phase4/2026-10-12_eq17/row_column_3ch/checkpoint.pt --output-root data/results/phase4/new_run/tracker --plot-root outputs/figures/phase4/new_run --manifest data/manifests/phase4_new_run_manifest.json`。runner 默认 checkpoint 已指向该 Eq. (17) 权重；复跑时建议显式给出新输出目录。
- 关联输入 parity：同一 (distribution, scenario) 下三 tracker 的输入哈希一致（manifest `parity.ok`）。
- Phase 4 产物计数：48 个 tracker 指标 JSON、48 张 PNG；不是 96 张图。
- manifest：48 个 tracker 用例、16 组场景输入 parity，`parity.ok=true`、`mismatches=[]`；实际 checkpoint 为 `models/checkpoints/phase4/legacy/row_column/checkpoint.pt`。
- 既有产物完整性：交付记录称 Phase 0–3.1 哈希均匹配，且其产物未被覆盖；本次只修订 Phase 4 报告文字，没有重写旧阶段产物。
- 标签隔离：特征张量不依赖 ID；tracker 只接收坐标。

## 10. 下一步建议

1. 用作者补充材料校准 RDF 语义与 Eq. (34)；若确认，替换相应实现并重跑。
2. 引入轨迹历史特征（真实候选历史）替代"上一帧检测"合成表示。
3. 在 CTDA 之上调优评分参数（论文明确 `P_FA/P_B/P_D` 场景相关），而非改动 B 的符号。
4. 扩大 `lambda_a` 与 `pad_size` 的消融；当前仅用论文 2D 设定 `0.1 / 40`。

## 11. 后续调试复跑（2026-10-12）

### 已修复的代码问题

1. **NaN 根因：mask 轴用错。** 3ch 融合将单通道矩阵转置为 `(B, columns, rows)` 后，仍使用 `~row_mask` 作为 key padding mask，误屏蔽有效测量列，并让全空序列进入全屏蔽注意力。改为安全处理后的 `~col_mask`，见 `src/models/ctda/model.py`。
2. **Eq. (17) decoder 路径缺项。** 原实现只用跨通道 query 执行一次 MHA，漏掉通道内 self-attention、该层的残差归一化，以及 cross-attention 后的 FFN。现已补齐论文 Eq. (17) 的 self-attention → ElementMix query/cross-attention → FFN 路径，同文件可查。
3. **非有限值快速失败。** `src/training/train_ctda.py` 现在检查 logits、loss 和 gradient，避免再次静默保存 NaN 训练结果。

前向/反向的小型检查覆盖部分 padding 与空序列，输出和梯度均有限。提权后的主机检查确认 GPU 可见；GPU 3 当时忙碌，本次训练和推理用空闲 GPU 4。

### 同协议完整训练与关联-only 测试

`paper_assumption_v1/train` 1,800 序列按序列拆为 1,620 train / 180 validation，完整 80 帧，8 epochs、batch 256、Adam、lr `1e-4`、seed 20261012。关联测试为独立 200 序列、15,800 帧样本，阈值 0.1、一对一 KM。row-column 与最终 Eq. (17) 3ch 使用相同拆分和训练设置：

| 模型 | 最佳验证 focal loss | Accuracy | Precision | Recall | F1 | 测试耗时 |
|---|---:|---:|---:|---:|---:|---:|
| row_column | 0.01091 | 0.98615 | 0.74219 | 0.83582 | 0.78623 | 124.68 s |
| row_column_3ch（Eq. 17 修复） | 0.02162 | 0.93752 | 0.03358 | 0.03782 | 0.03557 | 179.08 s |

两模型均无 NaN；3ch 性能仍远低于 row-column。当前 2D 数据的第三通道 `range_rdf` 是对论文 z 轴 RDF 的假设替代，可能是主要原因之一，但还没有做通道消融来确认。准确率受大量负样本影响，不能单独作为质量结论。

最终 3ch checkpoint 和训练摘要在 `models/checkpoints/phase4/2026-10-12_eq17/row_column_3ch/`；同协议 row-column 权重在 `models/checkpoints/phase4/2026-10-12_maskfix/row_column/`。全测试集指标在 `data/results/phase4/2026-10-12_eq17/association/`。早期只修 mask、未补 Eq. (17) 的诊断权重保存在 `models/checkpoints/phase4/2026-10-12_maskfix/row_column_3ch/`，不能视为最终模型。

### 同场景端到端 tracker

两个同协议模型分别在相同的 S0–S7 × 两分布 16 用例上运行，两个 manifest 的输入 parity 均为 true：

| 模型 | 匹配数 | FP | FN | ID switch | fragmentation | 有限 RMSE 均值 | MOTA-like | 秒/场景 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| row_column | 175.63 | 146.81 | 296.25 | 51.13 | 39.44 | 5.161 | −0.025 | 0.758 |
| row_column_3ch（Eq. 17 修复） | 75.81 | 43.75 | 396.06 | 35.06 | 24.94 | 6.620 | 0.061 | 0.997 |

3ch 输出更少，FP、ID switch 与 fragmentation 下降，但 FN、RMSE 变差，匹配数不到 row-column 的一半。正 MOTA-like 是 FP 减少带来的，不能据此判定整体质量更好；当前 CTDA 跟踪结果仍不达标。

最终 manifest：`data/manifests/phase4_eq17_3ch_manifest.json`（3ch）与 `data/manifests/phase4_row_column_manifest.json`（row-column）；逐案 JSON 和轨迹图在 `data/results/phase4/2026-10-12_eq17/tracker/`、`outputs/figures/phase4/2026-10-12_eq17/`。旧 Phase 0–4 结果均未覆盖。
