# CTDA 退化诊断：小样本配对对照

从 `results/ctda_repaired_run/final.pt` 的第 8 轮同一权重开始，每组训练 400 步、相同批次顺序。32 个 fit 场景共 640 对，32 个独立验证场景共 320 对，未使用测试集选择修改。

| 对照 | 训练目标 | 补全通道残差 | Accuracy | 正关联 Precision | 正关联 Recall | 正关联 F1 |
|---|---|---|---:|---:|---:|---:|
| E_residual_masked | masked_focal | True | 67.68% | 48.73% | 92.85% | 63.91% |
| F_residual_balanced_structure | balanced_structured | True | 63.54% | 44.73% | 89.73% | 59.70% |

- A→B：只去掉 padding 对 loss 的贡献。
- B→C：进一步分别按正负类数量归一化 Focal Loss。
- C→D：加入整张关联矩阵的行/列分类（带未匹配类）和监督的匹配数量惩罚。
- B→E、D→F：单独检查式 (17) 未写出的通道 decoder 残差。
- 结构约束在还原后的完整关联矩阵计算，跨分块的一行/一列不会被当成独立目标。

单纯要求行列和≤1，仍允许全零矩阵；这里的行/列分类及匹配数量惩罚还要求恢复真值中存在的匹配。padding 不产生分类梯度，检查见 [协议](protocol.json)。新增损失超出论文式 (19)，结果属于修正诊断，不可冒充原论文训练目标的严格复现。实际数字见 [结果](results.json)。
