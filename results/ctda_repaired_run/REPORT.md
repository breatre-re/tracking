# 第二轮 CTDA 修正训练

**仿真数据 → RDF 分块 → CTDA 训练 → 测试关联 → 权重保存已全部完成。** 使用现有 v1 距离噪声数据，单个训练种子，8 epochs。

本轮从随机初始化重新训练，修正为有效区域 Focal Loss 和补全通道 decoder 残差，复用第一轮完全相同的 RDF 数据、划分和训练种子。正关联 F1 从第一轮测试的 0 提升到 61.24%，全零退化解除；假匹配仍多，尚未达到论文效果。原版约 69% Accuracy 来自全零行/列的正确判定，不代表其关联能力优于本轮。

| 指标 | 训练前验证 | 训练后验证 | 最终测试 |
|---|---:|---:|---:|
| 论文 Accuracy | 11.41% | 63.68% | 64.02% |
| 正关联 Precision | 5.91% | 45.48% | 45.56% |
| 正关联 Recall | 16.34% | 92.90% | 93.37% |
| 正关联 F1 | 8.68% | 61.06% | 61.24% |
| Focal Loss（有效区域元素均值） | 0.134937 | 0.001584 | 0.001551 |
| Focal Loss（含 padding 元素均值） | 0.125772 | 0.071642 | 0.073441 |

训练 1620 场景、162,000 对、222,133 块；验证 180 场景、1,800 对；测试 200 场景、20,000 对。最终结果使用固定第 8 轮权重；测试不参与阈值或权重选择。

本轮 loss_mode=masked_focal，channel_residuals=True。训练目标保留 α=.75、γ=2 的 Focal Loss，没有新增行列分类或匹配数量惩罚；一对一约束在推理时的全局分配中执行。

残差补全属于对式 (17) 与标准 Transformer decoder 的实现解释，padding 归约原文也未明确；不能将此配置宣称为作者代码已核实的设定。额外约束损失单列在 [诊断对照](../ctda_collapse_diagnosis/REPORT.md)。

论文参考 Accuracy 98.33%；本次差值 -34.31 个百分点。当前数据上的距离基线测试 Accuracy 99.90%、正关联 F1 99.85%。数据与模型存在已记录的补全假设，不能仅凭结果数值声称严格复现。Loss 0.0067 的原归约方式未报告，本次 loss 口径已经明确。

训练与评价用时 10.49 分钟（不含 RDF 缓存准备）。[配置](run_config.json)、[数值结果](summary.json)、[训练曲线](training_curve.png)、[逐轮日志](history.csv)、[最终权重](final.pt)、[最低验证 loss 权重](best_validation.pt)、[关联输出示例](test_association_0.npz)。

最终权重已在 CPU 重新加载，并从三个原始测试配对重新推理；与 GPU 保存的概率最大差值 2.38e-7，最终匹配一致。[重新加载核验](reload_check.json)。

重跑命令：

```bash
MPLCONFIGDIR=/tmp/mttracking-mpl /home/chenrui/miniconda3/envs/mttrack-repro/bin/python experiments/train_ctda.py --device cuda:4 --output results/ctda_repaired_run_repeat --loss-mode masked_focal --channel-residuals
```
