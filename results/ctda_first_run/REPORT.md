# 首轮完整 CTDA 实验

**仿真数据 → RDF 分块 → CTDA 训练 → 测试关联 → 权重保存已全部完成。** 使用现有 v1 距离噪声数据，单个训练种子，8 epochs。

**当前模型发生未匹配退化，尚未达到论文效果。** 重新加载第 8 轮权重后检查全部 2494 个验证块，有效区域最高概率为 0.1681，均低于固定阈值 0.2；正、负关联平均概率分别为 0.144614、0.144654，基本没有分离。Accuracy 上升到约 69% 主要来自正确判定全零行/列，不能视为学会了目标关联。

| 指标 | 训练前验证 | 训练后验证 | 最终测试 |
|---|---:|---:|---:|
| 论文 Accuracy | 7.37% | 69.09% | 69.43% |
| 正关联 Precision | 0.98% | 0.00% | 0.00% |
| 正关联 Recall | 2.70% | 0.00% | 0.00% |
| 正关联 F1 | 1.44% | 0.00% | 0.00% |
| Focal Loss（含 padding 元素均值） | 0.124423 | 0.005820 | 0.005948 |

训练 1620 场景、162,000 对、222,133 块；验证 180 场景、1,800 对；测试 200 场景、20,000 对。最终结果使用固定第 8 轮权重；测试不参与阈值或权重选择。

论文参考 Accuracy 98.33%；本次差值 -28.90 个百分点。当前数据上的距离基线测试 Accuracy 99.90%、正关联 F1 99.85%。数据与模型存在已记录的补全假设，不能仅凭结果数值声称严格复现。Loss 0.0067 的原归约方式未报告，本次 loss 口径已经明确。

训练与评价用时 9.81 分钟（不含 RDF 缓存准备）。[配置](run_config.json)、[数值结果](summary.json)、[训练曲线](training_curve.png)、[逐轮日志](history.csv)、[最终权重](final.pt)、[最低验证 loss 权重](best_validation.pt)、[关联输出示例](test_association_0.npz)。

最终权重已在 CPU 重新加载，并从 3 个原始测试配对重新计算 RDF 和推理；概率与 GPU 保存结果的最大差值为 8.94e-8，匹配输出一致。[重新加载检查](reload_check.json)、[原始数据推理示例](reloaded_inference.npz)。

后续首先在现有数据上排查正类学习、padding/loss 归约和交叉注意力实现；首轮最小目标已经完成，不再把数据难度校准作为训练的前置条件。

重跑命令：

```bash
MPLCONFIGDIR=/tmp/mttracking-mpl /home/chenrui/miniconda3/envs/mttrack-repro/bin/python experiments/train_ctda.py --device cuda:4 --output results/ctda_first_run_repeat
```
