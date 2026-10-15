# 首轮数据执行结果

**数据已落盘并通过读回检查。当前是依据论文表 2 与显式假设生成的复现候选，不是作者原始数据。** 具体选择见 [简短计划](../../notes/reproduction_plan.md)。

- 主数据：2000 多目标场景，每场景 101 帧；共 200,000 个相邻帧关联对，两种噪声解释共享真值和标签。
- 实际单目标轨迹共 26,292 条；这是将论文未明确的“2000 trajectories”解释为多目标场景的结果，不等于已取得作者同规模轨迹集。
- 场景划分：训练池 1800（fit 1620、validation 180），封存 test 200。
- 额外交叉诊断：100 场景，单列保存，不混入主训练或测试。
- 主数据正关联边 2,129,273，全零标签对 1,891；目标数 1–25，均值 13.146。
- 实测漏检率 0.10012；杂波均值 19.9753/帧。
- 距离噪声标准差 1000.646 m；XYZ 噪声标准差 [999.356, 1000.233, 999.99] m。
- 式 (1) 控制换算最大加速度误差 8.88e-15 m/s²；坐标往返最大误差 5.82e-11 m。
- 有限值、运动、坐标转换、真实 ID、标签一对一、划分无重叠检查通过；主数据全部 2000 场景从文件读回检查。
- 场景文件总大小约 1003.9 MiB；另有 manifest、统计、示例 40×40 块和图。

## 数据生成时的距离基线（当时未评估封存测试集）

固定门限 10000 m，允许未匹配的一对一距离分配。Accuracy 按论文式 (35)；F1 只统计真实正关联。

| 评价集 | 噪声版本 | 配对数 | 论文 Accuracy | 正关联 F1 |
|---|---|---:|---:|---:|
| validation | range_only | 1800 | 99.89% | 99.84% |
| validation | cartesian_xyz | 1800 | 99.90% | 99.85% |
| crossing_diagnostic | range_only | 2100 | 27.52% | 33.11% |
| crossing_diagnostic | cartesian_xyz | 2100 | 20.91% | 22.68% |

主验证集每 10 帧取一对；交叉诊断取 40–60 s。它们使用不同空间条件，诊断用于展示难度敏感性，不能当作论文测试结果。若主验证距离基线已接近或超过 98.33%，应先核查仿真假设，不能据模型高分宣称复现成功。

全零预测在主验证集的论文 Accuracy 为 69.09%，正关联 F1 为 0；这进一步说明必须同时查看正关联指标。当前主候选的距离基线约 99.9%，后续应先核对相邻帧配对、目标间距和杂波区域的假设，保留固定验证与封存测试，不通过调数据迎合论文数值。

## 文件读取

每个 NPZ 保存 `truth_state`（帧×目标×6）、速度、加速度、控制、极坐标真值、漏检掩码、杂波数量，以及扁平观测。

`observation_offsets[k:k+2]` 指定第 k 帧观测；`observation_target_id=-1` 表示杂波，其余是场景内目标 ID。`positive_edges` 与 `positive_edge_offsets` 给出每对的非零标签坐标，其余标签为零。

两种输入分别为 `observation_range_only_xyz`、`observation_cartesian_xyz`。`experiments/generate_data.py` 中的 `pair()` 可还原某个关联对；`rdf_blocks()` 按式 (6)–(10) 得到网络输入。`validation_example_blocks.npz` 是至多 128 块可直接读取的示例，包含两种输入、标签、有效区域和原始位置索引；完整原始数据可在后续训练时按需变换。

场景列表/划分/逐文件 SHA256：[manifest.json](manifest.json)；完整统计：[audit.json](audit.json)；实际参数：[config.json](config.json)；示意图：[trajectory_examples.png](trajectory_examples.png)。

重新生成（同配置确定性）：

```bash
/home/chenrui/miniconda3/envs/mttrack-repro/bin/python experiments/generate_data.py --output data/ctda_3d_v1
```

生成脚本拒绝覆盖已有 manifest，避免无意改动固定数据。需重生成时使用新的输出目录。后续数据难度核查及 CTDA 完整训练／测试已完成，XYZ 版仍为解释敏感性候选；当前实验进展见[复现计划](../../notes/reproduction_plan.md)，组会材料见[文件计划表](../../notes/group_meeting_file_plan.md)。
