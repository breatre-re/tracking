# Phase 1 2D 数据集报告

## 生成配置

配置文件：`configs/phase1_2d.yaml`

| 参数 | 值 |
|---|---:|
| steps | 80 |
| num_targets | 6 |
| detection_probability | 0.9 |
| measurement_std | 1.5 |
| clutter_rate（每帧泊松均值） | 1.5 |
| process_accel_std | 0.08 |
| area_size | 100.0 |

每条样本使用 `default_rng(seed_start + sample_index)` 独立生成。Smoke seed 为 7–10，训练 seed 为 20261000–20262799，测试 seed 为 20263000–20263199；训练和测试使用不重叠的随机种子范围。目标数、帧数、检测概率、测量噪声、杂波率、过程噪声和 seed 均可通过 YAML 配置调整。

这是论文规模的样本数设置，不代表论文完整数据分布复现。当前轨迹出生/消失规则沿用 baseline 仿真；每条轨迹固定为 6 个目标，相关论文未公开的随机分布不作推断。

## 数据格式和目录

```text
data/
  data/datasets/phase1/
    smoke/smoke_dataset.npz
    train/train_dataset.npz
    test/test_dataset.npz
  manifests/
    phase1_manifest.json
docs/reports/
  phase1_report.md
```

每个 split 保存为一个压缩 NPZ：

- `truth_states`: `[样本数, steps, num_targets, 4]`，每个目标的 `[x, y, vx, vy]`；未出生或已消失的位置为 NaN。
- `measurements`: `[总测量数, 2]`，所有帧展平的二维点。
- `measurement_truth_ids`: `[总测量数]`，真实目标 ID；`-1` 明确表示 clutter/false alarm。
- `frame_offsets`: `[样本数, steps + 1]`，定位每帧测量在展平数组中的边界。
- `birth_events` / `death_events`: `[事件数, 3]`，列为 `[sample_index, target_id, frame]`；death frame 是不再活动的排他边界。
- `missed_detection_events`: `[事件数, 3]`，列为 `[sample_index, frame, target_id]`。
- `clutter_events`: `[事件数, 3]`，列为 `[sample_index, frame, frame_local_measurement_index]`。
- `config_json`、`sample_seeds`、`split`、`schema_version`：完整生成配置及样本随机种子。

`phase1_manifest.json` 保存基础场景配置、各 split 的样本和测量数量、seed 区间、事件数、文件路径及 SHA-256。

## 数据规模与检查结果

| Split | 样本数 | 测量点数 | Birth / death | 漏检事件 | Clutter 事件 | NPZ 大小 |
|---|---:|---:|---:|---:|---:|---:|
| Smoke | 4 | 1,861 | 24 / 24 | 143 | 464 | 81,925 bytes |
| Train | 1,800 | 839,798 | 10,800 / 10,800 | 69,207 | 216,005 | 35,573,019 bytes |
| Test | 200 | 93,371 | 1,200 / 1,200 | 7,624 | 23,995 | 3,955,898 bytes |

- 三个 split 的 shape、帧偏移、标签范围和活动目标来源均验证通过。
- Smoke、train、test 各自样本内容均无重复；train/test 之间重复样本数为 0。
- 每个测量点均具有目标 ID 或 `-1` clutter 标记。
- 对同一配置和 seed 连续生成两份 NPZ，文件字节完全一致。
- loader 返回的 tracker detections 不带 `source_target_id`；baseline 入口再次只传递测量坐标。标签保留在离线数据侧车中，不作为 tracker 输入。
- 新数据读取 baseline 成功：80 帧；FN 1；FP 195；ID switches 15；平均位置误差 1.5684；MOTA-like 0.4520。

## 执行命令

```bash
conda run -n mttrack-repro python -m unittest discover -s tests -v
conda run -n mttrack-repro python -m src.simulation.generate_dataset
conda run -n mttrack-repro python -m src.simulation.generate_dataset --profile all
conda run -n mttrack-repro python src/tracking/baseline_tracker.py \
  --dataset data/datasets/phase1/smoke/smoke_dataset.npz --sample-index 0 \
  --output /tmp/phase1_dataset_baseline.json
```

单元测试结果：4 项通过；Python 语法检查通过；未发现运行错误。
