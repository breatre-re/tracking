# Phase 1.5：论文分布配置化报告

## 产物与运行

- 配置：[phase1_5_distributions.yaml](../../configs/phase1_5_distributions.yaml)
- Manifest：[phase1_5_manifest.json](../../data/manifests/phase1_5_manifest.json)
- 原有 Phase 1 数据目录和文件保持不变；新数据保存在：

```text
data/datasets/phase1/
  baseline_distribution/{train,test}/{train,test}_dataset.npz
  paper_assumption_v1/{train,test}/{train,test}_dataset.npz
data/manifests/phase1_5_manifest.json
```

运行命令：

```bash
conda run -n mttrack-repro python -m src.simulation.generate_dataset \
  --config configs/phase1_5_distributions.yaml --profile all --output-root data/datasets/phase1_5
conda run -n mttrack-repro python -m unittest discover -s tests -v
```

对两种分布的 train/test 样本均独立重生成了一次，并逐字节比较 NPZ；SHA-256 一致。NPZ 可读，manifest 哈希匹配，训练/测试样本内容无重复。用旧 baseline 分别读取两种分布的训练样本成功。加载器返回给 tracker 的 `Detection` 不包含 `source_target_id`；测量真值标签只在离线侧车数组中。

## 配置与分布统计

所有数据均为 80 帧/样本，训练 1,800 条、测试 200 条。

| 分布 | 目标数 | Train 平均目标数 | Test 平均目标数 | Train clutter/帧均值 ± 标准差 | Test clutter/帧均值 ± 标准差 | Train 测量点 | Test 测量点 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_distribution` | 固定 6 | 6.00 | 6.00 | 1.5000 ± 1.2229 | 1.4997 ± 1.2245 | 839,798 | 93,371 |
| `paper_assumption_v1` | 离散均匀 U{1,…,25} | 12.77 | 12.76 | 20.0035 ± 4.4668 | 19.9832 ± 4.4706 | 3,885,985 | 431,886 |

Paper assumption 实际训练目标数范围为 1–25，测试也覆盖 1–25。完整目标数直方图、事件计数、种子范围、文件 SHA-256 和配置都在 manifest 中。

## 与原 Phase 1 的差异

- 原有 `data/datasets/phase1/{train,test}/` 和 `data/manifests/phase1_manifest.json` 未被覆盖。
- 新生成的 `baseline_distribution` 与旧 Phase 1 在真值、测量、标签、帧偏移和事件数组上逐项完全相同；只使用独立的新文件封装与 schema 元数据。
- `paper_assumption_v1` 将每条样本目标数从固定 6 改为独立采样的 1–25，将 clutter 从 Poisson lambda=1.5 改为 lambda=20。因 clutter/帧增加，测量总量也相应上升。
- tracker 算法和关联逻辑没有修改。仅数据生成/存储支持按样本变化的目标数，loader 在恢复样本时裁掉 padding。

## Reproduction assumptions

1. 目标数按整数离散均匀分布 `U{1,…,25}`，每个样本独立抽取。
2. 表 8 的 Poisson lambda=20 暂解释为**每个时间步的 clutter 数均值**。如果论文未明确 lambda 的时间单位，此项是 reproduction assumption，不视为已确认的原论文语义。
3. 除目标数和 clutter rate 外，未指定参数沿用 Phase 1：80 帧、检测概率 0.9、测量噪声标准差 1.5、过程加速度噪声标准差 0.08、场景大小 100。
4. 出生/死亡时序仍沿用已有 baseline 生成规则；此规则不是对论文未公开分布的断言。
5. 每个 split 使用连续但不相交的 seed 区间。Paper train 为 30261000–30262799，test 为 30263000–30263199；baseline 分布沿用旧 Phase 1 的 seed 区间。

## 验证结果与注意事项

- 单元测试：5 项通过，其中覆盖变量目标数、来源标签隔离和重复生成的逐字节一致性。
- 四个 split（两分布的 train/test）均完成可读性、manifest hash、标签来源、样本唯一性与 train/test 交集检查。
- 两种分布均可由现有 baseline 从打包数据读取并执行；这只验证数据接口，不代表 paper 分布下 tracker 指标已达标。
- Paper 分布的一条训练样本 baseline smoke 指标包含 3,946 个 FP、118 次 ID switch、MOTA-like -5.214。lambda=20 下当前简单贪心关联显然不适合作为高杂波性能方案；此为当前 baseline 的能力限制，不是数据生成阻塞。CTDA 未实现、未训练。
