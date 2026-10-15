"""Small paired sensitivity experiment; uses validation/diagnostic scenes only.

Constant translations of individual trajectories change spacing but preserve
velocity, acceleration and equation (1). Sensor errors, IDs, detection masks,
and clutter counts are shared across the paired cases. Never opens test files.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_data import baseline, cartesian, frame, score, spherical


def observations(data, cfg, scale=1.0, local_clutter=False):
    offsets = data['observation_offsets']
    ids = data['observation_target_id']
    clean = data['observation_clean_xyz'].copy()
    middle = len(data['time_s']) // 2
    p = data['truth_state'][:, :, :3]
    center = p[middle].mean(0)
    shift = (scale - 1) * (p[middle] - center)
    for k in range(len(offsets) - 1):
        a, b = offsets[k:k + 2]
        valid = ids[a:b] >= 0
        clean[a:b][valid] += shift[ids[a:b][valid]]
    if local_clutter:
        mask = ids < 0
        low, high = np.array(cfg['clutter_xyz_low_m']), np.array(cfg['clutter_xyz_high_m'])
        # Same uniform draws mapped into a 10-km cube about the scene center.
        unit = (clean[mask] - low) / (high - low)
        clean[mask] = center + (unit - .5) * 10000
    target = ids >= 0
    radial_errors = data['observation_range_only_rtp'][:, 0] - spherical(data['observation_clean_xyz'])[:, 0]
    polar = spherical(clean)
    polar[target, 0] += radial_errors[target]
    xyz = clean.copy()
    xyz[target] += (data['observation_cartesian_xyz'] - data['observation_clean_xyz'])[target]
    return {'range_only': cartesian(polar), 'cartesian_xyz': xyz}, p + shift[None], clean


def make_pair(data, xyz, k, gap):
    a, b = data['observation_offsets'][k:k + 2]
    c, d = data['observation_offsets'][k + gap:k + gap + 2]
    id1, id2 = data['observation_target_id'][a:b], data['observation_target_id'][c:d]
    label = ((id1[:, None] == id2[None, :]) & (id1[:, None] >= 0)).astype(np.uint8)
    return xyz[a:b], xyz[c:d], id1, id2, label


def mutual_nearest(x, y, gate):
    out = np.zeros((len(x), len(y)), np.uint8)
    if not len(x) or not len(y): return out
    distance = np.linalg.norm(x[:, None] - y[None], axis=-1)
    cols = distance.argmin(1); rows = distance.argmin(0)
    ii = np.arange(len(x))
    keep = (rows[cols] == ii) & (distance[ii, cols] < gate)
    out[ii[keep], cols[keep]] = 1
    return out


def geometry(x, y, label, p1, p2):
    distances = np.linalg.norm(x[:, None] - y[None], axis=-1)
    ii, jj = np.nonzero(label)
    positive = distances[ii, jj]
    candidates = distances[ii].copy()
    if len(ii): candidates[np.arange(len(ii)), jj] = np.inf
    wrong = candidates.min(1) if len(y) else np.array([])
    keep = np.isfinite(wrong)
    spacing = np.linalg.norm(p1[:, None] - p1[None], axis=-1)
    np.fill_diagonal(spacing, np.inf)
    nn = spacing.min(1)
    nn = nn[np.isfinite(nn)]
    return {'true_edge_distance_m': positive.tolist(), 'wrong_nearest_distance_m': wrong[keep].tolist(),
            'nearest_margin_m': (wrong[keep] - positive[keep]).tolist(),
            'motion_distance_m': np.linalg.norm(p2 - p1, axis=-1).tolist(),
            'target_nn_spacing_m': nn.tolist(),
            'ambiguous': int((wrong[keep] <= positive[keep]).sum()), 'positive_examined': int(keep.sum())}


def extended_score(label, pred, id1, id2):
    out = score(label, pred)
    false = pred.astype(bool) & ~label.astype(bool)
    rr, cc = np.nonzero(false)
    out['false_clutter_clutter'] = int(((id1[rr] < 0) & (id2[cc] < 0)).sum())
    out['false_target_target'] = int(((id1[rr] >= 0) & (id2[cc] >= 0)).sum())
    out['false_target_clutter'] = int(((id1[rr] < 0) ^ (id2[cc] < 0)).sum())
    return out


def aggregate(records, bootstrap_rng):
    positives = sum(r['positive'] for r in records)
    predicted = sum(r['predicted'] for r in records); tp = sum(r['tp'] for r in records)
    scene_ids = sorted({r['scene_id'] for r in records})
    scene_means = np.array([np.mean([r['accuracy'] for r in records if r['scene_id'] == s]) for s in scene_ids])
    # Every case has the same anchors per scene; scene mean equals pair mean.
    indices = bootstrap_rng.integers(0, len(scene_means), (1000, len(scene_means)))
    ci = np.quantile(scene_means[indices].mean(1), [.025, .975])
    return {'pairs': len(records), 'scenes': len(scene_ids), 'accuracy': float(scene_means.mean()),
            'accuracy_ci95_scene_bootstrap': ci.tolist(),
            'precision': tp / predicted if predicted else 0., 'recall': tp / positives if positives else 0.,
            'f1': 2 * tp / (positives + predicted) if positives + predicted else 0.,
            'true_edges': positives, 'predicted_edges': predicted,
            'false_clutter_clutter': sum(r['false_clutter_clutter'] for r in records),
            'false_target_target': sum(r['false_target_target'] for r in records),
            'false_target_clutter': sum(r['false_target_clutter'] for r in records),
            'scene_accuracy': {str(s): float(m) for s, m in zip(scene_ids, scene_means)}}


def geometry_summary(records):
    out = {}
    for key in ['true_edge_distance_m', 'wrong_nearest_distance_m', 'nearest_margin_m', 'motion_distance_m', 'target_nn_spacing_m']:
        values = np.concatenate([np.array(r[key]) for r in records])
        out[key] = dict(zip(['p10', 'median', 'p90'], map(float, np.quantile(values, [.1, .5, .9])))) if len(values) else None
    n = sum(r['positive_examined'] for r in records)
    out['wrong_edge_closer_fraction'] = sum(r['ambiguous'] for r in records) / n if n else None
    out['positive_edges_examined'] = n
    return out


def run(root, output):
    cfg = json.loads((root / 'config.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    # All 180 validation scenes; no scene selection based on results.
    entries = [e for e in manifest['scenes'] if e['kind'] == 'main' and e['subset'] == 'validation']
    assert set(e['scene_id'] for e in entries).isdisjoint(manifest['split']['test'])
    cases, geometries = {}, {}
    density_anchors = [40, 45, 50, 55, 60]
    gap_anchors = list(range(0, 91, 10))
    def evaluate(key, scene_id, x, y, id1, id2, label, gate, method):
        prediction = baseline(x, y, gate) if method == 'KM' else mutual_nearest(x, y, gate)
        values = extended_score(label, prediction, id1, id2); values['scene_id'] = scene_id
        cases.setdefault(key, []).append(values)
    for number, entry in enumerate(entries):
        with np.load(root / entry['file'], allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        # Paired geometry: move targets by constant per-trajectory translations.
        for scale in [1., .1, .03]:
            for local in [False, True]:
                variants, positions, clean = observations(data, cfg, scale, local)
                case = f"density_s{scale:g}_{'local' if local else 'wide'}_clutter"
                for variant, xyz in variants.items():
                    gkey = f'{case}/{variant}'
                    for k in density_anchors:
                        x, y, id1, id2, label = make_pair(data, xyz, k, 1)
                        geometries.setdefault(gkey, []).append(geometry(x, y, label, positions[k], positions[k + 1]))
                        for gate in [2000., 5000., 10000.]:
                            for method in ['KM', 'mutual_NN']:
                                key = f'{case}/{variant}/{method}/gate{int(gate)}'
                                evaluate(key, entry['scene_id'], x, y, id1, id2, label, gate, method)
        # Gap varies on the original trajectories and original observations.
        for gap in [1, 5, 10]:
            conservative_speed_bound = cfg['initial_speed_mps'][1] + cfg['acceleration_mps2'][1] * data['time_s'][-1]
            adaptive_gate = conservative_speed_bound * gap * cfg['dt_seconds'] + 3 * np.sqrt(6) * cfg['measurement_noise_std_m']
            for variant in cfg['noise_variants']:
                xyz = data['observation_range_only_xyz' if variant == 'range_only' else 'observation_cartesian_xyz']
                gkey = f'gap{gap}/{variant}'
                for k in gap_anchors:
                    x, y, id1, id2, label = make_pair(data, xyz, k, gap)
                    geometries.setdefault(gkey, []).append(geometry(x, y, label, data['truth_state'][k, :, :3], data['truth_state'][k + gap, :, :3]))
                    for mode, gate in [('fixed', 10000.), ('displacement_bound', adaptive_gate)]:
                        key = f'gap{gap}/{variant}/KM/{mode}'
                        evaluate(key, entry['scene_id'], x, y, id1, id2, label, gate, 'KM')
        # Control check: scale=1, wide clutter is numerically the source data.
        control, _, _ = observations(data, cfg)
        assert np.allclose(control['range_only'], data['observation_range_only_xyz'], atol=1e-7)
        assert np.array_equal(control['cartesian_xyz'], data['observation_cartesian_xyz'])
        if (number + 1) % 30 == 0: print(f'checked validation scenes {number + 1}/{len(entries)}', flush=True)
    rng = np.random.default_rng(20261014)
    aggregate_cases = {key: aggregate(records, rng) for key, records in cases.items()}
    # Paired CI for changes in accuracy, resampling whole shared scenes.
    paired = {}
    for variant in cfg['noise_variants']:
        reference = aggregate_cases[f'density_s1_wide_clutter/{variant}/KM/gate10000']['scene_accuracy']
        for density, clutter in [(0.1, 'wide'), (.03, 'wide'), (1., 'local'), (.03, 'local')]:
            key = f'density_s{density:g}_{clutter}_clutter/{variant}/KM/gate10000'
            scores = aggregate_cases[key]['scene_accuracy']
            diffs = np.array([scores[s] - reference[s] for s in sorted(reference)])
            idx = rng.integers(0, len(diffs), (1000, len(diffs)))
            paired[key] = {'delta_accuracy': float(diffs.mean()), 'ci95': np.quantile(diffs[idx].mean(1), [.025, .975]).tolist()}
    result = {'purpose': 'Minimal data difficulty check, not CTDA training or paper performance reproduction',
              'validation_scenes': len(entries), 'sealed_test_opened': False,
              'source_manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
              'protocol': {'density_anchors_s': density_anchors, 'gap_anchors_s': gap_anchors,
                           'spacing_scales': [1, .1, .03], 'local_clutter_cube_side_m': 10000,
                           'gates_m': [2000, 5000, 10000], 'pair_gaps_s': [1, 5, 10],
                           'methods': ['KM', 'mutual_NN'], 'bootstrap_unit': 'whole scene', 'bootstrap_repetitions': 1000,
                           'shared': ['velocities', 'accelerations', 'target IDs', 'detection masks', 'clutter counts', 'sensor noise draws'],
                           'caveat': 'Density and clutter transformations are additional diagnostics, not inferred author settings.'},
              'geometry': {key: geometry_summary(records) for key, records in geometries.items()},
              'baselines': aggregate_cases, 'paired_accuracy_differences': paired}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['case', 'pairs', 'accuracy', 'precision', 'recall', 'f1', 'false_clutter_clutter', 'false_target_target', 'false_target_clutter'])
        for key, values in aggregate_cases.items():
            writer.writerow([key] + [values[k] for k in ['pairs', 'accuracy', 'precision', 'recall', 'f1', 'false_clutter_clutter', 'false_target_target', 'false_target_clutter']])
    plot_results(output, result)
    write_report(output, result)
    print(json.dumps({'output': str(output), 'validation_scenes': len(entries), 'baseline_conditions': len(cases)}, indent=2))


def plot_results(output, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {'range_only': '#2374ab', 'cartesian_xyz': '#e07a24'}
    for variant in colors:
        for clutter, linestyle in [('wide', '-'), ('local', '--')]:
            xx, yy = [], []
            for scale in [1., .1, .03]:
                key = f'density_s{scale:g}_{clutter}_clutter/{variant}/KM/gate10000'
                values = result['baselines'][key]
                xx.append(scale); yy.append(values['accuracy'])
            axes[0, 0].plot(xx, yy, marker='o', color=colors[variant], linestyle=linestyle, label=f'{variant}, {clutter} clutter')
        for mode, linestyle in [('fixed', '-'), ('displacement_bound', '--')]:
            ys = [result['baselines'][f'gap{gap}/{variant}/KM/{mode}']['accuracy'] for gap in [1, 5, 10]]
            axes[0, 1].plot([1, 5, 10], ys, marker='o', color=colors[variant], linestyle=linestyle, label=f'{variant}, {mode}')
        for clutter, linestyle in [('wide', '-'), ('local', '--')]:
            ys = [result['geometry'][f'density_s{scale:g}_{clutter}_clutter/{variant}']['wrong_edge_closer_fraction'] for scale in [1., .1, .03]]
            axes[1, 0].plot([1., .1, .03], ys, marker='o', color=colors[variant], linestyle=linestyle, label=f'{variant}, {clutter} clutter')
        for method, linestyle in [('KM', '-'), ('mutual_NN', '--')]:
            ys = [result['baselines'][f'density_s0.03_local_clutter/{variant}/{method}/gate{gate}']['accuracy'] for gate in [2000, 5000, 10000]]
            axes[1, 1].plot([2, 5, 10], ys, marker='o', color=colors[variant], linestyle=linestyle, label=f'{variant}, {method}')
    axes[0, 0].set(xscale='log', xlabel='Target spacing scale at t=50 s', ylabel='Paper accuracy', title='Paired spacing / clutter controls', ylim=(0, 1.02))
    axes[0, 1].set(xlabel='Pair gap (s)', ylabel='Paper accuracy', title='Original geometry, paired gap controls', ylim=(0, 1.02))
    axes[1, 0].set(xscale='log', xlabel='Target spacing scale at t=50 s', ylabel='Wrong edge closer fraction', title='Positive association ambiguity', ylim=(0, 1.02))
    axes[1, 1].set(xlabel='Distance gate (km)', ylabel='Paper accuracy', title='Dense targets + local clutter', ylim=(0, 1.02))
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / 'difficulty_controls.png', dpi=160); plt.close(fig)


def write_report(output, result):
    base = result['baselines']['density_s1_wide_clutter/cartesian_xyz/KM/gate10000']
    dense = result['baselines']['density_s0.03_wide_clutter/cartesian_xyz/KM/gate10000']
    local = result['baselines']['density_s1_local_clutter/cartesian_xyz/KM/gate10000']
    local_fp = sum(local[k] for k in ['false_clutter_clutter', 'false_target_target', 'false_target_clutter'])
    cc_fraction = local['false_clutter_clutter'] / local_fp
    delta = result['paired_accuracy_differences']['density_s0.03_wide_clutter/cartesian_xyz/KM/gate10000']
    rows = []
    for scale in [1., .1, .03]:
        for clutter in ['wide', 'local']:
            for variant in ['range_only', 'cartesian_xyz']:
                key = f'density_s{scale:g}_{clutter}_clutter/{variant}'
                s = result['baselines'][key + '/KM/gate10000']; g = result['geometry'][key]
                rows.append(f"| {scale:g} | {clutter} | {variant} | {s['accuracy']:.2%} | {s['f1']:.2%} | {s['recall']:.2%} | {g['wrong_edge_closer_fraction']:.2%} |")
    gaprows = []
    for gap in [1, 5, 10]:
        for variant in ['range_only', 'cartesian_xyz']:
            a = result['baselines'][f'gap{gap}/{variant}/KM/fixed']; b = result['baselines'][f'gap{gap}/{variant}/KM/displacement_bound']
            gaprows.append(f"| {gap} | {variant} | {a['accuracy']:.2%} | {b['accuracy']:.2%} | {b['recall']:.2%} |")
    gr = result['geometry']['density_s1_wide_clutter/range_only']; gc = result['geometry']['density_s1_wide_clutter/cartesian_xyz']
    report = f"""# 最小数据难度核查

使用固定的 {result['validation_scenes']} 个验证场景；封存测试场景未打开。未训练网络。本实验单独改变目标密度、杂波空间区域和配对间隔，所有选择均为额外诊断，不是作者配置的推定。

**结论：当前主候选数据过于稀疏，适合运行链路检查，不足以单独检验 CTDA 在近邻/交叉场景中的作用。此前交叉诊断的低分同时受到目标密集和杂波局部密集影响，不能将两者混为一个“交叉难度”。**

以 XYZ 噪声版为例：

- 主候选 Accuracy {base['accuracy']:.2%}；同窗口正确边比最近错误边短一个数量级，最近邻混淆率接近零。
- 只将 t=50 s 的目标间距缩放至 .03、保持原杂波，Accuracy 降至 {dense['accuracy']:.2%}，正关联 Recall {dense['recall']:.2%}。相对主候选下降 {-delta['delta_accuracy'] * 100:.2f} 个百分点，配对场景重采样 95% 区间为 [{-delta['ci95'][1] * 100:.2f}, {-delta['ci95'][0] * 100:.2f}] 个百分点，确实出现目标关联困难。
- 只将杂波集中到 10 km 立方体、保持目标稀疏，Accuracy 降至 {local['accuracy']:.2%}，但真实正关联 Recall 仍为 {local['recall']:.2%}；错误匹配中 {cc_fraction:.2%} 是杂波—杂波。该低分主要反映杂波抑制/未匹配判定的失败。
- 在较长配对间隔下，固定门限和按位移上界放宽的门限结果不同；不能靠增大间隔并保持窄门限来人为制造“模型难题”。

这项核查没有证明 CTDA 一定能解决这些情况；要判断可学习性及模型贡献，还需小样本训练与基线对照。下一步保留 v1 作为稀疏对照，训练样本应覆盖接近与交叉情况，并把目标间距和杂波密度分别控制、分别报告。不得因某个缩放条件分数接近 98.33% 就把它认定为作者配置。

## 主数据的尺度

同一 40–60 s 窗口，真实目标最近邻间距中位数 {gr['target_nn_spacing_m']['median'] / 1000:.2f} km；1 s 真实位移中位数 {gr['motion_distance_m']['median'] / 1000:.2f} km。

正确关联的含噪量测距离中位数：距离噪声版 {gr['true_edge_distance_m']['median'] / 1000:.2f} km、XYZ 噪声版 {gc['true_edge_distance_m']['median'] / 1000:.2f} km。最近错误候选距离中位数分别为 {gr['wrong_nearest_distance_m']['median'] / 1000:.2f}、{gc['wrong_nearest_distance_m']['median'] / 1000:.2f} km。

因此可以直接检查“错误候选远、真实匹配近”是否成立；错误候选比正确候选近的比例见表。

## 目标密度和杂波区域的配对对照

锚点固定为 40/45/50/55/60 s，每条件 900 对。每条轨迹只作一个固定平移，使 t=50 s 的位置围绕场景中心缩放；速度、加速度、检测、ID 与噪声保持相同。wide 使用原杂波区域；local 将相同杂波均匀随机数映射到场景中心的 10 km 立方体。两者分别变化，可区分目标混淆与杂波密集导致的假匹配。

表中是固定 10 km 门限的全局距离分配，允许未匹配。Accuracy 为论文式 (35)，F1/Recall 只统计正关联，混淆率为错误候选距离小于等于正确关联距离的比例。

| 目标间距缩放 | 杂波区域 | 噪声 | Accuracy | 正关联 F1 | 正关联 Recall | 最近邻混淆率 |
|---:|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 配对间隔的对照

保持原始轨迹与量测，只将时间间隔改为 1/5/10 s，使用相同 0/10/…/90 s 起点，每条件 1800 对。“位移上界”门限是由初始速度上限、加速度上限、序列长度和噪声推得，不使用真实目标 ID 或实际速度调门限。原固定 10 km 门限在较大间隔下可能截断正确边，不能将门限效应解释成目标混淆。

| 间隔 s | 噪声 | 固定门限 Accuracy | 位移上界门限 Accuracy | 位移上界门限 Recall |
|---:|---|---:|---:|---:|
{chr(10).join(gaprows)}

## 核查口径与产物

- 另比较 2/5/10 km 门限与双向最近邻法，检查结果对求解方法的敏感性，详见 [CSV](metrics.csv)。
- 置信区间按完整场景重采样 1000 次；目标/杂波变化的差值采用配对场景重采样，见 [JSON](results.json)。不能把同一轨迹中的帧对视作独立试验。
- 错误边分为杂波—杂波、目标—杂波、错误目标—目标，避免把密集杂波造成的低分全部归咎于航迹交叉。
- 对照条件（scale=1、wide）与原始文件数值一致；原始数据及划分保持不变，标签不作为算法输入。
- [诊断图](difficulty_controls.png)；[复现实验脚本](../../experiments/check_data_difficulty.py)。

重跑：

```bash
MPLCONFIGDIR=/tmp/mttracking-mpl /home/chenrui/miniconda3/envs/mttrack-repro/bin/python experiments/check_data_difficulty.py
```
"""
    (output / 'REPORT.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/ctda_3d_v1'))
    parser.add_argument('--output', type=Path, default=Path('results/data_difficulty_v1'))
    args = parser.parse_args()
    run(args.data, args.output)
