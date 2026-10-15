"""Paper-parameter 3D simulation, adjacent-frame labels and data audit.

Supplemental choices are explicit in data_config.json and reproduction_plan.md.
No CTDA training or full BLUE tracker is implemented here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def spherical(xyz):
    x, y, z = np.moveaxis(xyz, -1, 0)
    return np.stack((np.linalg.norm(xyz, axis=-1), np.arctan2(y, x),
                     np.arctan2(z, np.hypot(x, y))), axis=-1)


def cartesian(rtp):
    r, theta, phi = np.moveaxis(rtp, -1, 0)
    return np.stack((r * np.cos(phi) * np.cos(theta),
                     r * np.cos(phi) * np.sin(theta), r * np.sin(phi)), axis=-1)


def velocity_basis(velocity):
    speed = np.linalg.norm(velocity, axis=-1)
    pitch = np.arctan2(velocity[..., 1], np.hypot(velocity[..., 0], velocity[..., 2]))
    yaw = np.arctan2(-velocity[..., 2], velocity[..., 0])
    cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    ev = np.stack((cp * cy, sp, -cp * sy), -1)
    ep = np.stack((-sp * cy, cp, sp * sy), -1)
    ey = np.stack((-sy, np.zeros_like(sy), -cy), -1)
    return speed, pitch, yaw, ev, ep, ey


def controls_and_state(position, velocity, accel, gravity):
    speed, pitch, yaw, ev, ep, ey = velocity_basis(velocity)
    tangential = (accel * ev).sum(-1)
    lift_cos = (accel * ep).sum(-1) + gravity * np.cos(pitch)
    lift_sin = -(accel * ey).sum(-1)
    controls = np.stack((-tangential - gravity * np.sin(pitch),
                         np.hypot(lift_cos, lift_sin), np.arctan2(lift_sin, lift_cos)), -1)
    state = np.concatenate((position, speed[..., None], pitch[..., None], yaw[..., None]), -1)
    return controls, state


def equation1_reconstructed_acceleration(state, controls, gravity):
    """Independently apply printed rates then differentiate Cartesian velocity."""
    v, pitch, yaw = state[..., 3], state[..., 4], state[..., 5]
    ud, ul, ug = controls[..., 0], controls[..., 1], controls[..., 2]
    cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    vdot = -ud - gravity * sp
    pdot = (ul * np.cos(ug) - gravity * cp) / v
    ydot = -ul * np.sin(ug) / (v * cp)
    return np.stack((vdot * cp * cy - v * sp * cy * pdot - v * cp * sy * ydot,
                     vdot * sp + v * cp * pdot,
                     -vdot * cp * sy + v * sp * sy * pdot - v * cp * cy * ydot), -1)


def generate_scene(cfg, scene_id, crossing=False):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg['seed'], scene_id, int(crossing), 1])))
    obs_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg['seed'], scene_id, int(crossing), 2])))
    frames, dt = cfg['frames'], cfg['dt_seconds']
    n = int(rng.integers(cfg['target_count'][0], cfg['target_count'][1] + 1))
    speed = rng.uniform(*cfg['initial_speed_mps'], size=n)
    pitch = rng.uniform(*cfg['initial_pitch_rad'], size=n)
    yaw = rng.uniform(*cfg['initial_yaw_rad'], size=n)
    velocity0 = np.stack((speed * np.cos(pitch) * np.cos(yaw), speed * np.sin(pitch),
                          -speed * np.cos(pitch) * np.sin(yaw)), -1)
    direction = rng.normal(size=(frames, n, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    magnitude = rng.uniform(*cfg['acceleration_mps2'], size=(frames, n, 1))
    acceleration = magnitude * direction
    refresh = cfg['acceleration_refresh_frames']
    acceleration = acceleration[(np.arange(frames) // refresh) * refresh]
    if crossing:
        acceleration[:] = acceleration[0]
        # All targets pass a small shared region at t=50 s; separate diagnostic.
        crossing_time = (frames - 1) * dt / 2
        anchor = np.array([0., 0., 50000.]) + rng.normal(0, 1000, (n, 3))
        initial = anchor - velocity0 * crossing_time - .5 * acceleration[0] * crossing_time ** 2
    else:
        initial = rng.uniform(cfg['initial_xyz_low_m'], cfg['initial_xyz_high_m'], size=(n, 3))
    positions, velocities = np.empty((frames, n, 3)), np.empty((frames, n, 3))
    positions[0], velocities[0] = initial, velocity0
    for k in range(frames - 1):
        positions[k + 1] = positions[k] + velocities[k] * dt + .5 * acceleration[k] * dt ** 2
        velocities[k + 1] = velocities[k] + acceleration[k] * dt
    controls, state = controls_and_state(positions, velocities, acceleration, cfg['gravity_mps2'])
    in_range = (np.abs(positions) <= cfg['max_abs_detection_xyz_m']).all(-1)
    detected = (obs_rng.random((frames, n)) >= cfg['missing_rate']) & in_range
    clutter_counts = obs_rng.poisson(cfg['clutter_poisson_mean'], size=frames)
    if crossing:
        clutter_low, clutter_high = [-5000, -5000, 45000], [5000, 5000, 55000]
    else:
        clutter_low, clutter_high = cfg['clutter_xyz_low_m'], cfg['clutter_xyz_high_m']
    ids, clean, ranges, xyz_range, xyz_cart, range_noise, cart_noise = [], [], [], [], [], [], []
    offsets = [0]
    for k in range(frames):
        target_ids = np.flatnonzero(detected[k]).astype(np.int32)
        clutter = obs_rng.uniform(clutter_low, clutter_high, size=(int(clutter_counts[k]), 3))
        clean_k = np.concatenate((positions[k, target_ids], clutter))
        id_k = np.concatenate((target_ids, np.full(len(clutter), -1, dtype=np.int32)))
        # Only genuine detections receive sensor noise. Clutter is already a
        # measurement sampled uniformly in Cartesian space, shared by variants.
        polar = spherical(clean_k)
        rn = obs_rng.normal(0, cfg['measurement_noise_std_m'], len(target_ids))
        cn = obs_rng.normal(0, cfg['measurement_noise_std_m'], (len(target_ids), 3))
        polar[:len(target_ids), 0] += rn
        cart_k = clean_k.copy()
        cart_k[:len(target_ids)] += cn
        perm = obs_rng.permutation(len(clean_k))
        clean.append(clean_k[perm]); ids.append(id_k[perm]); ranges.append(polar[perm])
        xyz_range.append(cartesian(polar)[perm]); xyz_cart.append(cart_k[perm])
        range_noise.append(rn); cart_noise.append(cn)
        offsets.append(offsets[-1] + len(clean_k))
    flat_ids = np.concatenate(ids)
    positive_edges, label_offsets = [], [0]
    for k in range(frames - 1):
        a, b = ids[k], ids[k + 1]
        pairs = np.argwhere((a[:, None] == b[None, :]) & (a[:, None] >= 0)).astype(np.int32)
        positive_edges.append(pairs)
        label_offsets.append(label_offsets[-1] + len(pairs))
    return {
        'scene_id': np.array(scene_id, np.int32), 'is_crossing_diagnostic': np.array(crossing),
        'time_s': np.arange(frames) * dt,
        'truth_state': state, 'truth_velocity_xyz': velocities, 'acceleration_xyz': acceleration,
        'controls_uD_uL_uG': controls, 'truth_spherical': spherical(positions),
        'detected': detected, 'in_range': in_range, 'clutter_counts': clutter_counts,
        'observation_offsets': np.array(offsets, np.int64), 'observation_target_id': flat_ids,
        'observation_clean_xyz': np.concatenate(clean), 'observation_range_only_rtp': np.concatenate(ranges),
        'observation_range_only_xyz': np.concatenate(xyz_range), 'observation_cartesian_xyz': np.concatenate(xyz_cart),
        'positive_edge_offsets': np.array(label_offsets, np.int64), 'positive_edges': np.concatenate(positive_edges),
        'range_noise_samples': np.concatenate(range_noise), 'cartesian_noise_samples': np.concatenate(cart_noise)
    }


def frame(data, k, variant):
    a, b = data['observation_offsets'][k:k + 2]
    key = {'range_only': 'observation_range_only_xyz', 'cartesian_xyz': 'observation_cartesian_xyz'}[variant]
    return data[key][a:b], data['observation_target_id'][a:b]


def pair(data, k, variant):
    x, aid = frame(data, k, variant)
    y, bid = frame(data, k + 1, variant)
    label = np.zeros((len(x), len(y)), np.uint8)
    a, b = data['positive_edge_offsets'][k:k + 2]
    edges = data['positive_edges'][a:b]
    if len(edges): label[edges[:, 0], edges[:, 1]] = 1
    return x, y, label


def rdf_blocks(x, y, truth, bias=1., size=40):
    if not len(x) or not len(y): return []
    diff = y.T[:, None, :] - x.T[:, :, None]
    absolute = np.abs(diff)
    scale = np.sqrt(absolute.max((1, 2)) * (absolute.min((1, 2)) + bias))
    features = np.tanh(scale[:, None, None] / (2 * (absolute + bias))) * np.sign(diff)
    rows, cols = truth.shape
    nr, nc = ((rows + size - 1) // size) * size, ((cols + size - 1) // size) * size
    padded = np.zeros((3, nr, nc), np.float32); padded[:, :rows, :cols] = features
    labels = np.zeros((nr, nc), np.uint8); labels[:rows, :cols] = truth
    valid = np.zeros((nr, nc), bool); valid[:rows, :cols] = True
    return [(padded[:, i:i + size, j:j + size], labels[i:i + size, j:j + size],
             valid[i:i + size, j:j + size], i, j)
            for i in range(0, nr, size) for j in range(0, nc, size)]


def score(truth, prediction):
    rows, cols = truth.shape
    correct_r, correct_c = (truth == prediction).all(1), (truth == prediction).all(0)
    tp = int((truth.astype(bool) & prediction.astype(bool)).sum())
    return {'accuracy': float((correct_r.sum() + correct_c.sum()) / (rows + cols)) if rows + cols else 1.,
            'tp': tp, 'predicted': int(prediction.sum()), 'positive': int(truth.sum()),
            'true_rows_correct': int(correct_r[truth.any(1)].sum()), 'true_rows': int(truth.any(1).sum())}


def baseline(x, y, gate):
    prediction = np.zeros((len(x), len(y)), np.uint8)
    if not len(x) or not len(y): return prediction
    distance = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    # Own unmatched choices prevent forced invalid assignments from stealing
    # valid matches. This is a fixed baseline, not the paper's network threshold.
    cost = np.concatenate((np.where(distance < gate, distance - gate, 1e12),
                           np.zeros((len(x), len(x)))), axis=1)
    r, c = linear_sum_assignment(cost)
    keep = c < len(y)
    prediction[r[keep], c[keep]] = 1
    return prediction


def audit_scene(data, cfg):
    for key, value in data.items():
        if np.issubdtype(value.dtype, np.floating): assert np.isfinite(value).all(), key
    assert np.all(np.linalg.norm(data['acceleration_xyz'], axis=-1) <= 10 + 1e-10)
    reconstructed = equation1_reconstructed_acceleration(data['truth_state'], data['controls_uD_uL_uG'], cfg['gravity_mps2'])
    eq1_error = float(np.max(np.abs(reconstructed - data['acceleration_xyz'])))
    assert eq1_error < 1e-7, eq1_error
    pos = data['truth_state'][..., :3]
    velocity = data['truth_velocity_xyz']; dt = cfg['dt_seconds']
    assert np.allclose(pos[1:], pos[:-1] + velocity[:-1] * dt + .5 * data['acceleration_xyz'][:-1] * dt ** 2, atol=1e-8)
    roundtrip_error = float(np.max(np.abs(cartesian(data['truth_spherical']) - pos)))
    assert roundtrip_error < 1e-7
    assert np.allclose(cartesian(data['observation_range_only_rtp']), data['observation_range_only_xyz'], atol=1e-7)
    ids = data['observation_target_id']; edges = data['positive_edges']; offsets = data['positive_edge_offsets']
    for k in range(cfg['frames'] - 1):
        _, a = frame(data, k, 'range_only'); _, b = frame(data, k + 1, 'range_only')
        assert len(a[a >= 0]) == len(np.unique(a[a >= 0]))
        e = edges[offsets[k]:offsets[k + 1]]
        assert len(e) == len(np.intersect1d(a[a >= 0], b[b >= 0]))
        if len(e):
            assert (a[e[:, 0]] >= 0).all() and np.array_equal(a[e[:, 0]], b[e[:, 1]])
            assert len(np.unique(e[:, 0])) == len(e) == len(np.unique(e[:, 1]))
    assert (ids >= -1).all()
    return eq1_error, roundtrip_error


def summarize_scores(scores):
    tp = sum(s['tp'] for s in scores); pred = sum(s['predicted'] for s in scores)
    positive = sum(s['positive'] for s in scores)
    return {'pairs_evaluated': len(scores), 'accuracy_mean': float(np.mean([s['accuracy'] for s in scores])),
            'positive_precision': tp / pred if pred else 0., 'positive_recall': tp / positive if positive else 0.,
            'positive_f1': 2 * tp / (pred + positive) if pred + positive else 0.,
            'true_row_accuracy': sum(s['true_rows_correct'] for s in scores) / sum(s['true_rows'] for s in scores),
            'true_positive_edges': positive}


def assess(root, cfg, manifest):
    baseline_scores = {group: {v: [] for v in cfg['noise_variants']} for group in ['validation', 'crossing_diagnostic']}
    zero_scores = {group: [] for group in baseline_scores}
    stats = {'scenes_read_back_and_checked': 0, 'eq1_max_error': 0., 'coordinate_roundtrip_max_error_m': 0.,
             'eligible_detections': 0, 'detections': 0, 'clutter_total': 0, 'frames': 0,
             'range_noise_count': 0, 'range_noise_sum': 0., 'range_noise_sum2': 0.,
             'cartesian_noise_count': 0, 'cartesian_noise_sum': [0., 0., 0.], 'cartesian_noise_sum2': [0., 0., 0.],
             'negative_noisy_ranges': 0, 'outside_detection_range_states': 0,
             'target_count_min': 25, 'target_count_max': 1, 'target_count_sum': 0,
             'pairs': 0, 'positive_edges': 0, 'empty_positive_pairs': 0, 'max_frame_observations': 0}
    examples = {'range_only': [], 'cartesian_xyz': [], 'label': [], 'valid': [], 'index': []}
    for entry in manifest['scenes']:
        with np.load(root / entry['file'], allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        e1, e2 = audit_scene(data, cfg)
        if entry['kind'] == 'main':
            stats['scenes_read_back_and_checked'] += 1
            stats['eq1_max_error'] = max(stats['eq1_max_error'], e1)
            stats['coordinate_roundtrip_max_error_m'] = max(stats['coordinate_roundtrip_max_error_m'], e2)
            stats['eligible_detections'] += int(data['in_range'].sum())
            stats['detections'] += int(data['detected'].sum())
            stats['outside_detection_range_states'] += int((~data['in_range']).sum())
            stats['clutter_total'] += int(data['clutter_counts'].sum()); stats['frames'] += cfg['frames']
            rn, cn = data['range_noise_samples'], data['cartesian_noise_samples']
            stats['range_noise_count'] += len(rn); stats['range_noise_sum'] += float(rn.sum()); stats['range_noise_sum2'] += float((rn ** 2).sum())
            stats['cartesian_noise_count'] += len(cn)
            for axis in range(3):
                stats['cartesian_noise_sum'][axis] += float(cn[:, axis].sum())
                stats['cartesian_noise_sum2'][axis] += float((cn[:, axis] ** 2).sum())
            stats['negative_noisy_ranges'] += int((data['observation_range_only_rtp'][:, 0] < 0).sum())
            n = data['truth_state'].shape[1]
            stats['target_count_min'] = min(stats['target_count_min'], n); stats['target_count_max'] = max(stats['target_count_max'], n)
            stats['target_count_sum'] += n; stats['pairs'] += cfg['frames'] - 1
            stats['positive_edges'] += len(data['positive_edges'])
            stats['empty_positive_pairs'] += int((np.diff(data['positive_edge_offsets']) == 0).sum())
            stats['max_frame_observations'] = max(stats['max_frame_observations'], int(np.diff(data['observation_offsets']).max()))
        group = 'crossing_diagnostic' if entry['kind'] == 'crossing_diagnostic' else entry['subset']
        if group not in baseline_scores: continue
        if group == 'validation':
            ks = range(0, cfg['frames'] - 1, cfg['baseline_validation_pair_stride'])
        else:
            low, high = cfg['crossing_diagnostic_time_window_seconds']
            ks = [k for k in range(cfg['frames'] - 1) if low <= k * cfg['dt_seconds'] <= high]
        for k in ks:
            for variant in cfg['noise_variants']:
                x, y, label = pair(data, k, variant)
                baseline_scores[group][variant].append(score(label, baseline(x, y, cfg['baseline_gate_m'])))
                if variant == 'range_only': zero_scores[group].append(score(label, np.zeros_like(label)))
            if group == 'validation' and len(examples['label']) < 128:
                variants = {v: rdf_blocks(*pair(data, k, v), cfg['rdf_bias_m'], cfg['partition_size']) for v in cfg['noise_variants']}
                for block_id, first in enumerate(variants['range_only']):
                    if len(examples['label']) == 128: break
                    for v in cfg['noise_variants']: examples[v].append(variants[v][block_id][0])
                    examples['label'].append(first[1]); examples['valid'].append(first[2])
                    examples['index'].append([entry['scene_id'], k, first[3], first[4], *label.shape])
        if entry['scene_id'] % 250 == 0: print(f"audit readback {entry['kind']} {entry['scene_id']}", flush=True)
    stats['target_count_mean'] = stats['target_count_sum'] / cfg['scene_count']
    stats['missing_rate_observed'] = 1 - stats['detections'] / stats['eligible_detections']
    stats['clutter_mean_observed'] = stats['clutter_total'] / stats['frames']
    rn_mean = stats['range_noise_sum'] / stats['range_noise_count']
    stats['range_noise_mean_m'] = rn_mean
    stats['range_noise_std_m'] = float(np.sqrt(stats['range_noise_sum2'] / stats['range_noise_count'] - rn_mean ** 2))
    cn_mean = np.array(stats['cartesian_noise_sum']) / stats['cartesian_noise_count']
    stats['cartesian_noise_mean_xyz_m'] = cn_mean.tolist()
    stats['cartesian_noise_std_xyz_m'] = np.sqrt(np.array(stats['cartesian_noise_sum2']) / stats['cartesian_noise_count'] - cn_mean ** 2).tolist()
    tolerance = .02 if cfg['scene_count'] < 100 else .002
    assert abs(stats['missing_rate_observed'] - cfg['missing_rate']) < tolerance
    assert abs(stats['clutter_mean_observed'] - cfg['clutter_poisson_mean']) < (.8 if cfg['scene_count'] < 100 else .15)
    assert abs(stats['range_noise_std_m'] - 1000) < (60 if cfg['scene_count'] < 100 else 5)
    assert max(abs(s - 1000) for s in stats['cartesian_noise_std_xyz_m']) < (60 if cfg['scene_count'] < 100 else 5)
    for subset in ['fit', 'validation', 'test']:
        members = set(manifest['split'][subset])
        for other in ['fit', 'validation', 'test']:
            if subset != other: assert not members.intersection(manifest['split'][other])
    stats['data_checks_passed'] = True
    np.savez_compressed(root / 'validation_example_blocks.npz', **{key: np.array(value) for key, value in examples.items()})
    return {'data_audit': stats,
            'distance_baseline': {g: {v: summarize_scores(s) for v, s in variants.items()} for g, variants in baseline_scores.items()},
            'all_zero_baseline': {g: summarize_scores(s) for g, s in zero_scores.items()},
            'test_set_performance_evaluated': False,
            'baseline_note': 'Fixed 10000m gate, distance partial assignment; validation and added crossing diagnostic only. Not CTDA results.'}


def save_plots(root, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = next(e for e in manifest['scenes'] if e['kind'] == 'main' and e['subset'] == 'fit' and e['targets'] >= 5)
    diag = next(e for e in manifest['scenes'] if e['kind'] == 'crossing_diagnostic' and e['targets'] >= 5)
    fig = plt.figure(figsize=(12, 5))
    for number, entry in enumerate([chosen, diag], 1):
        with np.load(root / entry['file']) as data:
            ax = fig.add_subplot(1, 2, number, projection='3d')
            p = data['truth_state'][:, :, :3] / 1000
            for i in range(p.shape[1]): ax.plot(p[:, i, 0], p[:, i, 1], p[:, i, 2], lw=1)
            clutter = data['observation_range_only_xyz'][data['observation_target_id'] < 0][::8] / 1000
            ax.scatter(*clutter.T, s=2, c='gray', alpha=.25)
            ax.set(xlabel='x (km)', ylabel='y (km)', zlabel='z (km)',
                   title=f"{'Main candidate' if number == 1 else 'Added crossing diagnostic'} / scene {entry['scene_id']}")
    fig.tight_layout(); fig.savefig(root / 'trajectory_examples.png', dpi=150); plt.close(fig)


def write_results(root, cfg, manifest, result):
    stats = result['data_audit']
    rows = []
    for group, variants in result['distance_baseline'].items():
        for variant, values in variants.items():
            rows.append(f"| {group} | {variant} | {values['pairs_evaluated']} | {values['accuracy_mean']:.2%} | {values['positive_f1']:.2%} |")
    size = sum((root / entry['file']).stat().st_size for entry in manifest['scenes'])
    text = f"""# 首轮数据执行结果

**数据已落盘并通过读回检查。当前是依据论文表 2 与显式假设生成的复现候选，不是作者原始数据。** 具体选择见 [简短计划](../../notes/reproduction_plan.md)。

- 主数据：{cfg['scene_count']} 多目标场景，每场景 {cfg['frames']} 帧；共 {stats['pairs']:,} 个相邻帧关联对，两种噪声解释共享真值和标签。
- 实际单目标轨迹共 {stats['target_count_sum']:,} 条；这是将论文未明确的“2000 trajectories”解释为多目标场景的结果，不等于已取得作者同规模轨迹集。
- 场景划分：训练池 {len(manifest['split']['training_pool'])}（fit {len(manifest['split']['fit'])}、validation {len(manifest['split']['validation'])}），封存 test {len(manifest['split']['test'])}。
- 额外交叉诊断：{cfg['diagnostic_crossing_scenes']} 场景，单列保存，不混入主训练或测试。
- 主数据正关联边 {stats['positive_edges']:,}，全零标签对 {stats['empty_positive_pairs']:,}；目标数 {stats['target_count_min']}–{stats['target_count_max']}，均值 {stats['target_count_mean']:.3f}。
- 实测漏检率 {stats['missing_rate_observed']:.5f}；杂波均值 {stats['clutter_mean_observed']:.4f}/帧。
- 距离噪声标准差 {stats['range_noise_std_m']:.3f} m；XYZ 噪声标准差 {[round(v, 3) for v in stats['cartesian_noise_std_xyz_m']]} m。
- 式 (1) 控制换算最大加速度误差 {stats['eq1_max_error']:.3g} m/s²；坐标往返最大误差 {stats['coordinate_roundtrip_max_error_m']:.3g} m。
- 有限值、运动、坐标转换、真实 ID、标签一对一、划分无重叠检查通过；主数据全部 {stats['scenes_read_back_and_checked']} 场景从文件读回检查。
- 场景文件总大小约 {size / 2**20:.1f} MiB；另有 manifest、统计、示例 40×40 块和图。

## 距离基线（未评估封存测试集）

固定门限 10000 m，允许未匹配的一对一距离分配。Accuracy 按论文式 (35)；F1 只统计真实正关联。

| 评价集 | 噪声版本 | 配对数 | 论文 Accuracy | 正关联 F1 |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

主验证集每 10 帧取一对；交叉诊断取 40–60 s。它们使用不同空间条件，诊断用于展示难度敏感性，不能当作论文测试结果。若主验证距离基线已接近或超过 98.33%，应先核查仿真假设，不能据模型高分宣称复现成功。

全零预测在主验证集的论文 Accuracy 为 {result['all_zero_baseline']['validation']['accuracy_mean']:.2%}，正关联 F1 为 0；这进一步说明必须同时查看正关联指标。当前主候选的距离基线约 99.9%，后续应先核对相邻帧配对、目标间距和杂波区域的假设，保留固定验证与封存测试，不通过调数据迎合论文数值。

## 文件读取

每个 NPZ 保存 `truth_state`（帧×目标×6）、速度、加速度、控制、极坐标真值、漏检掩码、杂波数量，以及扁平观测。

`observation_offsets[k:k+2]` 指定第 k 帧观测；`observation_target_id=-1` 表示杂波，其余是场景内目标 ID。`positive_edges` 与 `positive_edge_offsets` 给出每对的非零标签坐标，其余标签为零。

两种输入分别为 `observation_range_only_xyz`、`observation_cartesian_xyz`。`experiments/generate_data.py` 中的 `pair()` 可还原某个关联对；`rdf_blocks()` 按式 (6)–(10) 得到网络输入。`validation_example_blocks.npz` 是至多 128 块可直接读取的示例，包含两种输入、标签、有效区域和原始位置索引；完整原始数据可在后续训练时按需变换。

场景列表/划分/逐文件 SHA256：[manifest.json](manifest.json)；完整统计：[audit.json](audit.json)；实际参数：[config.json](config.json)；示意图：[trajectory_examples.png](trajectory_examples.png)。

重新生成（同配置确定性）：

```bash
/home/chenrui/miniconda3/envs/mttrack-repro/bin/python experiments/generate_data.py --output data/ctda_3d_v1
```

生成脚本拒绝覆盖已有 manifest，避免无意改动固定数据。需重生成时使用新的输出目录。下一步先做数据难度对照，再以距离噪声版启动 CTDA 小样本训练，XYZ 版作解释敏感性对照；已有候选数据可以直接读取。
"""
    (root / 'RESULTS.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('data_config.json'))
    parser.add_argument('--output', type=Path, default=Path('data/ctda_3d_v1'))
    parser.add_argument('--scenes', type=int)
    parser.add_argument('--crossing-scenes', type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.scenes:
        cfg['scene_count'] = args.scenes; cfg['training_pool_scenes'] = int(.9 * args.scenes)
        cfg['validation_scenes_within_training_pool'] = max(1, cfg['training_pool_scenes'] // 10)
    if args.crossing_scenes is not None: cfg['diagnostic_crossing_scenes'] = args.crossing_scenes
    root = args.output
    if (root / 'manifest.json').exists(): raise SystemExit('Existing manifest: use a different output directory.')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'scenes').mkdir(exist_ok=True); (root / 'crossing_diagnostic').mkdir(exist_ok=True)
    (root / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    split_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg['seed'], 11])))
    ordering = split_rng.permutation(cfg['scene_count']).tolist()
    pool = ordering[:cfg['training_pool_scenes']]; test = ordering[cfg['training_pool_scenes']:]
    val = pool[:cfg['validation_scenes_within_training_pool']]; fit = pool[cfg['validation_scenes_within_training_pool']:]
    subset = {i: s for s, indices in [('fit', fit), ('validation', val), ('test', test)] for i in indices}
    manifest = {'status': cfg['status'], 'seed': cfg['seed'], 'split': {'training_pool': pool, 'fit': fit, 'validation': val, 'test': test}, 'scenes': []}
    for kind, count in [('main', cfg['scene_count']), ('crossing_diagnostic', cfg['diagnostic_crossing_scenes'])]:
        for i in range(count):
            data = generate_scene(cfg, i, crossing=kind != 'main')
            folder = 'scenes' if kind == 'main' else 'crossing_diagnostic'
            file = f'{folder}/scene_{i:06d}.npz'
            np.savez_compressed(root / file, **data)
            manifest['scenes'].append({'scene_id': i, 'kind': kind, 'subset': subset[i] if kind == 'main' else 'diagnostic_only',
                                       'targets': data['truth_state'].shape[1], 'file': file,
                                       'sha256': hashlib.sha256((root / file).read_bytes()).hexdigest()})
            if (i + 1) % 100 == 0 or i + 1 == count: print(f'generated {kind} {i + 1}/{count}', flush=True)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result = assess(root, cfg, manifest)
    (root / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    save_plots(root, manifest); write_results(root, cfg, manifest, result)
    print(json.dumps({'output': str(root), 'data_checks_passed': result['data_audit']['data_checks_passed'],
                      'main_pairs': result['data_audit']['pairs'], 'distance_baseline': result['distance_baseline']}, indent=2), flush=True)


if __name__ == '__main__': main()
