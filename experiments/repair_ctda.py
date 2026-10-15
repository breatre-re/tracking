"""Paired collapse diagnosis and correction on existing training/validation data.

Added balanced/structured objectives are experimental changes to the paper.
No held-out test data is used to select the correction.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ctda_model import CTDA, focal_loss_elements
from train_ctda import assignment
from generate_data import score, summarize_scores


class PairBlocks(Dataset):
    def __init__(self, folder, scenes=None, stride=1, seed=20261015):
        self.inputs = np.load(folder / 'inputs.npy', mmap_mode='r')
        self.labels = np.load(folder / 'labels.npy', mmap_mode='r')
        self.valid = np.load(folder / 'valid.npy', mmap_mode='r')
        self.origins = np.load(folder / 'origins.npy', mmap_mode='r')
        self.meta = np.load(folder / 'pair_metadata.npy')
        self.offsets = np.load(folder / 'pair_offsets.npy')
        available = np.unique(self.meta[:, 0])
        if scenes is not None:
            available = np.random.default_rng(seed).choice(available, min(scenes, len(available)), replace=False)
        self.scene_ids = available.tolist()
        self.indices = np.flatnonzero(np.isin(self.meta[:, 0], available) & (self.meta[:, 1] % stride == 0))

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        p = self.indices[index]; a, b = self.offsets[p:p + 2]
        return (self.inputs[a:b].copy(), self.labels[a:b].copy(), self.valid[a:b].copy(),
                self.origins[a:b].copy(), self.meta[p].copy())


def collate(items):
    inputs, labels, valid, groups = [], [], [], []
    maxrows = max(int(item[4][2]) for item in items)
    maxcols = max(int(item[4][3]) for item in items)
    targets = np.zeros((len(items), maxrows, maxcols), np.float32)
    masks = np.zeros_like(targets, bool)
    cursor = 0
    for index, (x, y, mask, origins, meta) in enumerate(items):
        sid, k, rows, cols = map(int, meta)
        inputs.append(x); labels.append(y); valid.append(mask)
        groups.append((cursor, cursor + len(x), origins, rows, cols, sid, k))
        cursor += len(x)
        masks[index, :rows, :cols] = True
        for label, (i, j) in zip(y, origins):
            nr, nc = min(40, rows - i), min(40, cols - j)
            targets[index, i:i + nr, j:j + nc] = label[:nr, :nc]
    return (torch.from_numpy(np.concatenate(inputs)), torch.from_numpy(np.concatenate(labels).astype(np.float32)),
            torch.from_numpy(np.concatenate(valid)), torch.from_numpy(targets), torch.from_numpy(masks), groups)


def reconstruct(logits, targets, groups):
    full = logits.new_zeros(targets.shape)
    for index, (start, end, origins, rows, cols, _, _) in enumerate(groups):
        for b, (i, j) in zip(range(start, end), origins):
            nr, nc = min(40, rows - i), min(40, cols - j)
            full[index, i:i + nr, j:j + nc] = logits[b, :nr, :nc]
    return full


def class_balanced_mean(values, labels, mask):
    pos, neg = mask & labels.bool(), mask & ~labels.bool()
    # Focal elements already include alpha=.75/.25; normalize each class
    # separately to prevent the number of negatives from setting the optimum.
    return (values[pos].sum() / pos.sum().clamp_min(1)
            + values[neg].sum() / neg.sum().clamp_min(1))


def row_class_mean(values, matched, valid):
    pos, neg = valid & matched, valid & ~matched
    count = (pos.any().float() + neg.any().float()).clamp_min(1)
    return (values[pos].sum() / pos.sum().clamp_min(1)
            + values[neg].sum() / neg.sum().clamp_min(1)) / count


def structured_terms(logits, targets, mask):
    # Categorical row/column labels include an explicit unmatched choice.
    # This penalizes missed true matches as well as selecting a wrong edge.
    b, m, n = logits.shape
    masked = logits.masked_fill(~mask, -10000.)
    row_logits = torch.cat((masked, logits.new_zeros((b, m, 1))), dim=-1)
    col_logits = torch.cat((masked.transpose(1, 2), logits.new_zeros((b, n, 1))), dim=-1)
    row_mass, col_mass = targets.sum(-1), targets.sum(-2)
    row_labels = torch.where(row_mass > 0, targets.argmax(-1), n)
    col_labels = torch.where(col_mass > 0, targets.argmax(-2), m)
    row_ce = torch.nn.functional.cross_entropy(row_logits.reshape(-1, n + 1), row_labels.reshape(-1), reduction='none').reshape(b, m)
    col_ce = torch.nn.functional.cross_entropy(col_logits.reshape(-1, m + 1), col_labels.reshape(-1), reduction='none').reshape(b, n)
    valid_rows, valid_cols = mask.any(-1), mask.any(-2)
    ce = .5 * (row_class_mean(row_ce, row_mass > 0, valid_rows) + row_class_mean(col_ce, col_mass > 0, valid_cols))
    p = torch.sigmoid(logits) * mask
    # Supervised cardinality consistency: true matched rows/cols want one,
    # truly unmatched ones want zero. Pure <=1 constraints allow all-zero.
    mass = .5 * (((p.sum(-1) - row_mass).square()[valid_rows].mean())
                 + ((p.sum(-2) - col_mass).square()[valid_cols].mean()))
    return ce, mass


def objective(logits, labels, valid, targets, full_mask, groups, mode, cardinality_weight=.01):
    elements = focal_loss_elements(logits, labels)
    if mode == 'legacy': return elements.mean(), {}
    if mode == 'masked_focal': return elements[valid].mean(), {}
    loss = class_balanced_mean(elements, labels, valid)
    if mode == 'balanced_structured':
        full = reconstruct(logits, targets, groups)
        ce, mass = structured_terms(full, targets, full_mask)
        return loss + .2 * ce + cardinality_weight * mass, {'balanced_focal': loss.detach(), 'row_column_ce': ce.detach(), 'cardinality_mse': mass.detach()}
    return loss, {'balanced_focal': loss.detach()}


def evaluate_pairs(model, dataset, device, batch=64):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=0)
    model.eval(); scores = []; pos_sum = neg_sum = 0.; pos_count = neg_count = 0
    with torch.inference_mode():
        for x, labels, valid, targets, mask, groups in loader:
            logits = model(x.to(device))
            full = reconstruct(logits, targets.to(device), groups).sigmoid().cpu().numpy()
            targets = targets.numpy()
            for i, (_, _, _, rows, cols, _, _) in enumerate(groups):
                p, truth = full[i, :rows, :cols], targets[i, :rows, :cols].astype(np.uint8)
                scores.append(score(truth, assignment(p, .2)))
                positives = truth.astype(bool)
                pos_sum += float(p[positives].sum()); pos_count += int(positives.sum())
                neg_sum += float(p[~positives].sum()); neg_count += int((~positives).sum())
    summary = summarize_scores(scores)
    summary.update(positive_probability_mean=pos_sum / pos_count, negative_probability_mean=neg_sum / neg_count)
    return summary


def checks():
    y = torch.zeros(1, 2, 2); y[0, 0, 0] = 1; mask = torch.ones_like(y, dtype=torch.bool)
    missing = torch.full_like(y, -10., requires_grad=True)
    correct = torch.tensor([[[10., -10.], [-10., -10.]]], requires_grad=True)
    ce0, mass0 = structured_terms(missing, y, mask); ce1, mass1 = structured_terms(correct, y, mask)
    assert ce0 > ce1 and mass0 > mass1
    (ce0 + mass0).backward(); assert missing.grad[0, 0, 0] < 0 and torch.isfinite(missing.grad).all()
    # Invalid/padding elements have exactly zero loss gradient under masking.
    logits = torch.zeros(1, 2, 2, requires_grad=True); valid = torch.tensor([[[True, True], [False, False]]])
    focal_loss_elements(logits, y)[valid].mean().backward(); assert (logits.grad[~valid] == 0).all()
    return {'true_match_omission_is_penalized': True, 'gradient_pushes_missed_true_logit_up': True,
            'padding_loss_gradient_zero': True, 'correct_assignment_has_lower_structure_loss': True}


def run(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    original = args.run
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(original / 'final.pt', map_location='cpu', weights_only=False)
    cache_root = Path(checkpoint['configuration'].get('cache_root', str(original / 'cache')))
    fit = PairBlocks(cache_root / 'fit', scenes=32, stride=5)
    validation = PairBlocks(cache_root / 'validation', scenes=32, stride=1)
    assert set(fit.scene_ids).isdisjoint(validation.scene_ids)
    protocol = {'fit_scenes': fit.scene_ids, 'validation_scenes': validation.scene_ids,
                'fit_pairs': len(fit), 'validation_pairs': len(validation), 'steps_per_case': args.steps,
                'pairs_per_batch': 64, 'seed': 20261015, 'optimizer': 'Adam', 'learning_rate': 1e-4,
                'warm_start': str(original / 'final.pt'), 'held_out_test_used': False,
                'structure_ce_weight': .2, 'cardinality_mse_weight': args.cardinality_weight,
                'checks': checks(),
                'paper_deviations': ['class-wise focal normalization and added structured losses are experimental',
                                     'residual completion is an alternative interpretation of incomplete Eq17']}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    cases = [('A_legacy', 'legacy', False), ('B_masked_focal', 'masked_focal', False),
             ('C_balanced', 'balanced', False), ('D_balanced_structure', 'balanced_structured', False),
             ('E_residual_masked', 'masked_focal', True), ('F_residual_balanced_structure', 'balanced_structured', True)]
    if args.cases:
        requested = set(args.cases.split(','))
        assert requested <= {case[0] for case in cases}
        cases = [case for case in cases if case[0] in requested]
    results = {}; started = time.perf_counter()
    for name, mode, residuals in cases:
        torch.manual_seed(20261015); np.random.seed(20261015)
        config = dict(checkpoint['configuration']['model']); config['channel_residuals'] = residuals
        model = CTDA(**config).to(device); model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loader = DataLoader(fit, batch_size=64, shuffle=True, collate_fn=collate, num_workers=0,
                            generator=torch.Generator().manual_seed(20261015))
        iterator = iter(loader); model.train(); losses = []; case_start = time.perf_counter()
        for step in range(1, args.steps + 1):
            try: batch = next(iterator)
            except StopIteration: iterator = iter(loader); batch = next(iterator)
            x, labels, valid, targets, mask, groups = batch
            x, labels, valid = x.to(device), labels.to(device), valid.to(device)
            targets, mask = targets.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, _ = objective(logits, labels, valid, targets, mask, groups, mode, args.cardinality_weight)
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step(); losses.append(float(loss.item()))
            if step % 100 == 0: print(f'{name} step {step}/{args.steps} objective {np.mean(losses[-100:]):.5f} elapsed {time.perf_counter()-case_start:.1f}s', flush=True)
        metrics = evaluate_pairs(model, validation, device)
        results[name] = {'objective': mode, 'channel_residuals': residuals, 'validation': metrics,
                         'last_100_step_objective_mean': float(np.mean(losses[-100:])), 'elapsed_seconds': time.perf_counter() - case_start}
        torch.save({'model_state_dict': model.state_dict(), 'model_config': config, 'objective': mode,
                    'steps': args.steps, 'validation': metrics, 'protocol': protocol}, output / f'{name}.pt')
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(name + ' validation ' + json.dumps(metrics), flush=True)
    (output / 'elapsed_seconds.json').write_text(json.dumps({'seconds': time.perf_counter() - started}) + '\n')
    rows = [f"| {name} | {r['objective']} | {r['channel_residuals']} | {r['validation']['accuracy_mean']:.2%} | {r['validation']['positive_precision']:.2%} | {r['validation']['positive_recall']:.2%} | {r['validation']['positive_f1']:.2%} |"
            for name, r in results.items()]
    report = f"""# CTDA 退化诊断：小样本配对对照

从 `{original / 'final.pt'}` 的第 8 轮同一权重开始，每组训练 {args.steps} 步、相同批次顺序。32 个 fit 场景共 {len(fit)} 对，32 个独立验证场景共 {len(validation)} 对，未使用测试集选择修改。

| 对照 | 训练目标 | 补全通道残差 | Accuracy | 正关联 Precision | 正关联 Recall | 正关联 F1 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

- A→B：只去掉 padding 对 loss 的贡献。
- B→C：进一步分别按正负类数量归一化 Focal Loss。
- C→D：加入整张关联矩阵的行/列分类（带未匹配类）和监督的匹配数量惩罚。
- B→E、D→F：单独检查式 (17) 未写出的通道 decoder 残差。
- 结构约束在还原后的完整关联矩阵计算，跨分块的一行/一列不会被当成独立目标。

单纯要求行列和≤1，仍允许全零矩阵；这里的行/列分类及匹配数量惩罚还要求恢复真值中存在的匹配。padding 不产生分类梯度，检查见 [协议](protocol.json)。新增损失超出论文式 (19)，结果属于修正诊断，不可冒充原论文训练目标的严格复现。实际数字见 [结果](results.json)。
"""
    (output / 'REPORT.md').write_text(report, encoding='utf-8')
    print('DIAGNOSIS COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('results/ctda_first_run'))
    parser.add_argument('--output', type=Path, default=Path('results/ctda_collapse_diagnosis'))
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--cases', default='')
    parser.add_argument('--cardinality-weight', type=float, default=.01)
    args = parser.parse_args()
    run(args)
