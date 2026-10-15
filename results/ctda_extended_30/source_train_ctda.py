"""One complete CTDA experiment: RDF preparation, training, evaluation, weights."""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

from ctda_model import CTDA, focal_loss_elements
from generate_data import baseline, pair, rdf_blocks, score, summarize_scores


def prepare_cache(data_root, output, variant):
    manifest = json.loads((data_root / 'manifest.json').read_text())
    cfg = json.loads((data_root / 'config.json').read_text())
    entries = {e['scene_id']: e for e in manifest['scenes'] if e['kind'] == 'main'}
    source_hash = hashlib.sha256((data_root / 'manifest.json').read_bytes()).hexdigest()
    root = output / 'cache'
    root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split, stride in [('fit', 1), ('validation', 10), ('test', 1)]:
        folder = root / split
        folder.mkdir(exist_ok=True)
        stamp = folder / 'metadata.json'
        if stamp.exists():
            saved = json.loads(stamp.read_text())
            assert saved['manifest_sha256'] == source_hash and saved['variant'] == variant
            summaries[split] = saved
            continue
        scene_ids = sorted(manifest['split'][split])
        pairs, blocks = 0, 0
        for sid in scene_ids:
            with np.load(data_root / entries[sid]['file'], allow_pickle=False) as raw:
                counts = np.diff(raw['observation_offsets'])
            for k in range(0, cfg['frames'] - 1, stride):
                pairs += 1
                blocks += math.ceil(int(counts[k]) / 40) * math.ceil(int(counts[k + 1]) / 40)
        inputs = np.lib.format.open_memmap(folder / 'inputs.npy', mode='w+', dtype=np.float32, shape=(blocks, 3, 40, 40))
        labels = np.lib.format.open_memmap(folder / 'labels.npy', mode='w+', dtype=np.uint8, shape=(blocks, 40, 40))
        valid = np.lib.format.open_memmap(folder / 'valid.npy', mode='w+', dtype=np.bool_, shape=(blocks, 40, 40))
        origins = np.lib.format.open_memmap(folder / 'origins.npy', mode='w+', dtype=np.int16, shape=(blocks, 2))
        metadata = np.empty((pairs, 4), np.int32)
        offsets = np.empty(pairs + 1, np.int64)
        bidx, pidx, positives = 0, 0, 0
        for number, sid in enumerate(scene_ids):
            with np.load(data_root / entries[sid]['file'], allow_pickle=False) as raw:
                data = {k: raw[k] for k in raw.files}
            for k in range(0, cfg['frames'] - 1, stride):
                x, y, truth = pair(data, k, variant)
                metadata[pidx] = [sid, k, len(x), len(y)]
                offsets[pidx] = bidx
                for features, target, mask, i, j in rdf_blocks(x, y, truth, cfg['rdf_bias_m'], cfg['partition_size']):
                    inputs[bidx], labels[bidx], valid[bidx], origins[bidx] = features, target, mask, [i, j]
                    positives += int(target.sum()); bidx += 1
                pidx += 1
            if (number + 1) % 200 == 0: print(f'cache {split}: {number + 1}/{len(scene_ids)} scenes', flush=True)
        offsets[pidx] = bidx
        assert bidx == blocks and pidx == pairs
        for array in [inputs, labels, valid, origins]: array.flush()
        np.save(folder / 'pair_metadata.npy', metadata)
        np.save(folder / 'pair_offsets.npy', offsets)
        summaries[split] = {'manifest_sha256': source_hash, 'variant': variant, 'scenes': len(scene_ids),
                            'pair_stride': stride, 'pairs': pairs, 'blocks': blocks,
                            'positive_labels': positives, 'input_dtype': 'float32', 'padding_label': 0}
        stamp.write_text(json.dumps(summaries[split], indent=2) + '\n')
        print(f'cache {split} complete: {pairs} pairs / {blocks} blocks', flush=True)
    return summaries


class Blocks(Dataset):
    def __init__(self, folder):
        self.inputs = np.load(folder / 'inputs.npy', mmap_mode='r')
        self.labels = np.load(folder / 'labels.npy', mmap_mode='r')
        self.valid = np.load(folder / 'valid.npy', mmap_mode='r')

    def __len__(self): return len(self.inputs)

    def __getitem__(self, index):
        return (torch.from_numpy(self.inputs[index].copy()),
                torch.from_numpy(self.labels[index].astype(np.float32)),
                torch.from_numpy(self.valid[index].copy()))


def assignment(probability, threshold):
    rows, cols = probability.shape
    result = np.zeros((rows, cols), np.uint8)
    if not rows or not cols: return result
    costs = np.concatenate((np.where(probability >= threshold, -probability, 1e6), np.zeros((rows, rows))), axis=1)
    r, c = linear_sum_assignment(costs)
    keep = c < cols
    result[r[keep], c[keep]] = 1
    assert (result.sum(0) <= 1).all() and (result.sum(1) <= 1).all()
    return result


def evaluate(model, folder, device, threshold, batch_size=512, workers=4, save_examples=None):
    dataset = Blocks(folder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == 'cuda')
    probabilities = np.empty((len(dataset), 40, 40), np.float32)
    model.eval()
    elements_sum = valid_sum = 0.
    elements_count = valid_count = 0
    offset = 0
    with torch.inference_mode():
        for x, y, mask in loader:
            x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            logits = model(x)
            losses = focal_loss_elements(logits, y)
            elements_sum += losses.sum().item(); elements_count += losses.numel()
            valid_sum += losses[mask].sum().item(); valid_count += int(mask.sum().item())
            p = torch.sigmoid(logits).cpu().numpy()
            probabilities[offset:offset + len(p)] = p; offset += len(p)
    pair_meta = np.load(folder / 'pair_metadata.npy')
    offsets = np.load(folder / 'pair_offsets.npy')
    origins = np.load(folder / 'origins.npy', mmap_mode='r')
    metrics = []
    weight_correct, weight_total = 0., 0
    examples = []
    for index, (sid, k, rows, cols) in enumerate(pair_meta):
        matrix, truth = np.zeros((rows, cols), np.float32), np.zeros((rows, cols), np.uint8)
        for b in range(offsets[index], offsets[index + 1]):
            i, j = origins[b]; nr, nc = min(40, rows - i), min(40, cols - j)
            matrix[i:i + nr, j:j + nc] = probabilities[b, :nr, :nc]
            truth[i:i + nr, j:j + nc] = dataset.labels[b, :nr, :nc]
        predicted = assignment(matrix, threshold)
        value = score(truth, predicted); metrics.append(value)
        weight_correct += value['accuracy'] * (rows + cols); weight_total += rows + cols
        if save_examples is not None and len(examples) < 3:
            examples.append({'scene_id': int(sid), 'frame': int(k), 'probability': matrix, 'truth': truth, 'prediction': predicted})
    total = summarize_scores(metrics)
    total.update({'focal_loss_including_padding': elements_sum / elements_count,
                  'focal_loss_valid_elements': valid_sum / valid_count,
                  'accuracy_row_column_weighted': weight_correct / weight_total,
                  'association_threshold': threshold})
    if save_examples is not None:
        for n, example in enumerate(examples):
            np.savez_compressed(save_examples / f'test_association_{n}.npz', **example)
    return total


def runtime_check():
    torch.set_num_threads(2); torch.manual_seed(20261013)
    model = CTDA()
    x = torch.randn(2, 3, 40, 40)
    y = torch.eye(40).expand(2, -1, -1)
    output = model(x); assert output.shape == (2, 40, 40)
    loss = focal_loss_elements(output, y).mean(); loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    nonfinite = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not missing and not nonfinite and torch.isfinite(loss)
    p = np.array([[.9, .3], [.8, .1]], np.float32)
    pred = assignment(p, .2)
    assert pred.sum() == 2 and (pred.sum(0) <= 1).all() and (pred.sum(1) <= 1).all()
    assert not assignment(np.full((3, 4), .1), .2).any()
    print(json.dumps({'model_output_shape': list(output.shape), 'parameters': sum(p.numel() for p in model.parameters()),
                      'all_parameters_receive_finite_gradients': True, 'matching_checks': True}, indent=2))


def plot_history(output, history, masked=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [h['epoch'] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, [h['train_loss'] for h in history], marker='o', label='Train focal loss')
    loss_key = 'focal_loss_valid_elements' if masked else 'focal_loss_including_padding'
    ax[0].plot(epochs, [h['validation'][loss_key] for h in history], marker='o', label='Validation focal loss')
    ax[0].set(xlabel='Epoch', ylabel='Element mean focal loss', yscale='log')
    ax[1].plot(epochs, [h['validation']['accuracy_mean'] for h in history], marker='o', label='Paper accuracy')
    ax[1].plot(epochs, [h['validation']['positive_f1'] for h in history], marker='o', label='Positive F1')
    ax[1].set(xlabel='Epoch', ylabel='Validation score', ylim=(0, 1.02))
    for a in ax: a.grid(alpha=.2); a.legend()
    fig.tight_layout(); fig.savefig(output / 'training_curve.png', dpi=160); plt.close(fig)


def test_distance_baseline(data_root, variant):
    manifest = json.loads((data_root / 'manifest.json').read_text())
    metrics = []
    for entry in manifest['scenes']:
        if entry['kind'] != 'main' or entry['subset'] != 'test': continue
        with np.load(data_root / entry['file'], allow_pickle=False) as saved: data = {k: saved[k] for k in saved.files}
        for k in range(len(data['time_s']) - 1):
            x, y, truth = pair(data, k, variant)
            metrics.append(score(truth, baseline(x, y, 10000.)))
    return summarize_scores(metrics)


def run(args):
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    if args.check_only: runtime_check(); return
    cache_root = args.cache_root or output / 'cache'
    if args.cache_root is None:
        cache = prepare_cache(args.data, output, args.variant)
    else:
        expected_hash = hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest()
        cache = {split: json.loads((cache_root / split / 'metadata.json').read_text()) for split in ['fit', 'validation', 'test']}
        for value in cache.values():
            assert value['manifest_sha256'] == expected_hash and value['variant'] == args.variant
    (output / 'cache_summary.json').write_text(json.dumps(cache, indent=2) + '\n')
    if args.prepare_only: return
    if (output / 'final.pt').exists(): raise SystemExit('Run already complete; use a different output directory.')
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        assert torch.cuda.is_available(), 'CUDA device is not visible in this execution context.'
        torch.cuda.set_device(device); torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = CTDA(channel_residuals=args.channel_residuals).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    configuration = {'data': str(args.data), 'variant': args.variant, 'model': model.config,
                     'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
                     'optimizer': 'Adam', 'learning_rate': 1e-4, 'focal_alpha': .75, 'focal_gamma': 2,
                     'association_threshold': .2, 'training_loss_reduction': ('mean of valid elements, padding excluded' if args.loss_mode == 'masked_focal' else 'mean of every element including padding'),
                     'loss_mode': args.loss_mode, 'cache_root': str(cache_root),
                     'attention_padding_mask': False, 'device': args.device,
                     'parameter_count': sum(p.numel() for p in model.parameters()), 'torch_version': torch.__version__,
                     'cuda_build': torch.version.cuda, 'mixed_precision': False,
                     'assumptions': ['d_model=40, FFN=160, ReLU; independent weights across branches and channels',
                                     '2 encoder/decoder layers per row-column branch, 1 per channel branch',
                                     'Eq12/13 printed cross Q/K/V; Eq17 residual interpretation recorded by channel_residuals',
                                     'padding treatment recorded by loss_mode; paper reduction was not reported',
                                     'probability-sum assignment with dummy unmatched choices, threshold=.2',
                                     'training uses 1620 fit scenes; validation 180 from the 1800 paper training pool',
                                     'one seed for first complete run; simulated data assumptions remain'],
                     'model_source_sha256': hashlib.sha256(Path(__file__).with_name('ctda_model.py').read_bytes()).hexdigest(),
                     'trainer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if device.type == 'cuda': configuration['gpu_name'] = torch.cuda.get_device_name(device)
    (output / 'run_config.json').write_text(json.dumps(configuration, indent=2) + '\n')
    dataset = Blocks(cache_root / 'fit')
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0, generator=generator)
    started = time.perf_counter(); history = []
    print(f'Training CTDA: {configuration["parameter_count"]} parameters, {len(dataset)} blocks, {len(loader)} steps/epoch', flush=True)
    before = evaluate(model, cache_root / 'validation', device, .2, workers=args.workers)
    (output / 'initial_validation.json').write_text(json.dumps(before, indent=2) + '\n')
    print('initial validation ' + json.dumps(before), flush=True)
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss, total_samples = 0., 0
        epoch_start = time.perf_counter()
        for step, (x, y, valid) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            elements = focal_loss_elements(logits, y)
            loss = elements[valid].mean() if args.loss_mode == 'masked_focal' else elements.mean()
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at epoch {epoch} step {step}')
            loss.backward(); optimizer.step()
            weight = int(valid.sum().item()) if args.loss_mode == 'masked_focal' else len(x)
            total_loss += loss.item() * weight; total_samples += weight
            if step % 100 == 0 or step == len(loader):
                print(f'epoch {epoch}/{args.epochs} step {step}/{len(loader)} loss {total_loss / total_samples:.6f} elapsed {time.perf_counter() - epoch_start:.1f}s', flush=True)
        validation = evaluate(model, cache_root / 'validation', device, .2, workers=args.workers)
        record = {'epoch': epoch, 'train_loss': total_loss / total_samples, 'validation': validation,
                  'elapsed_seconds': time.perf_counter() - epoch_start}
        history.append(record)
        checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch, 'configuration': configuration, 'validation': validation}
        torch.save(checkpoint, output / 'last.pt')
        selection_key = 'focal_loss_valid_elements' if args.loss_mode == 'masked_focal' else 'focal_loss_including_padding'
        if validation[selection_key] < best_loss:
            best_loss = validation[selection_key]; torch.save(checkpoint, output / 'best_validation.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(f'epoch {epoch} validation ' + json.dumps(validation), flush=True)
    # Fixed final epoch is the primary result, not test-based model selection.
    torch.save(checkpoint, output / 'final.pt')
    test = evaluate(model, cache_root / 'test', device, .2, workers=args.workers, save_examples=output)
    distance = test_distance_baseline(args.data, args.variant)
    summary = {'workflow_completed': True, 'epochs_completed': args.epochs, 'initial_validation': before,
               'final_validation': history[-1]['validation'], 'test': test, 'test_distance_baseline': distance,
               'training_and_evaluation_seconds': time.perf_counter() - started,
               'checkpoint': 'final.pt', 'paper_accuracy_reference': .9833,
               'accuracy_gap_percentage_points': 100 * (test['accuracy_mean'] - .9833),
               'caveat': 'First complete run with explicit assumptions; not proof of strict reproduction.'}
    if device.type == 'cuda': summary['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot_history(output, history, masked=args.loss_mode == 'masked_focal')
    with (output / 'history.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1'])
        for h in history:
            v = h['validation']; writer.writerow([h['epoch'], h['train_loss'], v['focal_loss_including_padding'], v['accuracy_mean'], v['positive_precision'], v['positive_recall'], v['positive_f1']])
    report = f"""# 首轮完整 CTDA 实验

**仿真数据 → RDF 分块 → CTDA 训练 → 测试关联 → 权重保存已全部完成。** 使用现有 v1 距离噪声数据，单个训练种子，{args.epochs} epochs。

| 指标 | 训练前验证 | 训练后验证 | 最终测试 |
|---|---:|---:|---:|
| 论文 Accuracy | {before['accuracy_mean']:.2%} | {history[-1]['validation']['accuracy_mean']:.2%} | {test['accuracy_mean']:.2%} |
| 正关联 Precision | {before['positive_precision']:.2%} | {history[-1]['validation']['positive_precision']:.2%} | {test['positive_precision']:.2%} |
| 正关联 Recall | {before['positive_recall']:.2%} | {history[-1]['validation']['positive_recall']:.2%} | {test['positive_recall']:.2%} |
| 正关联 F1 | {before['positive_f1']:.2%} | {history[-1]['validation']['positive_f1']:.2%} | {test['positive_f1']:.2%} |
| Focal Loss（有效区域元素均值） | {before['focal_loss_valid_elements']:.6f} | {history[-1]['validation']['focal_loss_valid_elements']:.6f} | {test['focal_loss_valid_elements']:.6f} |
| Focal Loss（含 padding 元素均值） | {before['focal_loss_including_padding']:.6f} | {history[-1]['validation']['focal_loss_including_padding']:.6f} | {test['focal_loss_including_padding']:.6f} |

训练 {cache['fit']['scenes']} 场景、{cache['fit']['pairs']:,} 对、{cache['fit']['blocks']:,} 块；验证 {cache['validation']['scenes']} 场景、{cache['validation']['pairs']:,} 对；测试 {cache['test']['scenes']} 场景、{cache['test']['pairs']:,} 对。最终结果使用固定第 {args.epochs} 轮权重；测试不参与阈值或权重选择。

本轮 loss_mode={args.loss_mode}，channel_residuals={args.channel_residuals}。训练目标保留 α=.75、γ=2 的 Focal Loss，没有新增行列分类或匹配数量惩罚；一对一约束在推理时的全局分配中执行。

论文参考 Accuracy 98.33%；本次差值 {summary['accuracy_gap_percentage_points']:+.2f} 个百分点。当前数据上的距离基线测试 Accuracy {distance['accuracy_mean']:.2%}、正关联 F1 {distance['positive_f1']:.2%}。数据与模型存在已记录的补全假设，不能仅凭结果数值声称严格复现。Loss 0.0067 的原归约方式未报告，本次 loss 口径已经明确。

训练与评价用时 {summary['training_and_evaluation_seconds'] / 60:.2f} 分钟（不含 RDF 缓存准备）。[配置](run_config.json)、[数值结果](summary.json)、[训练曲线](training_curve.png)、[逐轮日志](history.csv)、[最终权重](final.pt)、[最低验证 loss 权重](best_validation.pt)、[关联输出示例](test_association_0.npz)。

重跑命令：

```bash
MPLCONFIGDIR=/tmp/mttracking-mpl /home/chenrui/miniconda3/envs/mttrack-repro/bin/python experiments/train_ctda.py --device cuda:4 --output {args.output}_repeat --loss-mode {args.loss_mode}{' --channel-residuals' if args.channel_residuals else ''}
```
"""
    (output / 'REPORT.md').write_text(report, encoding='utf-8')
    print('COMPLETED ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/ctda_3d_v1'))
    parser.add_argument('--output', type=Path, default=Path('results/ctda_first_run'))
    parser.add_argument('--variant', choices=['range_only', 'cartesian_xyz'], default='range_only')
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--loss-mode', choices=['legacy', 'masked_focal'], default='legacy')
    parser.add_argument('--channel-residuals', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    run(parser.parse_args())
