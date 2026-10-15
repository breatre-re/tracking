"""CNN/CTDA comparison on identical cached RDF, labels and evaluation code."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_model import AssociationCNN, runtime_check
from ctda_model import focal_loss_elements
from train_ctda import Blocks, evaluate, plot_history


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main(args):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mttracking-mpl')
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    np.random.seed(20261013)
    device = torch.device(args.device)
    if device.type == 'cuda':
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(20261013)
        torch.cuda.reset_peak_memory_stats(device)
    if args.check_only:
        print(json.dumps(runtime_check(device), indent=2), flush=True)
        return
    reference = json.loads((args.reference / 'run_config.json').read_text())
    assert reference['epochs'] == 40 and reference['loss_mode'] == 'masked_focal'
    cache = Path(reference['cache_root'])
    manifest_hash = hashlib.sha256((Path(reference['data']) / 'manifest.json').read_bytes()).hexdigest()
    metadata = {}
    for split in ['fit', 'validation', 'test']:
        metadata[split] = json.loads((cache / split / 'metadata.json').read_text())
        assert metadata[split]['manifest_sha256'] == manifest_hash and metadata[split]['variant'] == reference['variant']
    fit_scenes = set(np.load(cache / 'fit' / 'pair_metadata.npy')[:, 0].tolist())
    val_scenes = set(np.load(cache / 'validation' / 'pair_metadata.npy')[:, 0].tolist())
    test_scenes = set(np.load(cache / 'test' / 'pair_metadata.npy')[:, 0].tolist())
    assert not fit_scenes & val_scenes and not fit_scenes & test_scenes and not val_scenes & test_scenes
    args.output.mkdir(parents=True, exist_ok=False)
    model = AssociationCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=reference['learning_rate'])
    parameters = sum(p.numel() for p in model.parameters())
    cfg = {key: reference[key] for key in ['data', 'variant', 'seed', 'batch_size', 'optimizer', 'learning_rate',
                                          'focal_alpha', 'focal_gamma', 'association_threshold', 'training_loss_reduction',
                                          'loss_mode', 'cache_root']}
    cfg.update({'model_type': 'plain_dense_cnn', 'model': model.config, 'epochs': 40,
                'parameter_count': parameters, 'ctda_parameter_count': reference['parameter_count'],
                'parameter_difference_fraction': parameters / reference['parameter_count'] - 1,
                'device': args.device, 'reference_ctda_run': str(args.reference), 'torch_version': torch.__version__,
                'cuda_build': torch.version.cuda, 'mixed_precision': False,
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'architecture_origin': 'Locally specified plain CNN; paper Table4 does not specify complete CNN architecture',
                'controlled_variables': 'Same cached RDF/labels/valid masks, scene splits, optimizer/lr/batch/focal, threshold and global assignment',
                'test_policy': 'Fixed epoch8 and epoch40 checkpoints; evaluate only after all40 epochs, no test-based selection',
                'manifest_sha256': manifest_hash})
    if device.type == 'cuda': cfg['gpu_name'] = torch.cuda.get_device_name(device)
    for name in ['cnn_model.py', 'train_cnn.py', 'train_ctda.py', 'ctda_model.py']:
        source = Path(__file__).with_name(name).read_bytes()
        (args.output / f'source_{name}').write_bytes(source)
        cfg[f'{name}_sha256'] = hashlib.sha256(source).hexdigest()
    write_json(args.output / 'run_config.json', cfg)
    write_json(args.output / 'cache_summary.json', metadata)
    dataset = Blocks(cache / 'fit')
    generator = torch.Generator().manual_seed(reference['seed'])
    loader = DataLoader(dataset, batch_size=reference['batch_size'], shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0, generator=generator)
    threshold = reference['association_threshold']
    started = time.perf_counter()
    history = []
    initial = evaluate(model, cache / 'validation', device, threshold, workers=args.workers)
    write_json(args.output / 'initial_validation.json', initial)
    print('PROTOCOL ' + json.dumps(cfg), flush=True)
    print('INITIAL ' + json.dumps(initial), flush=True)
    best_loss = float('inf')
    for epoch in range(1, 41):
        model.train()
        total_loss = 0.
        total_elements = 0
        epoch_started = time.perf_counter()
        for step, (x, y, valid) in enumerate(loader, 1):
            x, y, valid = x.to(device, non_blocking=True), y.to(device, non_blocking=True), valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            elements = focal_loss_elements(model(x), y, cfg['focal_alpha'], cfg['focal_gamma'])
            loss = elements[valid].mean()
            assert torch.isfinite(loss), (epoch, step)
            loss.backward()
            optimizer.step()
            count = int(valid.sum().item())
            total_loss += loss.item() * count
            total_elements += count
            if step % 400 == 0 or step == len(loader):
                print(f'epoch {epoch}/40 step {step}/{len(loader)} loss {total_loss / total_elements:.6f}', flush=True)
        validation = evaluate(model, cache / 'validation', device, threshold, workers=args.workers)
        record = {'epoch': epoch, 'train_loss': total_loss / total_elements, 'validation': validation,
                  'elapsed_seconds': time.perf_counter() - epoch_started}
        history.append(record)
        checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch, 'configuration': cfg, 'validation': validation, 'torch_rng_state': torch.get_rng_state(),
                      'loader_generator_state': generator.get_state()}
        if device.type == 'cuda': checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state(device)
        torch.save(checkpoint, args.output / 'last.pt')
        if epoch == 8: torch.save(checkpoint, args.output / 'epoch_08.pt')
        if validation['focal_loss_valid_elements'] < best_loss:
            best_loss = validation['focal_loss_valid_elements']
            torch.save(checkpoint, args.output / 'best_validation.pt')
        write_json(args.output / 'history.json', history)
        print('EPOCH ' + json.dumps(record), flush=True)
    torch.save(checkpoint, args.output / 'final.pt')
    training_seconds = time.perf_counter() - started
    test40 = evaluate(model, cache / 'test', device, threshold, workers=args.workers, save_examples=args.output)
    old = torch.load(args.output / 'epoch_08.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(old['model_state_dict'])
    test8 = evaluate(model, cache / 'test', device, threshold, workers=args.workers)
    summary = {'workflow_completed': True, 'epochs_completed': 40, 'training_and_validation_seconds': training_seconds,
               'total_seconds': time.perf_counter() - started, 'initial_validation': initial,
               'final_validation': history[-1]['validation'], 'test': test40, 'test_epoch8': test8,
               'parameter_count': parameters, 'checkpoint': 'final.pt', 'test_selection': 'Fixed8 and40; no tuning on test'}
    if device.type == 'cuda': summary['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
    write_json(args.output / 'summary.json', summary)
    plot_history(args.output, history, masked=True)
    with (args.output / 'history.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_valid_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1'])
        for h in history:
            v = h['validation']
            writer.writerow([h['epoch'], h['train_loss'], v['focal_loss_valid_elements'], v['accuracy_mean'],
                             v['positive_precision'], v['positive_recall'], v['positive_f1']])
    lines = ['# 同数据 CNN 基线', '', '传统 CNN：4 个 5×5 卷积层，通道 3→64→128→128→64，每层 ReLU／dropout=.3；1×1 输出层生成 40×40 logits。无注意力、池化或残差，局部感受野 17×17。', '',
             f'参数 {parameters:,}，与 CTDA 的 {reference["parameter_count"]:,} 相差 {cfg["parameter_difference_fraction"]:+.2%}。论文未提供完整 CNN 配置，此网络是自行明确的对照模型。', '',
             '复用 CTDA 的原始 RDF 缓存、标签、有效掩码、场景划分、Adam／学习率／batch／Focal Loss 及阈值 .2／全局匹配。固定训练 40 轮，8 轮和 40 轮结果均在训练完成后评价，未使用测试选择配置。', '',
             '| CNN 完整测试 | Accuracy | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for epoch, value in [(8, test8), (40, test40)]:
        lines.append(f'| {epoch} 轮 | ' + ' | '.join(f'{value[k]:.2%}' for k in ['accuracy_mean', 'positive_precision', 'positive_recall', 'positive_f1']) + ' |')
    lines.extend(['', '[配置](run_config.json)、[完整数值](summary.json)、[曲线](training_curve.png)、[日志](history.csv)、[8 轮权重](epoch_08.pt)、[40 轮权重](final.pt)。'])
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('COMPLETED ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--reference', type=Path, default=Path('results/ctda_extended_40'))
    parser.add_argument('--output', type=Path, default=Path('results/cnn_same_data_40'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    main(parser.parse_args())
