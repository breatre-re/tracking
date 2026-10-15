"""Controlled epoch extension of an existing CTDA run; no new data or losses."""
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

from ctda_model import CTDA, focal_loss_elements
from train_ctda import Blocks, evaluate, plot_history


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def fixed_fit_subset(source, destination):
    """64 spread-out fit scenes, every tenth pair; copy complete pair blocks."""
    destination.mkdir(parents=True, exist_ok=False)
    meta = np.load(source / 'pair_metadata.npy')
    offsets = np.load(source / 'pair_offsets.npy')
    scenes = np.unique(meta[:, 0])
    chosen = scenes[np.linspace(0, len(scenes) - 1, min(64, len(scenes)), dtype=int)]
    selected = np.flatnonzero(np.isin(meta[:, 0], chosen) & (meta[:, 1] % 10 == 0))
    block_indices = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in selected])
    for name in ['inputs', 'labels', 'valid', 'origins']:
        array = np.load(source / f'{name}.npy', mmap_mode='r')
        np.save(destination / f'{name}.npy', array[block_indices])
    np.save(destination / 'pair_metadata.npy', meta[selected])
    new_offsets = np.r_[0, np.cumsum(offsets[selected + 1] - offsets[selected])]
    np.save(destination / 'pair_offsets.npy', new_offsets)
    specification = {'source': str(source), 'selection': '64 evenly spaced sorted fit scene IDs, frame % 10 == 0',
                     'scene_ids': chosen.tolist(), 'pairs': len(selected), 'blocks': len(block_indices),
                     'purpose': 'Fixed train diagnostics in eval mode, not validation/model selection'}
    write_json(destination / 'metadata.json', specification)
    return specification


def main(args):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mttracking-mpl')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    saved = torch.load(args.resume, map_location='cpu', weights_only=False)
    start_epoch = saved['epoch']
    assert args.epochs > start_epoch
    original = saved['configuration']
    assert original['loss_mode'] == 'masked_focal' and original['model']['channel_residuals']
    model_hash = hashlib.sha256(Path(__file__).with_name('ctda_model.py').read_bytes()).hexdigest()
    assert model_hash == original['model_source_sha256'], 'Model source changed; epoch-only comparison invalid.'
    cache_root = Path(original['cache_root'])
    manifest_hash = hashlib.sha256((Path(original['data']) / 'manifest.json').read_bytes()).hexdigest()
    for split in ['fit', 'validation', 'test']:
        metadata = json.loads((cache_root / split / 'metadata.json').read_text())
        assert metadata['manifest_sha256'] == manifest_hash and metadata['variant'] == original['variant']
    model = CTDA(**original['model']).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=original['learning_rate'])
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    assert len(optimizer.state) == len(saved['optimizer_state_dict']['state'])
    saved_steps = [int(state['step'].item()) for state in saved['optimizer_state_dict']['state'].values()]
    assert len(set(saved_steps)) == 1
    assert all(int(state['step'].item()) == saved_steps[0] for state in optimizer.state.values())
    if 'torch_rng_state' in saved:
        torch.set_rng_state(saved['torch_rng_state'].cpu())
        if device.type == 'cuda' and 'cuda_rng_state' in saved:
            torch.cuda.set_rng_state(saved['cuda_rng_state'].cpu(), device)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    train_folder = output / 'fixed_train_subset'
    subset = fixed_fit_subset(cache_root / 'fit', train_folder)
    validation_scenes = np.unique(np.load(cache_root / 'validation' / 'pair_metadata.npy')[:, 0])
    assert not np.intersect1d(subset['scene_ids'], validation_scenes).size
    configuration = dict(original)
    configuration.update({'epochs': args.epochs, 'device': args.device, 'resume_checkpoint': str(args.resume),
                          'resume_checkpoint_sha256': hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                          'resume_epoch': start_epoch, 'resume_optimizer_step': saved_steps[0],
                          'continuation_seed': args.seed, 'fixed_train_subset': subset,
                          'rng_recovery': ('Saved torch/CUDA/loader states restored' if 'torch_rng_state' in saved
                                           else 'Original checkpoint omitted RNG states; restarted torch/loader RNG with recorded seed'),
                          'epoch_selection': f'Fixed total epoch {args.epochs}; test used only at the end',
                          'extension_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write_json(output / 'run_config.json', configuration)
    for filename in ['continue_ctda.py', 'ctda_model.py', 'train_ctda.py']:
        (output / f'source_{filename}').write_bytes(Path(__file__).with_name(filename).read_bytes())
    generator = torch.Generator().manual_seed(args.seed)
    if 'loader_generator_state' in saved:
        generator.set_state(saved['loader_generator_state'].cpu())
    dataset = Blocks(cache_root / 'fit')
    loader = DataLoader(dataset, batch_size=original['batch_size'], shuffle=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=device.type == 'cuda', generator=generator)
    threshold = original['association_threshold']
    started = time.perf_counter()
    before = evaluate(model, cache_root / 'validation', device, threshold, workers=args.workers)
    previous = saved['validation']
    diffs = {key: abs(before[key] - previous[key]) for key in
             ['accuracy_mean', 'positive_precision', 'positive_recall', 'positive_f1', 'focal_loss_valid_elements']}
    assert max(diffs.values()) < 1e-6, f'Resume validation mismatch: {diffs}'
    fit_before = evaluate(model, train_folder, device, threshold, workers=args.workers)
    resume_check = {'model_loaded_strictly': True, 'optimizer_states_restored': len(optimizer.state),
                    'optimizer_step_restored': saved_steps[0], 'original_validation_differences': diffs,
                    'batch_size': original['batch_size'], 'steps_per_epoch': len(loader),
                    'optimizer_learning_rates': [g['lr'] for g in optimizer.param_groups],
                    'padding_excluded_from_loss': True}
    write_json(output / 'resume_check.json', resume_check)
    write_json(output / 'initial_validation.json', before)
    history = [{'epoch': start_epoch, 'train_loss': None, 'validation': before,
                'fixed_train': fit_before, 'elapsed_seconds': 0., 'resumed_baseline': True}]
    write_json(output / 'history.json', history)
    print('RESUME ' + json.dumps(resume_check), flush=True)
    print('BASELINE ' + json.dumps(history[0]), flush=True)
    best_loss = before['focal_loss_valid_elements']
    best_epoch = start_epoch
    # Save the inherited baseline too, so best_validation is always a real checkpoint.
    torch.save(saved, output / 'best_validation.pt')
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, total_elements = 0., 0
        for step, (x, y, valid) in enumerate(loader, 1):
            x, y, valid = x.to(device, non_blocking=True), y.to(device, non_blocking=True), valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = focal_loss_elements(model(x), y, original['focal_alpha'], original['focal_gamma'])[valid].mean()
            assert torch.isfinite(loss), (epoch, step)
            loss.backward()
            optimizer.step()
            count = int(valid.sum().item())
            total_loss += loss.item() * count
            total_elements += count
            if step % 200 == 0 or step == len(loader):
                print(f'epoch {epoch}/{args.epochs} step {step}/{len(loader)} loss {total_loss / total_elements:.6f}', flush=True)
        validation = evaluate(model, cache_root / 'validation', device, threshold, workers=args.workers)
        fit = evaluate(model, train_folder, device, threshold, workers=args.workers)
        record = {'epoch': epoch, 'train_loss': total_loss / total_elements, 'validation': validation,
                  'fixed_train': fit, 'elapsed_seconds': time.perf_counter() - epoch_start}
        history.append(record)
        checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch, 'configuration': configuration, 'validation': validation,
                      'torch_rng_state': torch.get_rng_state(), 'loader_generator_state': generator.get_state()}
        if device.type == 'cuda': checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state(device)
        torch.save(checkpoint, output / 'last.pt')
        if validation['focal_loss_valid_elements'] < best_loss:
            best_epoch = epoch
            best_loss = validation['focal_loss_valid_elements']
            torch.save(checkpoint, output / 'best_validation.pt')
        write_json(output / 'history.json', history)
        print('EPOCH ' + json.dumps(record), flush=True)
    torch.save(checkpoint, output / 'final.pt')
    # Epoch count was fixed before training; test never selects epoch/hyperparameters.
    test = evaluate(model, cache_root / 'test', device, threshold, workers=args.workers, save_examples=output)
    old_summary = json.loads((args.resume.parent / 'summary.json').read_text())
    summary = {'workflow_completed': True, 'epochs_completed': args.epochs, 'additional_epochs': args.epochs - start_epoch,
               'initial_validation': before, 'final_validation': history[-1]['validation'],
               'fixed_train_before': fit_before, 'fixed_train_final': history[-1]['fixed_train'],
               'test': test, 'previous_test': old_summary['test'],
               'test_distance_baseline': old_summary['test_distance_baseline'],
               'best_valid_loss_epoch': best_epoch, 'checkpoint': 'final.pt',
               'paper_accuracy_reference': .9833,
               'accuracy_gap_percentage_points': 100 * (test['accuracy_mean'] - .9833),
               'training_and_evaluation_seconds': time.perf_counter() - started,
               'caveat': 'Epoch-extension diagnostic beyond paper 8 epochs; simulation/model assumptions remain.'}
    if device.type == 'cuda': summary['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
    write_json(output / 'summary.json', summary)
    plot_history(output, history[1:], masked=True)
    with (output / 'history.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_online_loss', 'train_eval_loss', 'val_valid_loss', 'train_accuracy', 'val_accuracy',
                         'train_f1', 'val_f1', 'val_precision', 'val_recall'])
        for h in history:
            tr, va = h['fixed_train'], h['validation']
            writer.writerow([h['epoch'], h['train_loss'], tr['focal_loss_valid_elements'], va['focal_loss_valid_elements'],
                             tr['accuracy_mean'], va['accuracy_mean'], tr['positive_f1'], va['positive_f1'],
                             va['positive_precision'], va['positive_recall']])
    lines = ['# CTDA 延长轮次对照', '',
             f'从第 {start_epoch} 轮恢复模型和 Adam 状态，继续至固定第 {args.epochs} 轮。数据、模型、Focal Loss、batch=256、lr=1e-4、阈值 .2 均保持原设置。', '',
             ('恢复已保存的 torch/CUDA/loader 随机数状态。' if 'torch_rng_state' in saved else
              '旧检查点未保存随机数状态，续训随机序列重新初始化；本轮检查点补存 RNG 状态。') +
             '延长轮次属于收敛诊断，超出论文的 8 轮设置。', '',
             f'| 指标 | 原 {start_epoch} 轮测试 | 延长后测试 |', '|---|---:|---:|']
    for label, key in [('Accuracy', 'accuracy_mean'), ('Precision', 'positive_precision'), ('Recall', 'positive_recall'), ('F1', 'positive_f1')]:
        lines.append(f'| {label} | {old_summary["test"][key]:.2%} | {test[key]:.2%} |')
    lines.extend(['', '固定训练子集：64 个 fit 场景、640 对；仅作 eval 模式诊断，与验证场景无交叉。',
                  '验证：180 场景、1800 对；最终测试：200 场景、20000 对。测试只在预定终点执行，不用于选择轮次。', '',
                  '[完整结果](summary.json)、[同口径训练/验证曲线数据](history.csv)、[恢复检查](resume_check.json)、[配置](run_config.json)、[最终权重](final.pt)。'])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('COMPLETED ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path, default=Path('results/ctda_repaired_run/final.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/ctda_extended_20'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args())
