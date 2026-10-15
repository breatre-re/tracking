"""Audit and compare trained CNN/CTDA checkpoints with identical evaluation."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from cnn_model import AssociationCNN
from ctda_model import CTDA
from generate_data import pair, rdf_blocks
from train_ctda import assignment


def load_cnn(saved):
    cfg = saved['configuration']['model']
    model = AssociationCNN(**{key: cfg[key] for key in ['channels', 'kernel_size', 'dropout']})
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.eval()


def audit_cnn(folder):
    saved = torch.load(folder / 'final.pt', map_location='cpu', weights_only=False)
    cfg = saved['configuration']
    model = load_cnn(saved)
    root = Path(cfg['data'])
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = {e['scene_id']: e for e in manifest['scenes'] if e['kind'] == 'main'}
    data_cfg = json.loads((root / 'config.json').read_text())
    checks = []
    for file in sorted(folder.glob('test_association_*.npz')):
        with np.load(file) as example:
            sid, k = int(example['scene_id']), int(example['frame'])
            with np.load(root / entries[sid]['file']) as raw:
                data = {name: raw[name] for name in raw.files}
            x, y, truth = pair(data, k, cfg['variant'])
            blocks = rdf_blocks(x, y, truth, data_cfg['rdf_bias_m'], data_cfg['partition_size'])
            features = torch.from_numpy(np.stack([b[0] for b in blocks]).astype(np.float32))
            with torch.inference_mode(): probabilities = torch.sigmoid(model(features)).numpy()
            matrix = np.zeros(truth.shape, np.float32)
            for p, (_, _, _, i, j) in zip(probabilities, blocks):
                nr, nc = min(40, len(x) - i), min(40, len(y) - j)
                matrix[i:i + nr, j:j + nc] = p[:nr, :nc]
            delta = float(np.abs(matrix - example['probability']).max())
            assert delta < .001, 'CPU/full-FP32 versus cuDNN/TF32 tolerance exceeded'
            assert np.array_equal(truth, example['truth'])
            assert np.array_equal(assignment(matrix, cfg['association_threshold']), example['prediction'])
            checks.append({'scene_id': sid, 'frame': k, 'maximum_probability_difference': delta,
                           'truth_identical': True, 'matching_identical': True})
    assert len(checks) == 3
    steps = {int(s['step'].item()) for s in saved['optimizer_state_dict']['state'].values()}
    fit_blocks = json.loads((folder / 'cache_summary.json').read_text())['fit']['blocks']
    expected_steps = 40 * ((fit_blocks + cfg['batch_size'] - 1) // cfg['batch_size'])
    assert steps == {expected_steps}
    for name in ['cnn_model.py', 'train_cnn.py', 'train_ctda.py', 'ctda_model.py']:
        assert hashlib.sha256((folder / f'source_{name}').read_bytes()).hexdigest() == cfg[f'{name}_sha256']
    result = {'raw_pair_cpu_reload': checks, 'adam_final_step': expected_steps,
              'source_snapshot_hashes_match': True, 'cpu_gpu_tolerance': .001,
              'note': 'Same float32 tensor dtype; CUDA convolutions use framework-default cuDNN TF32 permission'}
    (folder / 'reload_check.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def benchmark(cnn_folder, ctda_folder, device):
    cnn_saved = torch.load(cnn_folder / 'final.pt', map_location='cpu', weights_only=False)
    ct_saved = torch.load(ctda_folder / 'final.pt', map_location='cpu', weights_only=False)
    cache = Path(cnn_saved['configuration']['cache_root'])
    inputs = np.load(cache / 'validation' / 'inputs.npy', mmap_mode='r')
    x = torch.from_numpy(inputs[:256].copy()).to(device)
    result = {'device': str(device), 'gpu_name': torch.cuda.get_device_name(device),
              'batch_size': 256, 'input_shape': list(x.shape), 'dtype': str(x.dtype), 'warmup_batches': 10,
              'timed_batches': 30, 'scope': 'Model forward only; cached GPU input; no RDF/loss/matching/transfers',
              'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'models': {}}
    for name, saved in [('CTDA', ct_saved), ('CNN', cnn_saved)]:
        if name == 'CTDA':
            model = CTDA(**saved['configuration']['model'])
            model.load_state_dict(saved['model_state_dict'], strict=True)
        else:
            model = load_cnn(saved)
        model = model.to(device).eval()
        samples = []
        with torch.inference_mode():
            for _ in range(10): model(x)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(30):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(x)
                torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - start)
        median = float(np.median(samples))
        result['models'][name] = {'median_batch_ms': 1000 * median, 'median_ms_per_block': 1000 * median / 256,
                                  'blocks_per_second': 256 / median, 'mean_batch_ms': 1000 * float(np.mean(samples)),
                                  'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20}
        del model
        torch.cuda.empty_cache()
    return result


def main(args):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mttracking-mpl')
    torch.set_num_threads(4)
    cnn_cfg = json.loads((args.cnn / 'run_config.json').read_text())
    ct_cfg = json.loads((args.ctda / 'run_config.json').read_text())
    shared = ['data', 'variant', 'seed', 'batch_size', 'optimizer', 'learning_rate', 'focal_alpha', 'focal_gamma',
              'association_threshold', 'training_loss_reduction', 'loss_mode', 'cache_root', 'epochs']
    assert all(cnn_cfg[k] == ct_cfg[k] for k in shared)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_cnn(args.cnn)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    timing = benchmark(args.cnn, args.ctda, device)
    (output / 'inference_benchmark.json').write_text(json.dumps(timing, indent=2) + '\n')
    summary = json.loads((args.cnn / 'summary.json').read_text())
    ct8 = json.loads((Path('results/ctda_repaired_run') / 'summary.json').read_text())['test']
    ct40 = json.loads((args.ctda / 'summary.json').read_text())['test']
    values = [('CTDA', 8, ct8), ('CNN', 8, summary['test_epoch8']), ('CTDA', 40, ct40), ('CNN', 40, summary['test'])]
    keys = ['accuracy_mean', 'positive_precision', 'positive_recall', 'positive_f1', 'focal_loss_valid_elements']
    with (output / 'comparison.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'epochs', 'parameters', 'test_pairs', *keys])
        for name, epoch, value in values:
            writer.writerow([name, epoch, cnn_cfg['parameter_count'] if name == 'CNN' else ct_cfg['parameter_count'],
                             value['pairs_evaluated'], *[value[key] for key in keys]])
    history = json.loads((args.cnn / 'history.json').read_text())
    ct_history = {}
    for folder in ['ctda_repaired_run', 'ctda_extended_20', 'ctda_extended_30', 'ctda_extended_40']:
        for h in json.loads((Path('results') / folder / 'history.json').read_text()):
            ct_history.setdefault(h['epoch'], h)
    assert sorted(ct_history) == list(range(1, 41)) and len(history) == 40
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, hs in [('CTDA', list(ct_history.values())), ('CNN', history)]:
        epochs = [h['epoch'] for h in hs]
        for ax, key in zip(axes, ['accuracy_mean', 'positive_f1', 'focal_loss_valid_elements']):
            scale = 1 if key == 'focal_loss_valid_elements' else 100
            ax.plot(epochs, [scale * h['validation'][key] for h in hs], label=name)
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel('Validation paper accuracy (%)')
    axes[1].set_ylabel('Validation positive F1 (%)')
    axes[2].set(ylabel='Validation valid focal loss', yscale='log')
    fig.tight_layout()
    fig.savefig(output / 'validation_comparison.png', dpi=170)
    plt.close(fig)
    result = {'shared_configuration_checked': {key: cnn_cfg[key] for key in shared}, 'cnn_audit': audit,
              'cnn_minus_ctda_40_accuracy_pp': 100 * (summary['test']['accuracy_mean'] - ct40['accuracy_mean']),
              'cnn_minus_ctda_40_f1_pp': 100 * (summary['test']['positive_f1'] - ct40['positive_f1'])}
    (output / 'comparison_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# CNN 与 CTDA 同数据对照', '', '仅比较网络结构：两者共用相同 RDF、标签、有效掩码、划分、Focal Loss、Adam/lr/batch、阈值 .2 和全局一对一匹配。输入已经包含论文的 RDF，故本实验不能单独衡量 RDF 的收益。', '',
             '| 模型 | 轮次 | Accuracy | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|---:|']
    for name, epoch, value in values:
        lines.append(f'| {name} | {epoch} | ' + ' | '.join(f'{value[key]:.2%}' for key in keys[:-1]) + ' |')
    lines.extend(['', '主对照为 40 轮，8 轮作为论文轮次参考；每组测试相同 200 场景、20,000 对。参数量：CTDA 847,222，CNN 824,449（少 2.69%）。', '',
                  'CNN 使用 4 层 5×5 卷积，局部感受野 17×17；CTDA 在 40×40 块内做行列及通道交叉注意力。CNN 的局部邻域属于关联矩阵的邻接位置；量测排列本身没有物理空间邻接含义。', '',
                  '| 同卡纯前向（batch=256） | CTDA | CNN |', '|---|---:|---:|'])
    lines.append('| 批次中位耗时（ms） | ' + ' | '.join(f'{timing["models"][name]["median_batch_ms"]:.3f}' for name in ['CTDA', 'CNN']) + ' |')
    lines.extend(['', '耗时来自相同 RTX4090、GPU 常驻 float32 输入和 eval 模式，各预热10批、计时30批；不含数据读取、RDF、Loss 或匹配，采用记录的 PyTorch 默认 TF32 权限。', '',
                  '结论仅适用于当前仿真数据与这个明确指定的 CNN。当前数据较稀疏，距离基线 Accuracy 99.90%；本次一组种子对照不代表所有目标密度或真实数据。论文 CNN 92.13%／CTDA 98.33% 为不同数据与配置下的参考；原文没有完整 CNN 架构，不能将本网络称为已核实的作者 CNN。', '',
                  '[数值表](comparison.csv)、[验证曲线](validation_comparison.png)、[控制条件核验](comparison_checks.json)、[推理测速](inference_benchmark.json)、[CNN 训练与权重](../cnn_same_data_40/REPORT.md)。'])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'comparison': result, 'inference': timing}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', type=Path, default=Path('results/cnn_same_data_40'))
    parser.add_argument('--ctda', type=Path, default=Path('results/ctda_extended_40'))
    parser.add_argument('--output', type=Path, default=Path('results/cnn_ctda_comparison'))
    parser.add_argument('--device', default='cuda:4')
    main(parser.parse_args())
