"""Plot comparable train/validation diagnostics and verify a saved CTDA run."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ctda_model import CTDA
from generate_data import pair, rdf_blocks
from train_ctda import assignment


def main(output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mttracking-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = json.loads((output / 'history.json').read_text())
    cfg_for_plot = json.loads((output / 'run_config.json').read_text())
    parent = Path(cfg_for_plot['resume_checkpoint']).parent
    seen = {output.resolve()}
    while (parent / 'history.json').exists():
        assert parent.resolve() not in seen, 'Checkpoint ancestry contains a cycle.'
        seen.add(parent.resolve())
        previous = json.loads((parent / 'history.json').read_text())
        if previous and all('fixed_train' in h for h in previous):
            parent_cfg = json.loads((parent / 'run_config.json').read_text())
            assert parent_cfg['fixed_train_subset'] == cfg_for_plot['fixed_train_subset']
            history = [h for h in previous if h['epoch'] < history[0]['epoch']] + history
            parent = Path(parent_cfg['resume_checkpoint']).parent
        else:
            break
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key, label in [('fixed_train', 'Fixed train subset'), ('validation', 'Validation')]:
        values = [h[key] for h in history]
        axes[0].plot(epochs, [v['focal_loss_valid_elements'] for v in values], 'o-', label=label)
        axes[1].plot(epochs, [100 * v['accuracy_mean'] for v in values], 'o-', label=label + ' accuracy')
        axes[1].plot(epochs, [100 * v['positive_f1'] for v in values], '--', label=label + ' F1')
    for metric, label in [('positive_precision', 'Precision'), ('positive_recall', 'Recall')]:
        axes[2].plot(epochs, [100 * h['validation'][metric] for h in history], 'o-', label=label)
    axes[0].set(yscale='log', ylabel='Valid-element focal loss (eval mode)')
    axes[1].set(ylabel='Association score (%)', ylim=(0, 102))
    axes[2].set(ylabel='Validation score (%)', ylim=(0, 102))
    for ax in axes:
        ax.set_xlabel('Total epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'convergence_diagnostics.png', dpi=170)
    plt.close(fig)
    torch.set_num_threads(4)
    saved = torch.load(output / 'final.pt', map_location='cpu', weights_only=False)
    cfg = saved['configuration']
    model = CTDA(**cfg['model'])
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval()
    root = Path(cfg['data'])
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = {e['scene_id']: e for e in manifest['scenes'] if e['kind'] == 'main'}
    checks = []
    for file in sorted(output.glob('test_association_*.npz')):
        with np.load(file) as example:
            sid, k = int(example['scene_id']), int(example['frame'])
            with np.load(root / entries[sid]['file']) as raw:
                data = {key: raw[key] for key in raw.files}
            x, y, truth = pair(data, k, cfg['variant'])
            blocks = rdf_blocks(x, y, truth, 1., 40)
            features = torch.from_numpy(np.stack([b[0] for b in blocks]).astype(np.float32))
            with torch.inference_mode():
                probabilities = torch.sigmoid(model(features)).numpy()
            matrix = np.zeros(truth.shape, np.float32)
            for p, (_, _, _, i, j) in zip(probabilities, blocks):
                nr, nc = min(40, len(x) - i), min(40, len(y) - j)
                matrix[i:i + nr, j:j + nc] = p[:nr, :nc]
            diff = float(np.max(np.abs(matrix - example['probability'])))
            assert diff < 1e-5
            assert np.array_equal(truth, example['truth'])
            assert np.array_equal(assignment(matrix, cfg['association_threshold']), example['prediction'])
            checks.append({'scene_id': sid, 'frame': k, 'maximum_probability_difference': diff,
                           'truth_identical': True, 'matching_identical': True})
    assert len(checks) == 3
    steps = {int(s['step'].item()) for s in saved['optimizer_state_dict']['state'].values()}
    resume_check = json.loads((output / 'resume_check.json').read_text())
    expected = resume_check['optimizer_step_restored'] + (saved['epoch'] - cfg['resume_epoch']) * resume_check['steps_per_epoch']
    assert steps == {expected}
    model_hash = hashlib.sha256((output / 'source_ctda_model.py').read_bytes()).hexdigest()
    extension_hash = hashlib.sha256((output / 'source_continue_ctda.py').read_bytes()).hexdigest()
    assert model_hash == cfg['model_source_sha256'] and extension_hash == cfg['extension_source_sha256']
    result = {'cpu_reload_raw_pair_checks': checks, 'adam_final_step': expected,
              'source_snapshot_hashes_match': True, 'epochs_completed': saved['epoch']}
    (output / 'reload_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
