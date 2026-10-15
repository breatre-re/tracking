"""Frozen-weight validation diagnosis; thresholds are diagnostic, not new results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from compare_cnn_ctda import load_cnn
from ctda_model import CTDA
from train_ctda import Blocks, assignment
from generate_data import score, summarize_scores


def diagnose(model, cache, ids, device):
    dataset = Blocks(cache)
    loader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=False)
    probabilities = np.empty((len(dataset), 40, 40), np.float32)
    offset = 0
    model.to(device).eval()
    with torch.inference_mode():
        for x, _, _ in loader:
            p = torch.sigmoid(model(x.to(device))).cpu().numpy()
            probabilities[offset:offset + len(p)] = p
            offset += len(p)
    mask = np.asarray(dataset.valid)
    p = probabilities[mask]
    y = np.asarray(dataset.labels)[mask].astype(bool)
    ranks = rankdata(p)
    npos, nneg = int(y.sum()), int((~y).sum())
    auc = float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    ordered = y[np.argsort(-p, kind='stable')]
    ap = float((np.cumsum(ordered)[ordered] / np.arange(1, len(ordered) + 1)[ordered]).sum() / npos)
    meta = np.load(cache / 'pair_metadata.npy')
    offsets = np.load(cache / 'pair_offsets.npy')
    origins = np.load(cache / 'origins.npy')
    thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    values = {t: [] for t in thresholds}
    false_types = {t: {'clutter_clutter': 0, 'target_clutter': 0, 'wrong_target_target': 0} for t in thresholds}
    false_scores = {t: [] for t in thresholds}
    row_top_correct = col_top_correct = total_positive = 0
    for index, (sid, k, rows, cols) in enumerate(meta):
        matrix = np.zeros((rows, cols), np.float32)
        truth = np.zeros((rows, cols), np.uint8)
        for b in range(offsets[index], offsets[index + 1]):
            i, j = origins[b]
            nr, nc = min(40, rows - i), min(40, cols - j)
            matrix[i:i + nr, j:j + nc] = probabilities[b, :nr, :nc]
            truth[i:i + nr, j:j + nc] = dataset.labels[b, :nr, :nc]
        ti, tj = np.where(truth)
        total_positive += len(ti)
        if rows and cols:
            row_top_correct += int((matrix[ti, tj] >= matrix.max(1)[ti]).sum())
            col_top_correct += int((matrix[ti, tj] >= matrix.max(0)[tj]).sum())
        aid, bid = ids[(int(sid), int(k))]
        for t in thresholds:
            prediction = assignment(matrix, t)
            values[t].append(score(truth, prediction))
            fi, fj = np.where(prediction.astype(bool) & ~truth.astype(bool))
            ar, br = aid[fi] >= 0, bid[fj] >= 0
            false_types[t]['clutter_clutter'] += int((~ar & ~br).sum())
            false_types[t]['target_clutter'] += int((ar ^ br).sum())
            false_types[t]['wrong_target_target'] += int((ar & br).sum())
            false_scores[t].extend(matrix[fi, fj].tolist())
    metrics = []
    for t in thresholds:
        result = summarize_scores(values[t])
        result.update({'threshold': t, 'false_edge_types': false_types[t],
                       'false_edge_probability_quantiles': np.quantile(false_scores[t], [.1, .5, .9]).tolist() if false_scores[t] else []})
        metrics.append(result)
    return {'valid_cells': len(p), 'positive_cells': npos, 'pointwise_roc_auc': auc,
            'pointwise_average_precision': ap,
            'true_edge_top_in_row_fraction': row_top_correct / total_positive,
            'true_edge_top_in_column_fraction': col_top_correct / total_positive,
            'positive_probability_quantiles_01_10_50_90_99': np.quantile(p[y], [.01, .1, .5, .9, .99]).tolist(),
            'negative_probability_quantiles_50_90_99_999': np.quantile(p[~y], [.5, .9, .99, .999]).tolist(),
            'threshold_diagnostics': metrics}


def main(device):
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    root = Path('results')
    ctda = torch.load(root / 'ctda_extended_40/final.pt', map_location='cpu', weights_only=False)
    cnn = torch.load(root / 'cnn_same_data_40/final.pt', map_location='cpu', weights_only=False)
    cfg = ctda['configuration']
    cache = Path(cfg['cache_root']) / 'validation'
    meta = np.load(cache / 'pair_metadata.npy')
    data_root = Path(cfg['data'])
    manifest = json.loads((data_root / 'manifest.json').read_text())
    entries = {e['scene_id']: e for e in manifest['scenes'] if e['kind'] == 'main'}
    ids = {}
    for sid in np.unique(meta[:, 0]):
        with np.load(data_root / entries[int(sid)]['file']) as raw:
            offsets = raw['observation_offsets']
            target_ids = raw['observation_target_id']
        for _, k, rows, cols in meta[meta[:, 0] == sid]:
            aid = target_ids[offsets[k]:offsets[k + 1]]
            bid = target_ids[offsets[k + 1]:offsets[k + 2]]
            assert len(aid) == rows and len(bid) == cols
            ids[(int(sid), int(k))] = (aid, bid)
    model = CTDA(**cfg['model'])
    model.load_state_dict(ctda['model_state_dict'], strict=True)
    results = {'protocol': 'Fixed40 weights, validation only180scenes/1800pairs, unchanged primary threshold.2; no training or test tuning',
               'CTDA': diagnose(model, cache, ids, device)}
    model.cpu()
    results['CNN'] = diagnose(load_cnn(cnn), cache, ids, device)
    output = root / 'cnn_ctda_validation_diagnosis'
    output.mkdir(exist_ok=True)
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    with (output / 'thresholds.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'threshold', 'accuracy', 'precision', 'recall', 'f1', 'false_clutter_clutter', 'false_target_clutter', 'false_wrong_targets'])
        for name in ['CTDA', 'CNN']:
            for value in results[name]['threshold_diagnostics']:
                writer.writerow([name, value['threshold'], value['accuracy_mean'], value['positive_precision'],
                                 value['positive_recall'], value['positive_f1'], *value['false_edge_types'].values()])
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:4')
    main(torch.device(parser.parse_args().device))
