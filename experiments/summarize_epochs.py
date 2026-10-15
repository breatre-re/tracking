"""Summarize the fixed 8/20/30/40-epoch controls for a research presentation."""
import csv
import json
from pathlib import Path


def main():
    root = Path('results')
    runs = [(8, 'ctda_repaired_run'), (20, 'ctda_extended_20'),
            (30, 'ctda_extended_30'), (40, 'ctda_extended_40')]
    summaries = {epoch: json.loads((root / name / 'summary.json').read_text()) for epoch, name in runs}
    output = root / 'ctda_epoch_comparison'
    output.mkdir(exist_ok=True)
    keys = ['accuracy_mean', 'positive_precision', 'positive_recall', 'positive_f1', 'focal_loss_valid_elements']
    with (output / 'test_metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'total_epochs', 'split', 'pairs', *keys, 'note'])
        for epoch, name in runs:
            value = summaries[epoch]['test']
            writer.writerow([name, epoch, 'test', value['pairs_evaluated'], *[value[k] for k in keys],
                             'Same corrected model, data, batch=256, threshold=.2; scores are fractions'])
        distance = summaries[40]['test_distance_baseline']
        writer.writerow(['distance_baseline', '', 'test', distance['pairs_evaluated'],
                         *[distance[k] for k in keys[:-1]], '', 'Fixed 10000 m distance gate'])
        writer.writerow(['paper_reference', 8, 'reported', '', .9833, '', '', '', '',
                         'Author data/settings not fully aligned; reported Loss .0067 has unknown reduction'])
    records = {}
    for epoch, name in runs[1:]:
        for h in json.loads((root / name / 'history.json').read_text()):
            records.setdefault(h['epoch'], h)
    assert sorted(records) == list(range(8, 41))
    with (output / 'validation_by_epoch.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_eval_accuracy', 'val_accuracy', 'train_eval_f1', 'val_f1',
                         'val_precision', 'val_recall', 'train_eval_valid_loss', 'val_valid_loss'])
        for epoch, h in sorted(records.items()):
            tr, va = h['fixed_train'], h['validation']
            writer.writerow([epoch, tr['accuracy_mean'], va['accuracy_mean'], tr['positive_f1'], va['positive_f1'],
                             va['positive_precision'], va['positive_recall'], tr['focal_loss_valid_elements'], va['focal_loss_valid_elements']])
    v30, v35, v40 = [records[e]['validation'] for e in [30, 35, 40]]
    tail = [records[e]['validation'] for e in range(36, 41)]
    changes = {}
    for key in ['accuracy_mean', 'positive_f1']:
        changes[key] = {'gain_30_to_40_pp': 100 * (v40[key] - v30[key]),
                        'gain_35_to_40_pp': 100 * (v40[key] - v35[key]),
                        'range_36_to_40_pp': 100 * (max(v[key] for v in tail) - min(v[key] for v in tail))}
    plateau = all(abs(v['gain_35_to_40_pp']) <= .2 and v['range_36_to_40_pp'] <= .3 for v in changes.values())
    result = {'epoch_count_fixed_before_training': 40, 'test_used_for_saturation_decision': False,
              'validation_metric_changes': changes, 'association_metrics_near_plateau': plateau,
              'criterion': 'Both |gain35to40|<=0.2 pp and range36to40<=0.3 pp; diagnostic heuristic, not paper criterion',
              'valid_loss_relative_drop_30_to_40': 1 - v40['focal_loss_valid_elements'] / v30['focal_loss_valid_elements'],
              'valid_loss_relative_drop_35_to_40': 1 - v40['focal_loss_valid_elements'] / v35['focal_loss_valid_elements'],
              'train_minus_validation_accuracy_pp': 100 * (records[40]['fixed_train']['accuracy_mean'] - v40['accuracy_mean'])}
    (output / 'saturation_results.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 轮次对照与平台判断', '',
             '采用同一修正模型、仿真数据、场景划分、batch=256、lr=1e-4、Focal Loss 及阈值 .2。判平台仅使用验证集；40 轮在训练前固定。', '',
             '| 完整测试 | Accuracy | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for epoch, _ in runs:
        value = summaries[epoch]['test']
        lines.append(f'| {epoch} 轮 | ' + ' | '.join(f'{value[k]:.2%}' for k in keys[:-1]) + ' |')
    lines.append('| 距离基线 | ' + ' | '.join(f'{distance[k]:.2%}' for k in keys[:-1]) + ' |')
    lines.extend(['| 论文参考（8 轮） | 98.33% | 未报告 | 未报告 | 未报告 |', '',
                  '四组完整测试均为同一 200 场景／20,000 对；验证为固定 180 场景／1,800 对。', '',
                  '| 验证指标 | 30→40 增益（百分点） | 35→40 增益（百分点） | 36–40 波动范围（百分点） |',
                  '|---|---:|---:|---:|'])
    for label, key in [('Accuracy', 'accuracy_mean'), ('F1', 'positive_f1')]:
        value = changes[key]
        lines.append(f'| {label} | {value["gain_30_to_40_pp"]:.3f} | {value["gain_35_to_40_pp"]:.3f} | {value["range_36_to_40_pp"]:.3f} |')
    lines.extend(['', '**判断：' + ('关联指标接近平台。' if plateau else '尚未观察到关联指标饱和。') + '**',
                  '上述结论使用预定经验判据，不证明全局最优，也不等同于论文结论。',
                  f'有效区域验证 Loss 30→40 相对下降 {result["valid_loss_relative_drop_30_to_40"]:.2%}；35→40 相对下降 {result["valid_loss_relative_drop_35_to_40"]:.2%}。',
                  f'第 40 轮训练子集／验证 Accuracy 为 {records[40]["fixed_train"]["accuracy_mean"]:.2%}／{v40["accuracy_mean"]:.2%}；训练子集用于诊断，与验证集分离。', '',
                  '当前复现完成了仿真到训练及测试的流程。作者数据、部分网络细节、Loss 归约未核实；40 轮也超出论文 8 轮设定。数值接近参考值不能作为严格复现证明。原第 8 轮检查点缺少随机数状态，首次续训重启了随机序列；后续恢复了已保存状态。', '',
                  '[全部测试数值](test_metrics.csv)、[8–40 轮验证数值](validation_by_epoch.csv)、[判据与统计](saturation_results.json)、[8–40 轮曲线](../ctda_extended_40/convergence_diagnostics.png)、[40 轮权重与核验](../ctda_extended_40/REPORT.md)。'])
    (output / 'saturation_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
