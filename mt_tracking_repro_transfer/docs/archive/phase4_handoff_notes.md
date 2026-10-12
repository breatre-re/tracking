# Phase 4 handoff bundle

This folder contains a copy of the requested Phase 4 evidence. Source artifacts in
the project directories were not moved or modified.

## Contents

- `phase4_manifest.json` — original 48-case manifest, including tracker metrics,
  scenario configurations, input hashes, and actual checkpoint hash.
- `training_logs/row_column_3ch_training_summary.json` — complete persisted
  training summary and per-epoch history for the failed 3-channel run.
- `training_logs/row_column_3ch_epoch_log.txt` — readable rendering of the
  persisted epoch history. The original terminal stdout was not saved, so this
  is reconstructed from the summary and is not represented as a raw console log.
- `training_logs/row_column_3ch_run_details.json` — training protocol, data
  limits, and failure status. `row_column_training_summary.json` is included as
  a comparison.
- `phase4_ctda.yaml` — Phase 4 model, dataset, training, inference, and score
  configuration.
- `checkpoint_inventory.json` — original paths, sizes, and SHA-256 values for
  the requested checkpoints. `row_column_3ch/checkpoint.pt` does not exist
  because that training run never produced a valid checkpoint.
- `checkpoints/row_column_checkpoint.pt` — copy of the available checkpoint
  actually used by the Phase 4 tracker manifest.
- `nan_cases/` — the two `paper_assumption_v1` cases whose recorded position
  RMSE is NaN (`S2`, seed 12022; `S6`, seed 12066). Each contains:
  - `tracker_input_positions.npz`: inference measurements and frame offsets,
    without truth IDs.
  - `offline_truth_and_labels.npz`: truth states and measurement source IDs for
    offline evaluation only; this file is not tracker input.
  - `tracker_output_metrics.json`: copied per-case output metrics JSON.
  - `ctda_matching_results.json`: per-frame CTDA track-ID/measurement-index
    matches and emitted estimates, reproduced with the manifest checkpoint.
- `source/phase4_runner.py` — Phase 4 runner.
- `source/ctda_features.py` — dataset-to-tensor / feature conversion module.
- `source/ctda_adapter.py`, `source/ctda_tracker.py`,
  `source/ctda_model.py`, and `source/ctda_objective.py` — adjacent code needed
  to interpret the association path.
- `nan_case_index.json` — case paths, seeds, hashes, and replay consistency.

The S2 and S6 association replays reproduced the manifest input hashes exactly.
Their total association counts match the per-case audit JSON (`1636` and `1489`,
respectively). These are CTDA association outputs, not evaluator
`matched_detections`; the latter is zero for both cases. The RMSE NaNs remain
unchanged in the copied original output JSON.

## Integrity

`checkpoint_inventory.json` provides checkpoint provenance. Other file hashes
can be recomputed with SHA-256 from the files in this directory.
