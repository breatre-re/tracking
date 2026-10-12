# Phase 4 reproduction assumptions

This document separates **source-confirmed** facts (from the local paper PDF) from
**reproduction assumptions** and **unresolved ambiguities**. Nothing here should
be cited as paper-confirmed unless it is in the "confirmed" column.

Paper source: `references/papers/Zhang 等 - 2026 - A robust multiple target tracking framework with Transformer-based data association and dynamic traj.pdf`
Zhang, He, Yao et al., *Aerospace Science and Technology* 168 (2026) 111007.

## 1. Evidence table

| # | Item | Paper location | Status |
|---|---|---|---|
| 1 | Association input is a **relative distance feature (RDF) matrix per spatial axis**, `C_x, C_y, C_z` | Section 3.1, Eq. (6)-(7), p. 4 | confirmed |
| 2 | RDF element form `c_ij = (2/(1+exp((|d|+bias)/mid)) - 1) * sign(d)`, `mid = max|d| * (min|d| + bias)` | Eq. (6)-(7) | consistent-with-spec (the printed sentence "elements near +/-1 mean similar" is **inverted** relative to the printed expression - see section 3) |
| 3 | Padding to a standard size with zero pads, partitioned into `N_pad x N_pad` sub-matrices | Section 3.2, Eq. (8)-(10) | confirmed |
| 4 | Row-column cross Transformer: per channel, `h1e = ColEncoder(C)`, `h2e = RowEncoder(C^T)^T` | Section 3.3.1, Eq. (11) | confirmed |
| 5 | Two-MHA cross-fed decoders | Eq. (12)-(13) | confirmed |
| 6 | Mixing strategies: element-wise (Eq. 14) or fully-connected (Eq. 15); paper favours element-wise | Eq. (14)-(15) | confirmed |
| 7 | 3-channel cross Transformer: per-channel encoders, cross-channel query, `W_{3x1}` fusion + sigmoid | Section 3.3.2, Eq. (16)-(18) | confirmed |
| 8 | No positional encoding (row/column order carries no meaning) | Section 3.3, p. 5 | confirmed |
| 9 | Focal loss with `alpha_t = 0.75`; paper prints `lambda_t = 2` for the focusing parameter | Section 3.3.3, Eq. (19) | confirmed (`gamma_t = 2` used) |
| 10 | Threshold `lambda_a` then Kuhn-Munkres one-to-one assignment | Section 3.4, Eq. (20), Algorithm 1 | confirmed |
| 11 | Padded input/output dimension 40, 4 heads, dropout 0.3 | Section 5.1.2 / Table 3 | confirmed |
| 12 | Training: 8 epochs, batch 256, Adam, lr `1e-4` | Section 5.1.2 / Table 3 | confirmed |
| 13 | 2D association threshold `lambda_a = 0.1` | Section 5.2 (2D experiment) | confirmed |
| 14 | Train on `paper_assumption_v1/train`, evaluate on its non-overlapping `test` | plan + Phase 1.5 data | confirmed (protocol) |

## 2. Reproduction assumptions (engineering, not paper-confirmed)

| Area | Assumption |
|---|---|
| Candidate history | Packed data has no trajectory histories, so a **row = the previous frame's true detection** (a synthesized candidate proxy). |
| 2D third channel | The paper's channels are `x, y, z`. Our data is 2D, so the third channel is a **sign-free range RDF** (`range_rdf`). |
| Row/column batching | We process the padded `P x P` matrix directly instead of the paper's sub-matrix batching (under-specified for our data). |
| Layer counts | `enc_layers = 2`, `dec_layers = 2` (not printed). |
| FFN width | `ffn_dim = 2 * d_model`. |
| Standard baseline | `standard_transformer` = one encoder over `2P` row/column tokens + dot-product head (no row-column cross structure). |
| Padding value | Padded cells set to `-1.0`; the model also zeroes padded cells so padding cannot affect valid outputs. |
| Truncation | At most `pad_size = 40` rows/columns are kept per frame (deterministic, as stored). |
| Checkpoint selection | Best validation loss on a **train-internal held-out sequence split** (`validation_fraction = 0.1`); the test split is never used for selection or tuning. |
| `x_bias` | `1e-3` (paper says "small quantity"). |
| Determinism | Seeds fixed; `cudnn.deterministic` enabled. Strict GPU bitwise reproducibility is not guaranteed by the installed stack. |

## 3. Unresolved ambiguities (preserved, not silently resolved)

1. **RDF semantics.** The printed expression yields `~0` for very small relative
   distances and `~+/-1` for large ones, while the accompanying sentence claims
   `+/-1` means "similar". The expression is implemented as printed; the
   contradiction is recorded here. A learned model can invert any monotone
   feature, so both readings remain viable.
2. **Eq. (34) Gaussian normalization.** The paper's closed form
   `ln(D/(2*pi*lambda_c)) - ln|Sigma| - d^2/2` places the `|Sigma|` and `1/2`
   factors ambiguously. Phase 4 keeps the standard multivariate Gaussian
   log-likelihood from Eq. (31) and does **not** adopt Eq. (34)'s closed form.
3. **`P_FA` vs Poisson clutter rate `lambda_c`.** Eq. (31) uses `ln P_FA`, while
   Eq. (34) contains `ln(D/(2*pi*lambda_c))`. The relation between them is not
   printed. Both are exposed separately in configuration and are never equated.
4. **Score hyperparameters** `lambda_s, lambda_t, S_c, P_B, P_FA, P_D` remain
   scene-dependent/empirical (Remark 1). Phase 4 reuses the Phase 3 values.
5. **Missed-detection increment sign.** Eq. (30) is source-confirmed as
   `ln((1-P_FA)/(1-P_D))` (positive when `P_D > P_FA`); Phase 4 keeps variant B.
   Variant A remains diagnostic only (Phase 3.1).
6. **Training data cardinality.** "2000 trajectories" is treated as 2000
   sequences (Phase 1.5 data), not 2000 individual target tracks.

## 4. Label isolation

Ground-truth ids (`measurement_truth_ids`) are used **only** to build the
supervision matrix in `build_example`. Feature tensors depend on positions only,
which is asserted by `tests/test_phase4_features.py::test_labels_do_not_enter_features`.
At inference the tracker calls `CTDAAssociation.associate(predicted_positions,
measurements, gate)` with positions only; no id is available to the model.
