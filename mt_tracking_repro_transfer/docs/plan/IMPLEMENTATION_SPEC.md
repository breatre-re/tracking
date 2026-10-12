# 2D Tracking Reproduction Implementation Spec

Source: Zhang et al., “A robust multiple target tracking framework with Transformer-based data association and dynamic trajectory management in challenging environments”, Aerospace Science and Technology 168 (2026) 111007.

This document specifies the paper-aligned 2D tracking experiment in Section 5.2. It is a reproduction contract for the implementation team; it does not define the main code structure.

## 1. Scope and data flow

The system is position-only multi-target tracking:

```text
ground-truth target states -> noisy range/bearing detections + clutter/misses
                           -> CTDA association
                           -> per-track BLUE/EKF-style update
                           -> score-based track management
                           -> confirmed labeled trajectories
```

Do not substitute the paper's 3D model from Eqs. (1)-(2) for this 2D benchmark. The 2D benchmark is introduced in Section 5.2.1 and uses Eqs. (37)-(39).

## 2. 2D state and motion model

State ordering must be:

```text
x_k = [x, x_dot, y, y_dot, omega]^T
```

where `(x, y)` is planar position, `(x_dot, y_dot)` is velocity, and `omega` is angular rate.

Discrete dynamics:

```text
x_(k+1) = F(omega) x_k + nu_k
```

For nonzero `omega`, use:

```text
F(omega) =
[[1, sin(omega)/omega, 0, -(1-cos(omega))/omega, 0],
 [0, cos(omega),       0, -sin(omega),            0],
 [0, (1-cos(omega))/omega, 1, sin(omega)/omega,  0],
 [0, sin(omega),       0, cos(omega),             0],
 [0, 0,                0, 0,                    1]]
```

Use the continuous-turn limit as `omega -> 0` to avoid division by zero. The paper prints the process-noise input matrix as:

```text
G = [[1/2, 0],
     [1,   0],
     [0, 1/2],
     [0, 1],
     [0, 0]]
```

The paper states `nu ~ N(0, Gamma)` and gives `sigma_omega` and `sigma_u`; its PDF typesetting is dimensionally ambiguous around `Gamma = diag([sigma_omega^2 G G^T, sigma_u^2])`. Preserve the paper values below, but make the exact covariance construction an explicit implementation setting and report it with results.

## 3. Observation model

Each true detection is a two-component range/bearing measurement:

```text
z_k = h(x_k) + w_k
h(x_k) = [ sqrt(x^2 + y^2), atan2(y, x) ]^T
```

`w_k` is zero-mean white measurement noise. Use range noise standard deviation `sigma_r` and bearing noise standard deviation `sigma_theta`. Angles are in radians. Use `atan2`, not a one-argument arctangent.

The tracking input at each time step is an unordered set of measurements containing true detections and false alarms. The association matrix is one-to-one: each row/column has at most one positive association.

## 4. 2D dataset and scenario parameters

Use these Section 5.2.1 parameters:

| Parameter | Value |
|---|---:|
| Initial speed range | Uniform `(0, 30)` m/s |
| Initial heading range | Uniform `(0, 2*pi)` rad |
| Turning-rate range | Uniform `(-0.05, 0.05)` rad/s |
| Target count range | Uniform/integer `(1, 25)` |
| Missing-detection rate | `0.1` |
| Poisson clutter parameter `lambda_c` | `20` |
| Surveillance radius `R` | `2000` m |
| Process-noise parameter `sigma_omega` | `15 m/s^2` as printed; verify units in any author-provided source |
| Process/input-noise parameter `sigma_u` | `pi/180` rad/s |
| Range-noise std. `sigma_r` | `5` m |
| Bearing-noise std. `sigma_theta` | `pi/360` rad |
| Generated trajectories | `2000` |
| Training split | `1800` |
| Test split | `200` |

The scenarios include target birth, death, crossing, missed detections, and clutter. The paper says the 2D generator is aligned with reference [50]; it does not provide a complete reproducible seed, time horizon, birth/death distributions, or initial-position distribution in Section 5.2.1. These must be treated as unresolved reproduction inputs, not silently invented paper parameters.

For the paper's reported training setup, use the default NumPy RNG and one consistent default seed across uniform and Gaussian draws if the seed can be obtained. The actual seed is not printed in the paper.

## 5. Association and filtering contract

At time `k`:

1. Predict every candidate and maintained track to obtain `x_hat_l(k|k-1)`.
2. Run CTDA on predicted track positions and the current measurement set.
3. Convert predicted probabilities to a binary association matrix using thresholding and Kuhn-Munkres/Hungarian one-to-one assignment.
4. For an associated pair, update that track with the paper's BLUE equations (Eqs. (23)-(25)); for an unassociated track, carry forward the prediction (Eqs. (28)-(29)).
5. Create a candidate track for every unassociated measurement.
6. Update scores and apply lifecycle rules.

The 2D CTDA training details reported with the overall MTT experiment are: padded input/output dimension `40`, `4` attention heads, dropout `0.3`, `8` epochs, batch size `256`, Adam learning rate `1e-4`. Section 5.2 reports association threshold `lambda_a = 0.1`; the separate 3D association experiment in Section 5.1 reports `lambda_a = 0.2`. Use `0.1` for this 2D tracking reproduction and record the distinction.

## 6. Score dynamics and track lifecycle

Maintain two disjoint label sets:

```text
L_c = candidate/unconfirmed tracks
L_m = maintained/confirmed tracks
```

For each track `l`, implement the score equation:

```text
S_l(k) = lambda_s S_l(k-1) + DeltaS_l(k)
         - lambda_s^lambda_t DeltaS_l(k-lambda_t)
```

The increments are:

- New unassociated measurement: `DeltaS = log(P_B / P_FA)`.
- Missed detection/unassociated track: `DeltaS = log((1-P_FA)/(1-P_D))`; state and covariance remain at prediction.
- Successful association: `DeltaS = log p(x_hat_l(k|k) | x_hat_l(k|k-1)) - log(P_FA)`, with the Gaussian density and Mahalanobis term in Eqs. (31)-(34).

Lifecycle rules:

- Candidate with `S_l >= S_c`: move from `L_c` to `L_m`.
- Candidate with `0 <= S_l < S_c`: keep as candidate.
- Any candidate with `S_l < 0`: delete its filter and label.
- Maintained track with `S_l >= 0`: continue tracking.
- Maintained track with `S_l < 0`: terminate and retire its label.

The paper does not publish numeric values for `lambda_s`, `lambda_t`, `S_c`, `P_B`, `P_FA`, or `P_D`; it says they are scene-dependent/empirically configured. The implementation must expose them as configuration fields and report the chosen values. Do not claim exact reproduction of the score manager until these values are obtained from the authors or supplementary code.

## 7. Evaluation protocol

For the 2D benchmark, run 100 Monte Carlo simulations on the test set. Report:

1. OSPA distance.
2. OSPA localization component.
3. OSPA cardinality component.
4. Association accuracy where association matrices are available.
5. Runtime comparison.
6. Qualitative trajectory plots showing clutter, crossings, and missed detections.
7. Identity-switch count and track continuity as additional reproduction diagnostics; the paper's trajectory discussion emphasizes label maintenance and no identity switches in the representative case.

The reported OSPA settings are cutoff `c = 100` and order `r = 1`. Implement the paper's Eqs. (40)-(42), including the cardinality penalty and the decomposition into localization and cardinality terms.

Reference values from the paper's Table 9 (use only as comparison targets, not as guaranteed acceptance criteria):

| Method | OSPA distance | OSPA location | OSPA cardinality |
|---|---:|---:|---:|
| PHD | 17.122 | 9.691 | 7.431 |
| CPHD | 16.430 | 11.601 | 4.829 |
| GLMB | 16.299 | 11.752 | 4.546 |
| Ours | **15.148** | 10.266 | 4.882 |

Reported runtime comparison in Table 10: PHD `0.3883 s`, CPHD `0.6653 s`, GLMB `88.37 s`, accelerated GLMB `41.62 s`, Ours `0.2730 s`. Runtime is hardware- and implementation-dependent.

## 8. Known paper ambiguities / do not silently resolve

- The paper does not print the random seed, time horizon, initial positions, or exact birth/death distributions for the 2D generator.
- The process-noise covariance expression around Eq. (38) is typeset ambiguously; retain a configurable covariance implementation.
- Score-manager hyperparameters `lambda_s`, `lambda_t`, `S_c`, `P_B`, `P_FA`, and `P_D` are not numerically specified.
- The 3D experiment uses association threshold `0.2`; the 2D overall MTT experiment uses `0.1`. Do not mix them.
- “2000 trajectories” refers to generated dataset sequences in the paper's wording; confirm with the authors whether this means 2000 complete multi-target scenes or 2000 individual target tracks before matching dataset cardinality exactly.

## Paper locations

- 2D model: Eqs. (37)-(39), Section 5.2.1, pp. 11-12.
- 2D parameters and split: Table 8, p. 12.
- Score dynamics: Eqs. (26)-(34), Sections 4.3-4.4, pp. 8-9.
- Tracking loop: Algorithm 2, p. 9.
- OSPA: Eqs. (40)-(42), Section 5.2.2, p. 13.
- Benchmark results: Tables 9-10, pp. 13 and 15.
