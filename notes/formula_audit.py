"""Independent scalar/matrix audits of the paper reading; Python stdlib only.

This is not a CTDA implementation, dataset generator, or training verification.
Run: python notes/formula_audit.py
"""

import itertools
import json
import math
from pathlib import Path


def paper_accuracy(truth, prediction):
    m, n = len(truth), len(truth[0])
    rows = sum(a == b for a, b in zip(truth, prediction))
    cols = sum(all(truth[i][j] == prediction[i][j] for i in range(m)) for j in range(n))
    return (rows + cols) / (m + n)


def element_accuracy(truth, prediction):
    return sum(a == b for ar, br in zip(truth, prediction) for a, b in zip(ar, br)) / (len(truth) * len(truth[0]))


def matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def ct_matrix(omega):
    if omega == 0:
        sinc, cosc = 1.0, 0.0
    else:
        sinc = math.sin(omega) / omega
        cosc = 2 * math.sin(omega / 2) ** 2 / omega
    c, s = math.cos(omega), math.sin(omega)
    return [[1, sinc, 0, -cosc, 0], [0, c, 0, -s, 0],
            [0, cosc, 1, sinc, 0], [0, s, 0, c, 0], [0, 0, 0, 0, 1]]


def audit():
    checks = []

    def record(name, passed, evidence):
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    # Formula 6 versus its stable algebraic equivalent.
    diffs, bias = [-1000000, -1000, -1, 0, 1, 1000, 1000000], 1.0
    scale = math.sqrt(max(map(abs, diffs)) * (min(map(abs, diffs)) + bias))
    error = 0.0
    features = []
    for d in diffs:
        sign = (d > 0) - (d < 0)
        original = (2 / (1 + math.exp(-scale / (abs(d) + bias))) - 1) * sign
        stable = math.tanh(scale / (2 * (abs(d) + bias))) * sign
        error = max(error, abs(original - stable))
        features.append(stable)
    record("eq6_sigmoid_tanh_equivalence", error < 1e-12, {"maximum_absolute_error": error})
    record("eq6_exact_zero_distance_maps_to_zero", features[3] == 0,
           {"zero_distance_feature": features[3], "interpretation": "Confirmed edge case, not a correction."})
    record("eq6_nonzero_distance_magnitude_decreases", abs(features[4]) > abs(features[5]) > abs(features[6]),
           {"positive_distances": [1, 1000, 1000000], "features": features[4:]})

    # Formula 19 versus alpha_t * (1-p_t)^gamma * negative log likelihood.
    focal_error = 0.0
    for y in [0, 1]:
        for p in [1e-9, 0.01, 0.2, 0.5, 0.99, 1 - 1e-9]:
            expanded = -0.75 * y * (1 - p) ** 2 * math.log(p) - 0.25 * (1 - y) * p ** 2 * math.log1p(-p)
            pt, alpha_t = (p, 0.75) if y else (1 - p, 0.25)
            compact = -alpha_t * (1 - pt) ** 2 * math.log(pt)
            focal_error = max(focal_error, abs(expanded - compact))
    record("eq19_focal_loss_equivalence", focal_error < 1e-7,
           {"maximum_absolute_error": focal_error, "alpha_positive": 0.75, "gamma": 2})

    # The paper metric is row/column exact correctness, not cell accuracy.
    identity = [[int(i == j) for j in range(40)] for i in range(40)]
    zero = [[0] * 40 for _ in range(40)]
    score, cell = paper_accuracy(identity, zero), element_accuracy(identity, zero)
    record("eq35_zero_prediction_counterexample", score == 0 and math.isclose(cell, 0.975),
           {"paper_accuracy": score, "element_accuracy": cell})
    missed = [row.copy() for row in identity]
    missed[0][0] = 0
    score, cell = paper_accuracy(identity, missed), element_accuracy(identity, missed)
    record("eq35_one_missed_edge", math.isclose(score, 0.975) and math.isclose(cell, 0.999375),
           {"paper_accuracy": score, "element_accuracy": cell})
    record("eq35_all_zero_truth_degeneracy", paper_accuracy(zero, zero) == 1,
           {"paper_accuracy": paper_accuracy(zero, zero), "warning": "Does not demonstrate recovery of any true association."})

    # Check the proposed interpretation of Eq26 independently of its recursion.
    increments, decay, window = [1, -0.5, 2, 0.3, -1, 3, 0, -2], 0.9, 3
    recursive, max_error = 0.0, 0.0
    for k, inc in enumerate(increments):
        expired = increments[k - window] if k >= window else 0.0
        recursive = decay * recursive + inc - decay ** window * expired
        direct = sum(decay ** q * increments[k - q] for q in range(min(window, k + 1)))
        max_error = max(max_error, abs(recursive - direct))
    record("eq26_finite_window_equivalence", max_error < 1e-12,
           {"maximum_absolute_error": max_error, "assumption": "initial score and pre-birth increments are zero"})

    # Reproduce the problematic sign as printed, without silently fixing it.
    printed_miss = math.log((1 - 0.1) / (1 - 0.9))
    record("eq30_printed_miss_increment_is_positive", printed_miss > 0,
           {"P_D": 0.9, "P_FA": 0.1, "printed_increment": printed_miss,
            "candidate_llr_reversal": -printed_miss, "status": "author confirmation needed"})

    lam = 20.0
    probability, term = 0.0, math.exp(-lam)
    for i in range(1, 201):
        term *= lam / i
        probability += term
    record("eq34_poisson_nonempty_probability_not_mean",
           math.isclose(probability, 1 - math.exp(-lam), abs_tol=1e-12) and not math.isclose(probability, lam),
           {"sum_probability_i_ge_1": probability, "closed_form": 1 - math.exp(-lam), "mean_count": lam})

    stationary = [100.0, 0.0, 250.0, 0.0, 0.0]
    # As printed in Figure/page 12, Eq38: wrong first-column entries and F[2,2].
    printed_matrix = ct_matrix(0.0)
    for i in [1, 2, 3]:
        printed_matrix[i][0] = 1
    printed_matrix[2][2] = 0
    printed_state = matvec(printed_matrix, stationary)
    corrected_state = matvec(ct_matrix(0.0), stationary)
    record("eq38_printed_stationary_state_counterexample",
           printed_state != stationary and corrected_state == stationary,
           {"initial": stationary, "printed_next": printed_state, "reference_model_next": corrected_state})
    moving = [100.0, 10.0, 250.0, -5.0, 0.0]
    zero_turn = matvec(ct_matrix(0.0), moving)
    small_turn = matvec(ct_matrix(1e-9), moving)
    record("eq38_reference_zero_turn_limit",
           zero_turn == [110.0, 10.0, 245.0, -5.0, 0.0] and max(abs(a - b) for a, b in zip(zero_turn, small_turn)) < 1e-6,
           {"zero_turn_next": zero_turn, "small_turn_next": small_turn, "dt_seconds": 1})

    table9 = {"PHD": [17.122, 9.691, 7.431], "CPHD": [16.430, 11.601, 4.829],
              "GLMB": [16.299, 11.752, 4.546], "CTDA_framework": [15.148, 10.266, 4.882]}
    errors = {name: abs(total - loc - card) for name, (total, loc, card) in table9.items()}
    # Each of three independently rounded entries can differ by 0.0005.
    record("table9_order_one_ospa_additivity_with_rounding", max(errors.values()) <= 0.0015,
           {"absolute_errors": errors, "three_entry_rounding_bound": 0.0015,
            "note": "GLMB has a 0.001 discrepancy, compatible with three-decimal rounding."})

    # A globally thresholded partial assignment differs from forced assignment
    # followed by deleting invalid edges. Exhaustive enumeration needs no SciPy.
    probabilities, threshold = [[0.30, 0.25], [0.19, 0.0]], 0.2
    forced = max(itertools.permutations(range(2)), key=lambda perm: sum(probabilities[i][j] for i, j in enumerate(perm)))
    forced_retained = [(i, j) for i, j in enumerate(forced) if probabilities[i][j] >= threshold]
    forced_value = sum(probabilities[i][j] for i, j in forced_retained)
    edges = [(i, j) for i in range(2) for j in range(2) if probabilities[i][j] >= threshold]
    candidates = []
    for size in range(len(edges) + 1):
        for chosen in itertools.combinations(edges, size):
            if len({i for i, _ in chosen}) == size and len({j for _, j in chosen}) == size:
                candidates.append(chosen)
    partial = max(candidates, key=lambda chosen: sum(probabilities[i][j] for i, j in chosen))
    partial_value = sum(probabilities[i][j] for i, j in partial)
    record("thresholded_partial_assignment_counterexample", partial_value > forced_value,
           {"probabilities": probabilities, "threshold": threshold, "forced_then_filtered": forced_retained,
            "forced_retained_value": forced_value, "partial_matching": partial, "partial_value": partial_value})

    # Literal Algorithm 1: the first A=0-based condition passes even if M=0.
    a, mask, output_probability = [[0]], [[0]], [[0.0]]
    condition_on_other_a = True  # No other row or column has a nonzero entry.
    if condition_on_other_a:
        a[0][0] = 1
    record("algorithm1_missing_mask_counterexample", a[0][0] == 1 and mask[0][0] == 0,
           {"probability": output_probability[0][0], "threshold": 0.2, "literal_A": a,
            "status": "Printed condition can select a forbidden edge."})

    return {"status": "formula_reading_audit_only", "checks_passed": sum(c["passed"] for c in checks),
            "checks_total": len(checks), "checks": checks,
            "caveat": "Confirmed numerical issues do not establish the authors' implementation or trained performance."}


if __name__ == "__main__":
    result = audit()
    destination = Path(__file__).with_name("formula_audit_results.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["checks_passed"], "total": result["checks_total"], "output": str(destination)}, ensure_ascii=False))
    if result["checks_passed"] != result["checks_total"]:
        raise SystemExit(1)
