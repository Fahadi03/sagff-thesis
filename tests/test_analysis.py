"""The statistics behind the registered analyses (docs/PROTOCOL.md §10), checked
against reference values computed independently of src/analysis.py: scipy called
directly, rank-based formulas written out by hand, and a worked Holm example."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src import analysis as A
from src.train import GATE_BINS, gate_stats

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Spearman
# --------------------------------------------------------------------------- #
def test_spearman_matches_scipy_and_the_rank_formula():
    x = [1, 1, 2, 2, 3, 3, 4, 4]
    y = [0.31, -0.20, 0.05, 0.40, 0.90, 0.12, 0.75, 0.66]
    res = A.spearman(x, y)
    ref = stats.spearmanr(x, y)
    assert res["rho"] == pytest.approx(ref.statistic, abs=1e-12)
    assert res["p"] == pytest.approx(ref.pvalue, abs=1e-12)
    assert res["n"] == len(x)
    # rho is the Pearson correlation of the (average) ranks
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    assert res["rho"] == pytest.approx(np.corrcoef(rx, ry)[0, 1], abs=1e-12)


def test_spearman_of_a_perfectly_increasing_series_is_one():
    assert A.spearman([1, 2, 3, 4, 5], [-0.3, -0.2, 0.0, 0.1, 0.4])["rho"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Wilcoxon and the matched-pairs rank-biserial correlation
# --------------------------------------------------------------------------- #
def test_wilcoxon_matches_scipy_on_the_differences():
    a = [0.71, 0.64, 0.80, 0.55, 0.62, 0.70, 0.66]
    b = [0.69, 0.65, 0.74, 0.50, 0.63, 0.61, 0.60]
    res = A.wilcoxon(a, b)
    ref = stats.wilcoxon(np.subtract(a, b), zero_method="wilcox", alternative="two-sided")
    assert res["statistic"] == pytest.approx(ref.statistic)
    assert res["p"] == pytest.approx(ref.pvalue, abs=1e-12)


def test_rank_biserial_worked_example():
    # d = a - b = [1, 2, -3, 4, 5]; |d| ranks 1..5; T+ = 1+2+4+5 = 12, T- = 3
    # rank-biserial = (T+ - T-) / (T+ + T-) = 9 / 15
    res = A.wilcoxon([1, 2, 0, 4, 5], [0, 0, 3, 0, 0])
    assert res["rank_biserial"] == pytest.approx(9 / 15)
    assert res["median_diff"] == pytest.approx(2.0)
    assert (res["n"], res["n_nonzero"]) == (5, 5)


def test_wilcoxon_drops_zero_differences_from_ranks_but_not_from_n():
    # d = [0, 1, -2]: zeros dropped (zero_method="wilcox"); ranks of |1|, |-2| are 1, 2
    res = A.wilcoxon([1, 2, 0], [1, 1, 2])
    assert (res["n"], res["n_nonzero"]) == (3, 2)
    assert res["rank_biserial"] == pytest.approx((1 - 2) / 3)
    assert res["median_diff"] == pytest.approx(0.0)


def test_all_positive_differences_give_rank_biserial_plus_one():
    a = np.arange(1, 11) / 100
    res = A.wilcoxon(list(a + 0.01), list(a))
    assert res["rank_biserial"] == pytest.approx(1.0)
    assert res["statistic"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Holm
# --------------------------------------------------------------------------- #
def test_holm_textbook_example():
    # Sorted p: 0.005, 0.01, 0.03, 0.04 with m = 4 -> 4*0.005 = 0.02, 3*0.01 = 0.03,
    # 2*0.03 = 0.06, 1*0.04 = 0.04 -> enforce monotonicity -> 0.06.
    adj = A.holm([0.01, 0.04, 0.03, 0.005], m=4)
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_with_a_family_larger_than_the_tests_given():
    # m = 9 over 2 p-values: 9*0.001 = 0.009, then max(0.009, 8*0.02 = 0.16)
    assert A.holm([0.02, 0.001], m=9) == pytest.approx([0.16, 0.009])


def test_holm_caps_at_one_and_never_lowers_a_p_value():
    ps = [0.2, 0.5, 0.9, 0.04]
    adj = A.holm(ps, m=9)
    assert all(a <= 1.0 for a in adj)
    assert all(a >= p for a, p in zip(adj, ps))


# --------------------------------------------------------------------------- #
# observed direction and status
# --------------------------------------------------------------------------- #
def test_observed_direction_rules():
    assert A.observed({"kind": "Spearman", "rho": 0.4}) == "> 0"
    assert A.observed({"kind": "Spearman", "rho": -0.4}) == "< 0"
    assert A.observed({"kind": "Wilcoxon", "rank_biserial": 0.8, "median_diff": 0.03}) == "> 0"
    amb = A.observed({"kind": "Wilcoxon", "rank_biserial": -0.08, "median_diff": 0.009})
    assert amb.startswith("ambiguous") and "> 0" in amb and "< 0" in amb


def test_status_rules():
    assert A.status(None, "> 0", 0.001) == "no prediction"
    assert A.status("> 0", "> 0", 0.0499) == "met"
    assert A.status("> 0", "< 0", 0.01) == "failed"
    assert A.status("> 0", "> 0", 0.05) == "non-significant"
    assert A.status("> 0", "ambiguous: x", 0.01) == "significant, direction ambiguous"


# --------------------------------------------------------------------------- #
# the registered tests on synthetic run logs with known answers
# --------------------------------------------------------------------------- #
def _synthetic_runs() -> dict:
    """Logs in which every registered quantity is known by construction."""
    R = {}
    for ti, t in enumerate(A.TRACKS):
        for k in A.KS:
            for si, s in enumerate(A.SEEDS):
                rank, j = A.K_RANK[k], 0.001 * si + 0.0001 * ti

                def log(disp, f1, conv=True):
                    gate = {"gate_displacement": {"mean": disp},
                            "gate_displacement_final_epoch": {"mean": disp - 0.01},
                            "gate_displacement_converged": {"mean": disp + 0.01} if conv else None}
                    return {"gate": gate, "metrics": {"test": {"macro_f1": f1}}}
                base = 0.5 + 0.05 * rank
                R[(t, k, s, "sagff", False)] = log(-0.3 + 0.1 * rank + j, base + 0.03 - 0.005 * rank - j)
                R[(t, k, s, "sagff-noise-perepoch", False)] = log(0.2 + j, base - 0.2)
                R[(t, k, s, "sagff-permuted-fixed", False)] = log(-0.5 - j, base - 0.3, conv=(si != 0))
                R[(t, k, s, "baseline-cap", False)] = log(0.0, base)
                R[(t, k, s, "concat-cap", False)] = log(0.0, base + 0.001 * (si - 2))
    return R


def test_registered_tests_pair_and_sign_correctly():
    R, dropped = _synthetic_runs(), []
    out = A.registered_tests(R, dropped)
    best = {x["test"]: x for x in out["best epoch (primary)"]}
    # S1: displacement increases strictly with k inside every seed
    for t in A.TRACKS:
        x = best[f"S1 {t}"]
        ref = stats.spearmanr([A.K_RANK[k] for k in A.KS for _ in A.SEEDS],
                              [R[(t, k, s, "sagff", False)]["gate"]["gate_displacement"]["mean"]
                               for k in A.KS for s in A.SEEDS])
        assert (x["n"], x["rho"]) == (20, pytest.approx(ref.statistic))
        assert x["observed"] == "> 0"
    # S2a: sagff displacement is below the per-epoch control in every pair
    assert best["S2a"]["n"] == 60 and best["S2a"]["rank_biserial"] == pytest.approx(-1.0)
    # S2b: sagff displacement is above the fixed control in every pair
    assert best["S2b"]["rank_biserial"] == pytest.approx(1.0) and best["S2b"]["status"] == "no prediction"
    # S3: displacement against the F1 gain over baseline-cap, recomputed directly
    xs = [R[(t, k, s, "sagff", False)]["gate"]["gate_displacement"]["mean"]
          for t in A.TRACKS for k in A.KS for s in A.SEEDS]
    ys = [R[(t, k, s, "sagff", False)]["metrics"]["test"]["macro_f1"]
          - R[(t, k, s, "baseline-cap", False)]["metrics"]["test"]["macro_f1"]
          for t in A.TRACKS for k in A.KS for s in A.SEEDS]
    assert best["S3"]["rho"] == pytest.approx(stats.spearmanr(xs, ys).statistic)
    # S5: sagff minus concat-cap, recomputed with scipy on the paired differences
    d = [R[(t, k, s, "sagff", False)]["metrics"]["test"]["macro_f1"]
         - R[(t, k, s, "concat-cap", False)]["metrics"]["test"]["macro_f1"]
         for t in A.TRACKS for k in A.KS for s in A.SEEDS]
    assert best["S5"]["p"] == pytest.approx(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    assert best["S5"]["median_diff"] == pytest.approx(float(np.median(d)))  # sagff minus concat-cap
    assert best["S5"]["observed"] == "> 0"
    # S4: concat-cap minus baseline-cap; the differences are 0.001 * (seed index - 2)
    d4 = [R[(t, k, s, "concat-cap", False)]["metrics"]["test"]["macro_f1"]
          - R[(t, k, s, "baseline-cap", False)]["metrics"]["test"]["macro_f1"]
          for t in A.TRACKS for k in A.KS for s in A.SEEDS]
    assert best["S4"]["median_diff"] == pytest.approx(float(np.median(d4)))
    assert best["S4"]["n_nonzero"] == sum(abs(x) > 1e-12 for x in d4)
    # Holm over the eight p-values of the readout with m = 9
    ps = [x["p"] for x in out["best epoch (primary)"]]
    assert [x["p_holm"] for x in out["best epoch (primary)"]] == pytest.approx(A.holm(ps, 9))


def test_runs_without_a_convergence_readout_are_dropped_from_that_readout_only():
    R, dropped = _synthetic_runs(), []
    out = A.registered_tests(R, dropped)
    s2b = {r: {x["test"]: x for x in xs}["S2b"] for r, xs in out.items()}
    # seed 13 of the fixed control has no convergence-anchored readout: 3 tracks x 4 k pairs
    assert s2b["convergence-anchored (sensitivity)"]["n"] == 60 - 12
    assert s2b["best epoch (primary)"]["n"] == 60
    assert sum(1 for d in dropped if d[0] == "S2b") == 12


# --------------------------------------------------------------------------- #
# gate statistics of a run (src.train.gate_stats) and the committed run logs
# --------------------------------------------------------------------------- #
def test_gate_stats_against_numpy():
    g = np.random.default_rng(0).uniform(0, 1, size=(7, 5))
    s = gate_stats(g, "gate_")
    assert s["gate_mean"] == round(float(g.mean()), 6)
    assert s["gate_std"] == round(float(g.std()), 6)
    assert s["gate_mean_per_dim"] == [round(float(v), 6) for v in g.mean(axis=0)]
    assert len(s["gate_hist"]) == GATE_BINS and sum(s["gate_hist"]) == g.size
    edges = np.linspace(0, 1, GATE_BINS + 1)
    assert s["gate_hist"] == [int(((g >= lo) & (g < hi)).sum()) for lo, hi in zip(edges[:-2], edges[1:-1])] \
        + [int(((g >= edges[-2]) & (g <= 1)).sum())]


def test_committed_displacements_equal_readout_minus_initialisation():
    """Every committed sagff-head run: each displacement equals its readout minus the
    gate mean at initialisation (R7), to the logged rounding."""
    problems = []
    for p in sorted((ROOT / "results" / "runs").glob("*.json")):
        log = json.loads(p.read_text(encoding="utf-8"))
        g = log["gate"]
        if not g:
            continue
        pairs = [("gate_displacement", "gate_mean"), ("gate_displacement_final_epoch", "gate_final_mean")]
        if g["gate_displacement_converged"] is not None:
            pairs.append(("gate_displacement_converged", "gate_converged_mean"))
        for derived, readout in pairs:
            if abs(g[derived]["mean"] - (g[readout] - g["gate_init_mean"])) > 2e-6:
                problems.append(f"{p.stem}: {derived}")
    assert not problems, problems[:10]
