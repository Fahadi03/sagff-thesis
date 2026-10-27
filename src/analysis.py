"""JSON run logs -> tables and figures (docs/PROTOCOL.md §7, §10, R8).

Every number comes from code: run results from results/runs/*.json only; the two
quantities the run logs do not contain - the leakage dose (T3) and the data-
cleaning summary (T5) - come from the existing generators that compute them from
the split files and raw data (scripts/leakage_dose.py, src/data.py). Nothing is
typed by hand.

Run anomalies are read from results/runs_anomalies.md (its machine-readable
block), never from a constant here, and exclude a run only from the table kinds
they name (here: timing). Every run stays in every pre-specified analysis.

Pre-specified analyses (§10), computed exactly as registered:
  S1   per track: Spearman(k rank, sagff gate displacement), n = 20
  S2a  Wilcoxon: sagff vs sagff-noise-perepoch gate displacement, n = 60
  S2b  Wilcoxon: sagff vs sagff-permuted-fixed gate displacement, n = 60
  S3   Spearman(sagff displacement, Macro-F1 sagff - baseline-cap), n = 60
  S4   Wilcoxon: concat-cap vs baseline-cap test Macro-F1, n = 60
  S5   Wilcoxon: sagff vs concat-cap test Macro-F1, n = 60
on three gate readouts: best epoch (primary), final epoch and convergence-anchored
(sensitivity); S4 and S5 do not involve the gate and are identical on all three.
All tests two-sided, alpha 0.05, scipy.stats.spearmanr / wilcoxon(zero_method="wilcox").

Holm family size. §10 says "the nine tests below (S1 x 3 tracks, S2 x 2, S3, S4,
S5)", which enumerates EIGHT tests. The main tables use m = 9 (the stated count,
the more conservative reading, per §9 "take the option that weakens our claim");
m = 8 (the enumerated list) is reported in the exploratory section only. Holm is
applied within each readout's family.

Status of a registered prediction (from the Holm-adjusted p, m = 9):
  met              significant, observed direction = predicted
  failed           significant, observed direction opposite to predicted
  non-significant  Holm-adjusted p >= 0.05
  no prediction    S2b (none registered)

Usage:  python -m src.analysis all        # every table and figure
        python -m src.analysis timing     # the timing table only
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.train import GATE_BINS

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
ANOMALIES = ROOT / "results" / "runs_anomalies.md"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
TRACKS = ("bnintent30-bn", "bnintent30-en", "banking77")
KS = ("5", "10", "20", "full")
K_RANK = {k: i + 1 for i, k in enumerate(KS)}
SEEDS = (13, 42, 87, 100, 2026)
HEADS = ("baseline", "concat", "baseline-cap", "concat-cap", "sagff")
CONTROLS = ("sagff-noise-perepoch", "sagff-permuted-fixed")
READOUTS = {
    "best epoch (primary)": lambda g: g["gate_displacement"]["mean"],
    "final epoch (sensitivity)": lambda g: g["gate_displacement_final_epoch"]["mean"],
    "convergence-anchored (sensitivity)":
        lambda g: (g["gate_displacement_converged"] or {}).get("mean"),
}
HOLM_M_STATED, HOLM_M_ENUMERATED = 9, 8
ALPHA = 0.05


def n5_note(n: int = len(SEEDS)) -> str:
    """The per-cell n = 5 note on attainable p-values. Every number in it is computed."""
    min_p = 2 / 2 ** n
    n_s1 = len(KS) * len(SEEDS)
    n_pair = len(TRACKS) * len(KS) * len(SEEDS)
    return (f"**Note on n = {n}.** Each cell of this table summarises n = {n} seeds. The smallest attainable "
            f"two-sided exact p for n = {n} with no zero differences is {min_p:.4f}, so no per-cell test can "
            f"reach alpha = {ALPHA} by construction. These cells are reported for effect size and seed "
            f"consistency; all inferential weight rests on the registered tests at n = {n_s1} and "
            f"n = {n_pair} (docs/PROTOCOL.md §10, `T2_preregistered_tests.md`).")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_runs() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*.json"))]


def load_exclusions() -> list[dict]:
    """The machine-readable block of results/runs_anomalies.md (a JSON list)."""
    if not ANOMALIES.exists():
        return []
    text = ANOMALIES.read_text(encoding="utf-8")
    blocks = re.findall(r"## Machine-readable exclusions\s+```json\s*(.*?)```", text, flags=re.S)
    if len(blocks) != 1:
        raise ValueError(f"{ANOMALIES} must contain exactly one machine-readable exclusions block")
    return json.loads(blocks[0])


def excluded_from(kind: str) -> dict[str, str]:
    """{run_id: reason} for runs excluded from tables of `kind` (e.g. "timing")."""
    return {e["run_id"]: e["reason"] for e in load_exclusions() if kind in e["exclude_from"]}


def index(runs: list[dict]) -> dict:
    """{(track, k, seed, condition, dirty): log} for runs with status ok."""
    out = {}
    for log in runs:
        if log["status"] == "ok":
            c = log["config"]
            out[(c["track"], c["k"], c["seed"], c["condition"], c["dirty"])] = log
    return out


def converged_threshold(R: dict) -> float:
    """training.converged_loss_threshold, as recorded in the run logs."""
    return next(iter(R.values()))["config"]["training"]["converged_loss_threshold"]


def f1(log: dict) -> float:
    return log["metrics"]["test"]["macro_f1"]


def optimizer_steps(log: dict) -> int:
    n, bs = log["splits"]["sizes"]["train"], log["config"]["training"]["batch_size"]
    return log["training_log"]["epochs_run"] * (-(-n // bs))


def msd(vals: list[float], fmt: str = "{:.3f}") -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    sd_fmt = fmt.replace("+", "")  # an SD has no sign
    return f"{fmt.format(statistics.mean(vals))} ± {sd_fmt.format(sd)}" + ("" if len(vals) == len(SEEDS) else f" (n={len(vals)})")


def write(name: str, lines: list[str]) -> str:
    text = "\n".join(lines) + "\n"
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_bytes(text.encode("utf-8"))
    return text


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def spearman(x: list[float], y: list[float]) -> dict:
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return {"kind": "Spearman", "statistic": float(r.statistic), "rho": float(r.statistic),
            "p": float(r.pvalue), "n": len(x), "n_nonzero": None, "median_diff": None,
            "rank_biserial": None}


def wilcoxon(x: list[float], y: list[float]) -> dict:
    """Two-sided paired Wilcoxon on x - y, zero_method='wilcox' (zeros dropped)."""
    from scipy.stats import rankdata
    from scipy.stats import wilcoxon as sp_wilcoxon
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    r = sp_wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    ranks = rankdata(np.abs(nz))
    t_pos, t_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    rb = float((t_pos - t_neg) / (t_pos + t_neg)) if len(nz) else 0.0
    return {"kind": "Wilcoxon", "statistic": float(r.statistic), "rho": None, "p": float(r.pvalue),
            "n": len(d), "n_nonzero": int(len(nz)), "median_diff": float(np.median(d)),
            "rank_biserial": rb}


def holm(ps: list[float], m: int) -> list[float]:
    """Holm step-down adjusted p-values for len(ps) <= m hypotheses."""
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for j, i in enumerate(order):
        running = max(running, min(1.0, (m - j) * ps[i]))
        adj[i] = running
    return adj


def _sign(v: float) -> str:
    return "> 0" if v > 0 else ("< 0" if v < 0 else "0")


def observed(res: dict) -> str:
    """Spearman: sign of rho. Wilcoxon: sign of the rank-biserial correlation, UNLESS
    the median paired difference has the other sign; then the direction is reported
    as ambiguous with both signs, never resolved in either favour."""
    if res["kind"] == "Spearman":
        return _sign(res["rho"])
    rb, med = _sign(res["rank_biserial"]), _sign(res["median_diff"])
    if rb != med:
        return f"ambiguous: median paired difference {med}, rank-biserial {rb}"
    return rb


def status(predicted: str | None, obs: str, p_holm: float) -> str:
    if predicted is None:
        return "no prediction"
    if p_holm >= ALPHA:
        return "non-significant"
    if obs.startswith("ambiguous"):
        return "significant, direction ambiguous"
    return "met" if obs == predicted else "failed"


# --------------------------------------------------------------------------- #
# the registered tests
# --------------------------------------------------------------------------- #
def registered_tests(R: dict, dropped: list) -> dict:
    """{readout: [test dict, ...]} for the eight registered tests on each readout."""
    out = {}
    for readout, get in READOUTS.items():
        tests = []
        # S1 per track
        for t in TRACKS:
            xs, ys = [], []
            for k in KS:
                for s in SEEDS:
                    log = R.get((t, k, s, "sagff", False))
                    v = get(log["gate"]) if log else None
                    if v is None:
                        dropped.append((f"S1 {t}", readout, f"k={k} seed={s}",
                                        "run missing" if log is None else "no convergence-anchored readout"))
                        continue
                    xs.append(K_RANK[k])
                    ys.append(v)
            tests.append({"test": f"S1 {t}", "comparison": "Spearman(k rank, sagff displacement)",
                          "predicted": "> 0", **spearman(xs, ys)})
        # S2a / S2b
        for name, ctrl, pred in (("S2a", CONTROLS[0], "< 0"), ("S2b", CONTROLS[1], None)):
            a, b = [], []
            for t in TRACKS:
                for k in KS:
                    for s in SEEDS:
                        la, lb = R.get((t, k, s, "sagff", False)), R.get((t, k, s, ctrl, False))
                        va = get(la["gate"]) if la else None
                        vb = get(lb["gate"]) if lb else None
                        if va is None or vb is None:
                            why = [w for w, v in (("sagff", va), (ctrl, vb)) if v is None]
                            dropped.append((name, readout, f"{t} k={k} seed={s}",
                                            "no convergence-anchored readout for " + " and ".join(why)))
                            continue
                        a.append(va)
                        b.append(vb)
            tests.append({"test": name, "comparison": f"sagff - {ctrl} (displacement)",
                          "predicted": pred, **wilcoxon(a, b)})
        # S3
        xs, ys = [], []
        for t in TRACKS:
            for k in KS:
                for s in SEEDS:
                    ls, lc = R.get((t, k, s, "sagff", False)), R.get((t, k, s, "baseline-cap", False))
                    v = get(ls["gate"]) if ls else None
                    if v is None or lc is None:
                        dropped.append(("S3", readout, f"{t} k={k} seed={s}",
                                        "no convergence-anchored readout for sagff" if ls else "run missing"))
                        continue
                    xs.append(v)
                    ys.append(f1(ls) - f1(lc))
        tests.append({"test": "S3", "comparison": "Spearman(sagff displacement, F1 sagff - baseline-cap)",
                      "predicted": "< 0", **spearman(xs, ys)})
        # S4, S5 (no gate involved: identical on every readout)
        for name, first, second in (("S4", "concat-cap", "baseline-cap"), ("S5", "sagff", "concat-cap")):
            a, b = [], []
            for t in TRACKS:
                for k in KS:
                    for s in SEEDS:
                        la, lb = R.get((t, k, s, first, False)), R.get((t, k, s, second, False))
                        if la is None or lb is None:
                            dropped.append((name, readout, f"{t} k={k} seed={s}", "run missing"))
                            continue
                        a.append(f1(la))
                        b.append(f1(lb))
            tests.append({"test": name, "comparison": f"{first} - {second} (test Macro-F1)",
                          "predicted": "> 0", **wilcoxon(a, b)})
        ps = [x["p"] for x in tests]
        for x, h9, h8 in zip(tests, holm(ps, HOLM_M_STATED), holm(ps, HOLM_M_ENUMERATED)):
            x["p_holm"], x["p_holm_m8"] = h9, h8
            x["observed"] = observed(x)
            x["status"] = status(x["predicted"], x["observed"], x["p_holm"])
            x["status_m8"] = status(x["predicted"], x["observed"], x["p_holm_m8"])
        out[readout] = tests
    return out


def descriptive_control_trends(R: dict) -> list[dict]:
    """§10: the S1 correlation for both controls, registered as DESCRIPTIVE (no Holm)."""
    rows = []
    for readout, get in READOUTS.items():
        for ctrl in CONTROLS:
            for t in TRACKS:
                xs, ys = [], []
                for k in KS:
                    for s in SEEDS:
                        log = R.get((t, k, s, ctrl, False))
                        v = get(log["gate"]) if log else None
                        if v is not None:
                            xs.append(K_RANK[k])
                            ys.append(v)
                rows.append({"readout": readout, "condition": ctrl, "track": t, **spearman(xs, ys)})
    return rows


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def fmt_opt(v, f="{:.4f}") -> str:
    return "" if v is None else f.format(v)


def t1(R: dict) -> str:
    L = ["# T1. Test Macro-F1", "", "Generated by `python -m src.analysis all`. Do not edit by hand.", "",
         f"Mean ± SD (sample SD, ddof = 1) over seeds {list(SEEDS)}; clean condition.", "",
         n5_note(), "",
         "| track | k | " + " | ".join(HEADS) + " |", "|" + " --- |" * (2 + len(HEADS))]
    for t in TRACKS:
        for k in KS:
            L.append(f"| {t} | {k} | " + " | ".join(
                msd([f1(R[(t, k, s, h, False)]) for s in SEEDS if (t, k, s, h, False) in R]) for h in HEADS) + " |")
    return write("T1_test_macro_f1.md", L)


def t2(tests: dict, desc: list[dict]) -> str:
    head = ["test", "comparison", "n", "n non-zero", "statistic", "rho", "median paired difference",
            "rank-biserial", "p", f"Holm p (m={HOLM_M_STATED})", "predicted", "observed", "status"]
    L = ["# T2. Pre-specified analyses (docs/PROTOCOL.md §10)", "",
         "Generated by `python -m src.analysis all`. Do not edit by hand.", "",
         "All tests two-sided, alpha = 0.05. Wilcoxon: statistic W from `scipy.stats.wilcoxon` "
         "(zero_method='wilcox'), differences taken as first minus second in the comparison column, "
         "observed direction = sign of the matched-pairs rank-biserial correlation. Spearman: statistic "
         "= rho. Holm-adjusted p within each readout's family, m = 9 as stated in §10 (the §10 list "
         "enumerates eight tests; m = 8 is reported in the exploratory section). Status uses the "
         "Holm-adjusted p (m = 9).", ""]
    amb = sorted({x["test"] for rows in tests.values() for x in rows if x["observed"].startswith("ambiguous")})
    L += ["**Direction ambiguity.** For a Wilcoxon test whose median paired difference and matched-pairs "
          "rank-biserial correlation disagree in sign, the observed direction is reported as ambiguous "
          "with both signs; neither sign is chosen. Tests affected: "
          + (", ".join(amb) if amb else "none") + ". Its status is determined by the Holm-adjusted p "
          "as for every other test.", ""]
    for readout, rows in tests.items():
        L += [f"## Readout: {readout}", "", "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        for x in rows:
            L.append("| " + " | ".join([
                x["test"], x["comparison"], str(x["n"]), "" if x["n_nonzero"] is None else str(x["n_nonzero"]),
                f"{x['statistic']:.4f}", fmt_opt(x["rho"]), fmt_opt(x["median_diff"]),
                fmt_opt(x["rank_biserial"]), fmt_p(x["p"]), fmt_p(x["p_holm"]),
                x["predicted"] or "none registered", x["observed"], x["status"]]) + " |")
        L.append("")
    L += ["---", "", "## Registered as descriptive (§10): S1 correlation for the two controls", "",
          "Not part of the Holm family; no prediction registered.", "",
          "| readout | condition | track | n | rho | p |", "| --- | --- | --- | --- | --- | --- |"]
    for r in desc:
        L.append(f"| {r['readout']} | {r['condition']} | {r['track']} | {r['n']} | {r['rho']:.4f} | {fmt_p(r['p'])} |")
    L += ["", "---", "", "## EXPLORATORY (not registered in §10)", "",
          "Holm adjustment with m = 8, the number of tests the §10 list actually enumerates "
          "(the registered text states m = 9).", "",
          "| readout | test | p | Holm p (m=8) | status with m=9 | status with m=8 |",
          "| --- | --- | --- | --- | --- | --- |"]
    n_diff = 0
    for readout, rows in tests.items():
        for x in rows:
            n_diff += x["status"] != x["status_m8"]
            L.append(f"| {readout} | {x['test']} | {fmt_p(x['p'])} | {fmt_p(x['p_holm_m8'])} | "
                     f"{x['status']} | {x['status_m8']} |")
    n_all = sum(len(r) for r in tests.values())
    L += ["", f"Tests whose status differs between m = 9 and m = 8: {n_diff} of {n_all}."]
    return write("T2_preregistered_tests.md", L)


def t3(R: dict) -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from leakage_dose import BASES, dose
    L = ["# T3. Leaky condition against clean condition (bnintent30-bn)", "",
         "Generated by `python -m src.analysis all`. Do not edit by hand.", "",
         "Test Macro-F1, mean ± SD (ddof = 1) over seeds; the difference is leaky minus clean for the "
         "same k and seed, then mean ± SD over seeds. Leaky = official assignment after step 1, steps 2-4 "
         "not applied; its test split has more rows than the clean one, so the two columns are scored "
         "on different test sets. Leakage dose (computed from the dirty split files by "
         "`scripts/leakage_dose.py`, not from run logs): test rows whose dedup key also occurs in the "
         "run's train split, mean (min-max) over seeds.", "",
         n5_note(), "",
         "| k | head | clean | leaky | leaky − clean, paired by seed | leakage dose, test rows with key in train |",
         "| --- | --- | --- | --- | --- | --- |"]
    for k in KS:
        d = [dose(BASES["dirty"], k, s) for s in SEEDS]
        dose_txt = (f"{statistics.mean(x['train'] for x in d):.1f} "
                    f"({min(x['train'] for x in d)}-{max(x['train'] for x in d)}) of {d[0]['n_test']}")
        for h in ("sagff", "baseline"):
            c = [f1(R[("bnintent30-bn", k, s, h, False)]) for s in SEEDS]
            l = [f1(R[("bnintent30-bn", k, s, h, True)]) for s in SEEDS]
            paired = [b - a for a, b in zip(c, l)]  # same k, same seed: leaky minus clean
            L.append(f"| {k} | {h} | {msd(c)} | {msd(l)} | {msd(paired, '{:+.3f}')} | {dose_txt} |")
    return write("T3_leaky_vs_clean.md", L)


def t4(runs: list[dict]) -> str:
    counts = defaultdict(set)
    for log in runs:
        if log["status"] == "ok":
            counts[(log["config"]["head"], log["config"]["track"])].add(log["head"]["param_count"])
    L = ["# T4. Head parameter counts", "", "Generated by `python -m src.analysis all`. Do not edit by hand.",
         "", "Trainable parameters, read from the run logs (`head.param_count`).", "",
         "| head | " + " | ".join(TRACKS) + " |", "|" + " --- |" * (1 + len(TRACKS))]
    for h in HEADS:
        L.append(f"| {h} | " + " | ".join(", ".join(f"{v:,}" for v in sorted(counts[(h, t)])) for t in TRACKS) + " |")
    return write("T4_head_parameters.md", L)


def t5() -> str:
    from src import data as D
    tracks, rep = D.build_tracks()
    L = ["# T5. Data-cleaning summary", "", "Generated by `python -m src.analysis all` from `src/data.py` "
         "(raw data, not run logs). Do not edit by hand. Full report: `results/tables/data_cleaning.md`.", "",
         "## Rows removed per step (dedup key, applied), train/val/test", "",
         "| dataset | step | rule | removed (train/val/test) | remaining (train/val/test) |",
         "| --- | --- | --- | --- | --- |"]
    for name, dr in (("BNIntent30", rep["bnintent30"]), ("BANKING77", rep["banking77"])):
        L.append(f"| {name} | raw | | | {sum(dr['raw'].values())} ({D._fmt(dr['raw'])}) |")
        for s in dr["steps"]:
            L.append(f"| {name} | {s['step']} | {s['name']} | {sum(s['removed'].values())} ({D._fmt(s['removed'])}) "
                     f"| {sum(s['remaining'].values())} ({D._fmt(s['remaining'])}) |")
        L.append(f"| {name} | final | | | {sum(dr['final'].values())} ({D._fmt(dr['final'])}) |")
    L += ["", "## Duplicate counts under the comparison-key levels (steps 3 and 4, one track's text at a time)", ""]
    L += D._key_level_table(rep)
    L += ["", "## Smallest training classes after cleaning", "", "| track | 1st | 2nd | 3rd |", "| --- | --- | --- | --- |"]
    for name, t in tracks.items():
        L.append(f"| {name} | " + " | ".join(f"{c} ({n})" for c, n in D.smallest_train_classes(t)) + " |")
    return write("T5_data_cleaning_summary.md", L)


def t6(R: dict) -> str:
    cols = [("init", lambda g: g["gate_init_mean"]), ("best", lambda g: g["gate_mean"]),
            ("final", lambda g: g["gate_final_mean"]), ("converged", lambda g: g.get("gate_converged_mean")),
            ("displacement best", READOUTS["best epoch (primary)"]),
            ("displacement final", READOUTS["final epoch (sensitivity)"]),
            ("displacement converged", READOUTS["convergence-anchored (sensitivity)"])]
    L = ["# T6. Gate statistics (sagff, test split)", "",
         "Generated by `python -m src.analysis all`. Do not edit by hand.", "",
         "Gate mean over all test examples and dimensions, mean ± SD over seeds (n shown where fewer "
         "than 5 runs have the readout): at initialisation, at the best (early-stopping) epoch, at the "
         f"final trained epoch, at the first epoch with training loss < {converged_threshold(R)} (converged); and the "
         "displacement (readout minus initialisation).", "",
         n5_note(), "",
         "| track | k | " + " | ".join(c for c, _ in cols) + " |", "|" + " --- |" * (2 + len(cols))]
    for t in TRACKS:
        for k in KS:
            gs = [R[(t, k, s, "sagff", False)]["gate"] for s in SEEDS if (t, k, s, "sagff", False) in R]
            L.append(f"| {t} | {k} | " + " | ".join(msd([fn(g) for g in gs], "{:.4f}") for _, fn in cols) + " |")
    return write("T6_gate_statistics.md", L)


def t7(R: dict) -> tuple[str, dict]:
    """Descriptive: per (track, k) paired Macro-F1 differences with an uncorrected,
    exploratory Wilcoxon per cell. Not part of the registered family."""
    pairs = (("T7a", "sagff", "baseline-cap"), ("T7b", "baseline", "baseline-cap"),
             ("T7c", "sagff", "baseline"))
    min_p = 2 / 2 ** len(SEEDS)
    L = ["# T7. Test Macro-F1 differences by track and k (DESCRIPTIVE; not part of the registered family)", "",
         "Generated by `python -m src.analysis all`. Do not edit by hand.", "",
         f"Paired differences over seeds {list(SEEDS)} (same track, k, seed), mean ± SD (ddof = 1). "
         "Per cell: two-sided paired Wilcoxon signed-rank (`scipy.stats.wilcoxon`, zero_method='wilcox'), "
         "**EXPLORATORY, without any multiplicity correction**. With n = 5 pairs and no zero differences "
         f"the smallest attainable two-sided exact p is {min_p:.4f}.", "", n5_note(), ""]
    data = {}
    for label, a, b in pairs:
        L += [f"## {label}. {a} − {b}", "",
              "| track | k | mean ± SD | median paired difference | n | n non-zero | W | p (exploratory, uncorrected) | rank-biserial |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for t in TRACKS:
            for k in KS:
                da = [f1(R[(t, k, s, a, False)]) for s in SEEDS]
                db = [f1(R[(t, k, s, b, False)]) for s in SEEDS]
                d = [x - y for x, y in zip(da, db)]
                w = wilcoxon(da, db)
                data[(a, b, t, k)] = d
                L.append(f"| {t} | {k} | {msd(d, '{:+.3f}')} | {w['median_diff']:+.4f} | {w['n']} | "
                         f"{w['n_nonzero']} | {w['statistic']:.1f} | {fmt_p(w['p'])} | {w['rank_biserial']:+.4f} |")
        L.append("")
    return write("T7_macro_f1_differences.md", L), data


def headline(R: dict) -> str:
    """Gate displacement and the two Macro-F1 gaps of sagff, side by side."""
    disp = READOUTS["best epoch (primary)"]
    L = ["# Headline numbers by track and k", "", "Generated by `python -m src.analysis all`. Do not edit by hand.",
         "", f"Mean ± SD (ddof = 1) over seeds {list(SEEDS)}. Gate displacement: `sagff`, best-epoch readout "
         "(gate mean on the test split minus the gate mean at initialisation). Macro-F1 differences: test "
         "Macro-F1 of `sagff` minus that of the named head, paired by seed (same track, k, seed).", "",
         n5_note(), "",
         "| track | k | sagff gate displacement | sagff − baseline (Macro-F1) | sagff − baseline-cap (Macro-F1) |",
         "| --- | --- | --- | --- | --- |"]
    for t in TRACKS:
        for k in KS:
            sg = [R[(t, k, s, "sagff", False)] for s in SEEDS]
            gap = lambda h: [f1(x) - f1(R[(t, k, s, h, False)]) for x, s in zip(sg, SEEDS)]
            L.append(f"| {t} | {k} | {msd([disp(x['gate']) for x in sg], '{:+.4f}')} | "
                     f"{msd(gap('baseline'), '{:+.3f}')} | {msd(gap('baseline-cap'), '{:+.3f}')} |")
    return write("headline_numbers.md", L)


def predictions_table(tests: dict) -> str:
    L = ["# Registered predictions: observed outcome", "",
         "Generated by `python -m src.analysis all`. Do not edit by hand. Status from the Holm-adjusted "
         "p (m = 9), alpha 0.05.", "",
         "| readout | test | predicted | observed | p | Holm p (m=9) | status |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for readout, rows in tests.items():
        for x in rows:
            L.append(f"| {readout} | {x['test']} | {x['predicted'] or 'none registered'} | {x['observed']} | "
                     f"{fmt_p(x['p'])} | {fmt_p(x['p_holm'])} | {x['status']} |")
    return write("predictions_vs_observed.md", L)


def timing_table() -> str:
    """Wall time and per-step time by (track, condition, k), timing anomalies excluded."""
    excluded = excluded_from("timing")
    cells = defaultdict(list)
    for log in load_runs():
        if log["status"] != "ok" or log["run_id"] in excluded:
            continue
        c = log["config"]
        key = (c["track"], c["condition"] + (" (dirty)" if c["dirty"] else ""), c["k"])
        cells[key].append((log["wall_clock_s"], 1000 * log["wall_clock_s"] / optimizer_steps(log)))
    L = ["# Run timing", "", "Generated by `python -m src.analysis timing`. Do not edit by hand.", "",
         "Wall-clock time per run and per optimizer step, mean (min-max) over seeds. `wall_clock_s` is "
         "wall time, not a compute measurement (docs/PROTOCOL.md §9). Excluded from this table only, as "
         "recorded in `results/runs_anomalies.md`:"]
    L += [f"- `{rid}`: {reason}" for rid, reason in sorted(excluded.items())] or ["- none"]
    L += ["", "| track | condition | k | runs | wall s | ms per step |", "| --- | --- | --- | --- | --- | --- |"]
    for (t, c, k) in sorted(cells, key=lambda x: (x[0], x[1], KS.index(x[2]))):
        w = [x[0] for x in cells[(t, c, k)]]
        m = [x[1] for x in cells[(t, c, k)]]
        L.append(f"| {t} | {c} | {k} | {len(w)} | {statistics.mean(w):.1f} ({min(w):.1f}-{max(w):.1f}) | "
                 f"{statistics.mean(m):.1f} ({min(m):.1f}-{max(m):.1f}) |")
    return write("run_timing.md", L)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "v"]
SINGLE_COL, DOUBLE_COL = 3.35, 6.9
X_LABEL = "Training examples per class, k"
DISP_LABEL = "Gate displacement (unitless)"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    return plt


def _style(i: int) -> dict:
    return {"color": OKABE_ITO[i % len(OKABE_ITO)], "linestyle": LINESTYLES[i % len(LINESTYLES)],
            "marker": MARKERS[i % len(MARKERS)], "markersize": 4, "linewidth": 1.2, "capsize": 2,
            "elinewidth": 0.8}


def _legend_below(fig, axes, ncol: int) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=ncol, frameon=False)


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    # no CreationDate: the PDF bytes then depend only on the data (reproducible figures)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")


def _series(R, track, condition, value):
    """Per k: (mean, sd, n) over seeds of value(log); None values skipped."""
    out = []
    for k in KS:
        vals = [value(R[(track, k, s, condition, False)]) for s in SEEDS if (track, k, s, condition, False) in R]
        vals = [v for v in vals if v is not None]
        out.append((statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals)))
    return out


def _k_axis(ax):
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels(KS)
    ax.set_xlabel(X_LABEL)


def figures(R: dict) -> list[str]:
    plt = _mpl()
    caps = []
    disp = READOUTS["best epoch (primary)"]
    err = ("Error bars: ±1 sample standard deviation (ddof = 1) over the five seeds "
           f"({', '.join(map(str, SEEDS))}).")

    # F1 -------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    for i, t in enumerate(TRACKS):
        ser = _series(R, t, "sagff", lambda l: disp(l["gate"]))
        off = (i - 1) * 0.06
        ax.errorbar([j + off for j in range(len(KS))], [m for m, _, _ in ser], yerr=[s for _, s, _ in ser],
                    label=t, **_style(i))
    ax.axhline(0, color="0.5", linewidth=0.6, linestyle=(0, (1, 2)))
    _k_axis(ax)
    ax.set_ylabel(DISP_LABEL)
    ax.legend(frameon=False)
    _save(fig, "F1_gate_displacement_vs_k")
    plt.close(fig)
    ns = sorted({n for t in TRACKS for _, _, n in _series(R, t, "sagff", lambda l: disp(l["gate"]))})
    caps.append(("F1", "Gate displacement of SAGFF against the number of training examples per class (k), "
                 "one line per dataset track. Displacement = mean gate value on the test split at the epoch "
                 "selected by early stopping minus the mean gate value at initialisation; negative values "
                 "mean a shift toward the handcrafted-feature branch, positive values toward the encoder. "
                 f"Points: mean over seeds (n = {', '.join(map(str, ns))} runs per point). {err} Points are "
                 "offset horizontally for legibility; the dotted line marks zero displacement."))

    # F2 -------------------------------------------------------------------- #
    fig, axes = plt.subplots(1, len(TRACKS), figsize=(DOUBLE_COL, 2.4), sharey=True)
    conds = ("sagff",) + CONTROLS
    for ax, t in zip(axes, TRACKS):
        for i, c in enumerate(conds):
            ser = _series(R, t, c, lambda l: disp(l["gate"]))
            off = (i - 1) * 0.06
            ax.errorbar([j + off for j in range(len(KS))], [m for m, _, _ in ser],
                        yerr=[s for _, s, _ in ser], label=c, **_style(i))
        ax.axhline(0, color="0.5", linewidth=0.6, linestyle=(0, (1, 2)))
        ax.set_title(t)
        _k_axis(ax)
    axes[0].set_ylabel(DISP_LABEL)
    _legend_below(fig, axes, len(conds))
    _save(fig, "F2_gate_displacement_with_controls")
    plt.close(fig)
    caps.append(("F2", "Gate displacement (best-epoch readout, as in F1) against k for SAGFF with real "
                 "handcrafted features and for the two permutation controls: per-epoch re-permuted features "
                 "(sagff-noise-perepoch) and one fixed permutation per split (sagff-permuted-fixed). One panel "
                 f"per dataset track. Points: mean over five seeds. {err} Points are offset horizontally for "
                 "legibility."))

    # F3 -------------------------------------------------------------------- #
    fig, axes = plt.subplots(1, len(TRACKS), figsize=(DOUBLE_COL, 2.4), sharey=True)
    for ax, t in zip(axes, TRACKS):
        for i, h in enumerate(HEADS):
            ser = _series(R, t, h, f1)
            off = (i - 2) * 0.05
            ax.errorbar([j + off for j in range(len(KS))], [m for m, _, _ in ser],
                        yerr=[s for _, s, _ in ser], label=h, **_style(i))
        ax.set_title(t)
        _k_axis(ax)
    axes[0].set_ylabel("Test Macro-F1 (proportion, 0-1)")
    _legend_below(fig, axes, len(HEADS))
    _save(fig, "F3_macro_f1_vs_k")
    plt.close(fig)
    caps.append(("F3", "Test Macro-F1 against k for the five heads, one panel per dataset track, clean "
                 f"condition. Points: mean over five seeds. {err} Points are offset horizontally for "
                 "legibility."))

    # F4 -------------------------------------------------------------------- #
    fig, axes = plt.subplots(1, len(TRACKS), figsize=(DOUBLE_COL, 2.3), sharey=True)
    edges = np.linspace(0, 1, GATE_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    for ax, t in zip(axes, TRACKS):
        for i, k in enumerate(("5", "full")):
            h = np.sum([R[(t, k, s, "sagff", False)]["gate"]["gate_hist"] for s in SEEDS], axis=0)
            st = _style(i)
            ax.step(centres, h / h.sum(), where="mid", color=st["color"], linestyle=st["linestyle"],
                    linewidth=1.2, label=f"k = {k}")
            ax.plot(centres, h / h.sum(), linestyle="none", marker=st["marker"], color=st["color"], markersize=3)
        ax.set_title(t)
        ax.set_xlabel("Gate value g (unitless, 0-1)")
    axes[0].set_ylabel("Share of gate values")
    _legend_below(fig, axes, 2)
    _save(fig, "F4_gate_histograms_k5_vs_full")
    plt.close(fig)
    caps.append(("F4", "Distribution of SAGFF gate values on the test split at the best (early-stopping) "
                 "epoch, k = 5 against the full training set, one panel per dataset track. Each curve pools "
                 "all gate values (every test example and every dimension) of the five seeds into 20 equal "
                 "bins on [0, 1] and shows each bin's share of the values; no error bars."))

    # F5 -------------------------------------------------------------------- #
    fig, axes = plt.subplots(1, len(KS), figsize=(DOUBLE_COL, 2.3), sharey=True)
    trunc = []
    for ax, k in zip(axes, KS):
        for i, t in enumerate(TRACKS):
            curves = [[c["gate_mean"] for c in R[(t, k, s, "sagff", False)]["training_log"]["curve"]]
                      for s in SEEDS]
            n_ep = min(len(c) for c in curves)
            trunc.append(f"{t} k={k}: {n_ep}")
            arr = np.array([c[:n_ep] for c in curves])
            mean, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
            ep = np.arange(1, n_ep + 1)
            st = _style(i)
            ax.plot(ep, mean, color=st["color"], linestyle=st["linestyle"], linewidth=1.2, label=t,
                    marker=st["marker"], markevery=max(1, n_ep // 6), markersize=3)
            ax.fill_between(ep, mean - sd, mean + sd, color=st["color"], alpha=0.15, linewidth=0)
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Training epoch")
    axes[0].set_ylabel("Gate mean on test split (unitless, 0-1)")
    _legend_below(fig, axes, len(TRACKS))
    _save(fig, "F5_gate_trajectory")
    plt.close(fig)
    caps.append(("F5", "Per-epoch trajectory of the SAGFF gate mean on the test split, one panel per k, "
                 "one line per dataset track. Lines: mean over the five seeds; shaded bands: ±1 sample "
                 "standard deviation (ddof = 1) over seeds. Runs stop at different epochs (early stopping), "
                 "so each series is truncated at the shortest of its five runs: " + "; ".join(trunc) + "."))

    # F6 -------------------------------------------------------------------- #
    comparisons = (("sagff", "baseline-cap"), ("sagff", "baseline"), ("baseline", "baseline-cap"))
    fig, axes = plt.subplots(1, len(comparisons), figsize=(DOUBLE_COL, 2.4), sharey=True)
    for ax, (a, b) in zip(axes, comparisons):
        for i, t in enumerate(TRACKS):
            means, sds = [], []
            for k in KS:
                d = [f1(R[(t, k, s, a, False)]) - f1(R[(t, k, s, b, False)]) for s in SEEDS]
                means.append(statistics.mean(d))
                sds.append(statistics.stdev(d))
            off = (i - 1) * 0.06
            ax.errorbar([j + off for j in range(len(KS))], means, yerr=sds, label=t, **_style(i))
        ax.axhline(0, color="0.3", linewidth=0.8, linestyle="-")
        ax.set_title(f"{a} − {b}")
        _k_axis(ax)
    axes[0].set_ylabel("Test Macro-F1 difference (proportion)")
    _legend_below(fig, axes, len(TRACKS))
    _save(fig, "F6_macro_f1_differences_vs_k")
    plt.close(fig)
    caps.append(("F6", "Descriptive. Paired differences in test Macro-F1 against k, one panel per comparison: "
                 "SAGFF minus the parameter-matched featureless head (baseline-cap), SAGFF minus the small "
                 "featureless head (baseline), and baseline minus baseline-cap; one line per dataset track. "
                 "Differences are paired by seed (same track, k and seed) before averaging. Points: mean paired "
                 f"difference over the five seeds. {err} The panels share one y-axis; the solid horizontal line "
                 "in each panel marks zero difference. Points are offset horizontally for legibility. Not part "
                 "of the registered analyses."))

    lines = ["# Figure captions", "", "Generated by `python -m src.analysis all` from the run logs. "
             "Do not edit by hand. Each figure is saved as PDF and 300-dpi PNG in results/figures/.", ""]
    for name, text in caps:
        lines += [f"**{name}.** {text}", ""]
    (FIGURES / "captions.md").write_bytes("\n".join(lines).encode("utf-8"))
    return [n for n, _ in caps]


# --------------------------------------------------------------------------- #
def run_all_analyses() -> dict:
    runs = load_runs()
    R = index(runs)
    failed = [log["run_id"] for log in runs if log["status"] != "ok"]
    dropped: list = []
    tests = registered_tests(R, dropped)
    desc = descriptive_control_trends(R)
    out = {"failed": failed, "dropped": dropped, "n_runs": len(runs)}
    out["T1"], out["T2"] = t1(R), t2(tests, desc)
    out["T3"], out["T4"], out["T5"], out["T6"] = t3(R), t4(runs), t5(), t6(R)
    out["predictions"] = predictions_table(tests)
    out["T7"], out["T7_data"] = t7(R)
    out["headline"] = headline(R)
    timing_table()
    out["figures"] = figures(R)
    thr = runs[0]["config"]["training"]["converged_loss_threshold"]
    conv = defaultdict(lambda: [0, 0])
    for log in runs:
        if log["status"] == "ok" and not log["config"]["dirty"]:
            c = log["config"]
            reached = any(e["train_loss"] < thr for e in log["training_log"]["curve"])
            conv[(c["condition"], c["track"])][0] += 0 if reached else 1
            conv[(c["condition"], c["track"])][1] += 1
    conds = ("sagff",) + CONTROLS + ("baseline-cap", "concat-cap", "baseline", "concat")
    L = ["# Run accounting", "", "Generated by `python -m src.analysis all`. Do not edit by hand.", "",
         f"- run logs: {len(runs)}; failed runs: {len(failed)}" + (f" ({', '.join(failed)})" if failed else ""),
         f"- pairs absent from the convergence-anchored readout of a registered test: {len(dropped)}", "",
         "## Convergence as an outcome", "",
         f"A run has a convergence-anchored readout only if its mean training loss fell below {thr} before "
         "early stopping ended it. Not reaching that threshold is an outcome of training under the run's "
         "condition, recorded here per condition; it is not an exclusion of a failed run. Clean condition, "
         "runs whose training loss never fell below the threshold / runs:", "",
         "| condition | " + " | ".join(TRACKS) + " | all tracks |", "|" + " --- |" * (2 + len(TRACKS))]
    for c in conds:
        cells = [conv[(c, t)] for t in TRACKS]
        L.append(f"| {c} | " + " | ".join(f"{a} / {b}" for a, b in cells)
                 + f" | {sum(a for a, _ in cells)} / {sum(b for _, b in cells)} |")
    L += ["", "## Pairs absent from the convergence-anchored readout", "",
          "| test | readout | unit | reason |", "| --- | --- | --- | --- |"]
    L += [f"| {a} | {b} | {c} | {d.replace('no convergence-anchored readout for', 'training loss never fell below the threshold for')} |"
          for a, b, c, d in dropped]
    out["accounting"] = write("run_accounting.md", L)
    return out


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "timing":
        print(timing_table())
    elif what == "all":
        out = run_all_analyses()
        print(f"runs {out['n_runs']}, failed {len(out['failed'])}, dropped {len(out['dropped'])}; "
              f"tables in {TABLES.relative_to(ROOT)}, figures {out['figures']} in {FIGURES.relative_to(ROOT)}")
    else:
        sys.exit(f"unknown analysis {what!r}; implemented: all, timing")


if __name__ == "__main__":
    main()
