# Pre-registration and disclosure: manuscript-ready text

The paragraphs below can be lifted into the manuscript (methods, or an appendix on
analysis provenance). Commit hashes refer to the project repository.

---

**Pre-specified analyses.** All confirmatory statistics (S1–S5: the k-trend of
the gate displacement, its comparison with two permutation controls, the
correlation between gate displacement and test-set gain, and the two
capacity-matched model comparisons), their predicted directions, the use of
two-sided tests, Holm correction across the family of nine tests, and the rule
for failed runs were fixed in the repository at commit `44bd224`, before any run
of the experimental grid.

**Disclosure.** After the analysis plan was fixed, a 30-run stopping-point
diagnostic was run and examined (the SAGFF model on all three datasets, k = 5 and
the full training set, all five seeds), to establish whether the gate statistics
depend on the epoch at which training is stopped. Because all runs are
deterministic, these 30 runs are identical to the corresponding runs of the full
grid; they form a subset of the data for the k-trend analysis (S1) and were seen
after the plan was fixed. In response to the stopping behaviour they revealed,
which is a property of the training procedure rather than the direction of any
result, two sensitivity readouts were added at commit `32b865c`: the gate
trajectory over all epochs, and a convergence-anchored gate readout taken at the
first epoch at which the mean training loss falls below 0.01. The primary readout
and all five statistics are unchanged from `44bd224`. The direction of the primary
k-trend was not examined before the full grid had completed. The diagnostic runs
were retained, not deleted; regenerating them would reproduce identical values.

**Commitment.** The k-trend in the 30 diagnostic runs was neither reported nor
interpreted before the full grid was complete; those runs were used only to answer
the stopping-point question for which they were run.

**Direction of the stopping-point effect.** At k = 5, the gate at the epoch
selected by early stopping lies above the gate at the final trained epoch in every
diagnostic run, while at the full training set the two agree within seed-to-seed
variation. Under the predicted direction of the k-trend, the primary
(best-epoch) readout therefore yields a smaller trend than the final-epoch readout
would: the stopping point can only suppress the predicted effect, not inflate it.
We report the k-trend on all three readouts (best epoch, final epoch,
convergence-anchored).

**Decisions made after registration.** Four decisions were taken after the
analysis plan was registered at commit `44bd224` (11 October 2026). Before
registration, the only test-set results in existence were one smoke-test run
(commit `9b2036f`) and one unlogged sanity run of the fixed-permutation control;
the first test-set results after registration were those of the stopping
diagnostic described above (commit `9e44c19`, 12 October 2026).

1. *Feature-dimension re-selection* (commit `7e7f8f6`, 11 October 2026). Made after
   registration but before the stopping diagnostic and before any grid run, on
   validation data only; no test file was opened. The selection grid was extended
   downward until the chosen configuration was bracketed.
2. *Learning-rate re-selection* (commit `fdb9e52`, 12 October 2026). Made after
   registration but before the stopping diagnostic and before any grid run, on
   validation data only and with the baseline head only; test data was never
   loaded. The grid was extended to 1e-2; the selected value, 3e-3, was unchanged.
3. *Holm family size* (commit `e73d392`, 13 October 2026). Decided after the grid
   had completed, in the commit that first computed the registered tests, and
   therefore after results were visible. The registered text states that the
   multiplicity family consists of nine tests while enumerating eight (S1 for three
   datasets, S2a, S2b, S3, S4, S5). We treat this as a typographical error in the
   registration, not a change of plan, and use the conservative reading, m = 9, for
   every reported Holm-adjusted p-value and status; Holm adjustment with m = 8 is
   reported separately as exploratory. No test's status differs between the two in
   any of the 24 test-by-readout combinations
   (`results/tables/T2_preregistered_tests.md`, which computes this comparison).
4. *Reporting rule for the direction of a Wilcoxon test* (commits `e73d392` and
   `c56a7de`, 13 October 2026). Decided after results were seen; the registration
   defines neither part. The observed direction of a Wilcoxon test is the sign of
   the matched-pairs rank-biserial correlation; when the median paired difference
   has the opposite sign, the direction is reported as ambiguous, with both signs,
   and is not resolved in either favour. The ambiguity rule applies to one test,
   S4, which is non-significant under every reading (unadjusted p = 0.6063,
   Holm-adjusted p = 1.0000 on all three readouts). No test's status therefore
   depends on this rule.
