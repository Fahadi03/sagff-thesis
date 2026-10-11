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
