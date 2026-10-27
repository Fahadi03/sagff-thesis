# Manuscript notes: text the paper needs, collected as the project goes

Each item states where its numbers come from. Never copy a number into the
manuscript from here without re-reading the generated file it cites (R8).

## Reproducibility section: determinism evidence

The learning-rate selection was run twice: first from an uncommitted working
tree, then from a clean commit with a different handcrafted-feature dimension
(which the baseline head does not use). Every run present in both, 180 of 180,
was identical bit for bit: the full per-epoch training curve, the selected and
final epochs, and all validation metrics, compared with exact floating-point
equality. Source: `results/tables/lr_reproduction.md`
(`scripts/compare_lr_reproduction.py`). Adding the per-epoch gate trajectory and
the convergence-anchored readout did not perturb training: the 30 stopping-diagnostic
runs, made before those readouts existed, are identical to the corresponding grid
runs in the per-epoch curve, best and final epoch, all metrics and every gate
statistic, 30 of 30. Source: `results/tables/stopping_vs_grid.md`
(`scripts/compare_stopping_to_grid.py`). (An earlier note also claimed agreement
between the two feature-dimension selections; the earlier selection was run from
an uncommitted tree and its scores are not in the repository, so that claim cannot
be traced and is not to be used.)

## Footnote: validation of the per-epoch permutation control

For the per-epoch control (`sagff-noise-perepoch`), the k-matched validation
split seen by early stopping is scored under the same epoch's permutation of
the handcrafted features, while the validation metric reported after training is
recomputed under a single fixed evaluation permutation. The two therefore need not
coincide; both are scores of a branch that carries no stable content by
construction. (Code: `src/train.py`, module docstring.)

## Pre-registration and disclosure

See `docs/preregistration_disclosure.md`.
