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
(`scripts/compare_lr_reproduction.py`). The nine feature-dimension configurations
evaluated in both feature selections also agree to the printed precision
(`results/tables/feature_dim_selection.md`). Logging the per-epoch gate trajectory
was likewise verified not to perturb training (identical curve, metrics and gate
statistics to the run made before the logging existed).

## Footnote: validation of the per-epoch permutation control

For the per-epoch control (`sagff-noise-perepoch`), the k-matched validation
split seen by early stopping is scored under the same epoch's permutation of
the handcrafted features, while the validation metric reported after training is
recomputed under a single fixed evaluation permutation. The two therefore need not
coincide; both are scores of a branch that carries no stable content by
construction. (Code: `src/train.py`, module docstring.)

## Pre-registration and disclosure

See `docs/preregistration_disclosure.md`.
