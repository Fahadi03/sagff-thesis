"""§2 model-spec checks that are easy to get wrong silently."""
import numpy as np
import pytest
import torch
import yaml

from conftest import ROOT
from src.models import HEAD_ORDER, build_head, count_params
from src.train import CONDITIONS, permutation

CFG = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
HC = CFG["heads"]
D, FEAT_DIM, DROPOUT = HC["d_model"], HC["feat_dim"], HC["dropout"]
TOLERANCE = {"baseline-cap": 0.20, "concat-cap": 0.01}


def _inputs(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, D, generator=g), torch.randn(b, FEAT_DIM, generator=g)


def test_feat_dim_matches_frozen_feature_config():
    f = CFG["features"]
    assert FEAT_DIM == f["tfidf_dim"] + f["chargram_dim"] + f["stats_dim"], \
        "heads.feat_dim in configs/default.yaml does not match the frozen feature dimensions"


def test_sagff_gate_is_a_vector_in_unit_interval():
    head = build_head("sagff", 30, HC, DROPOUT).eval()
    e, f = _inputs()
    g = head(e, f)[1]["gate"]
    assert g.shape == (4, D), f"gate must be a VECTOR of shape (B, d_model)=(4, {D}); got {tuple(g.shape)}"
    assert torch.all((g > 0) & (g < 1)), "gate values must lie in (0, 1)"


def test_gate_bias_comes_from_config_and_gate_starts_near_half():
    """G1: explicit gate-bias init from config; mean gate at init ~ sigmoid(bias)."""
    torch.manual_seed(13)
    head = build_head("sagff", 30, HC, DROPOUT).eval()
    assert torch.all(head.gate.bias == HC["gate_bias_init"]), "gate bias not set from config"
    e, f = _inputs(b=256)
    with torch.no_grad():
        g = head(e, f)[1]["gate"]
    expected = 1 / (1 + np.exp(-HC["gate_bias_init"]))
    assert abs(float(g.mean()) - expected) < 0.02, f"mean gate at init {float(g.mean()):.4f}, expected ~{expected:.2f}"


@pytest.mark.parametrize("name", ["baseline", "baseline-cap"])
def test_featureless_heads_ignore_handcrafted_input(name):
    head = build_head(name, 30, HC, DROPOUT).eval()
    e, f1 = _inputs(seed=1)
    _, f2 = _inputs(seed=2)
    assert not head.uses_features
    assert torch.equal(head(e, f1)[0], head(e, f2)[0]), f"{name} output depends on h_feat"


@pytest.mark.parametrize("name", ["concat", "concat-cap", "sagff"])
def test_feature_heads_use_handcrafted_input(name):
    head = build_head(name, 30, HC, DROPOUT).eval()
    e, f1 = _inputs(seed=1)
    _, f2 = _inputs(seed=2)
    assert head.uses_features
    assert not torch.equal(head(e, f1)[0], head(e, f2)[0]), f"{name} ignores h_feat"


@pytest.mark.parametrize("name", list(TOLERANCE))
@pytest.mark.parametrize("n_classes", [30, 77])
def test_cap_heads_are_parameter_matched_to_sagff(name, n_classes):
    cap = count_params(build_head(name, n_classes, HC, DROPOUT))
    sagff = count_params(build_head("sagff", n_classes, HC, DROPOUT))
    assert abs(cap / sagff - 1) <= TOLERANCE[name], \
        f"{name}/sagff = {cap / sagff:.4f} at {n_classes} classes (tolerance {TOLERANCE[name]:.0%})"


def test_logged_param_counts_match_modules_and_order():
    assert list(HEAD_ORDER) == HC["order"]
    for name in HEAD_ORDER:
        for key, n in (("30_classes", 30), ("77_classes", 77)):
            got = count_params(build_head(name, n, HC, DROPOUT))
            assert got == HC["param_counts"][name][key], \
                f"{name} @ {n} classes: config says {HC['param_counts'][name][key]}, module has {got}"


def test_grid_order_permuted_right_after_sagff_and_baseline_cap_last():
    assert CONDITIONS.index("sagff-permuted") == CONDITIONS.index("sagff") + 1
    assert CONDITIONS[-1] == "baseline-cap"


def test_permutation_preserves_marginals_and_is_deterministic():
    """G3: permuting rows keeps the multiset of feature vectors, breaks the mapping."""
    uids = [f"u{i:03d}" for i in range(200)]
    p1, p2 = permutation(13, "test", uids), permutation(13, "test", uids)
    assert np.array_equal(p1, p2), "permutation is not deterministic"
    assert sorted(p1.tolist()) == list(range(200)), "not a permutation"
    assert (p1 == np.arange(200)).mean() < 0.05, "permutation leaves most rows in place"
    assert not np.array_equal(p1, permutation(42, "test", uids)), "seed does not change the permutation"
    assert not np.array_equal(p1, permutation(13, "train", uids)), "split does not change the permutation"
