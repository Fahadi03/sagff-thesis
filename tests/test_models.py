"""§2 model-spec checks that are easy to get wrong silently."""
import pytest
import torch
import yaml

from conftest import ROOT
from src.models import HEAD_ORDER, build_head, count_params

CFG = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
FEAT_DIM = CFG["heads"]["feat_dim"]
CAP_H = CFG["heads"]["baseline_cap_hidden"]
D = CFG["heads"]["d_model"]


def _inputs(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, D, generator=g), torch.randn(b, FEAT_DIM, generator=g)


def test_feat_dim_matches_frozen_feature_config():
    f = CFG["features"]
    assert FEAT_DIM == f["tfidf_dim"] + f["chargram_dim"] + f["stats_dim"], \
        "heads.feat_dim in configs/default.yaml does not match the frozen feature dimensions"


def test_sagff_gate_is_a_vector_in_unit_interval():
    head = build_head("sagff", 30, FEAT_DIM, CAP_H).eval()
    e, f = _inputs()
    _, aux = head(e, f)
    g = aux["gate"]
    assert g.shape == (4, D), f"gate must be a VECTOR of shape (B, d_model)=(4, {D}); got {tuple(g.shape)}"
    assert torch.all((g > 0) & (g < 1)), "gate values must lie in (0, 1)"


@pytest.mark.parametrize("name", ["baseline", "baseline-cap"])
def test_featureless_heads_ignore_handcrafted_input(name):
    head = build_head(name, 30, FEAT_DIM, CAP_H).eval()
    e, f1 = _inputs(seed=1)
    _, f2 = _inputs(seed=2)
    assert not head.uses_features
    assert torch.equal(head(e, f1)[0], head(e, f2)[0]), f"{name} output depends on h_feat"


@pytest.mark.parametrize("name", ["concat", "sagff"])
def test_feature_heads_use_handcrafted_input(name):
    head = build_head(name, 30, FEAT_DIM, CAP_H).eval()
    e, f1 = _inputs(seed=1)
    _, f2 = _inputs(seed=2)
    assert head.uses_features
    assert not torch.equal(head(e, f1)[0], head(e, f2)[0]), f"{name} ignores h_feat"


@pytest.mark.parametrize("n_classes", [30, 77])
def test_baseline_cap_is_parameter_matched_to_sagff(n_classes):
    cap = count_params(build_head("baseline-cap", n_classes, FEAT_DIM, CAP_H))
    sagff = count_params(build_head("sagff", n_classes, FEAT_DIM, CAP_H))
    assert abs(cap / sagff - 1) <= 0.20, f"baseline-cap/sagff = {cap / sagff:.3f} at {n_classes} classes"


def test_logged_param_counts_match_modules_and_order():
    assert list(HEAD_ORDER) == CFG["heads"]["order"] and HEAD_ORDER[-1] == "baseline-cap"
    for name in HEAD_ORDER:
        for key, n in (("30_classes", 30), ("77_classes", 77)):
            got = count_params(build_head(name, n, FEAT_DIM, CAP_H))
            assert got == CFG["heads"]["param_counts"][name][key], \
                f"{name} @ {n} classes: config says {CFG['heads']['param_counts'][name][key]}, module has {got}"
