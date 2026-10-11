"""The five fusion heads (CLAUDE.md §2). All take the cached frozen-encoder
embedding `e` (B, d_model); the feature heads also take `h_feat` (B, F).

    head          features  gate  definition
    baseline      no        no    Classifier(Dropout(LayerNorm(e)))
    concat        yes       no    Classifier(Dropout([h_enc ; h_feat_p]))
    baseline-cap  no        no    Classifier(Dropout(LayerNorm(W2 gelu(W1 h_enc))))
    concat-cap    yes       no    Classifier(Dropout(LayerNorm(W2 gelu(W1 [h_enc ; h_feat_p]))))
    sagff         yes       yes   g = sigmoid(W_g [h_enc ; h_feat_p]); h = g*h_enc + (1-g)*h_feat_p;
                                  Classifier(Dropout(h))

with h_enc = LayerNorm(e) and h_feat_p = LayerNorm(W_p h_feat). baseline-cap and
concat-cap are parameter-matched to sagff, so the two confirmatory comparisons
(baseline-cap vs concat-cap: does handcrafted information help?; concat-cap vs
sagff: does gating help?) are made at matched capacity.

Gate convention (never flip): g HIGH = relying on the encoder, g LOW = relying on
the handcrafted features. g is a VECTOR, one value per dimension: (B, d_model).
The gate bias is initialised to `heads.gate_bias_init` from configs/default.yaml
(0.0, so g = sigmoid(W_g x + 0) is approximately 0.5 at initialisation); the
weights keep PyTorch's default initialisation under the run seed.
"""
from __future__ import annotations

import torch
from torch import nn

HEAD_ORDER = ("baseline", "concat", "sagff", "concat-cap", "baseline-cap")
D_MODEL = 768  # xlm-roberta-base


class Baseline(nn.Module):
    uses_features = False

    def __init__(self, n_classes: int, d_model: int = D_MODEL, dropout: float = 0.1, **_):
        super().__init__()
        self.ln_enc = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, e, h_feat=None):
        return self.classifier(self.drop(self.ln_enc(e))), {}


class Concat(nn.Module):
    uses_features = True

    def __init__(self, n_classes: int, feat_dim: int, d_model: int = D_MODEL, dropout: float = 0.1, **_):
        super().__init__()
        self.ln_enc = nn.LayerNorm(d_model)
        self.proj = nn.Linear(feat_dim, d_model)           # W_p
        self.ln_feat = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * d_model, n_classes)

    def forward(self, e, h_feat):
        h = torch.cat([self.ln_enc(e), self.ln_feat(self.proj(h_feat))], dim=-1)
        return self.classifier(self.drop(h)), {}


class SAGFF(nn.Module):
    uses_features = True

    def __init__(self, n_classes: int, feat_dim: int, gate_bias_init: float,
                 d_model: int = D_MODEL, dropout: float = 0.1, **_):
        super().__init__()
        self.ln_enc = nn.LayerNorm(d_model)
        self.proj = nn.Linear(feat_dim, d_model)           # W_p
        self.ln_feat = nn.LayerNorm(d_model)
        self.gate = nn.Linear(2 * d_model, d_model)        # W_g -> VECTOR gate
        nn.init.constant_(self.gate.bias, gate_bias_init)  # explicit, from config (G1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, e, h_feat):
        h_enc = self.ln_enc(e)
        h_feat_p = self.ln_feat(self.proj(h_feat))
        g = torch.sigmoid(self.gate(torch.cat([h_enc, h_feat_p], dim=-1)))  # (B, d_model)
        h = g * h_enc + (1 - g) * h_feat_p   # g high = encoder, g low = handcrafted
        return self.classifier(self.drop(h)), {"gate": g}


class BaselineCap(nn.Module):
    uses_features = False

    def __init__(self, n_classes: int, cap_hidden: int, d_model: int = D_MODEL, dropout: float = 0.1, **_):
        super().__init__()
        self.ln_enc = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, cap_hidden)
        self.act = nn.GELU()
        self.w2 = nn.Linear(cap_hidden, d_model)
        self.ln_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, e, h_feat=None):
        h = self.ln_out(self.w2(self.act(self.w1(self.ln_enc(e)))))
        return self.classifier(self.drop(h)), {}


class ConcatCap(nn.Module):
    uses_features = True

    def __init__(self, n_classes: int, feat_dim: int, concat_cap_hidden: int,
                 d_model: int = D_MODEL, dropout: float = 0.1, **_):
        super().__init__()
        self.ln_enc = nn.LayerNorm(d_model)
        self.proj = nn.Linear(feat_dim, d_model)           # W_p
        self.ln_feat = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(2 * d_model, concat_cap_hidden)
        self.act = nn.GELU()
        self.w2 = nn.Linear(concat_cap_hidden, d_model)
        self.ln_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, e, h_feat):
        h = torch.cat([self.ln_enc(e), self.ln_feat(self.proj(h_feat))], dim=-1)
        h = self.ln_out(self.w2(self.act(self.w1(h))))
        return self.classifier(self.drop(h)), {}


HEADS = {"baseline": Baseline, "concat": Concat, "sagff": SAGFF,
         "concat-cap": ConcatCap, "baseline-cap": BaselineCap}


def build_head(name: str, n_classes: int, heads_cfg: dict, dropout: float) -> nn.Module:
    """Build a head from the `heads` section of configs/default.yaml."""
    return HEADS[name](n_classes=n_classes, feat_dim=heads_cfg["feat_dim"],
                       cap_hidden=heads_cfg["baseline_cap_hidden"],
                       concat_cap_hidden=heads_cfg["concat_cap_hidden"],
                       gate_bias_init=heads_cfg["gate_bias_init"],
                       d_model=heads_cfg["d_model"], dropout=dropout)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def cap_hidden_for(feat_dim: int, n_classes: int, d_model: int = D_MODEL, multiple: int = 64) -> int:
    """Hidden width H that makes baseline-cap's parameter count match sagff's,
    rounded to a multiple of 64. Both heads share LayerNorm(e) and a
    d_model->n_classes classifier, so those cancel:
        sagff extra   = W_p (F*d + d) + LN(d: 2d) + W_g (2d*d + d)
        cap   extra   = W1 (d*H + H) + W2 (H*d + d) + LN(2d)
    """
    sagff_extra = (feat_dim * d_model + d_model) + 2 * d_model + (2 * d_model * d_model + d_model)
    h = (sagff_extra - d_model - 2 * d_model) / (2 * d_model + 1)
    return int(round(h / multiple) * multiple)


def concat_cap_hidden_for(d_model: int = D_MODEL, multiple: int = 64) -> int:
    """Hidden width H that makes concat-cap match sagff. Both share LayerNorm(e),
    W_p, LayerNorm(h_feat_p) and a d_model->n_classes classifier, so:
        sagff extra   = W_g (2d*d + d)
        concat-cap    = W1 (2d*H + H) + W2 (H*d + d) + LN(2d)
    Independent of F and of the class count.
    """
    h = (2 * d_model * d_model + d_model - d_model - 2 * d_model) / (3 * d_model + 1)
    return int(round(h / multiple) * multiple)
