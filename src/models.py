"""The four fusion heads (CLAUDE.md §2). All take the cached frozen-encoder
embedding `e` (B, d_model); the feature heads also take `h_feat` (B, F).

    baseline      Classifier(Dropout(LayerNorm(e)))                       no features
    concat        Classifier(Dropout([h_enc ; h_feat_p]))                 fusion ablation
    sagff         g = sigmoid(W_g [h_enc ; h_feat_p]),  h = g*h_enc + (1-g)*h_feat_p
    baseline-cap  Classifier(Dropout(LayerNorm(W2 gelu(W1 h_enc))))       no features,
                  parameter-matched to sagff, so baseline-vs-sagff tests
                  information rather than capacity

with h_enc = LayerNorm(e) and h_feat_p = LayerNorm(W_p h_feat).

Gate convention (never flip): g HIGH = relying on the encoder, g LOW = relying on
the handcrafted features. g is a VECTOR, one value per dimension: (B, d_model).

HEAD_ORDER is also the grid order (§5): baseline-cap is last so it can be
dropped if time runs out.
"""
from __future__ import annotations

import torch
from torch import nn

HEAD_ORDER = ("baseline", "concat", "sagff", "baseline-cap")
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

    def __init__(self, n_classes: int, feat_dim: int, d_model: int = D_MODEL, dropout: float = 0.1, **_):
        super().__init__()
        self.ln_enc = nn.LayerNorm(d_model)
        self.proj = nn.Linear(feat_dim, d_model)           # W_p
        self.ln_feat = nn.LayerNorm(d_model)
        self.gate = nn.Linear(2 * d_model, d_model)        # W_g -> VECTOR gate
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


HEADS = {"baseline": Baseline, "concat": Concat, "sagff": SAGFF, "baseline-cap": BaselineCap}


def build_head(name: str, n_classes: int, feat_dim: int, cap_hidden: int,
               d_model: int = D_MODEL, dropout: float = 0.1) -> nn.Module:
    return HEADS[name](n_classes=n_classes, feat_dim=feat_dim, cap_hidden=cap_hidden,
                       d_model=d_model, dropout=dropout)


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
