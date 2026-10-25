# Citation checklist

Compliance item (Elsevier: AI-generated references "can be incorrect, hallucinated,
or fabricated"; fabricated references can lead to rejection). Every entry was
verified on 2026-10-26 against the publisher's or archive's own record (ACL
Anthology, arXiv, Crossref). Nothing below was filled in from memory. A citation
that cannot be confirmed is marked **not found**, and the sentence relying on it is
dropped.

Status: **confirmed** (claim matches the source) · **corrected** (source found,
claim needed changing) · **not found** (no source located) · **pending** (not yet
available to check).

## Scope

- **Draft manuscript sections: PENDING.** The drafts (`docs/drafts/methodology_draft.md`,
  `results_draft.md`, `manuscript_remaining_sections.md`,
  `thesis_journey_and_defence.md`) were reported as added, but as of 2026-10-26 no
  `docs/drafts/` directory exists in the repository and no file of those names was
  found on Desktop, Downloads or Documents. The citations reported as marked
  `[verify]` in them are listed under "Pending" below. They are NOT verified: each
  needs the draft's own claim (authors, title, venue, year, DOI, and the sentence it
  supports) before it can be checked against the source.
- **`docs/PROTOCOL.md`:** three citations, all checked below (1-3).
- **Unavoidable method references:** the two datasets, BNIntent30's source
  dataset and the encoder are not cited in the protocol but will be cited in any
  manuscript of this study; checked below (4-7).

## Entries

| # | Claimed (where) | Verified record | DOI / identifier | Status | Source checked |
|---|---|---|---|---|---|
| 1 | Peters, Ruder & Smith (2019), feature-based transfer (PROTOCOL §0) | M. E. Peters, S. Ruder, N. A. Smith, "To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks", *Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP-2019)*, ACL, Florence, pp. 7-14, 2019 | 10.18653/v1/W19-4302 | confirmed | https://aclanthology.org/W19-4302/ |
| 2 | Dodge et al. 2020, seed instability of few-sample fine-tuning (PROTOCOL §0) | J. Dodge, G. Ilharco, R. Schwartz, A. Farhadi, H. Hajishirzi, N. Smith, "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping", arXiv:2002.06305, 2020. Abstract: "distinct random seeds can lead to substantially different results" | 10.48550/arXiv.2002.06305 | confirmed (preprint: no peer-reviewed venue listed; cite as arXiv) | https://arxiv.org/abs/2002.06305 |
| 3 | Gao et al., 2021, ACL 2021 §3, dev set of the same size as the few-shot training set (PROTOCOL §3.5) | T. Gao, A. Fisch, D. Chen, "Making Pre-trained Language Models Better Few-shot Learners", *Proceedings of the 59th Annual Meeting of the ACL and the 11th IJCNLP (Volume 1: Long Papers)*, pp. 3816-3830, 2021. §3 text ("development set ... of the same size as the few-shot training set") checked earlier against the paper's arXiv full text | 10.18653/v1/2021.acl-long.295 | confirmed | https://aclanthology.org/2021.acl-long.295/ ; https://ar5iv.labs.arxiv.org/html/2012.15723 |
| 4 | BNIntent30 dataset: arXiv:2312.10679 (`docs/data_license_request.md`; README, PROTOCOL and `configs/data.yaml` named only the GitHub repository) | M. Hasan, M. J. I. Basher, M. T. R. Shawon, "Bengali Intent Classification with Generative Adversarial BERT", *2023 26th International Conference on Computer and Information Technology (ICCIT)*, IEEE, pp. 1-6, 2023. Also arXiv:2312.10679 (17 Dec 2023) | 10.1109/ICCIT60459.2023.10440989 (arXiv: 10.48550/arXiv.2312.10679) | **corrected**: a peer-reviewed IEEE version exists; cite it, with the arXiv version as an optional second identifier | https://api.crossref.org (title query); https://arxiv.org/abs/2312.10679 |
| 5 | BANKING77 dataset (README, PROTOCOL §3) | I. Casanueva, T. Temčinas, D. Gerz, M. Henderson, I. Vulić, "Efficient Intent Detection with Dual Sentence Encoders", *Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI*, ACL, pp. 38-45, 2020 | 10.18653/v1/2020.nlp4convai-1.5 | confirmed | https://aclanthology.org/2020.nlp4convai-1.5/ |
| 6 | CLINC150, the source of BNIntent30 (PROTOCOL §4) | S. Larson, A. Mahendran, J. J. Peper, C. Clarke, A. Lee, P. Hill, J. K. Kummerfeld, K. Leach, M. A. Laurenzano, L. Tang, J. Mars, "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction", *Proceedings of EMNLP-IJCNLP 2019*, ACL, pp. 1311-1316, 2019 | 10.18653/v1/D19-1131 | confirmed. Note: the BNIntent30 paper spells it "CLINIC150"; the dataset's name is CLINC150 | https://aclanthology.org/D19-1131/ |
| 7 | xlm-roberta-base, the frozen encoder (PROTOCOL §5-6) | A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, V. Stoyanov, "Unsupervised Cross-lingual Representation Learning at Scale", *Proceedings of the 58th Annual Meeting of the ACL*, pp. 8440-8451, 2020 | 10.18653/v1/2020.acl-main.747 | confirmed | https://aclanthology.org/2020.acl-main.747/ |

## Corrections

1. **BNIntent30 (entry 4).** The repository cited only the arXiv preprint. A
   peer-reviewed version is indexed by Crossref: IEEE, ICCIT 2023, pp. 1-6,
   DOI 10.1109/ICCIT60459.2023.10440989. It is now the primary citation everywhere
   the dataset is cited (PROTOCOL §3.1, README, `configs/data.yaml` via
   `scripts/download_data.py`, `docs/data_license_request.md`), with the arXiv entry
   as a secondary pointer.
2. **CLINC150 spelling (entry 6).** The BNIntent30 paper spells its source dataset
   "CLINIC150". The dataset's name is CLINC150; our text must use the correct name.
   This is recorded in PROTOCOL §3.1 and in `configs/data.yaml`.

Not corrections, but worth knowing when writing the reference list: entry 2 is a
preprint with no peer-reviewed venue; entry 1 is a workshop paper; entry 6's name
is misspelled in the BNIntent30 paper.

## Pending (reported in the drafts as `[verify]`; drafts not yet in the repository)

Not checked, and not to be cited until checked against the draft's claim and the
source record. Listed as reported; several cannot be identified without the
draft's title or context (an author and year alone can match more than one paper).

| # | As reported | Status |
|---|---|---|
| P1 | Liang et al., Scientific Reports, 2025 | pending |
| P2 | Yang et al., Pattern Recognition, 2024 | pending |
| P3 | ESIE-BERT | pending |
| P4 | MICL | pending |
| P5 | MBCS | pending |
| P6 | Hochreiter & Schmidhuber, 1997 | pending |
| P7 | Srivastava et al., 2015 | pending |
| P8 | Reimers & Gurevych, 2019 | pending |
| P9 | Perez et al., 2021 | pending |
| P10 | Zhang et al., 2021 | pending |
