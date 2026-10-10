"""R9 enforcement: bnintent30-en and bnintent30-bn contain IDENTICAL rows."""
from conftest import check

EN, BN = "bnintent30-en", "bnintent30-bn"


def _is_bangla(s: str) -> bool:
    return any("ঀ" <= ch <= "৿" for ch in s)


def test_same_label_names(manifest):
    assert manifest["label_names"][EN] == manifest["label_names"][BN], \
        "label_names differ between the English and Bangla BNIntent30 tracks"


def test_identical_uids_order_and_labels_for_every_k_seed(split_files, manifest):
    problems = []
    combos = sorted({(k, s) for (t, k, s) in split_files if t == BN})
    expected = len(manifest["ks"]) * len(manifest["seeds"])
    if len(combos) != expected:
        problems.append(f"found {len(combos)} (k, seed) combos for {BN}, expected {expected}")
    for k, seed in combos:
        if (EN, k, seed) not in split_files:
            problems.append(f"k={k} seed={seed}: missing for {EN}")
            continue
        for s in ("train", "val", "test"):
            en, bn = split_files[(EN, k, seed)][s], split_files[(BN, k, seed)][s]
            if [r["uid"] for r in en] != [r["uid"] for r in bn]:
                problems.append(f"k={k} seed={seed} {s}: uid lists differ "
                                f"({len(en)} en rows vs {len(bn)} bn rows)")
            elif [r["label_id"] for r in en] != [r["label_id"] for r in bn]:
                problems.append(f"k={k} seed={seed} {s}: same uids but different labels")
    check(problems, "R9 violated: English and Bangla tracks diverge")


def test_languages_not_swapped(split_files):
    """Guard against a field mix-up: en text has no Bangla script, bn text (almost) all has."""
    en = split_files[(EN, "full", 13)]["train"]
    bn = split_files[(BN, "full", 13)]["train"]
    en_bangla = [r["uid"] for r in en if _is_bangla(r["text"])]
    bn_share = sum(_is_bangla(r["text"]) for r in bn) / len(bn)
    check(en_bangla, f"{EN} contains Bangla-script text")
    assert bn_share > 0.99, f"only {bn_share:.1%} of {BN} rows contain Bangla script"
