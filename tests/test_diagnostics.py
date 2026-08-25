import numpy as np
import torch

from wm811k_audit.diagnostics import dup_rate, global_eda, lot_share_rate, nn_hamming, split_diagnostics
from wm811k_audit.splits import carve_gold, split_lot, split_random


def test_dup_and_lot_share_rates():
    assert dup_rate(["a", "b", "c", "d"], ["a", "x", "d"]) == 0.5
    assert lot_share_rate(["l1", "l2", "l3"], ["l2", "l9"]) == 1 / 3
    assert np.isnan(dup_rate([], ["a"]))


def test_nn_hamming_matches_bruteforce():
    rng = np.random.default_rng(0)
    test = rng.integers(0, 3, (7, 5, 6), dtype=np.uint8)
    train = rng.integers(0, 3, (11, 5, 6), dtype=np.uint8)
    train[3] = test[2]  # exact duplicate -> distance 0
    brute = (test[:, None] != train[None]).sum(axis=(2, 3)).min(axis=1)
    got = nn_hamming(test, train, device="cpu", test_chunk=3, train_chunk=4)
    assert got.dtype == np.int32 and (got == brute).all() and got[2] == 0
    got_t = nn_hamming(torch.as_tensor(test), torch.as_tensor(train), device="cpu")
    assert (got_t == brute).all()


def test_split_diagnostics_lot_split_has_zero_lot_share(processed):
    maps64, meta = processed
    y, lots = meta["label9"].values, meta["lot_id"].values
    tr, te = split_lot(y, lots, seed=0)
    d, h = split_diagnostics(meta, torch.as_tensor(maps64), tr, te, device="cpu")
    assert d["lot_share_rate"] == 0.0
    assert len(h) == len(te) and d["nn_hamming_median"] >= 0
    tr2, te2 = split_random(y, seed=0)
    d2, _ = split_diagnostics(meta, torch.as_tensor(maps64), tr2, te2, device="cpu")
    assert d2["lot_share_rate"] > 0.5
    assert set(d2) == {"dup_rate", "dup_rate64", "dup_rate_defect", "lot_share_rate", "nn_hamming_mean",
                       "nn_hamming_median", "nn_hamming_p10", "nn_hamming_p25", "nn_hamming_p75", "nn_hamming_p90"}


def test_global_eda_keys_and_known_fixture_facts(processed):
    _, meta = processed
    e = global_eda(meta)
    assert e["n_labeled"] == len(meta)
    assert e["orig_split"]["lots_with_both"] == 0
    assert e["duplicates"]["n_rows_in_groups"] == 4 and e["duplicates"]["n_groups"] == 2
    assert e["duplicates"]["frac_groups_multi_lot"] == 0.5
    assert e["lots"]["n_lots"] == meta["lot_id"].nunique()
    assert set(e["shapes"]["top"][0]) == {"h", "w", "count"}
