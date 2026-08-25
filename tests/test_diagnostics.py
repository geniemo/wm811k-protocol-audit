import numpy as np
import pytest
import torch

from wm811k_audit.constants import NONE_IDX
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


def test_split_diagnostics_rates_match_bruteforce(processed):
    """Every rate in split_diagnostics must equal an independent set-based recomputation
    from meta, for both a random split and a lot-disjoint split -- not a hard-coded number,
    since which rows land on which side depends on the (fixed but opaque) splitter internals."""
    maps64, meta = processed
    y, lots = meta["label9"].values, meta["lot_id"].values

    for tr_idx, te_idx in (split_random(y, seed=0), split_lot(y, lots, seed=0)):
        tr, te = meta.iloc[tr_idx], meta.iloc[te_idx]
        d, h = split_diagnostics(meta, torch.as_tensor(maps64), tr_idx, te_idx, device="cpu")

        exp_dup = np.mean([hh in set(tr.raw_hash) for hh in te.raw_hash])
        exp_dup64 = np.mean([hh in set(tr.map64_hash) for hh in te.map64_hash])
        te_def = te[te.label9 != NONE_IDX]
        exp_dup_def = np.mean([hh in set(tr.raw_hash) for hh in te_def.raw_hash])
        exp_lot = np.mean([l in set(tr.lot_id) for l in te.lot_id])

        assert d["dup_rate"] == pytest.approx(exp_dup)
        assert d["dup_rate64"] == pytest.approx(exp_dup64)
        assert d["dup_rate_defect"] == pytest.approx(exp_dup_def)
        assert d["lot_share_rate"] == pytest.approx(exp_lot)
        assert d["nn_hamming_median"] == pytest.approx(np.median(h))

        # Derive -- don't assume -- whether a duplicate group actually straddles this
        # train/test partition: dup_rate (resp. dup_rate64) must be >0 exactly when one does.
        tr_pos, te_pos = set(tr_idx.tolist()), set(te_idx.tolist())
        for col, rate in (("raw_hash", d["dup_rate"]), ("map64_hash", d["dup_rate64"])):
            groups = meta.groupby(col)["row_id"].apply(list)
            groups = groups[groups.map(len) > 1]
            crosses = any(any(r in tr_pos for r in g) and any(r in te_pos for r in g) for g in groups)
            assert (rate > 0) == crosses


def test_global_eda_keys_and_known_fixture_facts(processed):
    _, meta = processed
    e = global_eda(meta)
    assert e["n_labeled"] == len(meta)
    assert e["orig_split"]["lots_with_both"] == 0
    assert e["duplicates"]["n_rows_in_groups"] == 4 and e["duplicates"]["n_groups"] == 2
    assert e["duplicates"]["frac_groups_multi_lot"] == 0.5
    assert e["lots"]["n_lots"] == meta["lot_id"].nunique()
    assert set(e["shapes"]["top"][0]) == {"h", "w", "count"}


def test_global_eda_bruteforce_facts(processed):
    """Cover the global_eda fields the first test leaves unchecked, against either an
    independent pandas recomputation or a fact the fixture's construction guarantees
    (lots 1-36 Training, 37-60 Test, strictly lot-disjoint, lot_num == lot number)."""
    _, meta = processed
    e = global_eda(meta)

    assert sum(e["class_counts"].values()) == len(meta)
    assert e["orig_split"]["runs_along_lot_order"] == 2
    assert e["orig_split"]["test_wafers_with_lot_in_training"] == 0.0
    assert e["orig_split"]["lot_num_range"]["Training"]["max"] < e["orig_split"]["lot_num_range"]["Test"]["min"]

    lot_defect_counts = meta.loc[meta["label9"] != NONE_IDX, "lot_id"].value_counts()
    multi_lots = lot_defect_counts[lot_defect_counts >= 2]
    n_classes_per_multi_lot = (meta[meta["lot_id"].isin(multi_lots.index) & (meta["label9"] != NONE_IDX)]
                                .groupby("lot_id")["failure_type"].nunique())
    exp_frac_single_class = float((n_classes_per_multi_lot == 1).mean()) if len(multi_lots) else float("nan")
    assert e["lots"]["lots_with_2plus_defects"] == len(multi_lots)
    assert e["lots"]["defect_wafers_in_such_lots"] == int(multi_lots.sum())
    assert e["lots"]["frac_single_class_among_them"] == pytest.approx(exp_frac_single_class)

    assert e["shapes"]["frac_over_64"] == pytest.approx(((meta["orig_h"] > 64) | (meta["orig_w"] > 64)).mean())
    assert e["duplicates64"]["n_rows_in_groups"] == int(meta["map64_hash"].duplicated(keep=False).sum())
    assert e["lots"]["labeled_per_lot_quantiles"]["50"] == pytest.approx(
        np.percentile(meta.groupby("lot_id").size(), 50))
