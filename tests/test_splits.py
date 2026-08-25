import numpy as np
import pytest

from wm811k_audit.constants import NONE_IDX
from wm811k_audit.splits import (CellSpec, all_cells, build_cell_split, cap_per_class, carve_gold, filter_classes,
                                 parse_cell_id, resolve_cap, split_lot, split_lot_ordered, split_original, split_random)


def test_all_cells_18_unique_in_nested_order():
    cells = all_cells()
    ids = [c.cell_id for c in cells]
    assert len(ids) == 18 and len(set(ids)) == 18
    assert ids[:4] == ["A1-B1-C1", "A1-B1-C2", "A1-B1-C3", "A1-B2-C1"]
    assert parse_cell_id("A3-B2-C3") == CellSpec("A3", "B2", "C3")
    assert CellSpec("A1", "B1", "C1").n_classes == 9 and CellSpec("A1", "B2", "C1").n_classes == 8
    with pytest.raises(ValueError):
        parse_cell_id("A9-B1-C1")


def test_carve_gold_is_lot_disjoint_and_stratified(processed):
    _, meta = processed
    pool, gold = carve_gold(meta["label9"].values, meta["lot_id"].values)
    assert len(pool) + len(gold) == len(meta) and len(np.intersect1d(pool, gold)) == 0
    assert set(meta["lot_id"].values[pool]).isdisjoint(set(meta["lot_id"].values[gold]))
    assert set(meta["label9"].values[gold]) == set(range(9))
    assert 0.12 < len(gold) / len(meta) < 0.30
    pool2, gold2 = carve_gold(meta["label9"].values, meta["lot_id"].values)
    assert (gold2 == gold).all()


def test_filter_classes():
    y = np.array([0, 8, 3, 8, 7])
    assert filter_classes(y, "B1").all()
    assert (filter_classes(y, "B2") == np.array([True, False, True, False, True])).all()


def test_resolve_cap_and_cap_per_class():
    y = np.array([0] * 10 + [1] * 3 + [2] * 6)
    assert resolve_cap(y, "C1") is None and resolve_cap(y, "C2") == 5000 and resolve_cap(y, "C3") == 3
    m = cap_per_class(y, 4, np.random.default_rng(0))
    assert np.bincount(y[m]).tolist() == [4, 3, 4]
    m2 = cap_per_class(y, 4, np.random.default_rng(0))
    assert (m == m2).all()
    assert cap_per_class(y, None, np.random.default_rng(0)).all()


def test_split_original():
    s = np.array(["Training", "Test", "Training", "", "Test"])
    tr, te = split_original(s)
    assert tr.tolist() == [0, 2] and te.tolist() == [1, 4]


def test_split_random_is_stratified_and_may_share_lots(processed):
    _, meta = processed
    y = meta["label9"].values
    tr, te = split_random(y, seed=0)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(y)
    assert 0.15 < len(te) / len(y) < 0.25
    frac_tr = np.bincount(y[tr], minlength=9) / len(tr)
    frac_te = np.bincount(y[te], minlength=9) / len(te)
    assert np.abs(frac_tr - frac_te).max() < 0.05
    lots = meta["lot_id"].values
    assert len(set(lots[tr]) & set(lots[te])) > 0


def test_split_lot_is_lot_disjoint(processed):
    _, meta = processed
    tr, te = split_lot(meta["label9"].values, meta["lot_id"].values, seed=1)
    lots = meta["lot_id"].values
    assert set(lots[tr]).isdisjoint(set(lots[te]))
    assert set(meta["label9"].values[te]) == set(range(9))


def test_split_lot_ordered_puts_later_lots_in_test(processed):
    _, meta = processed
    tr, te = split_lot_ordered(meta["lot_num"].values, meta["lot_id"].values, test_frac=0.2)
    nums = meta["lot_num"].values
    assert not np.isnan(nums[tr]).any() and not np.isnan(nums[te]).any()
    assert nums[te].min() >= nums[tr].max()
    lots = meta["lot_id"].values
    assert set(lots[tr]).isdisjoint(set(lots[te]))
    assert len(tr) + len(te) == int((~np.isnan(nums)).sum())
    assert 0.1 < len(te) / (len(tr) + len(te)) < 0.35


def test_build_cell_split_A3_B2_C3(processed):
    _, meta = processed
    pool, gold = carve_gold(meta["label9"].values, meta["lot_id"].values)
    meta_pool = meta.iloc[pool]
    cs = build_cell_split(CellSpec("A3", "B2", "C3"), meta_pool, seed=0)
    all_idx = np.concatenate([cs.train_idx, cs.test_idx])
    assert set(all_idx) <= set(pool) and len(np.intersect1d(all_idx, gold)) == 0
    y = meta["label9"].values[all_idx]
    assert NONE_IDX not in set(y) and cs.n_classes == 8
    counts = np.bincount(y, minlength=9)[:8]
    assert (counts == cs.cap_value).all()
    lots = meta["lot_id"].values
    assert set(lots[cs.train_idx]).isdisjoint(set(lots[cs.test_idx]))


def test_build_cell_split_A1_respects_original_labels(processed):
    _, meta = processed
    pool, _ = carve_gold(meta["label9"].values, meta["lot_id"].values)
    cs = build_cell_split(CellSpec("A1", "B1", "C1"), meta.iloc[pool], seed=2)
    assert (meta["orig_split"].values[cs.train_idx] == "Training").all()
    assert (meta["orig_split"].values[cs.test_idx] == "Test").all()
    assert cs.cap_value is None and cs.n_classes == 9


def test_build_cell_split_seed_changes_random_split_but_not_A1(processed):
    _, meta = processed
    pool, _ = carve_gold(meta["label9"].values, meta["lot_id"].values)
    a = build_cell_split(CellSpec("A2", "B1", "C1"), meta.iloc[pool], seed=0)
    b = build_cell_split(CellSpec("A2", "B1", "C1"), meta.iloc[pool], seed=1)
    assert not np.array_equal(a.test_idx, b.test_idx)
    c = build_cell_split(CellSpec("A1", "B1", "C1"), meta.iloc[pool], seed=0)
    d = build_cell_split(CellSpec("A1", "B1", "C1"), meta.iloc[pool], seed=1)
    assert np.array_equal(c.test_idx, d.test_idx)
