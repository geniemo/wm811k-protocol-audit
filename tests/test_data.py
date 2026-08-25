import numpy as np
import pandas as pd
import pytest

from wm811k_audit.constants import CLASS_NAMES, CLASS_TO_IDX
from wm811k_audit.data import (build_labeled_table, find_split_column, load_processed, make_lot_id,
                               parse_lot_num, save_processed, sha1_of_map, summarize, unwrap_label)


@pytest.mark.parametrize("cols,expected", [
    (["dieSize", "failureType", "lotName", "trainTestLabel", "waferIndex", "waferMap"], "trainTestLabel"),
    (["waferMap", "dieSize", "lotName", "waferIndex", "trianTestLabel", "failureType"], "trianTestLabel"),
])
def test_find_split_column(cols, expected):
    assert find_split_column(cols) == expected


def test_find_split_column_requires_exactly_one():
    with pytest.raises(KeyError):
        find_split_column(["waferMap", "failureType"])


@pytest.mark.parametrize("value,expected", [
    ("Training", "Training"), ("  Test ", "Test"), ("", None), ("   ", None),
    (np.array([["Center"]], dtype=object), "Center"), (np.array([], dtype=object), None),
    ([["Donut"]], "Donut"), ([], None), (0, None), (np.uint64(0), None), (None, None), (float("nan"), None),
])
def test_unwrap_label(value, expected):
    assert unwrap_label(value) == expected


def test_make_lot_id_singleton_for_missing():
    assert make_lot_id("lot12", 5) == ("lot12", False)
    assert make_lot_id(None, 5) == ("__singleton_5", True)
    assert make_lot_id("   ", 7) == ("__singleton_7", True)


def test_parse_lot_num():
    assert parse_lot_num("lot47542") == 47542
    assert parse_lot_num("weird") is None
    assert parse_lot_num(None) is None


def test_sha1_is_shape_sensitive():
    a = np.arange(12, dtype=np.uint8)
    assert sha1_of_map(a.reshape(3, 4)) != sha1_of_map(a.reshape(4, 3))
    assert sha1_of_map(a.reshape(3, 4)) == sha1_of_map(a.reshape(3, 4).copy())


def test_build_labeled_table_shapes_and_columns(processed, raw_df):
    maps64, meta = processed
    n_labeled = sum(isinstance(unwrap_label(v), str) for v in raw_df["failureType"])
    assert maps64.shape == (n_labeled, 64, 64) and maps64.dtype == np.uint8
    assert len(meta) == n_labeled
    assert set(np.unique(maps64).tolist()) <= {0, 1, 2}
    assert list(meta.columns) == ["row_id", "lot_name", "lot_id", "lot_num", "is_singleton_lot", "wafer_index",
                                  "die_size", "orig_h", "orig_w", "failure_type", "label9", "orig_split",
                                  "raw_hash", "map64_hash"]
    assert (meta["row_id"].values == np.arange(n_labeled)).all()


def test_build_labeled_table_labels_and_splits(processed):
    _, meta = processed
    assert set(meta["failure_type"]) <= set(CLASS_NAMES)
    assert (meta["label9"] == meta["failure_type"].map(CLASS_TO_IDX)).all()
    assert set(meta["orig_split"]) == {"Training", "Test"}
    nested = meta[(meta["lot_name"] == "lot1") & (meta["wafer_index"] == 13.0)]
    assert len(nested) == 1 and nested.iloc[0]["failure_type"] == "Center" and nested.iloc[0]["orig_split"] == "Training"
    assert not ((meta["lot_name"] == "lot1") & (meta["wafer_index"] == 14.0)).any()  # empty-array row is unlabeled


def test_build_labeled_table_singletons_and_lot_num(processed):
    _, meta = processed
    single = meta[meta["is_singleton_lot"]]
    assert len(single) == 3 and single["lot_id"].str.startswith("__singleton_").all()
    assert single["lot_num"].isna().all()
    assert single["lot_name"].isna().all()  # singleton rows have missing lot_name
    assert meta.loc[~meta["is_singleton_lot"], "lot_name"].notna().all()  # non-singleton rows have non-missing lot_name
    assert meta.loc[meta["lot_name"] == "lot7", "lot_num"].eq(7).all()


def test_build_labeled_table_hashes_detect_duplicates(processed):
    maps64, meta = processed
    vc = meta["raw_hash"].value_counts()
    assert (vc == 2).sum() == 2  # same-lot and cross-lot duplicate pairs
    assert meta["map64_hash"].nunique() <= meta["raw_hash"].nunique()


def test_save_load_roundtrip(processed, tmp_path):
    maps64, meta = processed
    save_processed(maps64, meta, tmp_path)
    m2, meta2 = load_processed(tmp_path)
    assert (m2 == maps64).all()
    pd.testing.assert_frame_equal(meta.reset_index(drop=True), meta2)
    assert (tmp_path / "summary.json").exists()
    # Verify lot_name missing-value representation survives the parquet roundtrip
    assert meta2.loc[meta2["is_singleton_lot"], "lot_name"].isna().all()
    assert meta2["lot_name"].dtype == meta["lot_name"].dtype  # dtype must survive roundtrip


def test_summarize_keys(processed):
    _, meta = processed
    s = summarize(meta)
    assert s["n_labeled"] == len(meta)
    assert set(s["class_counts"]) == set(CLASS_NAMES)
    assert s["split_counts"]["Training"] > 0 and s["split_counts"]["Test"] > 0
    assert s["orig_split_lots_with_both"] == 0
