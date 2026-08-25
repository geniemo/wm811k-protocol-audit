import numpy as np
import pandas as pd
import pytest

from wm811k_audit.constants import CLASS_NAMES

SHAPES = [(6, 21), (26, 26), (33, 29), (64, 64), (120, 80)]


def _make_map(rng, shape):
    h, w = shape
    m = rng.integers(1, 3, size=shape, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    r = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2
    m[r > 1] = 0
    return m


def make_raw_df(n_lots: int = 60, seed: int = 0, wafers_per_lot: int = 12) -> pd.DataFrame:
    """Synthetic frame shaped like the MIR pickle, plus Kaggle-style oddities.

    - lots 1..n_lots, first 60% 'Training', rest 'Test' (lot-disjoint like the real data)
    - ~40% none, rest uniform over the 9 class names (so every defect class has ~40 rows)
    - 20 unlabeled rows encoded as integer 0 (MIR style)
    - one Kaggle-style nested-array labeled row and one empty-array unlabeled row
    - 3 labeled rows with missing lotName
    - one exact duplicate inside the same lot, one exact duplicate across lots
    """
    rng = np.random.default_rng(seed)
    rows = []
    for lot in range(1, n_lots + 1):
        split = "Training" if lot <= int(n_lots * 0.6) else "Test"
        for w in range(1, wafers_per_lot + 1):
            cls = "none" if rng.random() < 0.4 else CLASS_NAMES[rng.integers(0, 9)]
            shape = SHAPES[rng.integers(0, len(SHAPES))]
            rows.append(dict(dieSize=float(shape[0] * shape[1]), failureType=cls, lotName=f"lot{lot}",
                             trainTestLabel=split, waferIndex=float(w), waferMap=_make_map(rng, shape)))
    for k in range(20):
        rows.append(dict(dieSize=676.0, failureType=0, lotName=f"lot{n_lots + 1}", trainTestLabel=0,
                         waferIndex=float(k + 1), waferMap=_make_map(rng, (26, 26))))
    rows.append(dict(dieSize=676.0, failureType=np.array([["Center"]], dtype=object), lotName="lot1",
                     trainTestLabel=np.array([["Training"]], dtype=object), waferIndex=13.0,
                     waferMap=_make_map(rng, (26, 26))))
    rows.append(dict(dieSize=676.0, failureType=np.array([], dtype=object), lotName="lot1",
                     trainTestLabel=np.array([], dtype=object), waferIndex=14.0,
                     waferMap=_make_map(rng, (26, 26))))
    for k in range(3):
        rows.append(dict(dieSize=676.0, failureType="Loc", lotName=None, trainTestLabel="Test",
                         waferIndex=1.0, waferMap=_make_map(rng, (26, 26))))
    same_lot_dup = dict(rows[0]); same_lot_dup["waferIndex"] = 99.0
    same_lot_dup["waferMap"] = rows[0]["waferMap"].copy()
    cross_lot_dup = dict(rows[1]); cross_lot_dup["lotName"] = f"lot{n_lots}"; cross_lot_dup["trainTestLabel"] = "Test"
    cross_lot_dup["waferMap"] = rows[1]["waferMap"].copy()
    rows.extend([same_lot_dup, cross_lot_dup])
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def raw_df():
    return make_raw_df()


@pytest.fixture(scope="session")
def processed(raw_df):
    from wm811k_audit.data import build_labeled_table
    return build_labeled_table(raw_df)
