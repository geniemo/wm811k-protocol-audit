"""Protocol cells and the fixed order in which they are applied:
pool -> class filter (B) -> per-class cap (C, seeded) -> split (A, seeded)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .constants import CAP_VALUES, GOLD_SEED, N_FOLDS, NONE_IDX

SPLITS = ("A1", "A2", "A3")
AUX_SPLITS = ("A4",)
CLASS_SETS = ("B1", "B2")
CAPS = ("C1", "C2", "C3")


@dataclass(frozen=True)
class CellSpec:
    split: str
    classes: str
    cap: str

    @property
    def cell_id(self) -> str:
        return f"{self.split}-{self.classes}-{self.cap}"

    @property
    def n_classes(self) -> int:
        return 9 if self.classes == "B1" else 8


def parse_cell_id(s: str) -> CellSpec:
    parts = s.split("-")
    if len(parts) != 3 or parts[0] not in SPLITS + AUX_SPLITS or parts[1] not in CLASS_SETS or parts[2] not in CAPS:
        raise ValueError(f"bad cell id {s!r}")
    return CellSpec(*parts)


def all_cells() -> list[CellSpec]:
    return [CellSpec(a, b, c) for a in SPLITS for b in CLASS_SETS for c in CAPS]


def _first_fold(splitter, y, groups=None):
    X = np.zeros((len(y), 1))
    tr, te = next(iter(splitter.split(X, y, groups)))
    return np.sort(tr), np.sort(te)


def carve_gold(labels, lot_ids, seed: int = GOLD_SEED, n_splits: int = N_FOLDS):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return _first_fold(sgkf, np.asarray(labels), np.asarray(lot_ids))


def filter_classes(labels, classes: str) -> np.ndarray:
    labels = np.asarray(labels)
    if classes == "B1":
        return np.ones(len(labels), dtype=bool)
    if classes == "B2":
        return labels != NONE_IDX
    raise ValueError(classes)


def resolve_cap(labels, cap_code: str) -> Optional[int]:
    v = CAP_VALUES[cap_code]
    if v == "min":
        counts = np.bincount(np.asarray(labels))
        return int(counts[counts > 0].min())
    return v


def cap_per_class(labels, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels)
    mask = np.ones(len(labels), dtype=bool)
    if cap is None:
        return mask
    for c in np.unique(labels):
        pos = np.flatnonzero(labels == c)
        if len(pos) > cap:
            mask[rng.choice(pos, size=len(pos) - cap, replace=False)] = False
    return mask


def split_original(orig_split):
    s = np.asarray(orig_split)
    return np.flatnonzero(s == "Training"), np.flatnonzero(s == "Test")


def split_random(labels, seed: int, n_splits: int = N_FOLDS):
    return _first_fold(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), np.asarray(labels))


def split_lot(labels, lot_ids, seed: int, n_splits: int = N_FOLDS):
    return _first_fold(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
                       np.asarray(labels), np.asarray(lot_ids))


def split_lot_ordered(lot_nums, lot_ids, test_frac: float = 0.2):
    nums = np.asarray(lot_nums, dtype=float)
    valid = ~np.isnan(nums)
    df = pd.DataFrame({"pos": np.flatnonzero(valid), "lot": np.asarray(lot_ids)[valid], "num": nums[valid]})
    order = df.groupby("lot")["num"].min().sort_values(kind="stable")
    sizes = df.groupby("lot").size().reindex(order.index)
    cum = sizes.cumsum() / sizes.sum()
    test_lots = set(cum.index[cum.values > (1 - test_frac)])
    is_test = df["lot"].isin(test_lots).values
    return df["pos"].values[~is_test], df["pos"].values[is_test]


@dataclass
class CellSplit:
    cell: CellSpec
    seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_classes: int
    cap_value: Optional[int]


def build_cell_split(cell: CellSpec, meta_pool: pd.DataFrame, seed: int) -> CellSplit:
    m = meta_pool[filter_classes(meta_pool["label9"].values, cell.classes)]
    cap_value = resolve_cap(m["label9"].values, cell.cap)
    m = m[cap_per_class(m["label9"].values, cap_value, np.random.default_rng(seed))]
    y = m["label9"].values
    if cell.split == "A1":
        tr, te = split_original(m["orig_split"].values)
    elif cell.split == "A2":
        tr, te = split_random(y, seed)
    elif cell.split == "A3":
        tr, te = split_lot(y, m["lot_id"].values, seed)
    elif cell.split == "A4":
        tr, te = split_lot_ordered(m["lot_num"].values, m["lot_id"].values)
    else:
        raise ValueError(cell.split)
    rid = m["row_id"].values
    return CellSplit(cell, seed, rid[tr], rid[te], cell.n_classes, cap_value)
