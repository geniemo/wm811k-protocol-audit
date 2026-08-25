"""One-time normalisation of the raw WM-811K pickle into fixed arrays + metadata."""
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .constants import CLASS_NAMES, CLASS_TO_IDX, IMG_SIZE
from .preprocess import resize_nearest

SPLIT_COL_PATTERN = re.compile(r"tr.*test.*label", re.IGNORECASE)
META_COLUMNS = ["row_id", "lot_name", "lot_id", "lot_num", "is_singleton_lot", "wafer_index", "die_size",
                "orig_h", "orig_w", "failure_type", "label9", "orig_split", "raw_hash", "map64_hash"]


def find_split_column(columns) -> str:
    hits = [c for c in columns if SPLIT_COL_PATTERN.search(str(c))]
    if len(hits) != 1:
        raise KeyError(f"expected exactly one train/test label column, found {hits}")
    return hits[0]


def unwrap_label(v) -> Optional[str]:
    """String labels pass through; nested arrays/lists unwrap to their first element;
    anything else (integer 0, empty array, None, NaN) means 'unlabeled'."""
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, np.ndarray):
        return unwrap_label(v.flat[0]) if v.size > 0 else None
    if isinstance(v, (list, tuple)):
        return unwrap_label(v[0]) if len(v) > 0 else None
    return None


def make_lot_id(lot_name, row_id: int) -> tuple[str, bool]:
    if isinstance(lot_name, str) and lot_name.strip():
        return lot_name.strip(), False
    return f"__singleton_{row_id}", True


def parse_lot_num(lot_name) -> Optional[int]:
    if not isinstance(lot_name, str):
        return None
    m = re.search(r"(\d+)", lot_name)
    return int(m.group(1)) if m else None


def sha1_of_map(arr) -> str:
    a = np.ascontiguousarray(arr, dtype=np.uint8)
    h = hashlib.sha1(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def build_labeled_table(df: pd.DataFrame, size: int = IMG_SIZE):
    split_col = find_split_column(df.columns)
    ftype = df["failureType"].map(unwrap_label)
    labeled_pos = np.flatnonzero(ftype.notna().values)
    n = len(labeled_pos)
    maps64 = np.empty((n, size, size), dtype=np.uint8)
    rows = []
    wm = df["waferMap"].values
    lots = df["lotName"].values
    splits = df[split_col].values
    widx = df["waferIndex"].values
    dsz = df["dieSize"].values
    ft = ftype.values
    for row_id, pos in enumerate(labeled_pos):
        m = np.asarray(wm[pos])
        maps64[row_id] = resize_nearest(m, size)
        lot_name = lots[pos] if isinstance(lots[pos], str) else None
        lot_id, single = make_lot_id(lot_name, row_id)
        name = ft[pos]
        if name not in CLASS_TO_IDX:
            raise ValueError(f"unknown failureType {name!r} at raw row {pos}")
        rows.append(dict(
            row_id=row_id, lot_name=lot_name, lot_id=lot_id, lot_num=parse_lot_num(lot_name),
            is_singleton_lot=single, wafer_index=float(widx[pos]), die_size=float(dsz[pos]),
            orig_h=int(m.shape[0]), orig_w=int(m.shape[1]), failure_type=name, label9=CLASS_TO_IDX[name],
            orig_split=unwrap_label(splits[pos]) or "", raw_hash=sha1_of_map(m),
            map64_hash=sha1_of_map(maps64[row_id])))
    meta = pd.DataFrame(rows, columns=META_COLUMNS)
    meta["lot_num"] = meta["lot_num"].astype(float)
    meta["label9"] = meta["label9"].astype(np.int64)
    meta["is_singleton_lot"] = meta["is_singleton_lot"].astype(bool)
    return maps64, meta


def summarize(meta: pd.DataFrame) -> dict:
    by_lot = meta.groupby("lot_id")["orig_split"].agg(lambda s: set(s))
    both = int(by_lot.map(lambda s: {"Training", "Test"} <= s).sum())
    return dict(
        n_labeled=int(len(meta)),
        class_counts={c: int((meta["failure_type"] == c).sum()) for c in CLASS_NAMES},
        split_counts={k: int(v) for k, v in meta["orig_split"].value_counts().items()},
        n_lots=int(meta["lot_id"].nunique()),
        n_singleton_lots=int(meta["is_singleton_lot"].sum()),
        n_unique_shapes=int(meta.groupby(["orig_h", "orig_w"]).ngroups),
        frac_over_64=float(((meta["orig_h"] > 64) | (meta["orig_w"] > 64)).mean()),
        n_exact_duplicate_rows=int(meta["raw_hash"].duplicated(keep=False).sum()),
        orig_split_lots_with_both=both,
    )


def save_processed(maps64: np.ndarray, meta: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labeled_maps64.npy", maps64)
    meta.to_parquet(out_dir / "labeled_meta.parquet", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summarize(meta), f, indent=2, ensure_ascii=False)


def load_processed(in_dir: Path):
    in_dir = Path(in_dir)
    maps64 = np.load(in_dir / "labeled_maps64.npy")
    meta = pd.read_parquet(in_dir / "labeled_meta.parquet")
    if not (meta["row_id"].values == np.arange(len(meta))).all():
        raise ValueError("meta row order must equal row_id (positional indexing relies on it)")
    return maps64, meta
