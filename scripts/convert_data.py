"""Convert the raw WM-811K pickle to data/processed/ and verify known dataset facts."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wm811k_audit.data import build_labeled_table, find_split_column, save_processed, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/raw/MIR-WM811K/Python/WM811K.pkl")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_pickle(args.pkl)
    print(f"loaded {len(df):,} rows in {time.time() - t0:.1f}s; columns={list(df.columns)}; "
          f"split column={find_split_column(df.columns)!r}")
    t1 = time.time()
    maps64, meta = build_labeled_table(df)
    print(f"built labeled table: {len(meta):,} rows in {time.time() - t1:.1f}s")
    save_processed(maps64, meta, Path(args.out))
    s = summarize(meta)
    print(json.dumps(s, indent=2, ensure_ascii=False))

    # Known facts for this dataset (MIR and Kaggle copies carry identical labels).
    assert s["n_labeled"] == 172950, s["n_labeled"]
    assert s["class_counts"]["none"] == 147431 and s["class_counts"]["Near-full"] == 149, s["class_counts"]
    assert s["split_counts"] == {"Test": 118595, "Training": 54355}, s["split_counts"]
    assert s["orig_split_lots_with_both"] == 0, "original split is expected to be lot-disjoint"
    assert set(np.unique(maps64).tolist()) <= {0, 1, 2}
    print("all dataset-fact checks passed")


if __name__ == "__main__":
    main()
