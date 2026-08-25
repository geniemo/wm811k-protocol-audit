"""Leakage diagnostics that depend only on the split, not on the model."""
import numpy as np
import pandas as pd
import torch

from .constants import CLASS_NAMES, NONE_IDX
from .preprocess import one_hot_maps


def dup_rate(test_hashes, train_hashes) -> float:
    test_hashes = list(test_hashes)
    if not test_hashes:
        return float("nan")
    s = set(train_hashes)
    return float(np.mean([h in s for h in test_hashes]))


def lot_share_rate(test_lots, train_lots) -> float:
    test_lots = list(test_lots)
    if not test_lots:
        return float("nan")
    s = set(train_lots)
    return float(np.mean([l in s for l in test_lots]))


def nn_hamming(test_maps, train_maps, device="cpu", test_chunk: int = 1024, train_chunk: int = 16384) -> np.ndarray:
    """Hamming distance from each test map to its nearest train map.
    Exact: one-hot the {0,1,2} maps, matches = dot product, computed in float32 with TF32 off."""
    test_t = torch.as_tensor(test_maps)
    train_t = torch.as_tensor(train_maps)
    n_pix = int(test_t.shape[1] * test_t.shape[2])
    out = np.full(len(test_t), n_pix, dtype=np.int32)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            for i in range(0, len(test_t), test_chunk):
                t = one_hot_maps(test_t[i:i + test_chunk].to(device)).reshape(-1, 3 * n_pix)
                best = torch.zeros(t.shape[0], device=device)
                for j in range(0, len(train_t), train_chunk):
                    r = one_hot_maps(train_t[j:j + train_chunk].to(device)).reshape(-1, 3 * n_pix)
                    best = torch.maximum(best, (t @ r.T).max(dim=1).values)
                out[i:i + test_chunk] = (n_pix - best).round().to(torch.int32).cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out


def split_diagnostics(meta: pd.DataFrame, maps, train_idx, test_idx, device="cpu"):
    tr = meta.iloc[train_idx]
    te = meta.iloc[test_idx]
    defect = te["label9"].values != NONE_IDX
    d = dict(
        dup_rate=dup_rate(te["raw_hash"].values, tr["raw_hash"].values),
        dup_rate64=dup_rate(te["map64_hash"].values, tr["map64_hash"].values),
        dup_rate_defect=dup_rate(te["raw_hash"].values[defect], tr["raw_hash"].values),
        lot_share_rate=lot_share_rate(te["lot_id"].values, tr["lot_id"].values),
    )
    maps_t = torch.as_tensor(maps)
    idx_te = torch.as_tensor(np.asarray(test_idx), dtype=torch.long, device=maps_t.device)
    idx_tr = torch.as_tensor(np.asarray(train_idx), dtype=torch.long, device=maps_t.device)
    h = nn_hamming(maps_t[idx_te], maps_t[idx_tr], device)
    d["nn_hamming_mean"] = float(h.mean()) if len(h) else float("nan")
    for q, name in [(50, "median"), (10, "p10"), (25, "p25"), (75, "p75"), (90, "p90")]:
        d[f"nn_hamming_{name}"] = float(np.percentile(h, q)) if len(h) else float("nan")
    return d, h


def _dup_stats(meta: pd.DataFrame, col: str) -> dict:
    vc = meta[col].value_counts()
    dup_hashes = vc[vc > 1].index
    g = meta[meta[col].isin(dup_hashes)]
    if len(g) == 0:
        return dict(n_rows_in_groups=0, frac_rows=0.0, n_groups=0, frac_groups_multi_lot=float("nan"),
                    rows_by_class={}, groups_spanning_orig_split=0)
    per = g.groupby(col)
    return dict(
        n_rows_in_groups=int(len(g)),
        frac_rows=float(len(g) / len(meta)),
        n_groups=int(len(dup_hashes)),
        frac_groups_multi_lot=float((per["lot_id"].nunique() > 1).mean()),
        rows_by_class={c: int(v) for c, v in g["failure_type"].value_counts().items()},
        groups_spanning_orig_split=int((per["orig_split"].nunique() > 1).sum()),
    )


def global_eda(meta: pd.DataFrame) -> dict:
    ct = pd.crosstab(meta["orig_split"], meta["failure_type"])
    by_lot = meta.groupby("lot_id")["orig_split"].agg(lambda s: set(s))
    tr_lots = set(meta.loc[meta["orig_split"] == "Training", "lot_id"])
    te = meta[meta["orig_split"] == "Test"]
    num = meta.dropna(subset=["lot_num"])
    lot_split = num.groupby("lot_id").agg(num=("lot_num", "min"), split=("orig_split", "first")).sort_values("num")
    seq = lot_split["split"].values
    runs = int(1 + (seq[1:] != seq[:-1]).sum()) if len(seq) else 0
    per_lot = meta.groupby("lot_id").size()
    defect = meta[meta["label9"] != NONE_IDX]
    g = defect.groupby("lot_id")["failure_type"]
    multi = g.size()
    multi = multi[multi >= 2]
    shapes = meta.groupby(["orig_h", "orig_w"]).size().sort_values(ascending=False)
    return dict(
        n_labeled=int(len(meta)),
        class_counts={c: int((meta["failure_type"] == c).sum()) for c in CLASS_NAMES},
        orig_split=dict(
            counts={k: int(v) for k, v in meta["orig_split"].value_counts().items()},
            crosstab={s: {c: int(ct.loc[s, c]) if c in ct.columns else 0 for c in CLASS_NAMES} for s in ct.index},
            lots_with_both=int(by_lot.map(lambda s: {"Training", "Test"} <= s).sum()),
            test_wafers_with_lot_in_training=float(te["lot_id"].isin(tr_lots).mean()) if len(te) else float("nan"),
            lot_num_range={s: dict(min=float(num.loc[num["orig_split"] == s, "lot_num"].min()),
                                   max=float(num.loc[num["orig_split"] == s, "lot_num"].max()),
                                   median=float(num.loc[num["orig_split"] == s, "lot_num"].median()))
                           for s in ("Training", "Test") if (num["orig_split"] == s).any()},
            runs_along_lot_order=runs,
        ),
        lots=dict(
            n_lots=int(meta["lot_id"].nunique()),
            n_singleton=int(meta["is_singleton_lot"].sum()),
            labeled_per_lot_quantiles={str(q): float(np.percentile(per_lot, q)) for q in (1, 10, 50, 90, 99, 100)},
            lots_with_2plus_defects=int(len(multi)),
            frac_single_class_among_them=float((g.nunique().loc[multi.index] == 1).mean()) if len(multi) else float("nan"),
            defect_wafers_in_such_lots=int(multi.sum()),
            n_defect_wafers=int(len(defect)),
        ),
        duplicates=_dup_stats(meta, "raw_hash"),
        duplicates64=_dup_stats(meta, "map64_hash"),
        shapes=dict(
            n_unique=int(len(shapes)),
            top=[dict(h=int(h), w=int(w), count=int(c)) for (h, w), c in shapes.head(10).items()],
            h_quantiles={str(q): float(np.percentile(meta["orig_h"], q)) for q in (0, 10, 50, 90, 100)},
            w_quantiles={str(q): float(np.percentile(meta["orig_w"], q)) for q in (0, 10, 50, 90, 100)},
            frac_over_64=float(((meta["orig_h"] > 64) | (meta["orig_w"] > 64)).mean()),
        ),
    )
