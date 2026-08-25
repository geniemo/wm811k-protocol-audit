"""Enumerate protocol cells, train the fixed model, evaluate on own-test and gold, append to results.csv."""
import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import NONE_IDX, PROCESSED_DIR, RESULTS_DIR
from .data import load_processed
from .diagnostics import split_diagnostics
from .metrics import classification_metrics, defect_metrics
from .model import build_model, count_params
from .splits import CellSpec, all_cells, build_cell_split, carve_gold, parse_cell_id
from .train import TrainConfig, predict, set_seed, train_fixed

RESULT_COLUMNS = [
    "run_id", "cell_id", "split", "classes", "cap", "cap_value", "seed", "model", "n_train", "n_test", "n_classes",
    "own_macro_f1", "own_bacc", "own_acc", "own_defect_f1", "own_defect_bacc",
    "gold_defect_f1", "gold_defect_bacc", "gold_full_macro_f1", "gold_full_bacc", "gold_none_f1",
    "dup_rate", "dup_rate64", "dup_rate_defect", "lot_share_rate",
    "nn_hamming_mean", "nn_hamming_median", "nn_hamming_p10", "nn_hamming_p25", "nn_hamming_p75", "nn_hamming_p90",
    "final_loss", "train_seconds", "n_params",
]


@dataclass
class Dataset:
    maps64: np.ndarray
    meta: pd.DataFrame
    pool_idx: np.ndarray
    gold_idx: np.ndarray


def ensure_gold(meta: pd.DataFrame, processed_dir: Path, min_share: float = 0.15, max_share: float = 0.25):
    processed_dir = Path(processed_dir)
    gpath, ppath = processed_dir / "gold_indices.npy", processed_dir / "pool_indices.npy"
    if gpath.exists() and ppath.exists():
        gold, pool = np.load(gpath), np.load(ppath)
    else:
        pool, gold = carve_gold(meta["label9"].values, meta["lot_id"].values)
        np.save(gpath, gold)
        np.save(ppath, pool)
    lots = meta["lot_id"].values
    if not set(lots[pool]).isdisjoint(set(lots[gold])):
        raise RuntimeError("gold and pool share lots")
    pool_set, gold_set = set(pool.tolist()), set(gold.tolist())
    if not pool_set.isdisjoint(gold_set):
        raise RuntimeError("gold and pool share row indices")
    if pool_set | gold_set != set(range(len(meta))):
        raise RuntimeError("gold and pool do not partition the labeled set")
    labels = meta["label9"].values
    n_classes = int(labels.max()) + 1
    gold_present = set(labels[gold].tolist())
    if gold_present != set(range(n_classes)):
        raise RuntimeError(f"gold set is missing classes {set(range(n_classes)) - gold_present}")
    gold_counts = np.bincount(labels[gold], minlength=n_classes)
    total_counts = np.bincount(labels, minlength=n_classes)
    shares = gold_counts / total_counts
    bad = {i: float(sh) for i, sh in enumerate(shares) if not (min_share <= sh <= max_share)}
    if bad:
        raise RuntimeError(f"gold class shares outside [{min_share}, {max_share}]: {bad}")
    return pool, gold


def load_dataset(processed_dir: Path) -> Dataset:
    maps64, meta = load_processed(processed_dir)
    pool, gold = ensure_gold(meta, processed_dir)
    return Dataset(maps64, meta, pool, gold)


def run_id_for(cell: CellSpec, seed: int, model_name: str) -> str:
    rid = f"{cell.cell_id}-s{seed}"
    return rid if model_name == "smallcnn" else f"{rid}-{model_name}"


def run_one(cell: CellSpec, seed: int, ds: Dataset, maps_t: torch.Tensor, labels_t: torch.Tensor, device,
            cfg: TrainConfig, out_root: Path, model_name: str = "smallcnn") -> dict:
    rid = run_id_for(cell, seed, model_name)
    out_dir = Path(out_root) / "runs" / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    y_all = ds.meta["label9"].values
    cs = build_cell_split(cell, ds.meta.iloc[ds.pool_idx], seed)
    gold = ds.gold_idx
    gold_defect = gold[y_all[gold] != NONE_IDX]
    y_own = y_all[cs.test_idx]
    own_def_mask = y_own != NONE_IDX

    def eval_fn(model):
        p_own = predict(model, maps_t, cs.test_idx, device)
        p_gd = predict(model, maps_t, gold_defect, device)
        return dict(own_macro_f1=classification_metrics(y_own, p_own, cs.n_classes)["macro_f1"],
                    gold_defect_f1=defect_metrics(y_all[gold_defect], p_gd, cs.n_classes)["defect_f1"])

    set_seed(seed)
    model = build_model(model_name, cs.n_classes)
    t0 = time.time()
    log = train_fixed(model, maps_t, labels_t, cs.train_idx, cfg, seed, device, eval_fn)
    train_seconds = time.time() - t0

    p_own = predict(model, maps_t, cs.test_idx, device)
    own = classification_metrics(y_own, p_own, cs.n_classes)
    own_def = defect_metrics(y_own[own_def_mask], p_own[own_def_mask], cs.n_classes)
    p_gd = predict(model, maps_t, gold_defect, device)
    gd = defect_metrics(y_all[gold_defect], p_gd, cs.n_classes)
    gf = classification_metrics(y_all[gold], predict(model, maps_t, gold, device), 9) if cs.n_classes == 9 else None
    diag, h = split_diagnostics(ds.meta, maps_t, cs.train_idx, cs.test_idx, device)

    config = dict(run_id=rid, cell_id=cell.cell_id, seed=seed, model=model_name, n_classes=cs.n_classes,
                  cap_value=cs.cap_value, n_train=int(len(cs.train_idx)), n_test=int(len(cs.test_idx)),
                  n_gold=int(len(gold)), n_gold_defect=int(len(gold_defect)), train_config=cfg.__dict__,
                  n_params=count_params(model))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(dict(config=config, own=own, own_defect=own_def, gold_defect=gd,
                                                          gold_full=gf, diagnostics=diag, train_log=log,
                                                          train_seconds=train_seconds), indent=2))
    pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    np.save(out_dir / "nn_hamming.npy", h)

    row = dict(run_id=rid, cell_id=cell.cell_id, split=cell.split, classes=cell.classes, cap=cell.cap,
               cap_value=cs.cap_value if cs.cap_value is not None else np.nan, seed=seed, model=model_name,
               n_train=len(cs.train_idx), n_test=len(cs.test_idx), n_classes=cs.n_classes,
               own_macro_f1=own["macro_f1"], own_bacc=own["bacc"], own_acc=own["acc"],
               own_defect_f1=own_def["defect_f1"], own_defect_bacc=own_def["defect_bacc"],
               gold_defect_f1=gd["defect_f1"], gold_defect_bacc=gd["defect_bacc"],
               gold_full_macro_f1=gf["macro_f1"] if gf else np.nan, gold_full_bacc=gf["bacc"] if gf else np.nan,
               gold_none_f1=gf["per_class_f1"][NONE_IDX] if gf else np.nan,
               final_loss=log[-1]["loss"], train_seconds=train_seconds, n_params=count_params(model), **diag)
    return {k: row[k] for k in RESULT_COLUMNS}


def completed_run_ids(csv_path: Path) -> set:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    return set(pd.read_csv(csv_path)["run_id"].astype(str))


def append_result(row: dict, csv_path: Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run protocol cells with the fixed model.")
    ap.add_argument("--cells", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--model", default="smallcnn")
    ap.add_argument("--aux-a4", action="store_true", help="also run the supplementary A4-B1-C1 cell")
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    ap.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--processed", default=str(PROCESSED_DIR))
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    cfg = TrainConfig(steps=args.steps, log_every=args.log_every)
    cells = all_cells() if args.cells == ["all"] else [parse_cell_id(c) for c in args.cells]
    if args.aux_a4:
        cells.append(CellSpec("A4", "B1", "C1"))
    ds = load_dataset(Path(args.processed))
    device = torch.device(args.device)
    maps_t = torch.as_tensor(ds.maps64).to(device)
    labels_t = torch.as_tensor(ds.meta["label9"].values.copy(), dtype=torch.long, device=device)
    csv_path = Path(args.out) / "results.csv"
    done = completed_run_ids(csv_path)
    todo = [(c, s) for c in cells for s in args.seeds if run_id_for(c, s, args.model) not in done]
    print(f"{len(done)} runs already in {csv_path}; {len(todo)} to run on {device}")
    for k, (cell, seed) in enumerate(todo, 1):
        rid = run_id_for(cell, seed, args.model)
        if rid in completed_run_ids(csv_path):
            print(f"[{k}/{len(todo)}] skip {rid} (already done)")
            continue
        t0 = time.time()
        row = run_one(cell, seed, ds, maps_t, labels_t, device, cfg, Path(args.out), args.model)
        append_result(row, csv_path)
        print(f"[{k}/{len(todo)}] {rid}: own_macro_f1={row['own_macro_f1']:.3f} gold_defect_f1={row['gold_defect_f1']:.3f} "
              f"lot_share={row['lot_share_rate']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    skipped = len(cells) * len(args.seeds) - len(todo)
    if skipped:
        print(f"skipped {skipped} already-completed runs")


if __name__ == "__main__":
    main()
