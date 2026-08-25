import json

import numpy as np
import pandas as pd
import pytest

from wm811k_audit.data import save_processed
from wm811k_audit.run import RESULT_COLUMNS, completed_run_ids, ensure_gold, main


def test_ensure_gold_caches_and_is_lot_disjoint(processed, tmp_path):
    maps64, meta = processed
    save_processed(maps64, meta, tmp_path)
    pool, gold = ensure_gold(meta, tmp_path)
    assert (tmp_path / "gold_indices.npy").exists() and (tmp_path / "pool_indices.npy").exists()
    pool2, gold2 = ensure_gold(meta, tmp_path)
    assert (gold == gold2).all() and (pool == pool2).all()
    lots = meta["lot_id"].values
    assert set(lots[pool]).isdisjoint(set(lots[gold]))


def test_main_runs_cells_writes_results_and_resumes(processed, tmp_path, capsys):
    maps64, meta = processed
    save_processed(maps64, meta, tmp_path)
    out = tmp_path / "results"
    argv = ["--processed", str(tmp_path), "--out", str(out), "--device", "cpu", "--steps", "20", "--log-every", "10",
            "--cells", "A2-B1-C3", "A3-B2-C1", "--seeds", "0"]
    main(argv)
    df = pd.read_csv(out / "results.csv")
    assert list(df.columns) == RESULT_COLUMNS and len(df) == 2
    assert set(df["run_id"]) == {"A2-B1-C3-s0", "A3-B2-C1-s0"}
    r = df.set_index("run_id").loc["A3-B2-C1-s0"]
    assert r["n_classes"] == 8 and np.isnan(r["gold_full_macro_f1"]) and r["lot_share_rate"] == 0.0
    r2 = df.set_index("run_id").loc["A2-B1-C3-s0"]
    assert r2["n_classes"] == 9 and 0.0 <= r2["gold_full_macro_f1"] <= 1.0 and r2["cap_value"] > 0
    m = json.loads((out / "runs" / "A2-B1-C3-s0" / "metrics.json").read_text())
    assert {"own", "own_defect", "gold_defect", "gold_full", "diagnostics", "train_log"} <= set(m)
    assert len(m["train_log"]) == 2 and (out / "runs" / "A2-B1-C3-s0" / "nn_hamming.npy").exists()
    assert completed_run_ids(out / "results.csv") == {"A2-B1-C3-s0", "A3-B2-C1-s0"}
    main(argv)  # resume: nothing new
    assert len(pd.read_csv(out / "results.csv")) == 2
    assert "skip" in capsys.readouterr().out.lower()


def _write_gold_pool(tmp_path, gold, pool):
    np.save(tmp_path / "gold_indices.npy", np.asarray(gold, dtype=np.int64))
    np.save(tmp_path / "pool_indices.npy", np.asarray(pool, dtype=np.int64))


def test_ensure_gold_passes_real_carve_on_fixture(processed, tmp_path):
    # spec-default band (15-25%) and full class presence, exercised on the real carve_gold output
    maps64, meta = processed
    save_processed(maps64, meta, tmp_path)
    pool, gold = ensure_gold(meta, tmp_path)
    labels = meta["label9"].values
    n_classes = int(labels.max()) + 1
    assert set(labels[gold].tolist()) == set(range(n_classes))


def test_ensure_gold_rejects_missing_class(tmp_path):
    # 2 lots, 2 classes; gold (lot "G") contains only class 0, class 1 lives entirely in pool
    meta = pd.DataFrame({
        "lot_id": ["G"] * 4 + ["P"] * 4,
        "label9": [0, 0, 0, 0, 0, 0, 1, 1],
    })
    gold, pool = np.arange(0, 4), np.arange(4, 8)
    _write_gold_pool(tmp_path, gold, pool)
    with pytest.raises(RuntimeError, match="missing classes"):
        ensure_gold(meta, tmp_path)


def test_ensure_gold_rejects_share_outside_band(tmp_path):
    # class 0: 1/10 in gold (10%, below the 15% floor); class 1: 4/20 in gold (20%, within band)
    meta = pd.DataFrame({
        "lot_id": ["G"] * 5 + ["P"] * 25,
        "label9": [0] + [1] * 4 + [0] * 9 + [1] * 16,
    })
    gold, pool = np.arange(0, 5), np.arange(5, 30)
    _write_gold_pool(tmp_path, gold, pool)
    with pytest.raises(RuntimeError, match="share"):
        ensure_gold(meta, tmp_path)
    # the same gold/pool passes once the band is widened enough to admit the 10% class
    ensure_gold(meta, tmp_path, min_share=0.05, max_share=0.5)


def test_aux_a4_cell(processed, tmp_path):
    maps64, meta = processed
    save_processed(maps64, meta, tmp_path)
    out = tmp_path / "results"
    main(["--processed", str(tmp_path), "--out", str(out), "--device", "cpu", "--steps", "10", "--log-every", "10",
          "--cells", "A4-B1-C1", "--seeds", "1"])
    df = pd.read_csv(out / "results.csv")
    assert df.iloc[0]["run_id"] == "A4-B1-C1-s1" and df.iloc[0]["lot_share_rate"] == 0.0
