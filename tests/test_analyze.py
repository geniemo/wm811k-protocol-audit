import json

import numpy as np
import pandas as pd
import pytest

from wm811k_audit.analyze import cell_summary, confusion_table, core_pairs, main_effects, write_confusion_table
from wm811k_audit.splits import all_cells


def _fake_results(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    base = {"A1": 0.70, "A2": 0.85, "A3": 0.72}
    for c in all_cells():
        for s in range(3):
            own = base[c.split] + (0.03 if c.classes == "B2" else 0) + {"C1": 0, "C2": -0.01, "C3": 0.05}[c.cap] + rng.normal(0, 0.005)
            gold = 0.70 + (0.02 if c.classes == "B2" else 0) + {"C1": 0, "C2": -0.02, "C3": -0.10}[c.cap] + rng.normal(0, 0.005)
            rows.append(dict(run_id=f"{c.cell_id}-s{s}", cell_id=c.cell_id, split=c.split, classes=c.classes, cap=c.cap,
                             seed=s, model="smallcnn", own_macro_f1=own, own_defect_f1=own - 0.02, gold_defect_f1=gold,
                             gold_full_macro_f1=gold + 0.05 if c.classes == "B1" else np.nan,
                             lot_share_rate=0.9 if c.split == "A2" else 0.0))
    return pd.DataFrame(rows)


def test_cell_summary_shape():
    s = cell_summary(_fake_results())
    assert len(s) == 18 and s["n_seeds"].eq(3).all()
    assert {"own_macro_f1_mean", "own_macro_f1_std", "gold_defect_f1_mean", "split", "classes", "cap"} <= set(s.columns)


def test_main_effects_recovers_planted_structure():
    me = main_effects(cell_summary(_fake_results()), "own_macro_f1")
    assert me["levels"]["split"]["A2"] > me["levels"]["split"]["A3"] + 0.1
    assert me["range"]["split"] > me["range"]["classes"]
    assert me["interaction_rms"] < 0.02


def test_core_pairs_keys_and_signs():
    cp = core_pairs(_fake_results())
    assert cp["split"]["A2-B1-C1"]["own_macro_f1_mean"] > cp["split"]["A3-B1-C1"]["own_macro_f1_mean"]
    assert cp["split"]["gap_own_macro_f1"] == pytest.approx(
        cp["split"]["A2-B1-C1"]["own_macro_f1_mean"] - cp["split"]["A3-B1-C1"]["own_macro_f1_mean"])
    assert cp["split"]["A2_within_model_gap"] == pytest.approx(
        cp["split"]["A2-B1-C1"]["own_defect_f1_mean"] - cp["split"]["A2-B1-C1"]["gold_defect_f1_mean"])
    assert cp["cap"]["A3-B1-C3"]["gold_defect_f1_mean"] < cp["cap"]["A3-B1-C1"]["gold_defect_f1_mean"]


def _write_fake_gold_confusion_run(runs_dir, cell_id, seed, row0, row1):
    d = runs_dir / f"{cell_id}-s{seed}"
    d.mkdir(parents=True)
    cm = [[0] * 9 for _ in range(9)]
    cm[0] = row0
    cm[1] = row1
    (d / "metrics.json").write_text(json.dumps({"gold_full": {"confusion": cm}}))


def test_confusion_table_sums_seeds_and_computes_recall(tmp_path):
    runs_dir = tmp_path / "runs"
    # seed 0: class 0 -> 8 correct + 2 to class 1 (support 10); class 1 -> 5 correct (support 5)
    _write_fake_gold_confusion_run(runs_dir, "FAKE-B1-C1", 0,
                                    row0=[8, 2, 0, 0, 0, 0, 0, 0, 0], row1=[0, 5, 0, 0, 0, 0, 0, 0, 0])
    # seed 1: class 0 -> all 10 misclassified as class 1; class 1 -> 5 correct
    _write_fake_gold_confusion_run(runs_dir, "FAKE-B1-C1", 1,
                                    row0=[0, 10, 0, 0, 0, 0, 0, 0, 0], row1=[0, 5, 0, 0, 0, 0, 0, 0, 0])

    t = confusion_table(runs_dir, "FAKE-B1-C1")
    assert t is not None
    assert t["support"][0] == pytest.approx(20) and t["support"][1] == pytest.approx(10)
    assert t["cm"][0].tolist() == pytest.approx([8, 12, 0, 0, 0, 0, 0, 0, 0])
    assert t["recall"][0] == pytest.approx(8 / 20)
    assert t["recall"][1] == pytest.approx(1.0)
    # unseen classes (zero support in every seed) must not raise a division error
    assert t["support"][2:].sum() == 0 and np.isfinite(t["norm"][2:]).all()


def test_confusion_table_returns_none_when_no_runs(tmp_path):
    assert confusion_table(tmp_path / "runs", "MISSING-CELL") is None


def test_write_confusion_table_writes_traceable_markdown(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_fake_gold_confusion_run(runs_dir, "FAKE-B1-C1", 0,
                                    row0=[8, 2, 0, 0, 0, 0, 0, 0, 0], row1=[0, 5, 0, 0, 0, 0, 0, 0, 0])
    _write_fake_gold_confusion_run(runs_dir, "FAKE-B1-C1", 1,
                                    row0=[0, 10, 0, 0, 0, 0, 0, 0, 0], row1=[0, 5, 0, 0, 0, 0, 0, 0, 0])
    out_path = tmp_path / "tables" / "confusion_gold_FAKE-B1-C1.md"
    t = write_confusion_table(runs_dir, "FAKE-B1-C1", out_path)
    assert t is not None
    text = out_path.read_text()
    assert "FAKE-B1-C1" in text
    assert "40.0" in text  # class-0 recall, 8/20
    assert "20" in text  # class-0 gold support


def test_seed_run_dirs_excludes_model_suffixed_runs(tmp_path):
    """`glob("A3-B1-C1-s*")` also matches "A3-B1-C1-s0-resnet18"; a table labelled as
    one model must not silently average another model's runs into it."""
    from wm811k_audit.analyze import seed_run_dirs

    for name in ["A3-B1-C1-s0", "A3-B1-C1-s1", "A3-B1-C1-s10",
                 "A3-B1-C1-s0-resnet18", "A3-B1-C1-s1-resnet18", "A3-B1-C1-sX"]:
        (tmp_path / name).mkdir()
    got = [d.name for d in seed_run_dirs(tmp_path, "A3-B1-C1")]
    assert got == ["A3-B1-C1-s0", "A3-B1-C1-s1", "A3-B1-C1-s10"]
