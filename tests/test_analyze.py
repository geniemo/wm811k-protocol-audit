import numpy as np
import pandas as pd
import pytest

from wm811k_audit.analyze import cell_summary, core_pairs, main_effects
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
