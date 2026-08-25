import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from wm811k_audit.metrics import classification_metrics, defect_metrics


def test_classification_metrics_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 500)
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 9, 500))
    m = classification_metrics(y, p, 9)
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert m["bacc"] == pytest.approx(balanced_accuracy_score(y, p))
    assert m["acc"] == pytest.approx((y == p).mean())
    assert len(m["per_class_f1"]) == 9 and np.array(m["confusion"]).shape == (9, 9)
    assert sum(m["support"]) == 500 and np.array(m["confusion"]).sum() == 500


def test_defect_metrics_treats_none_prediction_as_miss():
    y = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7])
    p = np.array([0, 8, 1, 1, 3, 4, 5, 6, 7])
    m = defect_metrics(y, p, n_pred_classes=9)
    # class 0: TP1 FN1 -> F1 2/3; class 1: TP1 FP1 -> 2/3; class 2: 0; classes 3..7 perfect -> 1
    assert m["per_class_f1"][0] == pytest.approx(2 / 3)
    assert m["per_class_f1"][1] == pytest.approx(2 / 3)
    assert m["per_class_f1"][2] == 0.0
    assert m["defect_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0 + 5) / 8)
    assert m["defect_bacc"] == pytest.approx((0.5 + 1 + 0 + 5) / 8)
    cm = np.array(m["confusion"])
    assert cm.shape == (8, 9) and cm[0, 8] == 1 and cm[2, 1] == 1 and cm.sum() == 9


def test_defect_metrics_averages_only_present_classes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 0])
    m = defect_metrics(y, p, n_pred_classes=8)
    assert m["n_classes_present"] == 2
    assert m["defect_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert np.array(m["confusion"]).shape == (8, 8)


def test_defect_metrics_rejects_none_in_truth():
    with pytest.raises(ValueError):
        defect_metrics(np.array([0, 8]), np.array([0, 8]), 9)
