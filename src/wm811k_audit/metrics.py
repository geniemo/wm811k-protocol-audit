"""Two metric families: the cell's own test set (as papers report it) and the common
gold-defect set (8 defect classes, identical negatives for every cell)."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from .constants import DEFECT_IDX


def classification_metrics(y_true, y_pred, n_classes: int) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))
    return dict(
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        bacc=float(balanced_accuracy_score(y_true, y_pred)),
        acc=float(accuracy_score(y_true, y_pred)),
        per_class_f1=f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        per_class_precision=precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        per_class_recall=recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        support=np.bincount(y_true, minlength=n_classes).tolist(),
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    )


def defect_metrics(y_true, y_pred, n_pred_classes: int) -> dict:
    """Mean F1 / mean recall over the defect classes present in y_true.
    A 9-class model predicting 'none' (8) for a defect wafer counts as a miss (FN)
    and is not a false positive for any defect class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size and (y_true.min() < 0 or y_true.max() > 7):
        raise ValueError("defect_metrics expects true labels in 0..7 (defect classes only)")
    if y_true.size == 0:
        return dict(
            defect_f1=float("nan"),
            defect_bacc=float("nan"),
            per_class_f1=[0.0] * 8,
            per_class_recall=[0.0] * 8,
            support=[0] * 8,
            n_classes_present=0,
            confusion=np.zeros((8, n_pred_classes), dtype=int).tolist(),
        )
    f1 = f1_score(y_true, y_pred, labels=DEFECT_IDX, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=DEFECT_IDX, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=8)[:8]
    present = support > 0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_pred_classes)))[:8]
    return dict(
        defect_f1=float(f1[present].mean()) if present.any() else float("nan"),  # empty case handled above
        defect_bacc=float(rec[present].mean()) if present.any() else float("nan"),  # empty case handled above
        per_class_f1=f1.tolist(),
        per_class_recall=rec.tolist(),
        support=support.tolist(),
        n_classes_present=int(present.sum()),
        confusion=cm.tolist(),
    )
