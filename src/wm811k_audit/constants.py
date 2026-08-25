from pathlib import Path

CLASS_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full", "none"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NONE_IDX = 8
DEFECT_IDX = list(range(8))
IMG_SIZE = 64
GOLD_SEED = 20260825
N_FOLDS = 5
CAP_VALUES = {"C1": None, "C2": 5000, "C3": "min"}
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
