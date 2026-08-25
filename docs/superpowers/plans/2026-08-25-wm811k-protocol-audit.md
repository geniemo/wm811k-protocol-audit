# WM-811K 평가 프로토콜 감사 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 모델·학습 절차·seed를 고정한 채 18개 평가 프로토콜 셀(분할 3 × 클래스 구성 2 × 클래스당 cap 3) × 3 seed를 돌려, as-reported / on-gold 두 열로 보고되는 성능 이동과 누수 진단을 산출하고 README·면접 노트로 정리한다.

**Architecture:** `src/wm811k_audit/` 패키지 하나. `data.py`(pkl→npy/parquet 1회 변환) → `splits.py`(gold carve, 셀 정의, B/C/A 적용 순서) → `model.py`+`train.py`(고정 CNN, 8,000 step 고정 학습) → `metrics.py`+`diagnostics.py`(as-reported/gold 지표, 누수 진단) → `run.py`(셀 순회·재개·results.csv) → `analyze.py`(표·그림). 데이터는 GPU에 uint8로 상주시키고 DataLoader 없이 인덱싱한다.

**Tech Stack:** Python 3.11 (conda env `wm811k`, 인터프리터 `/home/park/miniconda3/envs/wm811k/bin/python`), torch 2.11+cu128, scikit-learn 1.9, pandas 3.0, pyarrow, matplotlib, pytest. GPU: RTX 5070 Ti 16 GB.

**Spec:** `docs/superpowers/specs/2026-08-25-wm811k-protocol-audit-design.md` — 이 계획은 그 문서의 규격을 구현한다. 충돌 시 spec이 우선하고, spec을 바꾸면 전체를 다시 돌린다.

## Global Constraints

- 인터프리터는 항상 `/home/park/miniconda3/envs/wm811k/bin/python` (base env에는 아무것도 없다). 아래 `PY`로 표기.
- 클래스 순서 고정: `Center=0, Donut=1, Edge-Loc=2, Edge-Ring=3, Loc=4, Random=5, Scratch=6, Near-full=7, none=8`.
- 리사이즈: 64×64, 중심정렬 nearest (`floor((i+0.5)*H/64)`), 종횡비 무시. 입력은 3채널 one-hot.
- gold: `StratifiedGroupKFold(5, shuffle=True, random_state=20260825)`, groups=`lot_id`, fold 0 test.
- 학습 고정값: SmallCNN(32-64-128-128, GAP, Dropout 0.3), AdamW lr 1e-3 wd 1e-4, batch 256, **8,000 steps**, cosine→0, CE(가중치 없음), val 없음, early stopping 없음, 증강 없음, float32.
- seeds `{0,1,2}`. 셀 적용 순서: pool → B 필터 → C cap(seed) → A 분할(seed) → 학습(seed).
- 지표: own-test는 sklearn 기본 macro-F1/balanced accuracy; gold-defect는 결함 8클래스 중 support>0인 클래스의 평균 F1/평균 recall(B1 모델의 none 예측은 FN).
- 하지 않는다: 튜닝, 증강, 백본 비교(stretch의 ResNet-18은 동일 하이퍼파라미터의 강건성 확인일 뿐), raw accuracy 단독 보고.
- 커밋 메시지 끝에 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` 과 `Claude-Session: https://claude.ai/code/session_01SEKHtkGWsiPTaPUsgA9qcB` 트레일러.
- `data/`, `results/runs/`, `docs/interview_notes.md`는 gitignore. 결과 요약(`results/results.csv`, `results/tables/`, `results/figures/`)은 커밋한다.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `pyproject.toml` | 패키지 메타, pytest 설정 (src 레이아웃) |
| `src/wm811k_audit/constants.py` | 클래스 순서, 인덱스, 크기, gold seed, cap 정의 |
| `src/wm811k_audit/preprocess.py` | `nearest_index`, `resize_nearest`, `one_hot_maps` |
| `src/wm811k_audit/data.py` | pkl 컬럼 탐지·라벨 언랩·lot_id·해시·라벨 테이블 생성·저장/로드·요약 |
| `src/wm811k_audit/splits.py` | `CellSpec`, gold carve, B 필터, C cap, A1~A4 분할, `build_cell_split` |
| `src/wm811k_audit/model.py` | `SmallCNN`, (stretch) `ResNet18Adapted`, `build_model` |
| `src/wm811k_audit/train.py` | `TrainConfig`, `set_seed`, `train_fixed`, `predict` |
| `src/wm811k_audit/metrics.py` | `classification_metrics`, `defect_metrics` |
| `src/wm811k_audit/diagnostics.py` | `dup_rate`, `lot_share_rate`, `nn_hamming`, `split_diagnostics`, `global_eda` |
| `src/wm811k_audit/run.py` | 데이터 로드, gold 캐시, `run_one`, results.csv 누적·재개, CLI |
| `src/wm811k_audit/analyze.py` | 셀 요약표, 축별 주효과, 핵심 쌍, 그림 |
| `scripts/convert_data.py` | 실데이터 변환 + 사실 검증 assert |
| `scripts/eda.py` | 전역 EDA → `results/eda.json`, `results/tables/eda.md`, 샘플 그림 |
| `tests/conftest.py` | 합성 raw DataFrame fixture (`make_raw_df`) |
| `tests/test_*.py` | 모듈별 단위·통합 테스트 |

---

### Task 1: 패키지 스캐폴드 + 전처리 (`preprocess.py`)

**Files:**
- Create: `pyproject.toml`, `src/wm811k_audit/__init__.py`, `src/wm811k_audit/constants.py`, `src/wm811k_audit/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Produces: `constants.CLASS_NAMES: list[str]`, `CLASS_TO_IDX: dict[str,int]`, `NONE_IDX=8`, `DEFECT_IDX=[0..7]`, `IMG_SIZE=64`, `GOLD_SEED=20260825`, `N_FOLDS=5`, `CAP_VALUES={"C1":None,"C2":5000,"C3":"min"}`
- Produces: `preprocess.nearest_index(src_len:int, out_len:int)->np.ndarray[int64]`, `resize_nearest(arr:np.ndarray, size:int=64)->np.ndarray[uint8, (size,size)]`, `one_hot_maps(x:torch.Tensor[B,H,W])->torch.Tensor[float32, B,3,H,W]`

- [ ] **Step 1: 스캐폴드 작성**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "wm811k-audit"
version = "0.1.0"
description = "How much does the evaluation protocol move reported performance on WM-811K? Fixed-model factorial audit."
requires-python = ">=3.11"
dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "matplotlib", "torch", "tqdm"]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`src/wm811k_audit/__init__.py`:
```python
"""Evaluation-protocol audit on WM-811K with a fixed model."""
```

`src/wm811k_audit/constants.py`:
```python
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
```

설치: `PY -m pip install -e . --no-deps` (의존성은 이미 env에 있음).

- [ ] **Step 2: 실패하는 테스트 작성** — `tests/test_preprocess.py`

```python
import numpy as np
import pytest
import torch

from wm811k_audit.preprocess import nearest_index, one_hot_maps, resize_nearest


def test_nearest_index_identity_when_same_length():
    assert (nearest_index(64, 64) == np.arange(64)).all()


def test_nearest_index_upsample_replicates_blocks():
    idx = nearest_index(2, 64)
    assert (idx[:32] == 0).all() and (idx[32:] == 1).all()


def test_nearest_index_downsample_stays_in_range():
    idx = nearest_index(300, 64)
    assert idx.min() >= 0 and idx.max() <= 299 and (np.diff(idx) >= 0).all()


def test_resize_identity_on_64x64():
    a = np.random.default_rng(0).integers(0, 3, (64, 64), dtype=np.uint8)
    assert (resize_nearest(a) == a).all()


def test_resize_output_shape_dtype_values():
    a = np.random.default_rng(1).integers(0, 3, (26, 26), dtype=np.uint8)
    out = resize_nearest(a)
    assert out.shape == (64, 64) and out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1, 2}


def test_resize_downsample_uses_only_existing_values():
    a = np.full((120, 80), 2, dtype=np.uint8)
    a[0, 0] = 1
    out = resize_nearest(a)
    assert set(np.unique(out).tolist()) <= {1, 2}


def test_resize_rejects_non_2d():
    with pytest.raises(ValueError):
        resize_nearest(np.zeros((3, 4, 5), dtype=np.uint8))


def test_one_hot_shape_and_exclusive():
    x = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.uint8)
    oh = one_hot_maps(x)
    assert oh.shape == (1, 3, 2, 2) and oh.dtype == torch.float32
    assert torch.equal(oh.sum(dim=1), torch.ones(1, 2, 2))
    assert oh[0, 2, 1, 0] == 1 and oh[0, 1, 0, 1] == 1
```

- [ ] **Step 3: 실패 확인**

Run: `PY -m pytest tests/test_preprocess.py -v`
Expected: `ModuleNotFoundError: No module named 'wm811k_audit.preprocess'` (또는 ImportError)

- [ ] **Step 4: 구현** — `src/wm811k_audit/preprocess.py`

```python
"""Fixed preprocessing: nearest-neighbour resize and one-hot encoding.

Wafer-map pixels are discrete (0 = no die, 1 = pass, 2 = fail). Nearest-neighbour
keeps that vocabulary; bilinear/bicubic would invent values that do not exist.
"""
import numpy as np
import torch

from .constants import IMG_SIZE


def nearest_index(src_len: int, out_len: int) -> np.ndarray:
    """Centre-aligned nearest-neighbour source index for each output position."""
    if src_len <= 0 or out_len <= 0:
        raise ValueError("lengths must be positive")
    idx = np.floor((np.arange(out_len) + 0.5) * src_len / out_len).astype(np.int64)
    return np.clip(idx, 0, src_len - 1)


def resize_nearest(arr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D wafer map, got shape {a.shape}")
    rows = nearest_index(a.shape[0], size)
    cols = nearest_index(a.shape[1], size)
    return a[rows][:, cols].astype(np.uint8, copy=False)


def one_hot_maps(x: torch.Tensor, n_values: int = 3) -> torch.Tensor:
    """uint8/long [B,H,W] with values in {0,1,2} -> float32 [B,3,H,W]."""
    oh = torch.nn.functional.one_hot(x.long(), num_classes=n_values)
    return oh.permute(0, 3, 1, 2).float()
```

- [ ] **Step 5: 통과 확인**

Run: `PY -m pytest tests/test_preprocess.py -v`
Expected: 8 passed

- [ ] **Step 6: 커밋**

```bash
git add pyproject.toml src/wm811k_audit tests/test_preprocess.py
git commit -m "feat: package scaffold, constants, nearest-neighbour resize and one-hot encoding"
```

---

### Task 2: 데이터 정규화 (`data.py`) + 합성 fixture

**Files:**
- Create: `src/wm811k_audit/data.py`, `tests/conftest.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `preprocess.resize_nearest`, `constants.CLASS_TO_IDX`
- Produces:
  - `find_split_column(columns)->str`
  - `unwrap_label(v)->str|None`
  - `make_lot_id(lot_name, row_id:int)->tuple[str,bool]`
  - `parse_lot_num(lot_name)->int|None`
  - `sha1_of_map(arr)->str`
  - `build_labeled_table(df, size=64)->tuple[np.ndarray uint8 [N,64,64], pd.DataFrame meta]`
  - `save_processed(maps64, meta, out_dir)`, `load_processed(in_dir)->tuple[np.ndarray, pd.DataFrame]`
  - `summarize(meta)->dict`
  - meta 컬럼(고정): `row_id, lot_name, lot_id, lot_num(float, NaN 허용), is_singleton_lot, wafer_index, die_size, orig_h, orig_w, failure_type, label9, orig_split, raw_hash, map64_hash`
  - fixture: `tests/conftest.py::make_raw_df(n_lots=60, seed=0, wafers_per_lot=12)->pd.DataFrame`, pytest fixture `raw_df`, `processed` (=(maps64, meta))

- [ ] **Step 1: fixture 작성** — `tests/conftest.py`

```python
import numpy as np
import pandas as pd
import pytest

from wm811k_audit.constants import CLASS_NAMES

SHAPES = [(6, 21), (26, 26), (33, 29), (64, 64), (120, 80)]


def _make_map(rng, shape):
    h, w = shape
    m = rng.integers(1, 3, size=shape, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    r = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2
    m[r > 1] = 0
    return m


def make_raw_df(n_lots: int = 60, seed: int = 0, wafers_per_lot: int = 12) -> pd.DataFrame:
    """Synthetic frame shaped like the MIR pickle, plus Kaggle-style oddities.

    - lots 1..n_lots, first 60% 'Training', rest 'Test' (lot-disjoint like the real data)
    - ~40% none, rest uniform over the 9 class names (so every defect class has ~40 rows)
    - 20 unlabeled rows encoded as integer 0 (MIR style)
    - one Kaggle-style nested-array labeled row and one empty-array unlabeled row
    - 3 labeled rows with missing lotName
    - one exact duplicate inside the same lot, one exact duplicate across lots
    """
    rng = np.random.default_rng(seed)
    rows = []
    for lot in range(1, n_lots + 1):
        split = "Training" if lot <= int(n_lots * 0.6) else "Test"
        for w in range(1, wafers_per_lot + 1):
            cls = "none" if rng.random() < 0.4 else CLASS_NAMES[rng.integers(0, 9)]
            shape = SHAPES[rng.integers(0, len(SHAPES))]
            rows.append(dict(dieSize=float(shape[0] * shape[1]), failureType=cls, lotName=f"lot{lot}",
                             trainTestLabel=split, waferIndex=float(w), waferMap=_make_map(rng, shape)))
    for k in range(20):
        rows.append(dict(dieSize=676.0, failureType=0, lotName=f"lot{n_lots + 1}", trainTestLabel=0,
                         waferIndex=float(k + 1), waferMap=_make_map(rng, (26, 26))))
    rows.append(dict(dieSize=676.0, failureType=np.array([["Center"]], dtype=object), lotName="lot1",
                     trainTestLabel=np.array([["Training"]], dtype=object), waferIndex=13.0,
                     waferMap=_make_map(rng, (26, 26))))
    rows.append(dict(dieSize=676.0, failureType=np.array([], dtype=object), lotName="lot1",
                     trainTestLabel=np.array([], dtype=object), waferIndex=14.0,
                     waferMap=_make_map(rng, (26, 26))))
    for k in range(3):
        rows.append(dict(dieSize=676.0, failureType="Loc", lotName=None, trainTestLabel="Test",
                         waferIndex=1.0, waferMap=_make_map(rng, (26, 26))))
    same_lot_dup = dict(rows[0]); same_lot_dup["waferIndex"] = 99.0
    same_lot_dup["waferMap"] = rows[0]["waferMap"].copy()
    cross_lot_dup = dict(rows[1]); cross_lot_dup["lotName"] = f"lot{n_lots}"; cross_lot_dup["trainTestLabel"] = "Test"
    cross_lot_dup["waferMap"] = rows[1]["waferMap"].copy()
    rows.extend([same_lot_dup, cross_lot_dup])
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def raw_df():
    return make_raw_df()


@pytest.fixture(scope="session")
def processed(raw_df):
    from wm811k_audit.data import build_labeled_table
    return build_labeled_table(raw_df)
```

- [ ] **Step 2: 실패하는 테스트 작성** — `tests/test_data.py`

```python
import numpy as np
import pandas as pd
import pytest

from wm811k_audit.constants import CLASS_NAMES, CLASS_TO_IDX
from wm811k_audit.data import (build_labeled_table, find_split_column, load_processed, make_lot_id,
                               parse_lot_num, save_processed, sha1_of_map, summarize, unwrap_label)


@pytest.mark.parametrize("cols,expected", [
    (["dieSize", "failureType", "lotName", "trainTestLabel", "waferIndex", "waferMap"], "trainTestLabel"),
    (["waferMap", "dieSize", "lotName", "waferIndex", "trianTestLabel", "failureType"], "trianTestLabel"),
])
def test_find_split_column(cols, expected):
    assert find_split_column(cols) == expected


def test_find_split_column_requires_exactly_one():
    with pytest.raises(KeyError):
        find_split_column(["waferMap", "failureType"])


@pytest.mark.parametrize("value,expected", [
    ("Training", "Training"), ("  Test ", "Test"), ("", None), ("   ", None),
    (np.array([["Center"]], dtype=object), "Center"), (np.array([], dtype=object), None),
    ([["Donut"]], "Donut"), ([], None), (0, None), (np.uint64(0), None), (None, None), (float("nan"), None),
])
def test_unwrap_label(value, expected):
    assert unwrap_label(value) == expected


def test_make_lot_id_singleton_for_missing():
    assert make_lot_id("lot12", 5) == ("lot12", False)
    assert make_lot_id(None, 5) == ("__singleton_5", True)
    assert make_lot_id("   ", 7) == ("__singleton_7", True)


def test_parse_lot_num():
    assert parse_lot_num("lot47542") == 47542
    assert parse_lot_num("weird") is None
    assert parse_lot_num(None) is None


def test_sha1_is_shape_sensitive():
    a = np.arange(12, dtype=np.uint8)
    assert sha1_of_map(a.reshape(3, 4)) != sha1_of_map(a.reshape(4, 3))
    assert sha1_of_map(a.reshape(3, 4)) == sha1_of_map(a.reshape(3, 4).copy())


def test_build_labeled_table_shapes_and_columns(processed, raw_df):
    maps64, meta = processed
    n_labeled = sum(isinstance(unwrap_label(v), str) for v in raw_df["failureType"])
    assert maps64.shape == (n_labeled, 64, 64) and maps64.dtype == np.uint8
    assert len(meta) == n_labeled
    assert set(np.unique(maps64).tolist()) <= {0, 1, 2}
    assert list(meta.columns) == ["row_id", "lot_name", "lot_id", "lot_num", "is_singleton_lot", "wafer_index",
                                  "die_size", "orig_h", "orig_w", "failure_type", "label9", "orig_split",
                                  "raw_hash", "map64_hash"]
    assert (meta["row_id"].values == np.arange(n_labeled)).all()


def test_build_labeled_table_labels_and_splits(processed):
    _, meta = processed
    assert set(meta["failure_type"]) <= set(CLASS_NAMES)
    assert (meta["label9"] == meta["failure_type"].map(CLASS_TO_IDX)).all()
    assert set(meta["orig_split"]) == {"Training", "Test"}
    nested = meta[(meta["lot_name"] == "lot1") & (meta["wafer_index"] == 13.0)]
    assert len(nested) == 1 and nested.iloc[0]["failure_type"] == "Center" and nested.iloc[0]["orig_split"] == "Training"
    assert not ((meta["lot_name"] == "lot1") & (meta["wafer_index"] == 14.0)).any()  # empty-array row is unlabeled


def test_build_labeled_table_singletons_and_lot_num(processed):
    _, meta = processed
    single = meta[meta["is_singleton_lot"]]
    assert len(single) == 3 and single["lot_id"].str.startswith("__singleton_").all()
    assert single["lot_num"].isna().all()
    assert meta.loc[meta["lot_name"] == "lot7", "lot_num"].eq(7).all()


def test_build_labeled_table_hashes_detect_duplicates(processed):
    maps64, meta = processed
    vc = meta["raw_hash"].value_counts()
    assert (vc == 2).sum() == 2  # same-lot and cross-lot duplicate pairs
    assert meta["map64_hash"].nunique() <= meta["raw_hash"].nunique()


def test_save_load_roundtrip(processed, tmp_path):
    maps64, meta = processed
    save_processed(maps64, meta, tmp_path)
    m2, meta2 = load_processed(tmp_path)
    assert (m2 == maps64).all()
    pd.testing.assert_frame_equal(meta.reset_index(drop=True), meta2)
    assert (tmp_path / "summary.json").exists()


def test_summarize_keys(processed):
    _, meta = processed
    s = summarize(meta)
    assert s["n_labeled"] == len(meta)
    assert set(s["class_counts"]) == set(CLASS_NAMES)
    assert s["split_counts"]["Training"] > 0 and s["split_counts"]["Test"] > 0
    assert s["orig_split_lots_with_both"] == 0
```

- [ ] **Step 3: 실패 확인**

Run: `PY -m pytest tests/test_data.py -v`
Expected: ImportError (`wm811k_audit.data` 없음)

- [ ] **Step 4: 구현** — `src/wm811k_audit/data.py`

```python
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
```

- [ ] **Step 5: 통과 확인**

Run: `PY -m pytest tests/test_data.py -v`
Expected: 모두 passed. (roundtrip에서 `lot_name` None이 parquet 왕복 후 None으로 유지되는지 확인 — 실패하면 `assert_frame_equal(..., check_dtype=False)`가 아니라 원인을 보고 `lot_name`을 문자열/None으로 정규화한다.)

- [ ] **Step 6: 커밋**

```bash
git add src/wm811k_audit/data.py tests/conftest.py tests/test_data.py
git commit -m "feat: normalise raw pickle into 64x64 uint8 maps + metadata (lot ids, hashes, original split)"
```

---

### Task 3: 실데이터 변환 스크립트 + 변환 실행

**Files:**
- Create: `scripts/convert_data.py`
- Output: `data/processed/labeled_maps64.npy`, `labeled_meta.parquet`, `summary.json`

**Interfaces:**
- Consumes: `data.build_labeled_table`, `save_processed`, `find_split_column`, `summarize`

- [ ] **Step 1: 스크립트 작성** — `scripts/convert_data.py`

```python
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
    assert set(np.unique(maps64[:20000]).tolist()) <= {0, 1, 2}
    print("all dataset-fact checks passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행**

Run: `PY scripts/convert_data.py`
Expected: `loaded 811,457 rows ...`, `built labeled table: 172,950 rows`, summary JSON, `all dataset-fact checks passed`. `data/processed/`에 세 파일(맵 npy ≈ 708 MB).

- [ ] **Step 3: 커밋** (데이터는 gitignore, 스크립트만)

```bash
git add scripts/convert_data.py
git commit -m "feat: raw-to-processed conversion script with dataset-fact assertions"
```

---

### Task 4: 분할 (`splits.py`)

**Files:**
- Create: `src/wm811k_audit/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: `constants.GOLD_SEED, N_FOLDS, NONE_IDX, CAP_VALUES`
- Produces:
  - `CellSpec(split:str, classes:str, cap:str)` frozen dataclass; `.cell_id` (`"A2-B1-C3"`), `.n_classes` (9|8)
  - `parse_cell_id(s)->CellSpec`, `all_cells()->list[CellSpec]` (18개, A→B→C 중첩 순서)
  - `carve_gold(labels, lot_ids, seed=GOLD_SEED, n_splits=N_FOLDS)->(pool_idx, gold_idx)` 정렬된 positional 인덱스
  - `filter_classes(labels, classes)->bool mask`, `resolve_cap(labels, cap_code)->int|None`, `cap_per_class(labels, cap, rng)->bool mask`
  - `split_original(orig_split)->(train_pos, test_pos)`, `split_random(labels, seed)`, `split_lot(labels, lot_ids, seed)`, `split_lot_ordered(lot_nums, lot_ids, test_frac=0.2)`
  - `CellSplit(cell, seed, train_idx, test_idx, n_classes, cap_value)`; `build_cell_split(cell, meta_pool, seed)->CellSplit` — `train_idx/test_idx`는 **전역 row_id**

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_splits.py`

```python
import numpy as np
import pytest

from wm811k_audit.constants import NONE_IDX
from wm811k_audit.splits import (CellSpec, all_cells, build_cell_split, cap_per_class, carve_gold, filter_classes,
                                 parse_cell_id, resolve_cap, split_lot, split_lot_ordered, split_original, split_random)


def test_all_cells_18_unique_in_nested_order():
    cells = all_cells()
    ids = [c.cell_id for c in cells]
    assert len(ids) == 18 and len(set(ids)) == 18
    assert ids[:4] == ["A1-B1-C1", "A1-B1-C2", "A1-B1-C3", "A1-B2-C1"]
    assert parse_cell_id("A3-B2-C3") == CellSpec("A3", "B2", "C3")
    assert CellSpec("A1", "B1", "C1").n_classes == 9 and CellSpec("A1", "B2", "C1").n_classes == 8
    with pytest.raises(ValueError):
        parse_cell_id("A9-B1-C1")


def test_carve_gold_is_lot_disjoint_and_stratified(processed):
    _, meta = processed
    pool, gold = carve_gold(meta["label9"].values, meta["lot_id"].values)
    assert len(pool) + len(gold) == len(meta) and len(np.intersect1d(pool, gold)) == 0
    assert set(meta["lot_id"].values[pool]).isdisjoint(set(meta["lot_id"].values[gold]))
    assert set(meta["label9"].values[gold]) == set(range(9))
    assert 0.12 < len(gold) / len(meta) < 0.30
    pool2, gold2 = carve_gold(meta["label9"].values, meta["lot_id"].values)
    assert (gold2 == gold).all()


def test_filter_classes():
    y = np.array([0, 8, 3, 8, 7])
    assert filter_classes(y, "B1").all()
    assert (filter_classes(y, "B2") == np.array([True, False, True, False, True])).all()


def test_resolve_cap_and_cap_per_class():
    y = np.array([0] * 10 + [1] * 3 + [2] * 6)
    assert resolve_cap(y, "C1") is None and resolve_cap(y, "C2") == 5000 and resolve_cap(y, "C3") == 3
    m = cap_per_class(y, 4, np.random.default_rng(0))
    assert np.bincount(y[m]).tolist() == [4, 3, 4]
    m2 = cap_per_class(y, 4, np.random.default_rng(0))
    assert (m == m2).all()
    assert cap_per_class(y, None, np.random.default_rng(0)).all()


def test_split_original():
    s = np.array(["Training", "Test", "Training", "", "Test"])
    tr, te = split_original(s)
    assert tr.tolist() == [0, 2] and te.tolist() == [1, 4]


def test_split_random_is_stratified_and_may_share_lots(processed):
    _, meta = processed
    y = meta["label9"].values
    tr, te = split_random(y, seed=0)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(y)
    assert 0.15 < len(te) / len(y) < 0.25
    frac_tr = np.bincount(y[tr], minlength=9) / len(tr)
    frac_te = np.bincount(y[te], minlength=9) / len(te)
    assert np.abs(frac_tr - frac_te).max() < 0.05
    lots = meta["lot_id"].values
    assert len(set(lots[tr]) & set(lots[te])) > 0


def test_split_lot_is_lot_disjoint(processed):
    _, meta = processed
    tr, te = split_lot(meta["label9"].values, meta["lot_id"].values, seed=1)
    lots = meta["lot_id"].values
    assert set(lots[tr]).isdisjoint(set(lots[te]))
    assert set(meta["label9"].values[te]) == set(range(9))


def test_split_lot_ordered_puts_later_lots_in_test(processed):
    _, meta = processed
    tr, te = split_lot_ordered(meta["lot_num"].values, meta["lot_id"].values, test_frac=0.2)
    nums = meta["lot_num"].values
    assert not np.isnan(nums[tr]).any() and not np.isnan(nums[te]).any()
    assert nums[te].min() >= nums[tr].max()
    lots = meta["lot_id"].values
    assert set(lots[tr]).isdisjoint(set(lots[te]))
    assert len(tr) + len(te) == int((~np.isnan(nums)).sum())
    assert 0.1 < len(te) / (len(tr) + len(te)) < 0.35


def test_build_cell_split_A3_B2_C3(processed):
    _, meta = processed
    pool, gold = carve_gold(meta["label9"].values, meta["lot_id"].values)
    meta_pool = meta.iloc[pool]
    cs = build_cell_split(CellSpec("A3", "B2", "C3"), meta_pool, seed=0)
    all_idx = np.concatenate([cs.train_idx, cs.test_idx])
    assert set(all_idx) <= set(pool) and len(np.intersect1d(all_idx, gold)) == 0
    y = meta["label9"].values[all_idx]
    assert NONE_IDX not in set(y) and cs.n_classes == 8
    counts = np.bincount(y, minlength=9)[:8]
    assert (counts == cs.cap_value).all()
    lots = meta["lot_id"].values
    assert set(lots[cs.train_idx]).isdisjoint(set(lots[cs.test_idx]))


def test_build_cell_split_A1_respects_original_labels(processed):
    _, meta = processed
    pool, _ = carve_gold(meta["label9"].values, meta["lot_id"].values)
    cs = build_cell_split(CellSpec("A1", "B1", "C1"), meta.iloc[pool], seed=2)
    assert (meta["orig_split"].values[cs.train_idx] == "Training").all()
    assert (meta["orig_split"].values[cs.test_idx] == "Test").all()
    assert cs.cap_value is None and cs.n_classes == 9


def test_build_cell_split_seed_changes_random_split_but_not_A1(processed):
    _, meta = processed
    pool, _ = carve_gold(meta["label9"].values, meta["lot_id"].values)
    a = build_cell_split(CellSpec("A2", "B1", "C1"), meta.iloc[pool], seed=0)
    b = build_cell_split(CellSpec("A2", "B1", "C1"), meta.iloc[pool], seed=1)
    assert not np.array_equal(a.test_idx, b.test_idx)
    c = build_cell_split(CellSpec("A1", "B1", "C1"), meta.iloc[pool], seed=0)
    d = build_cell_split(CellSpec("A1", "B1", "C1"), meta.iloc[pool], seed=1)
    assert np.array_equal(c.test_idx, d.test_idx)
```

- [ ] **Step 2: 실패 확인**

Run: `PY -m pytest tests/test_splits.py -v`
Expected: ImportError

- [ ] **Step 3: 구현** — `src/wm811k_audit/splits.py`

```python
"""Protocol cells and the fixed order in which they are applied:
pool -> class filter (B) -> per-class cap (C, seeded) -> split (A, seeded)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .constants import CAP_VALUES, GOLD_SEED, N_FOLDS, NONE_IDX

SPLITS = ("A1", "A2", "A3")
AUX_SPLITS = ("A4",)
CLASS_SETS = ("B1", "B2")
CAPS = ("C1", "C2", "C3")


@dataclass(frozen=True)
class CellSpec:
    split: str
    classes: str
    cap: str

    @property
    def cell_id(self) -> str:
        return f"{self.split}-{self.classes}-{self.cap}"

    @property
    def n_classes(self) -> int:
        return 9 if self.classes == "B1" else 8


def parse_cell_id(s: str) -> CellSpec:
    parts = s.split("-")
    if len(parts) != 3 or parts[0] not in SPLITS + AUX_SPLITS or parts[1] not in CLASS_SETS or parts[2] not in CAPS:
        raise ValueError(f"bad cell id {s!r}")
    return CellSpec(*parts)


def all_cells() -> list[CellSpec]:
    return [CellSpec(a, b, c) for a in SPLITS for b in CLASS_SETS for c in CAPS]


def _first_fold(splitter, y, groups=None):
    X = np.zeros((len(y), 1))
    tr, te = next(iter(splitter.split(X, y, groups)))
    return np.sort(tr), np.sort(te)


def carve_gold(labels, lot_ids, seed: int = GOLD_SEED, n_splits: int = N_FOLDS):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return _first_fold(sgkf, np.asarray(labels), np.asarray(lot_ids))


def filter_classes(labels, classes: str) -> np.ndarray:
    labels = np.asarray(labels)
    if classes == "B1":
        return np.ones(len(labels), dtype=bool)
    if classes == "B2":
        return labels != NONE_IDX
    raise ValueError(classes)


def resolve_cap(labels, cap_code: str) -> Optional[int]:
    v = CAP_VALUES[cap_code]
    if v == "min":
        counts = np.bincount(np.asarray(labels))
        return int(counts[counts > 0].min())
    return v


def cap_per_class(labels, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels)
    mask = np.ones(len(labels), dtype=bool)
    if cap is None:
        return mask
    for c in np.unique(labels):
        pos = np.flatnonzero(labels == c)
        if len(pos) > cap:
            mask[rng.choice(pos, size=len(pos) - cap, replace=False)] = False
    return mask


def split_original(orig_split):
    s = np.asarray(orig_split)
    return np.flatnonzero(s == "Training"), np.flatnonzero(s == "Test")


def split_random(labels, seed: int, n_splits: int = N_FOLDS):
    return _first_fold(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), np.asarray(labels))


def split_lot(labels, lot_ids, seed: int, n_splits: int = N_FOLDS):
    return _first_fold(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
                       np.asarray(labels), np.asarray(lot_ids))


def split_lot_ordered(lot_nums, lot_ids, test_frac: float = 0.2):
    nums = np.asarray(lot_nums, dtype=float)
    valid = ~np.isnan(nums)
    df = pd.DataFrame({"pos": np.flatnonzero(valid), "lot": np.asarray(lot_ids)[valid], "num": nums[valid]})
    order = df.groupby("lot")["num"].min().sort_values(kind="stable")
    sizes = df.groupby("lot").size().reindex(order.index)
    cum = sizes.cumsum() / sizes.sum()
    test_lots = set(cum.index[cum.values > (1 - test_frac)])
    is_test = df["lot"].isin(test_lots).values
    return df["pos"].values[~is_test], df["pos"].values[is_test]


@dataclass
class CellSplit:
    cell: CellSpec
    seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_classes: int
    cap_value: Optional[int]


def build_cell_split(cell: CellSpec, meta_pool: pd.DataFrame, seed: int) -> CellSplit:
    m = meta_pool[filter_classes(meta_pool["label9"].values, cell.classes)]
    cap_value = resolve_cap(m["label9"].values, cell.cap)
    m = m[cap_per_class(m["label9"].values, cap_value, np.random.default_rng(seed))]
    y = m["label9"].values
    if cell.split == "A1":
        tr, te = split_original(m["orig_split"].values)
    elif cell.split == "A2":
        tr, te = split_random(y, seed)
    elif cell.split == "A3":
        tr, te = split_lot(y, m["lot_id"].values, seed)
    elif cell.split == "A4":
        tr, te = split_lot_ordered(m["lot_num"].values, m["lot_id"].values)
    else:
        raise ValueError(cell.split)
    rid = m["row_id"].values
    return CellSplit(cell, seed, rid[tr], rid[te], cell.n_classes, cap_value)
```

- [ ] **Step 4: 통과 확인**

Run: `PY -m pytest tests/test_splits.py -v`
Expected: 11 passed. (`StratifiedKFold`가 소수 클래스에 대해 UserWarning을 낼 수 있음 — 경고는 허용, 실패는 불허.)

- [ ] **Step 5: 커밋**

```bash
git add src/wm811k_audit/splits.py tests/test_splits.py
git commit -m "feat: protocol cells, lot-disjoint gold carve, class filter, per-class cap, A1-A4 splits"
```

---

### Task 5: 지표 (`metrics.py`)

**Files:**
- Create: `src/wm811k_audit/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `classification_metrics(y_true, y_pred, n_classes)->dict` keys: `macro_f1, bacc, acc, per_class_f1, per_class_precision, per_class_recall, support, confusion` (confusion은 `n_classes×n_classes` list)
  - `defect_metrics(y_true, y_pred, n_pred_classes)->dict` keys: `defect_f1, defect_bacc, per_class_f1, per_class_recall, support, n_classes_present, confusion` (confusion은 `8×n_pred_classes`); `y_true`는 0..7만 허용

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_metrics.py`

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `PY -m pytest tests/test_metrics.py -v`
Expected: ImportError

- [ ] **Step 3: 구현** — `src/wm811k_audit/metrics.py`

```python
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
    f1 = f1_score(y_true, y_pred, labels=DEFECT_IDX, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=DEFECT_IDX, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=8)[:8]
    present = support > 0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_pred_classes)))[:8]
    return dict(
        defect_f1=float(f1[present].mean()) if present.any() else float("nan"),
        defect_bacc=float(rec[present].mean()) if present.any() else float("nan"),
        per_class_f1=f1.tolist(),
        per_class_recall=rec.tolist(),
        support=support.tolist(),
        n_classes_present=int(present.sum()),
        confusion=cm.tolist(),
    )
```

- [ ] **Step 4: 통과 확인**

Run: `PY -m pytest tests/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/wm811k_audit/metrics.py tests/test_metrics.py
git commit -m "feat: own-test and gold-defect metric definitions"
```

---

### Task 6: 누수 진단 (`diagnostics.py`) + EDA 스크립트

**Files:**
- Create: `src/wm811k_audit/diagnostics.py`, `scripts/eda.py`
- Test: `tests/test_diagnostics.py`
- Output: `results/eda.json`, `results/tables/eda.md`, `results/figures/samples_per_class.png`

**Interfaces:**
- Consumes: `preprocess.one_hot_maps`, `constants.NONE_IDX, CLASS_NAMES`
- Produces:
  - `dup_rate(test_hashes, train_hashes)->float`, `lot_share_rate(test_lots, train_lots)->float`
  - `nn_hamming(test_maps, train_maps, device="cpu", test_chunk=1024, train_chunk=16384)->np.ndarray[int32]` (입력은 numpy uint8 또는 torch uint8 텐서)
  - `split_diagnostics(meta, maps, train_idx, test_idx, device)->tuple[dict, np.ndarray]` dict keys: `dup_rate, dup_rate64, dup_rate_defect, lot_share_rate, nn_hamming_mean, nn_hamming_median, nn_hamming_p10, nn_hamming_p25, nn_hamming_p75, nn_hamming_p90`
  - `global_eda(meta)->dict`

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_diagnostics.py`

```python
import numpy as np
import torch

from wm811k_audit.diagnostics import dup_rate, global_eda, lot_share_rate, nn_hamming, split_diagnostics
from wm811k_audit.splits import carve_gold, split_lot, split_random


def test_dup_and_lot_share_rates():
    assert dup_rate(["a", "b", "c", "d"], ["a", "x", "d"]) == 0.5
    assert lot_share_rate(["l1", "l2", "l3"], ["l2", "l9"]) == 1 / 3
    assert np.isnan(dup_rate([], ["a"]))


def test_nn_hamming_matches_bruteforce():
    rng = np.random.default_rng(0)
    test = rng.integers(0, 3, (7, 5, 6), dtype=np.uint8)
    train = rng.integers(0, 3, (11, 5, 6), dtype=np.uint8)
    train[3] = test[2]  # exact duplicate -> distance 0
    brute = (test[:, None] != train[None]).sum(axis=(2, 3)).min(axis=1)
    got = nn_hamming(test, train, device="cpu", test_chunk=3, train_chunk=4)
    assert got.dtype == np.int32 and (got == brute).all() and got[2] == 0
    got_t = nn_hamming(torch.as_tensor(test), torch.as_tensor(train), device="cpu")
    assert (got_t == brute).all()


def test_split_diagnostics_lot_split_has_zero_lot_share(processed):
    maps64, meta = processed
    y, lots = meta["label9"].values, meta["lot_id"].values
    tr, te = split_lot(y, lots, seed=0)
    d, h = split_diagnostics(meta, torch.as_tensor(maps64), tr, te, device="cpu")
    assert d["lot_share_rate"] == 0.0
    assert len(h) == len(te) and d["nn_hamming_median"] >= 0
    tr2, te2 = split_random(y, seed=0)
    d2, _ = split_diagnostics(meta, torch.as_tensor(maps64), tr2, te2, device="cpu")
    assert d2["lot_share_rate"] > 0.5
    assert set(d2) == {"dup_rate", "dup_rate64", "dup_rate_defect", "lot_share_rate", "nn_hamming_mean",
                       "nn_hamming_median", "nn_hamming_p10", "nn_hamming_p25", "nn_hamming_p75", "nn_hamming_p90"}


def test_global_eda_keys_and_known_fixture_facts(processed):
    _, meta = processed
    e = global_eda(meta)
    assert e["n_labeled"] == len(meta)
    assert e["orig_split"]["lots_with_both"] == 0
    assert e["duplicates"]["n_rows_in_groups"] == 4 and e["duplicates"]["n_groups"] == 2
    assert e["duplicates"]["frac_groups_multi_lot"] == 0.5
    assert e["lots"]["n_lots"] == meta["lot_id"].nunique()
    assert set(e["shapes"]["top"][0]) == {"h", "w", "count"}
```

- [ ] **Step 2: 실패 확인**

Run: `PY -m pytest tests/test_diagnostics.py -v`
Expected: ImportError

- [ ] **Step 3: 구현** — `src/wm811k_audit/diagnostics.py`

```python
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
```

- [ ] **Step 4: 통과 확인**

Run: `PY -m pytest tests/test_diagnostics.py -v`
Expected: 4 passed

- [ ] **Step 5: EDA 스크립트 작성** — `scripts/eda.py`

```python
"""Global EDA on the processed labeled set -> results/eda.json, results/tables/eda.md, sample figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from wm811k_audit.constants import CLASS_NAMES, PROCESSED_DIR, RESULTS_DIR
from wm811k_audit.data import load_processed
from wm811k_audit.diagnostics import global_eda

WAFER_CMAP = ListedColormap(["#f0efec", "#9ec5f4", "#0d366b"])  # no die / pass / fail


def write_markdown(e: dict, path: Path):
    lines = ["# WM-811K labeled set — EDA", "", f"- labeled wafers: {e['n_labeled']:,}",
             f"- lots: {e['lots']['n_lots']:,} (singleton lot ids: {e['lots']['n_singleton']})",
             f"- labeled wafers per lot, median: {e['lots']['labeled_per_lot_quantiles']['50']:.0f}", "",
             "## Class counts", "", "| class | count | share |", "|---|---:|---:|"]
    for c in CLASS_NAMES:
        n = e["class_counts"][c]
        lines.append(f"| {c} | {n:,} | {100 * n / e['n_labeled']:.2f}% |")
    o = e["orig_split"]
    lines += ["", "## Original Training/Test labels", "",
              f"- counts: {o['counts']}", f"- lots containing both Training and Test wafers: **{o['lots_with_both']}**",
              f"- Test wafers whose lot also has Training wafers: {o['test_wafers_with_lot_in_training']:.3f}",
              f"- lot-number range: {o['lot_num_range']}", f"- Training/Test runs along lot order: {o['runs_along_lot_order']}",
              "", "| split | " + " | ".join(CLASS_NAMES) + " |", "|---|" + "---:|" * len(CLASS_NAMES)]
    for s, row in o["crosstab"].items():
        lines.append(f"| {s} | " + " | ".join(f"{row[c]:,}" for c in CLASS_NAMES) + " |")
    L = e["lots"]
    lines += ["", "## Within-lot structure", "",
              f"- lots with >=2 defect wafers: {L['lots_with_2plus_defects']:,}; single-class among them: {L['frac_single_class_among_them']:.3f}",
              f"- defect wafers living in such lots: {L['defect_wafers_in_such_lots']:,} / {L['n_defect_wafers']:,}"]
    for key, title in (("duplicates", "Exact duplicates (raw maps)"), ("duplicates64", "Exact duplicates (after 64x64 resize)")):
        d = e[key]
        lines += ["", f"## {title}", "", f"- rows in duplicate groups: {d['n_rows_in_groups']:,} ({100 * d['frac_rows']:.2f}%), groups: {d['n_groups']:,}",
                  f"- groups spanning >1 lot: {d['frac_groups_multi_lot']:.3f}; groups spanning Training&Test: {d['groups_spanning_orig_split']:,}",
                  f"- rows by class: {d['rows_by_class']}"]
    S = e["shapes"]
    lines += ["", "## Map shapes", "", f"- unique shapes: {S['n_unique']}; H quantiles {S['h_quantiles']}; W quantiles {S['w_quantiles']}",
              f"- maps with H>64 or W>64 (downsampled): {100 * S['frac_over_64']:.2f}%",
              "- top shapes: " + ", ".join(f"{t['h']}x{t['w']} ({t['count']:,})" for t in S["top"])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_samples(maps64, meta, path: Path, per_class: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(per_class, 9, figsize=(13.5, 4.8), facecolor="#fcfcfb")
    for j, c in enumerate(CLASS_NAMES):
        rows = meta.index[meta["failure_type"] == c].values
        pick = rng.choice(rows, size=min(per_class, len(rows)), replace=False)
        for i in range(per_class):
            ax = axes[i, j]
            ax.set_axis_off()
            if i < len(pick):
                ax.imshow(maps64[pick[i]], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
            if i == 0:
                ax.set_title(c, fontsize=9, color="#0b0b0b")
    fig.suptitle("WM-811K labeled wafer maps after 64x64 nearest-neighbour resize (gray: no die, light: pass, dark: fail)",
                 fontsize=9, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    maps64, meta = load_processed(PROCESSED_DIR)
    e = global_eda(meta)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    with open(RESULTS_DIR / "eda.json", "w", encoding="utf-8") as f:
        json.dump(e, f, indent=2, ensure_ascii=False)
    write_markdown(e, RESULTS_DIR / "tables" / "eda.md")
    plot_samples(maps64, meta, RESULTS_DIR / "figures" / "samples_per_class.png")
    print(json.dumps({k: e[k] for k in ("n_labeled", "orig_split", "lots", "duplicates")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: EDA 실행**

Run: `PY scripts/eda.py`
Expected: `results/eda.json`, `results/tables/eda.md`, `results/figures/samples_per_class.png`; 출력에 `lots_with_both: 0`, `runs_along_lot_order: 16`, 중복 3.76% 근방.

- [ ] **Step 7: 커밋**

```bash
git add src/wm811k_audit/diagnostics.py tests/test_diagnostics.py scripts/eda.py results/eda.json results/tables/eda.md results/figures/samples_per_class.png
git commit -m "feat: leakage diagnostics (duplicates, lot sharing, nearest-neighbour Hamming) and global EDA"
```

---

### Task 7: 모델과 고정 학습 루프 (`model.py`, `train.py`)

**Files:**
- Create: `src/wm811k_audit/model.py`, `src/wm811k_audit/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `preprocess.one_hot_maps`
- Produces:
  - `SmallCNN(n_classes:int, in_ch:int=3, dropout:float=0.3)`; `build_model(name:str, n_classes:int)->nn.Module` (`"smallcnn"`; `"resnet18"`은 Task 11에서 추가); `count_params(model)->int`
  - `TrainConfig(steps=8000, batch_size=256, lr=1e-3, weight_decay=1e-4, log_every=1000)` frozen dataclass
  - `set_seed(seed)`; `train_fixed(model, maps_u8:torch.Tensor[N,64,64], labels:torch.Tensor[N] long, train_idx:np.ndarray, cfg, seed, device, eval_fn=None)->list[dict]` (모델은 in-place 학습, eval 모드로 반환; log row keys: `step, loss, lr, seconds` + eval_fn 반환 키)
  - `predict(model, maps_u8, idx, device, batch_size=2048)->np.ndarray[int64]`

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_train.py`

```python
import numpy as np
import torch

from wm811k_audit.model import SmallCNN, build_model, count_params
from wm811k_audit.train import TrainConfig, predict, set_seed, train_fixed


def _toy(n_per_class=40, seed=0):
    """Two trivially separable classes a GAP-CNN can learn: sparse random fails vs a dense fail block."""
    rng = np.random.default_rng(seed)
    maps = np.ones((2 * n_per_class, 64, 64), dtype=np.uint8)
    y = np.repeat([0, 1], n_per_class)
    for i in range(len(maps)):
        if y[i] == 0:
            maps[i, rng.integers(0, 64, 5), rng.integers(0, 64, 5)] = 2
        else:
            r0, c0 = rng.integers(0, 40), rng.integers(0, 40)
            maps[i, r0:r0 + 24, c0:c0 + 24] = 2
    return torch.as_tensor(maps), torch.as_tensor(y, dtype=torch.long)


def test_smallcnn_shapes_and_size():
    m = SmallCNN(9)
    out = m(torch.zeros(4, 3, 64, 64))
    assert out.shape == (4, 9)
    assert 100_000 < count_params(m) < 1_000_000
    assert build_model("smallcnn", 8)(torch.zeros(2, 3, 64, 64)).shape == (2, 8)


def test_train_fixed_learns_toy_and_logs():
    maps, y = _toy()
    idx = np.arange(len(y))
    cfg = TrainConfig(steps=60, batch_size=16, log_every=20)
    model = SmallCNN(2)
    log = train_fixed(model, maps, y, idx, cfg, seed=0, device="cpu", eval_fn=lambda m: {"probe": 1.0})
    assert [r["step"] for r in log] == [20, 40, 60] and all("probe" in r for r in log)
    assert log[-1]["loss"] < log[0]["loss"]
    pred = predict(model, maps, idx, device="cpu", batch_size=32)
    assert pred.shape == (len(y),) and (pred == y.numpy()).mean() > 0.9


def test_train_fixed_is_deterministic_on_cpu():
    maps, y = _toy()
    idx = np.arange(len(y))
    cfg = TrainConfig(steps=30, batch_size=16, log_every=30)
    preds = []
    for _ in range(2):
        set_seed(0)
        model = SmallCNN(2)
        train_fixed(model, maps, y, idx, cfg, seed=0, device="cpu")
        preds.append(predict(model, maps, idx, device="cpu"))
    assert (preds[0] == preds[1]).all()


def test_batch_size_is_capped_by_train_size():
    maps, y = _toy(n_per_class=5)
    model = SmallCNN(2)
    log = train_fixed(model, maps, y, np.arange(10), TrainConfig(steps=3, batch_size=256, log_every=3), seed=0, device="cpu")
    assert len(log) == 1
```

- [ ] **Step 2: 실패 확인**

Run: `PY -m pytest tests/test_train.py -v`
Expected: ImportError

- [ ] **Step 3: 구현** — `src/wm811k_audit/model.py`

```python
"""The one fixed model. No tuning happens here or anywhere else."""
import torch
from torch import nn


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    """conv(32)-pool-conv(64)-pool-conv(128)-pool-conv(128)-GAP-dropout-linear. ~0.24M params."""

    def __init__(self, n_classes: int, in_ch: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_ch, 32), nn.MaxPool2d(2),
            _block(32, 64), nn.MaxPool2d(2),
            _block(64, 128), nn.MaxPool2d(2),
            _block(128, 128), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(name: str, n_classes: int) -> nn.Module:
    if name == "smallcnn":
        return SmallCNN(n_classes)
    raise ValueError(f"unknown model {name!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
```

`src/wm811k_audit/train.py`:

```python
"""Fixed optimisation budget: same number of gradient steps for every cell."""
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .preprocess import one_hot_maps


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 8000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    log_every: int = 1000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _batches(n: int, batch_size: int, generator: torch.Generator):
    """Infinite stream of index batches; reshuffles when an epoch is exhausted, drops the remainder."""
    while True:
        perm = torch.randperm(n, generator=generator)
        for s in range(0, n - batch_size + 1, batch_size):
            yield perm[s:s + batch_size]


def train_fixed(model, maps_u8: torch.Tensor, labels: torch.Tensor, train_idx: np.ndarray, cfg: TrainConfig,
                seed: int, device, eval_fn: Optional[Callable] = None) -> list[dict]:
    set_seed(seed)
    model.to(device).train()
    idx_t = torch.as_tensor(np.asarray(train_idx), dtype=torch.long, device=device)
    n = len(idx_t)
    bs = min(cfg.batch_size, n)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    stream = _batches(n, bs, torch.Generator().manual_seed(seed))
    log, running, count, t0 = [], 0.0, 0, time.time()
    for step in range(1, cfg.steps + 1):
        b = idx_t[next(stream).to(device)]
        x = one_hot_maps(maps_u8[b.to(maps_u8.device)].to(device))
        y = labels[b.to(labels.device)].to(device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += float(loss.detach())
        count += 1
        if step % cfg.log_every == 0 or step == cfg.steps:
            row = dict(step=step, loss=running / count, lr=float(sched.get_last_lr()[0]), seconds=time.time() - t0)
            if eval_fn is not None:
                model.eval()
                row.update(eval_fn(model))
                model.train()
            log.append(row)
            running, count = 0.0, 0
    model.eval()
    return log


@torch.no_grad()
def predict(model, maps_u8: torch.Tensor, idx: np.ndarray, device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=maps_u8.device)
    out = []
    for s in range(0, len(idx_t), batch_size):
        x = one_hot_maps(maps_u8[idx_t[s:s + batch_size]].to(device))
        out.append(model(x).argmax(dim=1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.int64)
```

`maps_u8`/`labels`가 `device`와 다른 장치에 있어도 동작하도록 인덱스를 텐서의 장치로 옮겨 인덱싱한 뒤 `device`로 보낸다(실행 시 maps는 GPU 상주라 no-op).

- [ ] **Step 4: 통과 확인**

Run: `PY -m pytest tests/test_train.py -v`
Expected: 4 passed (CPU, 수 초)

- [ ] **Step 5: 커밋**

```bash
git add src/wm811k_audit/model.py src/wm811k_audit/train.py tests/test_train.py
git commit -m "feat: fixed SmallCNN and step-budgeted training loop with deterministic seeding"
```

---

### Task 8: 실험 실행기 (`run.py`) — 셀 순회, 재개, results.csv

**Files:**
- Create: `src/wm811k_audit/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: 모든 이전 모듈
- Produces:
  - `Dataset(maps64, meta, pool_idx, gold_idx)`; `ensure_gold(meta, processed_dir)->(pool_idx, gold_idx)` (`gold_indices.npy`/`pool_indices.npy` 캐시, lot-disjoint 검증)
  - `load_dataset(processed_dir)->Dataset`
  - `RESULT_COLUMNS: list[str]`
  - `run_one(cell, seed, ds, maps_t, labels_t, device, cfg, out_root, model_name="smallcnn")->dict` (row) — `results/runs/<run_id>/{config.json, metrics.json, train_log.csv, nn_hamming.npy}` 기록
  - `completed_run_ids(csv_path)->set[str]`, `append_result(row, csv_path)`
  - `main(argv=None)`; CLI: `--cells all|<ids...>`, `--seeds 0 1 2`, `--model smallcnn`, `--aux-a4`, `--steps`, `--log-every`, `--device`, `--processed`, `--out`
  - run_id 규칙: `"{cell_id}-s{seed}"`, smallcnn 외 모델은 `"-{model}"` 접미

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_run.py`

```python
import json

import numpy as np
import pandas as pd

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


def test_aux_a4_cell(processed, tmp_path):
    maps64, meta = processed
    save_processed(maps64, meta, tmp_path)
    out = tmp_path / "results"
    main(["--processed", str(tmp_path), "--out", str(out), "--device", "cpu", "--steps", "10", "--log-every", "10",
          "--cells", "A4-B1-C1", "--seeds", "1"])
    df = pd.read_csv(out / "results.csv")
    assert df.iloc[0]["run_id"] == "A4-B1-C1-s1" and df.iloc[0]["lot_share_rate"] == 0.0
```

- [ ] **Step 2: 실패 확인**

Run: `PY -m pytest tests/test_run.py -v`
Expected: ImportError

- [ ] **Step 3: 구현** — `src/wm811k_audit/run.py`

```python
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


def ensure_gold(meta: pd.DataFrame, processed_dir: Path):
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
    if len(pool) + len(gold) != len(meta):
        raise RuntimeError("gold/pool do not partition the labeled set")
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
    labels_t = torch.as_tensor(ds.meta["label9"].values, dtype=torch.long, device=device)
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
```

- [ ] **Step 4: 통과 확인**

Run: `PY -m pytest tests/test_run.py -v`
Expected: 3 passed. (합성 데이터 pool에서 `A2-B1-C3`의 cap은 pool 내 최소 클래스 수(≈30)이고 `StratifiedKFold(5)`가 돌아갈 만큼 크다. 실패하면 fixture 크기가 아니라 로직을 의심한다.)

- [ ] **Step 5: 전체 테스트**

Run: `PY -m pytest -q`
Expected: 전부 passed

- [ ] **Step 6: 커밋**

```bash
git add src/wm811k_audit/run.py tests/test_run.py
git commit -m "feat: experiment runner with gold cache, per-run artifacts, resumable results.csv"
```

---

### Task 9: 실데이터 실행 — gold 확정, 스모크, 본 실험 54 + 보조 3 run

**Files:**
- Output: `data/processed/gold_indices.npy`, `data/processed/pool_indices.npy`, `results/results.csv`, `results/runs/*`

- [ ] **Step 1: gold 확정과 스모크(짧은 step)**

Run:
```bash
PY -m wm811k_audit.run --cells A3-B1-C3 --seeds 0 --steps 200 --log-every 100 --out /tmp/claude-1000/-home-park-workspace-wm811k-protocol-audit/de126fa2-9e5c-4678-a0f0-95645703ee6b/scratchpad/smoke
PY - <<'EOF'
import numpy as np, pandas as pd
meta = pd.read_parquet("data/processed/labeled_meta.parquet")
gold = np.load("data/processed/gold_indices.npy"); pool = np.load("data/processed/pool_indices.npy")
print("gold", len(gold), "pool", len(pool), "gold frac %.3f" % (len(gold)/len(meta)))
print("gold class counts", np.bincount(meta.label9.values[gold], minlength=9).tolist())
print("pool class counts", np.bincount(meta.label9.values[pool], minlength=9).tolist())
import hashlib; print("gold sha1", hashlib.sha1(gold.tobytes()).hexdigest()[:12])
EOF
```
Expected: 스모크 run이 오류 없이 끝나고 results.csv 1행. gold ≈ 20% (34,000±3,000), 9클래스 모두 존재(Near-full ≈ 30). gold sha1 12자리를 기록해 README에 넣는다.

- [ ] **Step 2: 본 실험 1 run 시간 측정**

Run: `PY -m wm811k_audit.run --cells A3-B1-C1 --seeds 0` (기본 8,000 step, `results/`에 기록)
Expected: 1~3분. `results/results.csv` 1행. 출력의 `own_macro_f1`, `gold_defect_f1`가 0.5 이상이면 정상 범위(2605.14255의 ResNet18 macro-F1 0.85가 상한 참고치).

- [ ] **Step 3: 나머지 전부 백그라운드 실행**

Run (백그라운드, 로그는 scratchpad):
```bash
nohup PY -m wm811k_audit.run --cells all --seeds 0 1 2 --aux-a4 > <scratchpad>/full_run.log 2>&1 &
```
Expected: `57 runs`에서 이미 끝난 1개를 제외한 56개 순회. 완료 후 `results/results.csv` 57행. 중간에 끊기면 같은 명령을 다시 실행(재개).

- [ ] **Step 4: 완료 검증**

```bash
PY - <<'EOF'
import pandas as pd
df = pd.read_csv("results/results.csv")
print(len(df), "rows;", df.cell_id.nunique(), "cells;", df.groupby("cell_id").seed.nunique().min(), "seeds/cell min")
print(df.groupby("cell_id")[["own_macro_f1","gold_defect_f1","lot_share_rate","dup_rate"]].mean().round(3).to_string())
assert len(df) == 57 and (df.groupby("cell_id").seed.nunique() == 3).all()
assert (df[df.split.isin(["A1","A3","A4"])].lot_share_rate == 0).all()
EOF
```
Expected: 57 rows, 19 cells, 3 seeds each; A1/A3/A4에서 lot_share_rate 0.

- [ ] **Step 5: 커밋** (results.csv와 gold 해시 메모)

```bash
git add results/results.csv
git commit -m "results: 18 protocol cells x 3 seeds + A4 supplementary (fixed SmallCNN, 8000 steps)"
```

---

### Task 10: 분석 (`analyze.py`) — 표·축별 기여도·핵심 쌍·그림

**Files:**
- Create: `src/wm811k_audit/analyze.py`
- Test: `tests/test_analyze.py`
- Output: `results/tables/{cells.md,cells.csv,main_effects.md,core_pairs.md,seed_noise.md}`, `results/figures/{core_pair_split.png,core_pair_cap.png,main_effects.png,confusion_gold_A3-B1-C1.png,nn_hamming_A2_vs_A3.png}`

**Interfaces:**
- Consumes: `results/results.csv`, `results/runs/<run_id>/metrics.json`, `nn_hamming.npy`
- Produces:
  - `load_results(path)->pd.DataFrame`
  - `cell_summary(df, metrics=SUMMARY_METRICS)->pd.DataFrame` (index cell_id; columns `<metric>_mean`, `<metric>_std`, `n_seeds`, `split, classes, cap`)
  - `main_effects(summary, metric)->dict` keys: `levels` ({axis: {level: mean}}), `range` ({axis: max-min}), `interaction_rms`, `interaction_max_abs`
  - `core_pairs(df)->dict`
  - `write_tables(...)`, `plot_*` 함수들, `main(argv=None)`
- 색: 슬롯1 파랑 `#2a78d6`(as-reported / A2 random), 슬롯2 주황 `#eb6834`(on-gold / A3 lot), 슬롯3 아쿠아 `#1baf7a`(A1 original; 직접 라벨 병기), 순차 램프 파랑 `#cde2fb→#0d366b`, 잉크 `#0b0b0b/#52514e/#898781`, 격자 `#e1e0d9`, 표면 `#fcfcfb`. 검증기 통과 확인됨(2026-08-25). 이중 축 금지, 막대 직접 라벨은 값 텍스트(잉크색)로만.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_analyze.py`

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `PY -m pytest tests/test_analyze.py -v`
Expected: ImportError

- [ ] **Step 3: 구현** — `src/wm811k_audit/analyze.py`

```python
"""Turn results.csv + per-run artifacts into the tables and figures the README uses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .constants import CLASS_NAMES, RESULTS_DIR

SUMMARY_METRICS = ["own_macro_f1", "own_defect_f1", "gold_defect_f1", "gold_full_macro_f1"]
AXES = {"split": ["A1", "A2", "A3"], "classes": ["B1", "B2"], "cap": ["C1", "C2", "C3"]}
AXIS_LABELS = {"split": {"A1": "original", "A2": "random", "A3": "lot-group"},
               "classes": {"B1": "9 classes", "B2": "8 defect classes"},
               "cap": {"C1": "no cap", "C2": "cap 5000/class", "C3": "balanced (min class)"}}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["model"] == "smallcnn"].copy() if "model" in df.columns else df


def cell_summary(df: pd.DataFrame, metrics=SUMMARY_METRICS) -> pd.DataFrame:
    g = df.groupby("cell_id")
    out = pd.DataFrame({"split": g["split"].first(), "classes": g["classes"].first(), "cap": g["cap"].first(),
                        "n_seeds": g["seed"].nunique()})
    for m in metrics:
        if m in df.columns:
            out[f"{m}_mean"] = g[m].mean()
            out[f"{m}_std"] = g[m].std(ddof=0)
    return out


def main_effects(summary: pd.DataFrame, metric: str) -> dict:
    col = f"{metric}_mean"
    s = summary.dropna(subset=[col])
    s = s[s["split"].isin(AXES["split"])]
    levels = {ax: {lv: float(s.loc[s[ax] == lv, col].mean()) for lv in lvs if (s[ax] == lv).any()} for ax, lvs in AXES.items()}
    rng = {ax: (max(v.values()) - min(v.values())) if v else float("nan") for ax, v in levels.items()}
    # additive model y ~ 1 + A + B + C on cell means; residual = interaction
    X = [np.ones(len(s))]
    for ax, lvs in AXES.items():
        for lv in lvs[1:]:
            X.append((s[ax] == lv).astype(float).values)
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, s[col].values, rcond=None)
    resid = s[col].values - X @ beta
    return dict(metric=metric, levels=levels, range=rng, interaction_rms=float(np.sqrt(np.mean(resid ** 2))),
                interaction_max_abs=float(np.max(np.abs(resid))) if len(resid) else float("nan"))


def core_pairs(df: pd.DataFrame) -> dict:
    s = cell_summary(df)

    def cell(cid):
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in s.loc[cid].items()} if cid in s.index else None

    split = {cid: cell(cid) for cid in ("A2-B1-C1", "A3-B1-C1", "A1-B1-C1", "A4-B1-C1") if cid in s.index}
    if "A2-B1-C1" in s.index and "A3-B1-C1" in s.index:
        split["gap_own_macro_f1"] = float(s.loc["A2-B1-C1", "own_macro_f1_mean"] - s.loc["A3-B1-C1", "own_macro_f1_mean"])
        split["gap_gold_defect_f1"] = float(s.loc["A2-B1-C1", "gold_defect_f1_mean"] - s.loc["A3-B1-C1", "gold_defect_f1_mean"])
        split["A2_within_model_gap"] = float(s.loc["A2-B1-C1", "own_defect_f1_mean"] - s.loc["A2-B1-C1", "gold_defect_f1_mean"])
        split["A3_within_model_gap"] = float(s.loc["A3-B1-C1", "own_defect_f1_mean"] - s.loc["A3-B1-C1", "gold_defect_f1_mean"])
    cap = {cid: cell(cid) for cid in ("A3-B1-C1", "A3-B1-C2", "A3-B1-C3") if cid in s.index}
    if "A3-B1-C1" in s.index and "A3-B1-C3" in s.index:
        cap["gap_own_macro_f1"] = float(s.loc["A3-B1-C3", "own_macro_f1_mean"] - s.loc["A3-B1-C1", "own_macro_f1_mean"])
        cap["gap_gold_defect_f1"] = float(s.loc["A3-B1-C3", "gold_defect_f1_mean"] - s.loc["A3-B1-C1", "gold_defect_f1_mean"])
    return dict(split=split, cap=cap)


def _fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def write_tables(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = cell_summary(df)
    s.to_csv(out_dir / "cells.csv")
    lines = ["| cell | split | classes | cap | n_train | as-reported macro-F1 (own test) | gold defect-F1 (common) | gold 9-class macro-F1 | lot_share | dup_rate |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    ntr = df.groupby("cell_id")["n_train"].mean()
    ls = df.groupby("cell_id")["lot_share_rate"].mean()
    dr = df.groupby("cell_id")["dup_rate"].mean()
    for cid, r in s.iterrows():
        gf = _fmt(r["gold_full_macro_f1_mean"], r["gold_full_macro_f1_std"]) if not np.isnan(r["gold_full_macro_f1_mean"]) else "—"
        lines.append(f"| {cid} | {AXIS_LABELS['split'].get(r['split'], r['split'])} | {AXIS_LABELS['classes'][r['classes']]} | "
                     f"{AXIS_LABELS['cap'][r['cap']]} | {ntr[cid]:,.0f} | {_fmt(r['own_macro_f1_mean'], r['own_macro_f1_std'])} | "
                     f"{_fmt(r['gold_defect_f1_mean'], r['gold_defect_f1_std'])} | {gf} | {ls[cid]:.2f} | {dr[cid]:.3f} |")
    (out_dir / "cells.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    me_lines = ["| metric | axis | level means | range (max−min) | interaction RMS | interaction max |", "|---|---|---|---:|---:|---:|"]
    effects = {}
    for metric in ("own_macro_f1", "gold_defect_f1"):
        me = main_effects(s, metric)
        effects[metric] = me
        for ax in AXES:
            lv = ", ".join(f"{AXIS_LABELS[ax][k]}={v:.3f}" for k, v in me["levels"][ax].items())
            me_lines.append(f"| {metric} | {ax} | {lv} | {me['range'][ax]:.3f} | {me['interaction_rms']:.3f} | {me['interaction_max_abs']:.3f} |")
    (out_dir / "main_effects.md").write_text("\n".join(me_lines) + "\n", encoding="utf-8")

    cp = core_pairs(df)
    cp_lines = ["## Core pair 1 — random vs lot-group split (9 classes, no cap)", ""]
    for cid in ("A2-B1-C1", "A3-B1-C1", "A1-B1-C1", "A4-B1-C1"):
        if cid in cp["split"]:
            c = cp["split"][cid]
            cp_lines.append(f"- {cid}: own macro-F1 {_fmt(c['own_macro_f1_mean'], c['own_macro_f1_std'])}, own defect-F1 {_fmt(c['own_defect_f1_mean'], c['own_defect_f1_std'])}, "
                            f"gold defect-F1 {_fmt(c['gold_defect_f1_mean'], c['gold_defect_f1_std'])}")
    for k in ("gap_own_macro_f1", "gap_gold_defect_f1", "A2_within_model_gap", "A3_within_model_gap"):
        if k in cp["split"]:
            cp_lines.append(f"- {k}: {cp['split'][k]:+.3f}")
    cp_lines += ["", "## Core pair 2 — full vs balanced subset (lot-group split, 9 classes)", ""]
    for cid in ("A3-B1-C1", "A3-B1-C2", "A3-B1-C3"):
        if cid in cp["cap"]:
            c = cp["cap"][cid]
            cp_lines.append(f"- {cid}: own macro-F1 {_fmt(c['own_macro_f1_mean'], c['own_macro_f1_std'])}, gold defect-F1 {_fmt(c['gold_defect_f1_mean'], c['gold_defect_f1_std'])}")
    for k in ("gap_own_macro_f1", "gap_gold_defect_f1"):
        if k in cp["cap"]:
            cp_lines.append(f"- {k}: {cp['cap'][k]:+.3f}")
    (out_dir / "core_pairs.md").write_text("\n".join(cp_lines) + "\n", encoding="utf-8")

    noise = s[["own_macro_f1_std", "gold_defect_f1_std"]].describe().loc[["mean", "max"]]
    (out_dir / "seed_noise.md").write_text("| | own macro-F1 seed std | gold defect-F1 seed std |\n|---|---:|---:|\n" +
                                          "\n".join(f"| {i} | {r['own_macro_f1_std']:.4f} | {r['gold_defect_f1_std']:.4f} |" for i, r in noise.iterrows()) + "\n",
                                          encoding="utf-8")
    return dict(summary=s, effects=effects, core=cp)


def _bar_pair(ax, labels, means, stds, colors, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, width=0.55, color=colors, capsize=3, error_kw=dict(ecolor=INK2, lw=1))
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", visible=False)


def plot_core_pair_split(cp: dict, out: Path):
    cells = [c for c in ("A1-B1-C1", "A2-B1-C1", "A3-B1-C1") if c in cp["split"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    names = {"A1-B1-C1": "original", "A2-B1-C1": "random", "A3-B1-C1": "lot-group"}
    cols = {"A1-B1-C1": AQUA, "A2-B1-C1": BLUE, "A3-B1-C1": ORANGE}
    _bar_pair(axes[0], [names[c] for c in cells], [cp["split"][c]["own_macro_f1_mean"] for c in cells],
              [cp["split"][c]["own_macro_f1_std"] for c in cells], [cols[c] for c in cells],
              "As reported: macro-F1 on each protocol's own test set", "macro-F1 (9 classes)")
    _bar_pair(axes[1], [names[c] for c in cells], [cp["split"][c]["gold_defect_f1_mean"] for c in cells],
              [cp["split"][c]["gold_defect_f1_std"] for c in cells], [cols[c] for c in cells],
              "Same models on the common lot-disjoint gold set", "defect-F1 (8 classes)")
    fig.suptitle("Split protocol (9 classes, no cap) — bars: mean of 3 seeds, whiskers: seed std", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_core_pair_cap(cp: dict, out: Path):
    cells = [c for c in ("A3-B1-C1", "A3-B1-C2", "A3-B1-C3") if c in cp["cap"]]
    names = {"A3-B1-C1": "no cap", "A3-B1-C2": "cap 5000", "A3-B1-C3": "balanced"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _bar_pair(axes[0], [names[c] for c in cells], [cp["cap"][c]["own_macro_f1_mean"] for c in cells],
              [cp["cap"][c]["own_macro_f1_std"] for c in cells], [BLUE] * len(cells),
              "As reported: macro-F1 on each protocol's own test set", "macro-F1 (9 classes)")
    _bar_pair(axes[1], [names[c] for c in cells], [cp["cap"][c]["gold_defect_f1_mean"] for c in cells],
              [cp["cap"][c]["gold_defect_f1_std"] for c in cells], [ORANGE] * len(cells),
              "Same models on the common lot-disjoint gold set", "defect-F1 (8 classes)")
    fig.suptitle("Sample-selection protocol (lot-group split, 9 classes)", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_main_effects(effects: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, axis in zip(axes, AXES):
        for metric, color, label in (("own_macro_f1", BLUE, "as-reported (own test)"), ("gold_defect_f1", ORANGE, "on gold (common)")):
            lv = effects[metric]["levels"][axis]
            xs = np.arange(len(lv))
            ax.plot(xs, list(lv.values()), marker="o", ms=6, lw=2, color=color, label=label)
        ax.set_xticks(np.arange(len(lv)), [AXIS_LABELS[axis][k] for k in lv], rotation=0)
        ax.set_title(f"axis: {axis}", fontsize=10, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("level mean over the other axes")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Main effect of each protocol axis (18 cells, 3 seeds each)", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusion(runs_dir: Path, cell_id: str, out: Path):
    cms = []
    for d in sorted(runs_dir.glob(f"{cell_id}-s*")):
        m = json.loads((d / "metrics.json").read_text())
        if m.get("gold_full"):
            cms.append(np.array(m["gold_full"]["confusion"], dtype=float))
    if not cms:
        return
    cm = np.sum(cms, axis=0)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(9), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(9), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{100 * norm[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_title(f"{cell_id} on gold (row-normalised %, 3 seeds summed; n={int(cm.sum()):,})", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_nn_hamming(runs_dir: Path, out: Path):
    series = []
    for cid, color, label in (("A2-B1-C1", BLUE, "random split"), ("A3-B1-C1", ORANGE, "lot-group split")):
        d = runs_dir / f"{cid}-s0" / "nn_hamming.npy"
        if d.exists():
            series.append((np.load(d), color, label))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    hi = max(int(np.percentile(h, 99)) for h, _, _ in series)
    bins = np.linspace(0, max(hi, 1), 60)
    for h, color, label in series:
        ax.hist(h, bins=bins, histtype="step", lw=2, color=color, label=f"{label} (median {np.median(h):.0f})", density=True)
    ax.set_xlabel("Hamming distance from each test wafer to its nearest training wafer (64x64 dies)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("How close is the nearest training example? (seed 0, 9 classes, no cap)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)
    root = Path(args.results)
    df = load_results(root / "results.csv")
    tables = write_tables(df, root / "tables")
    figs = root / "figures"
    figs.mkdir(exist_ok=True)
    plot_core_pair_split(tables["core"], figs / "core_pair_split.png")
    plot_core_pair_cap(tables["core"], figs / "core_pair_cap.png")
    plot_main_effects(tables["effects"], figs / "main_effects.png")
    plot_confusion(root / "runs", "A3-B1-C1", figs / "confusion_gold_A3-B1-C1.png")
    plot_nn_hamming(root / "runs", figs / "nn_hamming_A2_vs_A3.png")
    print((root / "tables" / "core_pairs.md").read_text())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인**

Run: `PY -m pytest tests/test_analyze.py -v`
Expected: 3 passed

- [ ] **Step 5: 실데이터 분석 실행 + 그림 눈으로 확인**

Run: `PY -m wm811k_audit.analyze` 후 `results/figures/*.png`를 Read 도구로 열어 라벨 겹침·잘림·범례 유무를 확인한다(dataviz 절차 7). 문제 있으면 figsize/회전만 조정(데이터·색 규칙은 유지).

- [ ] **Step 6: 커밋**

```bash
git add src/wm811k_audit/analyze.py tests/test_analyze.py results/tables results/figures
git commit -m "feat: analysis tables (cells, main effects, core pairs, seed noise) and figures"
```

---

### Task 11 (stretch): 두 번째 고정 모델 ResNet-18 — 핵심 두 쌍만

**Files:**
- Modify: `src/wm811k_audit/model.py` (`ResNet18Adapted`, `build_model("resnet18")`)
- Test: `tests/test_train.py`에 1개 추가
- Output: `results/results.csv`에 `model=resnet18` 9행

- [ ] **Step 1: torchvision 설치**

Run: `PY -m pip install torchvision --index-url https://download.pytorch.org/whl/cu128`
Expected: torch 2.11과 호환되는 torchvision 설치. `PY -c "import torchvision; print(torchvision.__version__)"`

- [ ] **Step 2: 실패하는 테스트 추가** — `tests/test_train.py` 끝에

```python
def test_resnet18_adapted_shape():
    m = build_model("resnet18", 9)
    assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 9)
    assert count_params(m) > 10_000_000
```

- [ ] **Step 3: 구현** — `model.py`에 추가

```python
class ResNet18Adapted(nn.Module):
    """torchvision resnet18 from scratch; 3x3 stride-1 stem and no stem max-pool so 64x64 inputs keep an 8x8 final map."""

    def __init__(self, n_classes: int, in_ch: int = 3):
        super().__init__()
        from torchvision.models import resnet18
        net = resnet18(weights=None, num_classes=n_classes)
        net.conv1 = nn.Conv2d(in_ch, 64, 3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```
그리고 `build_model`에 `if name == "resnet18": return ResNet18Adapted(n_classes)`.

- [ ] **Step 4: 테스트 통과 확인** — `PY -m pytest tests/test_train.py -v`

- [ ] **Step 5: 9 run 실행** (동일 하이퍼파라미터, 튜닝 없음)

Run: `nohup PY -m wm811k_audit.run --model resnet18 --cells A2-B1-C1 A3-B1-C1 A3-B1-C3 --seeds 0 1 2 > <scratchpad>/resnet_run.log 2>&1 &`
Expected: results.csv에 `model=resnet18` 9행 추가(run_id 접미 `-resnet18`).

- [ ] **Step 6: 비교표** — `analyze.py`의 `load_results`는 smallcnn만 쓰므로, README용으로 아래를 `results/tables/second_model.md`에 기록

```bash
PY - <<'EOF'
import pandas as pd
df = pd.read_csv("results/results.csv")
g = df[df.cell_id.isin(["A2-B1-C1","A3-B1-C1","A3-B1-C3"])].groupby(["model","cell_id"])[["own_macro_f1","gold_defect_f1"]].agg(["mean","std"]).round(3)
open("results/tables/second_model.md","w").write(g.to_markdown() + "\n"); print(g)
EOF
```
(`to_markdown`에 `tabulate`가 필요하면 `PY -m pip install tabulate`.)

- [ ] **Step 7: 커밋**

```bash
git add src/wm811k_audit/model.py tests/test_train.py results/results.csv results/tables/second_model.md
git commit -m "feat(stretch): ResNet-18 with identical hyperparameters on the two core pairs"
```

---

### Task 12: README(프로토콜 카드·결과)와 면접 노트

**Files:**
- Create: `README.md`, `docs/interview_notes.md`(gitignore), `environment.yml`

- [ ] **Step 1: environment.yml**

```yaml
name: wm811k
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - pip:
      - --index-url https://download.pytorch.org/whl/cu128
      - torch
      - numpy
      - pandas
      - pyarrow
      - scikit-learn
      - scipy
      - matplotlib
      - pytest
      - tqdm
```

- [ ] **Step 2: README 작성** — 구조(각 절은 실제 숫자로 채운다; 숫자는 `results/tables/*.md`에서 복사)

1. 제목 + 한 줄 정의 (한국어 본문, 영어 요약 1문단)
2. **핵심 결과 3줄**: 대표값(`A3-B1-C1` gold 9-class macro-F1 mean±std) → 같은 모델 랜덤 분할 as-reported → 격차. "숫자 먼저, 조건 나중" 순서.
3. 왜 이 실험인가 (배포 시 새 lot; 논문마다 프로토콜이 다름 — 1차 자료 3개 링크)
4. **프로토콜 카드** 표: 데이터(MIR 배포본, 라벨 172,950), gold(StratifiedGroupKFold 5, seed 20260825, fold 0, sha1 12자리), 리사이즈, 인코딩, 모델, 학습 고정값, 지표 정의, seed
5. 실험 설계: 축 A/B/C 정의와 적용 순서, 18셀, 두 열(as-reported / on-gold)의 의미
6. 결과: `cells.md` 표, `core_pair_split.png`, `core_pair_cap.png`, `main_effects.png`, `confusion_gold_A3-B1-C1.png`, `nn_hamming_A2_vs_A3.png`, seed 노이즈 표, (stretch) second_model.md
7. 데이터 사실(EDA): 원본 분할 lot-disjoint·유사-시간, lot 내 결함 군집, 중복 3.76%(none 위주)
8. 해석과 한계: spec §7·§8 그대로(단정 금지 항목 포함)
9. 재현: 환경 → 다운로드(MIR URL) → `scripts/convert_data.py` → `scripts/eda.py` → `python -m wm811k_audit.run --cells all --seeds 0 1 2 --aux-a4` → `python -m wm811k_audit.analyze`; 테스트 `pytest`
10. 인용: Wu et al. 2015 (readme 요구 사항 두 개 인용 그대로)

- [ ] **Step 3: 면접 노트** — `docs/interview_notes.md` (gitignore): spec §5-8 형식으로 90초 요약(숫자 삽입), 예상 질문 6개와 답, 말하지 말아야 할 것, "모르는 영역" 문장.

- [ ] **Step 4: 최종 검증**

Run: `PY -m pytest -q` 전부 통과; README의 모든 숫자가 `results/tables/`와 일치하는지 grep으로 대조; 그림 파일 5개 존재.

- [ ] **Step 5: 커밋**

```bash
git add README.md environment.yml
git commit -m "docs: README with protocol card, results, reproduction steps"
```

---

## 계획 자체 점검 (작성 후 수행)

- spec §2.2 메타 컬럼 ↔ Task 2 `META_COLUMNS` 일치. spec §3.1 고정값 ↔ Task 7 `TrainConfig`/`SmallCNN` 일치. spec §3.2 gold ↔ Task 4 `carve_gold` + Task 8 `ensure_gold`. spec §3.3 적용 순서 ↔ `build_cell_split`. spec §3.4 세 집합 지표 ↔ Task 5 + Task 8 `run_one`. spec §3.5 진단 ↔ Task 6. spec §4 저장·재개 ↔ Task 8. spec §5 산출물 ↔ Task 10·12. spec §6 테스트 ↔ 각 Task.
- 함수명 일관성: `build_cell_split`, `carve_gold`, `split_diagnostics`, `defect_metrics`, `classification_metrics`, `train_fixed`, `predict`, `run_id_for`, `RESULT_COLUMNS` — 이후 Task에서 동일 이름 사용.
