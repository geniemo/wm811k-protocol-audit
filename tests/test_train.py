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


def test_resnet18_adapted_shape():
    m = build_model("resnet18", 9)
    assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 9)
    assert count_params(m) > 10_000_000
