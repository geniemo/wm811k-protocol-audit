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
