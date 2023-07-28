"""Test the util methods."""
from typing import Tuple

import numpy as np
import pytest
import pywt
import torch

from src.ptwt._util import (
    _as_wavelet,
    _fold_channels,
    _pad_symmetric,
    _pad_symmetric_1d,
    _unfold_channels,
)


class _MyHaarFilterBank(object):
    @property
    def filter_bank(self) -> Tuple[list, list, list, list]:
        """Unscaled Haar wavelet filters."""
        return (
            [1 / 2, 1 / 2.0],
            [-1 / 2.0, 1 / 2.0],
            [1 / 2.0, 1 / 2.0],
            [1 / 2.0, -1 / 2.0],
        )


@pytest.mark.parametrize(
    "wavelet",
    [
        "db4",
        pywt.Wavelet("sym4"),
        pywt.Wavelet("custom_wavelet_object", filter_bank=_MyHaarFilterBank()),
    ],
)
def test_as_wavelet(wavelet: str) -> None:
    """Test return types of _as_wavelet."""
    wavelet_result = _as_wavelet(wavelet)
    assert isinstance(wavelet_result, pywt.Wavelet)


@pytest.mark.parametrize("wavelet", ["invalid_wavelet_name"])
def test_failed_as_wavelet(wavelet: str) -> None:
    """Test expected errors for invalid input to _as_wavelet."""
    with pytest.raises(ValueError):
        wavelet = _as_wavelet(wavelet)


@pytest.mark.parametrize("size", [[5], [12], [19]])
@pytest.mark.parametrize(
    "pad_list", [(2, 2), (0, 0), (1, 0), (0, 1), (2, 1), (1, 2), (10, 10)]
)
def test_pad_symmetric_1d(size, pad_list):
    """Test symetric padding in a single dimension."""
    test_signal = np.random.randint(0, 9, size=size).astype(np.float32)
    my_pad = _pad_symmetric_1d(torch.from_numpy(test_signal), pad_list)
    np_pad = np.pad(test_signal, pad_list, mode="symmetric")
    assert np.allclose(np_pad, my_pad.numpy())


@pytest.mark.parametrize("size", [[6, 5], [5, 6], [5, 5], [9, 9], [3, 3]])
@pytest.mark.parametrize("pad_list", [[(1, 4), (4, 1)], [(2, 2), (3, 3)]])
def test_pad_symmetric(size, pad_list):
    """Test high-dimensional symetric padding."""
    array = np.random.randint(0, 9, size=size)
    my_pad = _pad_symmetric(torch.from_numpy(array), pad_list)
    np_pad = np.pad(array, pad_list, mode="symmetric")
    assert np.allclose(my_pad.numpy(), np_pad)


@pytest.mark.parametrize("size", [[20, 21, 22, 23], [1, 2, 3, 4], [4, 3, 2, 1]])
def test_fold(size):
    """Ensure channel folding works as expected."""
    array = torch.randn(*size).type(torch.float64)
    folded = _fold_channels(array)
    assert tuple(folded.shape) == (size[0] * size[1], size[2], size[3])
    rec = _unfold_channels(folded, size)
    np.allclose(array.numpy(), rec.numpy())
