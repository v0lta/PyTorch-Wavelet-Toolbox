"""Test the stationary wavelet transformation code."""

from typing import Optional

import numpy as np
import pytest
import pywt
import torch

from ptwt._stationary_transform import _iswt, _swt


@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("size", [[1, 32], [3, 64], [5, 64]])
@pytest.mark.parametrize("wavelet", ["db1", "db2"])
def test_swt_1d(level: Optional[int], size: int, wavelet: str) -> None:
    """Test the 1d swt."""
    signal = np.random.normal(size=size).astype(np.float64)
    ptwt_coeff = _swt(torch.from_numpy(signal), wavelet, level=level)
    pywt_coeff = pywt.swt(signal, wavelet, level, trim_approx=True, norm=False)
    test_list = []
    for a, b in zip(ptwt_coeff, pywt_coeff):
        test_list.extend([np.allclose(ael.numpy(), bel) for ael, bel in zip(a, b)])
    assert all(test_list)


@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("size", [[1, 32], [5, 64]])
@pytest.mark.parametrize("wavelet", ["db1", "db2"])
def test_iswt_1d(level: Optional[int], size: int, wavelet: str) -> None:
    """Ensure iswt inverts swt."""
    signal = np.random.normal(size=size).astype(np.float64)
    # signal = np.stack([np.arange(32)] * 3).astype(np.float64)
    ptwt_coeff = _swt(torch.from_numpy(signal), wavelet, level=level)
    rec = _iswt(ptwt_coeff, wavelet)
    assert np.allclose(rec.numpy(), signal)


@pytest.mark.parametrize("size", [[32, 64], [32, 128], [3, 32, 256]])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "sym3"])
@pytest.mark.parametrize("level", [1, 2, 3])  # None needs more work.
@pytest.mark.parametrize("axis", [1, -1])
def test_swt_1d_slow(level: Optional[int], size: int, wavelet: str, axis: int) -> None:
    """Test the 1d swt."""
    signal = np.random.normal(size=size).astype(np.float64)
    ptwt_coeff = _swt(torch.from_numpy(signal), wavelet, level=level, axis=axis)
    pywt_coeff = pywt.swt(
        signal, wavelet, level, trim_approx=True, norm=False, axis=axis
    )
    test_list = []
    for a, b in zip(ptwt_coeff, pywt_coeff):
        test_list.extend([np.allclose(ael.numpy(), bel) for ael, bel in zip(a, b)])
    assert all(test_list)
    rec = _iswt(ptwt_coeff, wavelet, axis=axis)
    assert np.allclose(rec.numpy(), signal)
