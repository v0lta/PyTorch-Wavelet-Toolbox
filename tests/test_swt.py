"""Test the stationary wavelet transformation code."""

import numpy as np
import pytest
import pywt
import torch

from src.ptwt.stationary_transform import swt


@pytest.mark.parametrize("size", [32, 64])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "sym4"])
def test_swt_1d(level, size, wavelet):
    """Test the 1d swt."""
    signal = np.expand_dims(np.arange(size).astype(np.float64), 0)
    ptwt_coeff = swt(torch.from_numpy(signal), wavelet, level=level)
    pywt_coeff = pywt.swt(signal, wavelet, level, trim_approx=True, norm=False)
    test_list = []
    for a, b in zip(ptwt_coeff, pywt_coeff):
        test_list.extend([np.allclose(ael.numpy(), bel) for ael, bel in zip(a, b)])
    assert all(test_list)
