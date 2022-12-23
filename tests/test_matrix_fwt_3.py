"""Test the 3d matrix-fwt code."""

from typing import List

import numpy as np
import pytest
import pywt
import torch

import src.ptwt as ptwt
from src.ptwt.matmul_transform import construct_boundary_a


def batch_dim_mm(matrix: torch.Tensor, batch_tensor: torch.Tensor, axis: int):
    dim_length = batch_tensor.shape[axis]
    res = torch.sparse.mm(matrix, batch_tensor.transpose(axis, -1).reshape(-1, dim_length).T).T
    return res.reshape(batch_tensor.shape).transpose(-1, axis)


@pytest.mark.parametrize("axis", [1, 2, 3])
def test_single_dim_mm(axis: int):
    length = 10
    test_tensor = torch.rand(4, length, length, length).type(torch.float64)

    pywt_dec_lo, pywt_dec_hi = pywt.wavedec(test_tensor.numpy(), pywt.Wavelet("Haar"), axis=axis, level=1)
    haar_mat = construct_boundary_a(pywt.Wavelet("haar"), length=length)
    dec_lo, dec_hi = batch_dim_mm(haar_mat, test_tensor, axis=axis).split(length//2, axis)
    assert np.allclose(pywt_dec_lo, dec_lo.numpy())
    assert np.allclose(pywt_dec_hi, dec_hi.numpy())