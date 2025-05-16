"""Test dtype support for the fwt code."""

# Written by moritz ( @ wolter.tech ) in 2025
import numpy as np
import pytest
import pywt
import torch
from scipy import datasets

from src.ptwt.conv_transform import _flatten_2d_coeff_lst
from src.ptwt.conv_transform_2 import wavedec2, waverec2


@pytest.mark.slow
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16]
)
def test_2d_wavedec_rec(dtype):
    """Ensure pywt.wavedec2 and ptwt.wavedec2 produce the same coefficients.

    wavedec2 and waverec2 must invert each other.
    """
    mode = "reflect"
    level = 2
    size = (32, 32)
    face = np.transpose(
        datasets.face()[256 : (512 + size[0]), 256 : (512 + size[1])], [2, 0, 1]
    ).astype(np.float32)
    wavelet = pywt.Wavelet("db2")
    to_transform = torch.from_numpy(face).to(torch.float32)
    coeff2d = wavedec2(to_transform, wavelet, mode=mode, level=level)
    pywt_coeff2d = pywt.wavedec2(face, wavelet, mode=mode, level=level)
    for pos, coeffs in enumerate(pywt_coeff2d):
        if type(coeffs) is tuple:
            for tuple_pos, tuple_el in enumerate(coeffs):
                assert (
                    tuple_el.shape == coeff2d[pos][tuple_pos].shape
                ), "pywt and ptwt should produce the same shapes."
        else:
            assert (
                coeffs.shape == coeff2d[pos].shape
            ), "pywt and ptwt should produce the same shapes."
    flat_coeff_list_pywt = np.concatenate(_flatten_2d_coeff_lst(pywt_coeff2d), -1)
    flat_coeff_list_ptwt = torch.cat(_flatten_2d_coeff_lst(coeff2d), -1)
    assert np.allclose(flat_coeff_list_pywt, flat_coeff_list_ptwt.numpy(), atol=1e-3)
    rec = waverec2(coeff2d, wavelet)
    rec = rec.numpy().squeeze().astype(np.float32)
    assert np.allclose(face, rec[:, : face.shape[1], : face.shape[2]], atol=1e-3)
