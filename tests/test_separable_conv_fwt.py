"""Separable transform test code."""

import numpy as np
import pytest
import pywt
import torch

from src.ptwt.matmul_transform_2 import MatrixWavedec2
from src.ptwt.matmul_transform_3 import MatrixWavedec3
from src.ptwt.separable_conv_transform import (
    _separable_conv_wavedecn,
    _separable_conv_waverecn,
    fswavedec2,
    fswavedec3,
    fswaverec2,
    fswaverec3,
)


@pytest.mark.parametrize("level", (1, 2))
@pytest.mark.parametrize(
    "shape", ((12, 12), (24, 12, 12), (12, 24, 12), (12, 12, 12, 12))
)
def test_separable_conv(shape, level) -> None:
    """Test the separable transforms."""
    data = np.random.randint(0, 9, shape)

    result = pywt.fswavedecn(data, "haar", levels=level)
    detail_keys = result.detail_keys()
    approx = result.approx
    details = [result[key] for key in detail_keys]
    flat_pywt_res = [approx]
    flat_pywt_res.extend(details)

    pt_data = torch.from_numpy(data).unsqueeze(0).type(torch.float64)
    ptwt_res = _separable_conv_wavedecn(pt_data, "haar", mode="reflect", level=level)
    ptwt_res_lists = [ptwt_res[0]]
    # product a proper order.
    ptwt_res_lists.extend(
        [
            ptwt_dict[key]
            for ptwt_dict in ptwt_res[1:]
            for key in sorted(ptwt_dict.keys())
            if len(key) == len(shape)
        ]
    )
    flat_ptwt_res = [
        tensor.numpy() for tensor_list in ptwt_res_lists for tensor in tensor_list
    ]
    # pywt and ptwt should produce identical coefficients.
    pywt_fine_scale = list(filter(lambda x: x.shape == approx.shape, flat_pywt_res))
    assert all(
        [
            np.allclose(ptwt_array, pywt_array)
            for ptwt_array, pywt_array in zip(flat_ptwt_res, pywt_fine_scale)
        ]
    )

    rec = _separable_conv_waverecn(ptwt_res, "haar")
    assert np.allclose(rec.numpy(), data)


@pytest.mark.parametrize("shape", [(5, 64, 64), (5, 65, 65), (5, 29, 29)])
@pytest.mark.parametrize("wavelet", ["haar", "db3", "sym5"])
def test_example_fs2d(shape, wavelet):
    """Test 2d fully separable padding."""
    data = torch.randn(*shape).type(torch.float64)
    coeff = fswavedec2(data, wavelet, level=2)
    rec = fswaverec2(coeff, wavelet)
    assert np.allclose(data.numpy(), rec[[slice(0, s) for s in shape]].numpy())


@pytest.mark.parametrize("shape", [(5, 64, 64, 64), (5, 65, 65, 65), (5, 29, 29, 29)])
@pytest.mark.parametrize("wavelet", ["haar", "db3", "sym5"])
def test_example_fs3d(shape, wavelet):
    """Test 3d fully separable padding."""
    data = torch.randn(*shape).type(torch.float64)
    coeff = fswavedec3(data, wavelet, level=2)
    rec = fswaverec3(coeff, wavelet)
    assert np.allclose(data.numpy(), rec[[slice(0, s) for s in shape]].numpy())


# test separable conv and mamul consistency for the Haar case.
@pytest.mark.slow
@pytest.mark.parametrize("level", [1, 2, 3, None])
@pytest.mark.parametrize("shape", [[5, 128, 128], [3, 2, 64, 64], [1, 1, 64, 64]])
def test_conv_mm_2d(level, shape):
    """Compare mm and conv fully separable results."""
    data = torch.randn(*shape).type(torch.float64)
    fs_conv_coeff = fswavedec2(data, "haar", level=level)
    fs_mm_coeff = MatrixWavedec2("haar", level, separable=True)(data)
    # compare coefficients
    assert len(fs_conv_coeff) == len(fs_mm_coeff)
    for c_conv, c_mm in zip(fs_conv_coeff, fs_mm_coeff):
        if isinstance(c_conv, torch.Tensor):
            assert np.allclose(c_conv.numpy(), c_mm.numpy())
        else:
            # (ll, (lh, hl, hh), ...)
            c_conv_list = [c_conv[key] for key in ("ad", "da", "dd")]
            assert all(
                np.allclose(c_el_conv.numpy(), c_el_mm.numpy())
                for c_el_conv, c_el_mm in zip(c_conv_list, c_mm)
            )
    rec = fswaverec2(fs_conv_coeff, "haar")
    assert np.allclose(data.numpy(), rec.numpy())


@pytest.mark.slow
@pytest.mark.parametrize("level", [1, 2, 3, None])
def test_conv_mm_3d(level):
    """Compare mm and conv 3d fully separable results."""
    shape = (5, 128, 128, 128)
    data = torch.randn(*shape).type(torch.float64)
    fs_conv_coeff = fswavedec3(data, "haar", level=level)
    fs_mm_coeff = MatrixWavedec3("haar", level)(data)
    # compare coefficients
    assert len(fs_conv_coeff) == len(fs_mm_coeff)
    for c_conv, c_mm in zip(fs_conv_coeff, fs_mm_coeff):
        if isinstance(c_conv, torch.Tensor):
            assert np.allclose(c_conv.numpy(), c_mm.numpy())
        else:
            keys = c_conv.keys()
            assert all(np.allclose(c_conv[key], c_mm[key]) for key in keys)
    rec = fswaverec3(fs_conv_coeff, "haar")
    assert np.allclose(data.numpy(), rec.numpy())
