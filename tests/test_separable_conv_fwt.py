"""Separable transform test code."""

import numpy as np
import pytest
import pywt
import torch

from src.ptwt.separable_conv_transform import (
    _separable_conv_wavedecn,
    _separable_conv_waverecn,
    fswavedec,
    fswavedec2,
    fswavedec3,
    fswaverec,
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


@pytest.mark.parametrize("shape", [(5, 64), (5, 65), (5, 29)])
@pytest.mark.parametrize("wavelet", ["haar", "db3", "sym5"])
def test_example_fs1d(shape, wavelet):
    """Test 1d fully separable padding."""
    data = torch.randn(*shape).type(torch.float64)
    coeff = fswavedec(data, wavelet, level=2)
    rec = fswaverec(coeff, wavelet)
    assert np.allclose(data.numpy(), rec[: shape[0], : shape[1]].numpy())


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
