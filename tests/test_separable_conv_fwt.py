"""Separable transform test code."""

import numpy as np
import pytest
import pywt
import torch

from src.ptwt.conv_transform import wavedec


def _separable_conv_dwtn_(input, wavelet, mode, key, dict) -> None:
    axis_total = len(input.shape) - 1
    if len(key) < axis_total:
        current_axis = len(key) + 1
        transposed = input.transpose(current_axis, -1)
        flat = transposed.reshape(-1, transposed.shape[-1])
        res_a, res_d = wavedec(flat, wavelet, 1, mode)
        res_a = res_a.reshape(list(transposed.shape[:-1]) + [res_a.shape[-1]])
        res_d = res_d.reshape(list(transposed.shape[:-1]) + [res_d.shape[-1]])
        res_a = res_a.transpose(-1, current_axis)
        res_d = res_d.transpose(-1, current_axis)
        _separable_conv_dwtn_(res_a, wavelet, mode, "a" + key, dict)
        _separable_conv_dwtn_(res_d, wavelet, mode, "d" + key, dict)
    else:
        dict[key] = input


def _separable_conv_wavedecn(input, wavelet, mode, levels):
    result = []
    approx = input
    for _ in range(levels):
        level_dict = {}
        _separable_conv_dwtn_(approx, wavelet, mode, "", level_dict)
        approx_key = "a" * (len(input.shape) - 1)
        approx = level_dict.pop(approx_key)
        result.append(level_dict)
    result.append(approx)
    return result[::-1]


@pytest.mark.parametrize("shape", ((12, 12), (12, 12, 12), (12, 24, 12)))
def test_separable_conv(shape) -> None:
    """Test the separable transforms."""
    data = np.random.randint(0, 9, shape)

    result = pywt.fswavedecn(data, "haar", levels=2)
    detail_keys = result.detail_keys()
    approx = result.approx
    details = [result[key] for key in detail_keys]
    flat_pywt_res = [approx]
    flat_pywt_res.extend(details)

    pt_data = torch.from_numpy(data).unsqueeze(0).type(torch.float64)
    ptwt_res = _separable_conv_wavedecn(pt_data, "haar", mode="reflect", levels=2)
    ptwt_res_lists = [ptwt_res[0]]
    ptwt_res_lists.extend(
        [tensor for ptwt_dict in ptwt_res[1:] for _, tensor in ptwt_dict.items()]
    )
    flat_ptwt_res = [
        tensor.numpy() for tensor_list in ptwt_res_lists for tensor in tensor_list
    ]

    pywt_fine_scale = list(filter(lambda x: x.shape == approx.shape, flat_pywt_res))
    assert all(
        [
            np.allclose(ptwt_tensor, pywt_array)
            for ptwt_tensor, pywt_array in zip(flat_ptwt_res, pywt_fine_scale)
        ]
    )

    pass
