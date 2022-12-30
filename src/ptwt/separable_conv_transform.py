"""Implement separable convolution based transforms.

Under the hood code in this module transforms all dimensions
individually using torch.nn.functional.conv1d and it's
transpose.
"""
# from typing import Union

from src.ptwt.conv_transform import wavedec, waverec


def _separable_conv_dwtn_(input, wavelet, mode, key, dict) -> None:
    axis_total = len(input.shape) - 1
    if len(key) == axis_total:
        dict[key] = input
    if len(key) < axis_total:
        current_axis = len(key) + 1
        transposed = input.transpose(-current_axis, -1)
        flat = transposed.reshape(-1, transposed.shape[-1])
        res_a, res_d = wavedec(flat, wavelet, 1, mode)
        res_a = res_a.reshape(list(transposed.shape[:-1]) + [res_a.shape[-1]])
        res_d = res_d.reshape(list(transposed.shape[:-1]) + [res_d.shape[-1]])
        res_a = res_a.transpose(-1, -current_axis)
        res_d = res_d.transpose(-1, -current_axis)
        _separable_conv_dwtn_(res_a, wavelet, mode, "a" + key, dict)
        _separable_conv_dwtn_(res_d, wavelet, mode, "d" + key, dict)


def _separable_conv_idwtn(in_dict, wavelet):
    done_dict = {}
    a_initial_keys = list(filter(lambda x: x[0] == "a", in_dict.keys()))
    for a_key in a_initial_keys:
        current_axis = len(a_key)
        d_key = "d" + a_key[1:]
        coeff_d = in_dict[d_key]
        d_shape = coeff_d.shape
        # undo any analysis padding.
        coeff_a = in_dict[a_key][tuple(slice(0, ds) for ds in d_shape)]
        trans_a, trans_d = (
            coeff.transpose(-1, -current_axis) for coeff in (coeff_a, coeff_d)
        )
        flat_a, flat_d = (
            coeff.reshape(-1, coeff.shape[-1]) for coeff in (trans_a, trans_d)
        )
        rec_ad = waverec((flat_a, flat_d), wavelet)
        rec_ad = rec_ad.reshape(list(trans_a.shape[:-1]) + [rec_ad.shape[-1]])
        rec_ad = rec_ad.transpose(-current_axis, -1)
        if a_key[1:]:
            done_dict[a_key[1:]] = rec_ad
        else:
            return rec_ad
    return _separable_conv_idwtn(done_dict, wavelet)


def _separable_conv_waverecn(coeff_list, wavelet):
    approx = coeff_list[0]
    for level_dict in coeff_list[1:]:
        keys = list(level_dict.keys())
        level_dict["a" * max(map(len, keys))] = approx
        approx = _separable_conv_idwtn(level_dict, wavelet)
    return approx


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
