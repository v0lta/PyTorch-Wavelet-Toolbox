"""Code for three dimensional padded transforms.

The functions here are based on torch.nn.functional.conv3d and it's transpose.
"""

from __future__ import annotations

from functools import partial
from itertools import product
from typing import Optional, Union, cast

import pywt
import torch

from ._util import (
    Wavelet,
    _as_wavelet,
    _check_same_device_dtype,
    _construct_nd_filt,
    _fwt_padn,
    _get_filter_tensors,
    _get_pad,
    _get_pad_removal_slice,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
)
from .constants import BoundaryMode, WaveletCoeffNd, WaveletDetailDict
from .conv_transform import _adjust_padding_at_reconstruction


def wavedec3(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "zero",
    level: Optional[int] = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
) -> WaveletCoeffNd:
    """Compute a three-dimensional wavelet transform.

    Args:
        data (torch.Tensor): The input data. For example of shape
            ``[batch_size, length, height, width]``
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        mode :
            The desired padding mode for extending the signal along the edges.
            Defaults to "zero". See :data:`ptwt.constants.BoundaryMode`.
        level (Optional[int]): The maximum decomposition level.
            This argument defaults to None.
        axes (tuple[int, int, int]): Compute the transform over these axes
            instead of the last three. Defaults to (-3, -2, -1).

    Returns:
        A tuple containing the wavelet coefficients,
        see :data:`ptwt.constants.WaveletCoeffNd`.

    Example:
        >>> import ptwt, torch
        >>> data = torch.randn(5, 16, 16, 16)
        >>> transformed = ptwt.wavedec3(data, "haar", level=2, mode="reflect")
    """
    data, ds = _preprocess_tensor(data, ndim=3, axes=axes)

    wavelet = _as_wavelet(wavelet)
    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    dec_filt = _construct_nd_filt(lo=dec_lo, hi=dec_hi, ndim=3)

    if level is None:
        level = pywt.dwtn_max_level(
            [data.shape[-1], data.shape[-2], data.shape[-3]], wavelet
        )

    coeff_keys = ["".join(key) for key in product(["a", "d"], repeat=3)]

    result_lst: list[WaveletDetailDict] = []
    res_lll = data
    for _ in range(level):
        res_lll = _fwt_padn(res_lll, wavelet, ndim=3, mode=mode)
        res = torch.nn.functional.conv3d(res_lll, dec_filt, stride=2)
        result_dict = {
            key: res.squeeze(1) for key, res in zip(coeff_keys, torch.split(res, 1, 1))
        }
        res_lll = result_dict.pop("aaa").unsqueeze(1)
        result_lst.append(result_dict)
    result_lst.reverse()
    coeffs: WaveletCoeffNd = res_lll.squeeze(1), *result_lst

    return _postprocess_coeffs(coeffs, ndim=3, ds=ds, axes=axes)


def waverec3(
    coeffs: WaveletCoeffNd,
    wavelet: Union[Wavelet, str],
    axes: tuple[int, int, int] = (-3, -2, -1),
) -> torch.Tensor:
    """Reconstruct a signal from wavelet coefficients.

    Args:
        coeffs (WaveletCoeffNd): The wavelet coefficient tuple
            produced by wavedec3, see :data:`ptwt.constants.WaveletCoeffNd`.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        axes (tuple[int, int, int]): Transform these axes instead of the
            last three. Defaults to (-3, -2, -1).

    Returns:
        The reconstructed four-dimensional signal tensor of shape
        ``[batch, depth, height, width]``.

    Raises:
        ValueError: If coeffs is not in a shape as returned from wavedec3 or
            if the dtype is not supported or if the provided axes input has length
            other than three or if the same axes it repeated three.

    Example:
        >>> import ptwt, torch
        >>> data = torch.randn(5, 16, 16, 16)
        >>> transformed = ptwt.wavedec3(data, "haar", level=2, mode="reflect")
        >>> reconstruction = ptwt.waverec3(transformed, "haar")
    """
    coeffs, ds = _preprocess_coeffs(coeffs, ndim=3, axes=axes)
    torch_device, torch_dtype = _check_same_device_dtype(coeffs)

    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    rec_filt = _construct_nd_filt(lo=rec_lo, hi=rec_hi, ndim=3)

    detail_keys = [
        "".join(key) for key in product(["a", "d"], repeat=3) if "".join(key) != "aaa"
    ]

    res_lll = coeffs[0]
    for c_pos, coeff_dict in enumerate(coeffs[1:], start=1):
        if not isinstance(coeff_dict, dict) or len(coeff_dict) != 7:
            raise ValueError(
                f"Unexpected detail coefficient type: {type(coeff_dict)}. Detail "
                "coefficients must be a dict containing 7 tensors as returned by "
                "wavedec3."
            )
        if any(coeff.shape != res_lll.shape for coeff in coeff_dict.values()):
            raise ValueError("All coefficients on each level must have the same shape")

        res_lll = torch.stack([res_lll] + [coeff_dict[key] for key in detail_keys], 1)
        res_lll = torch.nn.functional.conv_transpose3d(res_lll, rec_filt, stride=2)
        res_lll = res_lll.squeeze(1)

        # remove the padding
        if c_pos < len(coeffs) - 1:
            next_details = cast(WaveletDetailDict, coeffs[c_pos + 1])
            next_detail_shape = next_details["aad"].shape
        else:
            next_detail_shape = None

        _slice = partial(
            _get_pad_removal_slice,
            filt_len=filt_len,
            data_shape=res_lll.shape,
            next_detail_shape=next_detail_shape,
        )

        res_lll = res_lll[..., _slice(-3), _slice(-2), _slice(-1)]

    return _postprocess_tensor(res_lll, ndim=3, ds=ds, axes=axes)
