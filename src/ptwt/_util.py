"""Utility methods to compute wavelet decompositions."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union, cast, overload

import numpy as np
import pywt
import torch

from .constants import (
    BoundaryMode,
    OrthogonalizeMethod,
    Wavelet,
    WaveletCoeff2d,
    WaveletCoeffNd,
    WaveletDetailDict,
    WaveletDetailTuple2d,
)


def _translate_boundary_strings(pywt_mode: BoundaryMode) -> str:
    """Translate pywt mode strings to PyTorch mode strings.

    We support constant, zero, reflect, and periodic.
    Unfortunately, "constant" has different meanings in the
    Pytorch and PyWavelet communities.

    Raises:
        ValueError: If the padding mode is not supported.
    """
    if pywt_mode == "constant":
        return "replicate"
    elif pywt_mode == "zero":
        return "constant"
    elif pywt_mode == "reflect":
        return pywt_mode
    elif pywt_mode == "periodic":
        return "circular"
    elif pywt_mode == "symmetric":
        # pytorch does not support symmetric mode,
        # we have our own implementation.
        return pywt_mode
    raise ValueError(f"Padding mode not supported: {pywt_mode}")


def _as_wavelet(wavelet: Union[Wavelet, str]) -> Wavelet:
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        The input wavelet object or the pywt wavelet object described by the input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


def _get_len(wavelet: Union[tuple[torch.Tensor, ...], str, Wavelet]) -> int:
    """Get number of filter coefficients for various wavelet data types."""
    if isinstance(wavelet, tuple):
        return wavelet[0].shape[0]
    else:
        return len(_as_wavelet(wavelet))


def _get_filter_tensors(
    wavelet: Union[Wavelet, str],
    flip: bool,
    device: Union[torch.device, str],
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input wavelet to filter tensors.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        flip (bool): Flip filters left-right, if true.
        device (torch.device or str): PyTorch target device.
        dtype (torch.dtype): The data type sets the precision of the
            computation. Default: torch.float32.

    Returns:
        A tuple (dec_lo, dec_hi, rec_lo, rec_hi) containing
        the four filter tensors
    """
    wavelet = _as_wavelet(wavelet)
    device = torch.device(device)

    if isinstance(wavelet, tuple):
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet
    else:
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo, flip, device, dtype)
    dec_hi_tensor = _create_tensor(dec_hi, flip, device, dtype)
    rec_lo_tensor = _create_tensor(rec_lo, flip, device, dtype)
    rec_hi_tensor = _create_tensor(rec_hi, flip, device, dtype)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _create_tensor(
    filter_seq: Sequence[float], flip: bool, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if flip:
        if isinstance(filter_seq, torch.Tensor):
            return filter_seq.flip(-1).unsqueeze(0).to(device=device, dtype=dtype)
        else:
            return torch.tensor(filter_seq[::-1], device=device, dtype=dtype).unsqueeze(
                0
            )
    else:
        if isinstance(filter_seq, torch.Tensor):
            return filter_seq.unsqueeze(0).to(device=device, dtype=dtype)
        else:
            return torch.tensor(filter_seq, device=device, dtype=dtype).unsqueeze(0)


def _is_boundary_mode_supported(boundary_mode: Optional[OrthogonalizeMethod]) -> bool:
    return boundary_mode in typing.get_args(OrthogonalizeMethod)


def _is_dtype_supported(dtype: torch.dtype) -> bool:
    return dtype in [torch.float32, torch.float64]


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's outer for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul


def _pad_symmetric_1d(signal: torch.Tensor, pad_list: tuple[int, int]) -> torch.Tensor:
    padl, padr = pad_list
    dimlen = signal.shape[0]
    if padl > dimlen or padr > dimlen:
        if padl > dimlen:
            signal = _pad_symmetric_1d(signal, (dimlen, 0))
            padl = padl - dimlen
        if padr > dimlen:
            signal = _pad_symmetric_1d(signal, (0, dimlen))
            padr = padr - dimlen
        return _pad_symmetric_1d(signal, (padl, padr))
    else:
        cat_list = [signal]
        if padl > 0:
            cat_list.insert(0, signal[:padl].flip(0))
        if padr > 0:
            cat_list.append(signal[-padr::].flip(0))
        return torch.cat(cat_list, dim=0)


def _pad_symmetric(
    signal: torch.Tensor, pad_lists: Sequence[tuple[int, int]]
) -> torch.Tensor:
    if len(signal.shape) < len(pad_lists):
        raise ValueError("not enough dimensions to pad.")

    dims = len(signal.shape) - 1
    for pos, pad_list in enumerate(pad_lists[::-1]):
        current_axis = dims - pos
        signal = signal.transpose(0, current_axis)
        signal = _pad_symmetric_1d(signal, pad_list)
        signal = signal.transpose(current_axis, 0)
    return signal


def _get_pad(data_len: int, filt_len: int) -> tuple[int, int]:
    """Compute the required padding.

    Args:
        data_len (int): The length of the input vector.
        filt_len (int): The size of the used filter.

    Returns:
        A tuple (padr, padl). The first entry specifies how many numbers
        to attach on the right. The second entry covers the left side.
    """
    # pad to ensure we see all filter positions and
    # for pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    # we pad half of the total requried padding on each side.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    padr += data_len % 2

    return padr, padl


def _adjust_padding_at_reconstruction(
    res_ll_size: int, coeff_size: int, pad_end: int, pad_start: int
) -> tuple[int, int]:
    pred_size = res_ll_size - (pad_start + pad_end)
    next_size = coeff_size
    if next_size == pred_size:
        pass
    elif next_size == pred_size - 1:
        pad_end += 1
    else:
        raise AssertionError(
            "padding error, please check if dec and rec wavelets are identical."
        )
    return pad_end, pad_start


def _flatten_2d_coeff_lst(
    coeff_lst_2d: WaveletCoeff2d,
    flatten_tensors: bool = True,
) -> list[torch.Tensor]:
    """Flattens a sequence of tensor tuples into a single list.

    Args:
        coeff_lst_2d (WaveletCoeff2d): A pywt-style
            coefficient tuple of torch tensors.
        flatten_tensors (bool): If true, 2d tensors are flattened. Defaults to True.

    Returns:
        A single 1-d list with all original elements.
    """

    def _process_tensor(coeff: torch.Tensor) -> torch.Tensor:
        return coeff.flatten() if flatten_tensors else coeff

    flat_coeff_lst = [_process_tensor(coeff_lst_2d[0])]
    for coeff_tuple in coeff_lst_2d[1:]:
        flat_coeff_lst.extend(map(_process_tensor, coeff_tuple))
    return flat_coeff_lst


def _fold_axes(data: torch.Tensor, keep_no: int) -> tuple[torch.Tensor, list[int]]:
    """Fold unchanged leading dimensions into a single batch dimension.

    Args:
        data (torch.Tensor): The input data array.
        keep_no (int): The number of dimensions to keep.

    Returns:
        A tuple (result_tensor, input_shape) where result_tensor is the
        folded result array, and input_shape the shape of the original input.
    """
    dshape = list(data.shape)
    return (
        torch.reshape(data, [int(np.prod(dshape[:-keep_no]))] + dshape[-keep_no:]),
        dshape,
    )


def _unfold_axes(data: torch.Tensor, ds: list[int], keep_no: int) -> torch.Tensor:
    """Unfold i.e. [batch*channel,height,widht] to [batch,channel,height,width]."""
    return torch.reshape(data, ds[:-keep_no] + list(data.shape[-keep_no:]))


def _check_if_tensor(array: Any) -> torch.Tensor:
    if not isinstance(array, torch.Tensor):
        raise ValueError(
            "First element of coeffs must be the approximation coefficient tensor."
        )
    return array


def _check_axes_argument(axes: Sequence[int]) -> None:
    if len(set(axes)) != len(axes):
        raise ValueError("Cant transform the same axis twice.")


def _get_transpose_order(
    axes: Sequence[int], data_shape: Sequence[int]
) -> tuple[list[int], list[int]]:
    axes = list(map(lambda a: a + len(data_shape) if a < 0 else a, axes))
    all_axes = list(range(len(data_shape)))
    remove_transformed = list(filter(lambda a: a not in axes, all_axes))
    return remove_transformed, axes


def _swap_axes(data: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    _check_axes_argument(axes)
    front, back = _get_transpose_order(axes, list(data.shape))
    return torch.permute(data, front + back)


def _undo_swap_axes(data: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    _check_axes_argument(axes)
    front, back = _get_transpose_order(axes, list(data.shape))
    restore_sorted = torch.argsort(torch.tensor(front + back)).tolist()
    return torch.permute(data, restore_sorted)


@overload
def _map_result(
    data: WaveletCoeff2d,
    function: Callable[[torch.Tensor], torch.Tensor],
) -> WaveletCoeff2d: ...


@overload
def _map_result(
    data: WaveletCoeffNd,
    function: Callable[[torch.Tensor], torch.Tensor],
) -> WaveletCoeffNd: ...


def _map_result(
    data: Union[WaveletCoeff2d, WaveletCoeffNd],
    function: Callable[[torch.Tensor], torch.Tensor],
) -> Union[WaveletCoeff2d, WaveletCoeffNd]:
    approx = function(data[0])
    result_lst: list[
        Union[
            WaveletDetailDict,
            WaveletDetailTuple2d,
        ]
    ] = []
    for element in data[1:]:
        if isinstance(element, tuple):
            result_lst.append(
                WaveletDetailTuple2d(
                    function(element[0]),
                    function(element[1]),
                    function(element[2]),
                )
            )
        elif isinstance(element, dict):
            new_dict = {key: function(value) for key, value in element.items()}
            result_lst.append(new_dict)
        else:
            raise ValueError(f"Unexpected input type {type(element)}")

    # cast since we assume that the full list is of the same type
    cast_result_lst = cast(
        Union[list[WaveletDetailDict], list[WaveletDetailTuple2d]], result_lst
    )
    return approx, *cast_result_lst
