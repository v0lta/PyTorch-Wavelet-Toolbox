"""Utility methods to compute wavelet decompositions from a dataset."""

import typing
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import pywt
import torch

from .constants import OrthogonalizeMethod


class Wavelet(Protocol):
    """Wavelet object interface, based on the pywt wavelet object."""

    name: str
    dec_lo: Sequence[float]
    dec_hi: Sequence[float]
    rec_lo: Sequence[float]
    rec_hi: Sequence[float]
    dec_len: int
    rec_len: int
    filter_bank: Tuple[
        Sequence[float], Sequence[float], Sequence[float], Sequence[float]
    ]

    def __len__(self) -> int:
        """Return the number of filter coefficients."""
        return len(self.dec_lo)


def _as_wavelet(wavelet: Union[Wavelet, str]) -> Wavelet:
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        Wavelet: the input wavelet object or the pywt wavelet object described by the
            input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


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


def _get_len(wavelet: Union[Tuple[torch.Tensor, ...], str, Wavelet]) -> int:
    """Get number of filter coefficients for various wavelet data types."""
    if isinstance(wavelet, tuple):
        return wavelet[0].shape[0]
    else:
        return len(_as_wavelet(wavelet))


def _pad_symmetric_1d(signal: torch.Tensor, pad_list: Tuple[int, int]) -> torch.Tensor:
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
        return torch.cat(cat_list, axis=0)  # type: ignore


def _pad_symmetric(
    signal: torch.Tensor, pad_lists: List[Tuple[int, int]]
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


def _fold_axes(data: torch.Tensor, keep_no: int) -> Tuple[torch.Tensor, List[int]]:
    """Fold unchanged leading dimensions into a single batch dimension.

    Args:
        data ( torch.Tensor): The input data array.
        keep_no (int): The number of dimensions to keep.

    Returns:
        Tuple[ torch.Tensor, List[int]]:
            The folded result array, and the shape of the original input.
    """
    dshape = list(data.shape)
    return (
        torch.reshape(data, [int(np.prod(dshape[:-keep_no]))] + dshape[-keep_no:]),
        dshape,
    )


def _unfold_axes(data: torch.Tensor, ds: List[int], keep_no: int) -> torch.Tensor:
    """Unfold i.e. [batch*channel,height,widht] to [batch,channel,height,width]."""
    return torch.reshape(data, ds[:-keep_no] + list(data.shape[-keep_no:]))


def _check_if_tensor(array: Any) -> torch.Tensor:
    if not isinstance(array, torch.Tensor):
        raise ValueError(
            "First element of coeffs must be the approximation coefficient tensor."
        )
    return array


def _check_axes_argument(axes: List[int]) -> None:
    if len(set(axes)) != len(axes):
        raise ValueError("Cant transform the same axis twice.")


def _get_transpose_order(
    axes: List[int], data_shape: List[int]
) -> Tuple[List[int], List[int]]:
    axes = list(map(lambda a: a + len(data_shape) if a < 0 else a, axes))
    all_axes = list(range(len(data_shape)))
    remove_transformed = list(filter(lambda a: a not in axes, all_axes))
    return remove_transformed, axes


def _swap_axes(data: torch.Tensor, axes: List[int]) -> torch.Tensor:
    _check_axes_argument(axes)
    front, back = _get_transpose_order(axes, list(data.shape))
    return torch.permute(data, front + back)


def _undo_swap_axes(data: torch.Tensor, axes: List[int]) -> torch.Tensor:
    _check_axes_argument(axes)
    front, back = _get_transpose_order(axes, list(data.shape))
    restore_sorted = torch.argsort(torch.tensor(front + back)).tolist()
    return torch.permute(data, restore_sorted)


def _map_result(
    data: List[Union[torch.Tensor, Any]],  # following jax tree_map typing can be Any
    function: Callable[[Any], torch.Tensor],
) -> List[Union[torch.Tensor, Any]]:
    # Apply the given function to the input list of tensor and tuples.
    result_lst: List[Union[torch.Tensor, Any]] = []
    for element in data:
        if isinstance(element, torch.Tensor):
            result_lst.append(function(element))
        elif isinstance(element, tuple):
            result_lst.append(
                (function(element[0]), function(element[1]), function(element[2]))
            )
        elif isinstance(element, dict):
            new_dict = {}
            for key, value in element.items():
                new_dict[key] = function(value)
            result_lst.append(new_dict)
    return result_lst
