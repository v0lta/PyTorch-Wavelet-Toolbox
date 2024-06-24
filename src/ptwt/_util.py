"""Utility methods to compute wavelet decompositions from a dataset."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Protocol, Union, cast, overload

import numpy as np
import pywt
import torch

from .constants import (
    OrthogonalizeMethod,
    WaveletCoeff2d,
    WaveletCoeffNd,
    WaveletDetailDict,
    WaveletDetailTuple2d,
)


class Wavelet(Protocol):
    """Wavelet object interface, based on the pywt wavelet object."""

    name: str
    dec_lo: Sequence[float]
    dec_hi: Sequence[float]
    rec_lo: Sequence[float]
    rec_hi: Sequence[float]
    dec_len: int
    rec_len: int
    filter_bank: tuple[
        Sequence[float], Sequence[float], Sequence[float], Sequence[float]
    ]

    def __len__(self) -> int:
        """Return the number of filter coefficients."""
        return len(self.dec_lo)


class WaveletTensorTuple(NamedTuple):
    """Named tuple containing the wavelet filter bank to use in JIT code."""

    dec_lo: torch.Tensor
    dec_hi: torch.Tensor
    rec_lo: torch.Tensor
    rec_hi: torch.Tensor

    @property
    def dec_len(self) -> int:
        """Length of decomposition filters."""
        return len(self.dec_lo)

    @property
    def rec_len(self) -> int:
        """Length of reconstruction filters."""
        return len(self.rec_lo)

    @property
    def filter_bank(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter bank of the wavelet."""
        return self

    @classmethod
    def from_wavelet(cls, wavelet: Wavelet, dtype: torch.dtype) -> WaveletTensorTuple:
        """Construct Wavelet named tuple from wavelet protocol member."""
        return cls(
            torch.tensor(wavelet.dec_lo, dtype=dtype),
            torch.tensor(wavelet.dec_hi, dtype=dtype),
            torch.tensor(wavelet.rec_lo, dtype=dtype),
            torch.tensor(wavelet.rec_hi, dtype=dtype),
        )


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


def _get_len(wavelet: Union[tuple[torch.Tensor, ...], str, Wavelet]) -> int:
    """Get number of filter coefficients for various wavelet data types."""
    if isinstance(wavelet, tuple):
        return wavelet[0].shape[0]
    else:
        return len(_as_wavelet(wavelet))


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


def _preprocess_coeffs_1d(
    result_lst: Sequence[torch.Tensor], axis: int
) -> tuple[Sequence[torch.Tensor], list[int]]:
    if axis != -1:
        if isinstance(axis, int):
            result_lst = [coeff.swapaxes(axis, -1) for coeff in result_lst]
        else:
            raise ValueError("1d transforms operate on a single axis only.")

    # Fold axes for the wavelets
    ds = list(result_lst[0].shape)
    if len(ds) == 1:
        result_lst = [uf_coeff.unsqueeze(0) for uf_coeff in result_lst]
    elif len(ds) > 2:
        result_lst = [_fold_axes(uf_coeff, 1)[0] for uf_coeff in result_lst]
    return result_lst, ds


def _postprocess_coeffs_1d(
    result_list: list[torch.Tensor], ds: list[int], axis: int
) -> list[torch.Tensor]:
    if len(ds) == 1:
        result_list = [r_el.squeeze(0) for r_el in result_list]
    elif len(ds) > 2:
        # Unfold axes for the wavelets
        result_list = [_unfold_axes(fres, ds, 1) for fres in result_list]
    else:
        result_list = result_list

    if axis != -1:
        result_list = [coeff.swapaxes(axis, -1) for coeff in result_list]

    return result_list


def _preprocess_coeffs_2d(
    coeffs: WaveletCoeff2d, axes: tuple[int, int]
) -> tuple[WaveletCoeff2d, list[int]]:
    # swap axes if necessary
    if tuple(axes) != (-2, -1):
        if len(axes) != 2:
            raise ValueError("2D transforms work with two axes.")
        else:
            swap_fn = partial(_swap_axes, axes=axes)
            coeffs = _map_result(coeffs, swap_fn)

    # Fold axes for the wavelets
    ds = list(coeffs[0].shape)
    if len(ds) <= 1:
        raise ValueError("2d transforms require at least 2 input dimensions")
    elif len(ds) == 2:
        coeffs = _map_result(coeffs, lambda x: x.unsqueeze(0))
    elif len(ds) > 3:
        coeffs = _map_result(coeffs, lambda t: _fold_axes(t, 2)[0])
    return coeffs, ds


def _postprocess_coeffs_2d(
    coeffs: WaveletCoeff2d, ds: list[int], axes: tuple[int, int]
) -> WaveletCoeff2d:
    if len(ds) == 2:
        coeffs = _map_result(coeffs, lambda x: x.squeeze(0))
    elif len(ds) > 3:
        _unfold_axes2 = partial(_unfold_axes, ds=ds, keep_no=2)
        coeffs = _map_result(coeffs, _unfold_axes2)

    if tuple(axes) != (-2, -1):
        undo_swap_fn = partial(_undo_swap_axes, axes=axes)
        coeffs = _map_result(coeffs, undo_swap_fn)

    return coeffs


def _preprocess_tensor_1d(
    data: torch.Tensor, axis: int, add_channel_dim: bool = True
) -> tuple[torch.Tensor, list[int]]:
    """Preprocess input tensor dimensions.

    Args:
        data (torch.Tensor): An input tensor of any shape.
        axis (int): Compute the transform over this axis instead of the
            last one.
        add_channel_dim (bool): If True, ensures that the return has at
            least three axes by adding a new axis at dim 1.
            Defaults to True.

    Returns:
        A tuple (data, ds) where data is a data tensor of shape
        [new_batch, 1, to_process] and ds contains the original shape.

    Raises:
        ValueError: if ``axis`` is not a single int.
    """
    if axis != -1:
        if isinstance(axis, int):
            data = data.swapaxes(axis, -1)
        else:
            raise ValueError("1d transforms operate on a single axis only.")

    ds = list(data.shape)
    if len(ds) == 1:
        # assume time series
        data = data.unsqueeze(0)
    elif len(ds) > 2:
        data, ds = _fold_axes(data, 1)

    if add_channel_dim:
        data = data.unsqueeze(1)

    return data, ds


def _postprocess_tensor_1d(
    data: torch.Tensor, ds: list[int], axis: int
) -> torch.Tensor:
    if len(ds) == 1:
        data = data.squeeze(0)
    elif len(ds) > 2:
        data = _unfold_axes(data, ds, 1)

    if axis != -1:
        data = data.swapaxes(axis, -1)

    return data


def _preprocess_tensor_2d(
    data: torch.Tensor, axes: tuple[int, int], add_channel_dim: bool = True
) -> tuple[torch.Tensor, list[int]]:
    if tuple(axes) != (-2, -1):
        if len(axes) != 2:
            raise ValueError("2D transforms work with two axes.")
        else:
            data = _swap_axes(data, list(axes))

    # Preprocess multidimensional input.
    ds = list(data.shape)
    if len(ds) <= 1:
        raise ValueError("More than one input dimension required.")
    elif len(ds) == 2:
        data = data.unsqueeze(0)
    elif len(ds) >= 4:
        data, ds = _fold_axes(data, 2)

    if add_channel_dim:
        data = data.unsqueeze(1)

    return data, ds


def _postprocess_tensor_2d(
    data: torch.Tensor, ds: list[int], axes: tuple[int, int]
) -> torch.Tensor:
    if len(ds) == 2:
        data = data.squeeze(0)
    elif len(ds) > 3:
        data = _unfold_axes(data, ds, 2)

    if tuple(axes) != (-2, -1):
        data = _undo_swap_axes(data, axes)
    return data
