"""Utility methods to compute wavelet decompositions."""

from __future__ import annotations

import functools
import typing
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal, Optional, TypeAlias, Union, cast, overload

import numpy as np
import pywt
import torch
from more_itertools import grouper
from typing_extensions import ParamSpec, TypeVar

from .constants import (
    SUPPORTED_DTYPES,
    BoundaryMode,
    OrthogonalizeMethod,
    Wavelet,
    WaveletCoeff2d,
    WaveletCoeffNd,
    WaveletDetailDict,
    WaveletDetailTuple2d,
)

#: All the PyTorch boundary modes for :func:`torch.nn.functional.pad`
PyTorchBoundaryMode = Literal["replicate", "constant", "reflect", "circular"]

#: All the PyTorch boundary modes for :func:`torch.nn.functional.pad`
#: plus `symmetric` for the custom ptwt boundary
ExtendedPyTorchBoundaryMode = PyTorchBoundaryMode | Literal["symmetric"]

translation_dict: dict[BoundaryMode, ExtendedPyTorchBoundaryMode] = {
    "constant": "replicate",
    "zero": "constant",
    "reflect": "reflect",
    "periodic": "circular",
    # pytorch does not support symmetric mode,
    # we have our own implementation.
    "symmetric": "symmetric",
}

#: A hint for axes
AxisHint: TypeAlias = int | Sequence[int] | None


def _translate_boundary_strings(
    pywt_mode: BoundaryMode | None,
) -> ExtendedPyTorchBoundaryMode:
    """Translate pywt mode strings to PyTorch mode strings.

    We support ``constant``, ``zero``, ``reflect``,
    ``periodic`` and ``symmetric``.
    Unfortunately, ``constant`` has different meanings in the
    Pytorch and PyWavelet communities.

    Raises:
        ValueError: If the padding mode is not supported.
    """
    if pywt_mode is None:
        return translation_dict["reflect"]
    if pywt_mode in translation_dict:
        return translation_dict[pywt_mode]
    else:
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
    return_tensor = torch.as_tensor(
        data=filter_seq,
        dtype=dtype,
        device=device,
    ).unsqueeze(0)

    if flip:
        return_tensor = return_tensor.flip(-1)

    return return_tensor


def _is_orthogonalize_method_supported(
    orthogonalization: Optional[OrthogonalizeMethod],
) -> bool:
    return orthogonalization in typing.get_args(OrthogonalizeMethod)


def _is_dtype_supported(dtype: torch.dtype) -> bool:
    return dtype in SUPPORTED_DTYPES


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

    return padl, padr


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


def _check_same_device(
    tensor: torch.Tensor, torch_device: torch.device
) -> torch.Tensor:
    if torch_device != tensor.device:
        raise ValueError("coefficients must be on the same device")
    return tensor


def _check_same_dtype(tensor: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    if torch_dtype != tensor.dtype:
        raise ValueError("coefficients must have the same dtype")
    return tensor


def _check_same_device_dtype(
    coeffs: Union[list[torch.Tensor], WaveletCoeff2d, WaveletCoeffNd],
) -> tuple[torch.device, torch.dtype]:
    """Check coefficients for dtype and device consistency.

    Check that all coefficient tensors in `coeffs` have the same
    device and dtype.

    Args:
        coeffs (Wavelet coefficients): The resulting coefficients of
            a discrete wavelet transform. Can be either of
            `list[torch.Tensor]` (1d case),
            :data:`ptwt.constants.WaveletCoeff2d` (2d case) or
            :data:`ptwt.constants.WaveletCoeffNd` (Nd case).

    Returns:
        A tuple (device, dtype) with the shared device and dtype of
        all tensors in coeffs.
    """
    c = _check_if_tensor(coeffs[0])
    torch_device, torch_dtype = c.device, c.dtype

    # check for all tensors in `coeffs` that the device matches `torch_device`
    _coeff_tree_map(coeffs, partial(_check_same_device, torch_device=torch_device))
    # check for all tensors in `coeffs` that the dtype matches `torch_dtype`
    _coeff_tree_map(coeffs, partial(_check_same_dtype, torch_dtype=torch_dtype))

    return torch_device, torch_dtype


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
def _coeff_tree_map(
    coeffs: list[torch.Tensor],
    function: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]: ...


@overload
def _coeff_tree_map(
    coeffs: WaveletCoeff2d,
    function: Callable[[torch.Tensor], torch.Tensor],
) -> WaveletCoeff2d: ...


@overload
def _coeff_tree_map(
    coeffs: WaveletCoeffNd,
    function: Callable[[torch.Tensor], torch.Tensor],
) -> WaveletCoeffNd: ...


def _coeff_tree_map(
    coeffs: Union[list[torch.Tensor], WaveletCoeff2d, WaveletCoeffNd],
    function: Callable[[torch.Tensor], torch.Tensor],
) -> Union[list[torch.Tensor], WaveletCoeff2d, WaveletCoeffNd]:
    """Apply `function` to all tensor elements in `coeffs`.

    Applying a function to all tensors in the (potentially nested)
    coefficient data structure is a common requirement in coefficient
    pre- and postprocessing. This function saves us from having to loop
    over the coefficient data structures in processing.

    Conceptually, this function is inspired by the
    pytree processing philosophy of the JAX framework, see
    https://jax.readthedocs.io/en/latest/working-with-pytrees.html

    Raises:
        ValueError: If the input type is not supported.
    """
    approx = function(coeffs[0])
    result_lst: list[
        Union[
            torch.Tensor,
            WaveletDetailDict,
            WaveletDetailTuple2d,
        ]
    ] = []
    for element in coeffs[1:]:
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
        elif isinstance(element, torch.Tensor):
            result_lst.append(function(element))
        else:
            raise ValueError(f"Unexpected input type {type(element)}")

    if not result_lst:
        # if only approximation coeff:
        # use list iff data is a list
        return [approx] if isinstance(coeffs, list) else (approx,)
    elif isinstance(result_lst[0], torch.Tensor):
        # if the first detail coeff is tensor
        # -> all are tensors -> return a list
        return [approx] + cast(list[torch.Tensor], result_lst)
    else:
        # cast since we assume that the full list is of the same type
        cast_result_lst = cast(
            Union[list[WaveletDetailDict], list[WaveletDetailTuple2d]], result_lst
        )
        return approx, *cast_result_lst


# 1d case
@overload
def _preprocess_coeffs(
    coeffs: list[torch.Tensor],
    ndim: Literal[1],
    axes: AxisHint = ...,
    add_channel_dim: bool = False,
) -> tuple[list[torch.Tensor], list[int]]: ...


# 2d case
@overload
def _preprocess_coeffs(
    coeffs: WaveletCoeff2d,
    ndim: Literal[2],
    axes: AxisHint = ...,
    add_channel_dim: bool = False,
) -> tuple[WaveletCoeff2d, list[int]]: ...


# Nd case
@overload
def _preprocess_coeffs(
    coeffs: WaveletCoeffNd,
    ndim: int,
    axes: AxisHint = ...,
    add_channel_dim: bool = False,
) -> tuple[WaveletCoeffNd, list[int]]: ...


# list of nd tensors
@overload
def _preprocess_coeffs(
    coeffs: list[torch.Tensor],
    ndim: int,
    axes: AxisHint = ...,
    add_channel_dim: bool = False,
) -> tuple[list[torch.Tensor], list[int]]: ...


def _preprocess_coeffs(
    coeffs: Union[
        list[torch.Tensor],
        WaveletCoeff2d,
        WaveletCoeffNd,
    ],
    ndim: int,
    axes: AxisHint = None,
    add_channel_dim: bool = False,
) -> tuple[
    Union[
        list[torch.Tensor],
        WaveletCoeff2d,
        WaveletCoeffNd,
    ],
    list[int],
]:
    """Preprocess coeff tensor dimensions.

    For each coefficient tensor in `coeffs` the transformed axes
    as specified by `axes` are moved to be the last.
    Adds a batch dim if a coefficient tensor has none.
    If it has multiple batch dimensions, they are folded into a single
    batch dimension.

    Args:
        coeffs (Wavelet coefficients): The resulting coefficients of
            a discrete wavelet transform. Can be either of
            `list[torch.Tensor]` (1d case),
            :data:`ptwt.constants.WaveletCoeff2d` (2d case) or
            :data:`ptwt.constants.WaveletCoeffNd` (Nd case).
        ndim (int): The number of axes :math:`N` on which the transformation
            was applied.
        axes : Axes on which the transform was calculated.
        add_channel_dim (bool): If True, ensures that all returned coefficients
            have at least `:math:`N + 2` axes by potentially adding a new axis at dim 1.
            Defaults to False.

    Returns:
        A tuple ``(coeffs, ds)`` where ``coeffs`` are the transformed
        coefficients and ``ds`` contains the original shape of ``coeffs[0]``.
        If `add_channel_dim` is True, all coefficient tensors have
        :math:`N + 2` axes ([B, 1, c1, ..., cN]).
        otherwise :math:`N + 1` ([B, c1, ..., cN]).

    Raises:
        ValueError: If the input dtype is unsupported or `ndim` does not
            fit to the passed `axes` or `coeffs` dimensions.
    """
    if ndim <= 0:
        raise ValueError("Number of dimensions must be positive")

    torch_dtype = _check_if_tensor(coeffs[0]).dtype
    if not _is_dtype_supported(torch_dtype):
        raise ValueError(f"Input dtype {torch_dtype} not supported")

    axes = _ensure_axes(axes=axes, dim=ndim)
    if axes != _get_default_axes(ndim):
        # for all tensors in `coeffs`: swap the axes
        swap_fn = partial(_swap_axes, axes=axes)
        coeffs = _coeff_tree_map(coeffs, swap_fn)

    # Fold axes for the wavelets
    ds = list(coeffs[0].shape)
    if len(ds) < ndim:
        raise ValueError(f"At least {ndim} input dimensions required.")
    elif len(ds) == ndim:
        # for all tensors in `coeffs`: unsqueeze(0)
        coeffs = _coeff_tree_map(coeffs, lambda x: x.unsqueeze(0))
    elif len(ds) > ndim + 1:
        # for all tensors in `coeffs`: fold leading dims to batch dim
        coeffs = _coeff_tree_map(coeffs, lambda t: _fold_axes(t, ndim)[0])

    if add_channel_dim:
        # for all tensors in `coeffs`: add channel dim
        coeffs = _coeff_tree_map(coeffs, lambda x: x.unsqueeze(1))

    return coeffs, ds


# 1d case
@overload
def _postprocess_coeffs(
    coeffs: list[torch.Tensor],
    ndim: Literal[1],
    ds: list[int],
    axes: AxisHint = ...,
) -> list[torch.Tensor]: ...


# 2d case
@overload
def _postprocess_coeffs(
    coeffs: WaveletCoeff2d,
    ndim: Literal[2],
    ds: list[int],
    axes: AxisHint = ...,
) -> WaveletCoeff2d: ...


# Nd case
@overload
def _postprocess_coeffs(
    coeffs: WaveletCoeffNd,
    ndim: int,
    ds: list[int],
    axes: AxisHint = ...,
) -> WaveletCoeffNd: ...


# list of nd tensors
@overload
def _postprocess_coeffs(
    coeffs: list[torch.Tensor],
    ndim: int,
    ds: list[int],
    axes: AxisHint = ...,
) -> list[torch.Tensor]: ...


def _postprocess_coeffs(
    coeffs: Union[
        list[torch.Tensor],
        WaveletCoeff2d,
        WaveletCoeffNd,
    ],
    ndim: int,
    ds: list[int],
    axes: AxisHint = None,
) -> Union[
    list[torch.Tensor],
    WaveletCoeff2d,
    WaveletCoeffNd,
]:
    """Postprocess coeff tensor dimensions.

    This revereses the operations of :func:`_preprocess_coeffs`.

    Unfolds potentially folded batch dimensions and removes any added
    dimensions.
    The transformed axes as specified by `axes` are moved back to their
    original position.

    Args:
        coeffs (Wavelet coefficients): The preprocessed coefficients of
            a discrete wavelet transform. Can be either of
            `list[torch.Tensor]` (1d case),
            :data:`ptwt.constants.WaveletCoeff2d` (2d case) or
            :data:`ptwt.constants.WaveletCoeffNd` (Nd case).
        ndim (int): The number of axes :math:`N` on which the transformation was
            applied.
        ds (list of ints): The shape of the original first coefficient before
            preprocessing, i.e. of ``coeffs[0]``.
        axes : Axes on which the transform was calculated.

    Returns:
        The result of undoing the preprocessing operations on `coeffs`.

    Raises:
        ValueError: If `ndim` does not fit to the passed `axes`
            or `coeffs` dimensions.
    """
    if ndim <= 0:
        raise ValueError("Number of dimensions must be positive")

    axes = _ensure_axes(axes=axes, dim=ndim)

    # Fold axes for the wavelets
    if len(ds) < ndim:
        raise ValueError(f"At least {ndim} input dimensions required.")
    elif len(ds) == ndim:
        # for all tensors in `coeffs`: remove batch dim
        coeffs = _coeff_tree_map(coeffs, lambda x: x.squeeze(0))
    elif len(ds) > ndim + 1:
        # for all tensors in `coeffs`: unfold batch dim
        unfold_axes_fn = partial(_unfold_axes, ds=ds, keep_no=ndim)
        coeffs = _coeff_tree_map(coeffs, unfold_axes_fn)

    if axes != _get_default_axes(ndim):
        # for all tensors in `coeffs`: undo axes swapping
        undo_swap_fn = partial(_undo_swap_axes, axes=axes)
        coeffs = _coeff_tree_map(coeffs, undo_swap_fn)

    return coeffs


def _preprocess_tensor(
    data: torch.Tensor,
    ndim: int,
    *,
    axes: AxisHint = None,
    add_channel_dim: bool = True,
) -> tuple[torch.Tensor, list[int]]:
    """Preprocess input tensor dimensions.

    The transformed axes as specified by `axes` are moved to be the last.
    Adds a batch dim if `data` has none.
    If `data` has multiple batch dimensions, they are folded into a single
    batch dimension.

    Args:
        data (torch.Tensor): An input tensor with at least `ndim` axes.
        ndim (int): The number of axes :math:`N` on which the transformation is
            applied.
        axes : Axes on which the transform is calculated.
        add_channel_dim (bool): If True, ensures that the return has at
            least :math:`N + 2` axes by potentially adding a new axis at dim 1.
            Defaults to True.

    Returns:
        A tuple ``(data, ds)`` where ``data`` is the transformed data tensor
        and ``ds`` contains the original shape.
        If `add_channel_dim` is True,
        `data` has :math:`N + 2` axes ([B, 1, d1, ..., dN]).
        otherwise :math:`N + 1` ([B, d1, ..., dN]).
    """
    # interpreting data as the approximation coeffs of a 0-level FWT
    # allows us to reuse the `_preprocess_coeffs` code
    data_lst, ds = _preprocess_coeffs(
        [data], ndim=ndim, axes=axes, add_channel_dim=add_channel_dim
    )
    return data_lst[0], ds


def _postprocess_tensor(
    data: torch.Tensor, ndim: int, ds: list[int], axes: AxisHint | None = None
) -> torch.Tensor:
    """Postprocess input tensor dimensions.

    This revereses the operations of :func:`_preprocess_tensor`.

    Unfolds potentially folded batch dimensions and removes any added
    dimensions.
    The transformed axes as specified by `axes` are moved back to their
    original position.

    Args:
        data (torch.Tensor): An preprocessed input tensor.
        ndim (int): The number of axes :math:`N` on which the transformation is
            applied.
        ds (list of ints): The shape of the original input tensor before
            preprocessing.
        axes : Axes on which the transform was calculated.

    Returns:
        The result of undoing the preprocessing operations on `data`.
    """
    # interpreting data as the approximation coeffs of a 0-level FWT
    # allows us to reuse the `_postprocess_coeffs` code
    # return approx, *cast_result_lst
    return _postprocess_coeffs(coeffs=[data], ndim=ndim, ds=ds, axes=axes)[0]


Param = ParamSpec("Param")
RetType = TypeVar("RetType")


def _deprecated_alias(
    **aliases: str,
) -> Callable[[Callable[Param, RetType]], Callable[Param, RetType]]:
    """Handle deprecated function and method arguments.

    Use as follows::

        @_deprecated_alias(old_arg='new_arg')
        def myfunc(new_arg):
            ...

    Adapted from https://stackoverflow.com/a/49802489
    """

    def rename_kwargs(
        func_name: str,
        kwargs: Param.kwargs,  # type: ignore
        aliases: dict[str, str],
    ) -> None:
        """Rename deprecated kwarg.

        Raises:
            TypeError: If both arguments are present.
        """
        for alias, new in aliases.items():
            if alias in kwargs:
                if new in kwargs:
                    raise TypeError(
                        f"{func_name} received both {alias} and {new} as arguments!"
                        f" {alias} is deprecated, use {new} instead."
                    )
                warnings.warn(
                    message=(
                        f"`{alias}` is deprecated as an argument to `{func_name}`; use"
                        f" `{new}` instead."
                    ),
                    category=DeprecationWarning,
                    stacklevel=3,
                )
                kwargs[new] = kwargs.pop(alias)

    def deco(f: Callable[Param, RetType]) -> Callable[Param, RetType]:
        @functools.wraps(f)
        def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco


def _group_for_symmetric(padding: tuple[int, ...]) -> list[tuple[int, int]]:
    """Repack the padding tuple for symmetric padding."""
    return list(reversed(list(grouper(padding, 2))))  # type: ignore[arg-type]


def _get_padding_n(
    data: torch.Tensor, wavelet: Union[Wavelet, str], n: int
) -> tuple[int, ...]:
    wavelet_length = _get_len(wavelet)
    rv: list[int] = []
    for i in range(1, n + 1):
        rv.extend(_get_pad(data.shape[-i], wavelet_length))
    return tuple(rv)


def _ensure_axes(*, axes: AxisHint = None, dim: int) -> tuple[int, ...]:
    if axes is None:
        return _get_default_axes(dim)
    if isinstance(axes, int):
        if dim != 1:
            raise ValueError(f"tried passing single axis to {dim}D transform")
        return (axes,)
    if len(axes) != dim:
        raise ValueError(f"tried passing {len(axes)}D axes {axes} to {dim}D transform")
    _check_axes_argument(axes)
    return tuple(axes)


def _get_default_axes(n: int) -> tuple[int, ...]:
    """Get the default axes for a transformation.

    Args:
        n: The number of dimensions of the convolution

    Returns:
        A sequence of the default axes

    Raises:
        ValueError: If the dimension is not a natural number

    Examples:
        >>> _get_default_axes(1)
        (-1,)
        >>> _get_default_axes(2)
        (-2, -1)
        >>> _get_default_axes(3)
        (-3, -2, -1)
    """
    if n < 1:
        raise ValueError(f"only natural number dimensions are allowed. given: {n}")
    return tuple(range(-n, 0))


def _preprocess_deconstruction(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    ndim: int,
    axes: AxisHint = None,
) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    data, ds = _preprocess_tensor(data, ndim=ndim, axes=axes)
    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    dec_filt = _construct_nd_filt(dec_lo, dec_hi, n=ndim)
    return data, ds, dec_lo, dec_hi, dec_filt


def _construct_nd_filt(lo: torch.Tensor, hi: torch.Tensor, n: int) -> torch.Tensor:
    if n == 1:
        return _construct_1d_filt(lo, hi)
    elif n == 2:
        return _construct_2d_filt(lo, hi)
    elif n == 3:
        return _construct_3d_filt(lo, hi)
    else:
        raise NotImplementedError()


def _construct_1d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Construct one-dimensional filters."""
    return torch.stack([lo, hi], 0)


def _construct_2d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Construct two-dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        Stacked 2d-filters of dimension

        [2^2, 1, height, width].

        The four filters are ordered ll, lh, hl, hh.

    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    filt = filt.unsqueeze(1)
    return filt


def _construct_3d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Construct three-dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        Stacked 3d filters of dimension::

        [2^3, 1, length, height, width].

        The four filters are ordered ll, lh, hl, hh.
    """
    dim_size = lo.shape[-1]
    size = [dim_size] * 3
    lll = _outer(lo, _outer(lo, lo)).reshape(size)
    llh = _outer(lo, _outer(lo, hi)).reshape(size)
    lhl = _outer(lo, _outer(hi, lo)).reshape(size)
    lhh = _outer(lo, _outer(hi, hi)).reshape(size)
    hll = _outer(hi, _outer(lo, lo)).reshape(size)
    hlh = _outer(hi, _outer(lo, hi)).reshape(size)
    hhl = _outer(hi, _outer(hi, lo)).reshape(size)
    hhh = _outer(hi, _outer(hi, hi)).reshape(size)
    filt = torch.stack([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], 0)
    filt = filt.unsqueeze(1)
    return filt
