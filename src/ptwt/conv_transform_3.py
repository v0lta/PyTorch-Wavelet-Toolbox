"""Code for three dimensional padded transforms.

The functions here are based on torch.nn.functional.conv3d and it's transpose.
"""

from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import pywt
import torch

from ._util import (
    Wavelet,
    _as_wavelet,
    _check_axes_argument,
    _check_if_tensor,
    _fold_axes,
    _get_len,
    _is_dtype_supported,
    _map_result,
    _outer,
    _pad_symmetric,
    _swap_axes,
    _undo_swap_axes,
    _unfold_axes,
)
from .constants import BoundaryMode
from .conv_transform import (
    _adjust_padding_at_reconstruction,
    _get_filter_tensors,
    _get_pad,
    _translate_boundary_strings,
)


def _construct_3d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Construct three-dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        torch.Tensor: Stacked 3d filters of dimension::

        [8, 1, length, height, width].

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


def _fwt_pad3(
    data: torch.Tensor, wavelet: Union[Wavelet, str], *, mode: BoundaryMode
) -> torch.Tensor:
    """Pad data for the 3d-FWT.

    This function pads the last three axes.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode :
            The desired padding mode for extending the signal along the edges.
            See :data:`ptwt.constants.BoundaryMode`.

    Returns:
        The padded output tensor.

    """
    pytorch_mode = _translate_boundary_strings(mode)

    wavelet = _as_wavelet(wavelet)
    pad_back, pad_front = _get_pad(data.shape[-3], _get_len(wavelet))
    pad_bottom, pad_top = _get_pad(data.shape[-2], _get_len(wavelet))
    pad_right, pad_left = _get_pad(data.shape[-1], _get_len(wavelet))
    if pytorch_mode == "symmetric":
        data_pad = _pad_symmetric(
            data, [(pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right)]
        )
    else:
        data_pad = torch.nn.functional.pad(
            data,
            [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back],
            mode=pytorch_mode,
        )
    return data_pad


def wavedec3(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "zero",
    level: Optional[int] = None,
    axes: Tuple[int, int, int] = (-3, -2, -1),
) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Compute a three-dimensional wavelet transform.

    Args:
        data (torch.Tensor): The input data. For example of shape
            [batch_size, length, height, width]
        wavelet (Union[Wavelet, str]): The wavelet to transform with.
            ``pywt.wavelist(kind='discrete')`` lists possible choices.
        mode :
            The desired padding mode for extending the signal along the edges.
            Defaults to "zero". See :data:`ptwt.constants.BoundaryMode`.
        level (Optional[int]): The maximum decomposition level.
            This argument defaults to None.
        axes (Tuple[int, int, int]): Compute the transform over these axes
            instead of the last three. Defaults to (-3, -2, -1).

    Returns:
        list: A list with the lll coefficients and dictionaries
        with the filter order strings::

            ("aad", "ada", "add", "daa", "dad", "dda", "ddd")

        as keys. With a for the low pass or approximation filter and
        d for the high-pass or detail filter.

    Raises:
        ValueError: If the input has fewer than three dimensions or
            if the dtype is not supported or
            if the provided axes input has length other than three.

    Example:
        >>> import ptwt, torch
        >>> data = torch.randn(5, 16, 16, 16)
        >>> transformed = ptwt.wavedec3(data, "haar", level=2, mode="reflect")

    """
    if tuple(axes) != (-3, -2, -1):
        if len(axes) != 3:
            raise ValueError("3D transforms work with three axes.")
        else:
            _check_axes_argument(list(axes))
            data = _swap_axes(data, list(axes))

    ds = None
    if data.dim() < 3:
        raise ValueError("At least three dimensions are required for 3d wavedec.")
    elif len(data.shape) == 3:
        data = data.unsqueeze(1)
    else:
        data, ds = _fold_axes(data, 3)
        data = data.unsqueeze(1)

    if not _is_dtype_supported(data.dtype):
        raise ValueError(f"Input dtype {data.dtype} not supported")

    wavelet = _as_wavelet(wavelet)
    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    dec_filt = _construct_3d_filt(lo=dec_lo, hi=dec_hi)

    if level is None:
        level = pywt.dwtn_max_level(
            [data.shape[-1], data.shape[-2], data.shape[-3]], wavelet
        )

    result_lst: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = []
    res_lll = data
    for _ in range(level):
        if len(res_lll.shape) == 4:
            res_lll = res_lll.unsqueeze(1)
        res_lll = _fwt_pad3(res_lll, wavelet, mode=mode)
        res = torch.nn.functional.conv3d(res_lll, dec_filt, stride=2)
        res_lll, res_llh, res_lhl, res_lhh, res_hll, res_hlh, res_hhl, res_hhh = [
            sr.squeeze(1) for sr in torch.split(res, 1, 1)
        ]
        result_lst.append(
            {
                "aad": res_llh,
                "ada": res_lhl,
                "add": res_lhh,
                "daa": res_hll,
                "dad": res_hlh,
                "dda": res_hhl,
                "ddd": res_hhh,
            }
        )
    result_lst.append(res_lll)
    result_lst.reverse()

    if ds:
        _unfold_axes_fn = partial(_unfold_axes, ds=ds, keep_no=3)
        result_lst = _map_result(result_lst, _unfold_axes_fn)

    if tuple(axes) != (-3, -2, -1):
        undo_swap_fn = partial(_undo_swap_axes, axes=axes)
        result_lst = _map_result(result_lst, undo_swap_fn)

    return result_lst


def _waverec3d_fold_channels_3d_list(
    coeffs: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[
    List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    List[int],
]:
    # fold the input coefficients for processing conv2d_transpose.
    fold_coeffs: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = []
    ds = list(_check_if_tensor(coeffs[0]).shape)
    for coeff in coeffs:
        if isinstance(coeff, torch.Tensor):
            fold_coeffs.append(_fold_axes(coeff, 3)[0])
        else:
            new_dict = {}
            for key, value in coeff.items():
                new_dict[key] = _fold_axes(value, 3)[0]
            fold_coeffs.append(new_dict)
    return fold_coeffs, ds


def waverec3(
    coeffs: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    wavelet: Union[Wavelet, str],
    axes: Tuple[int, int, int] = (-3, -2, -1),
) -> torch.Tensor:
    """Reconstruct a signal from wavelet coefficients.

    Args:
        coeffs (list): The wavelet coefficient list produced by wavedec3.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        axes (Tuple[int, int, int]): Transform these axes instead of the
            last three. Defaults to (-3, -2, -1).

    Returns:
        torch.Tensor: The reconstructed four-dimensional signal of shape
        [batch, depth, height, width].

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
    if tuple(axes) != (-3, -2, -1):
        if len(axes) != 3:
            raise ValueError("3D transforms work with three axes")
        else:
            _check_axes_argument(list(axes))
            swap_axes_fn = partial(_swap_axes, axes=list(axes))
            coeffs = _map_result(coeffs, swap_axes_fn)

    wavelet = _as_wavelet(wavelet)
    ds = None
    # the Union[tensor, dict] idea is coming from pywt. We don't change it here.
    res_lll = _check_if_tensor(coeffs[0])
    if res_lll.dim() < 3:
        raise ValueError(
            "Three dimensional transforms require at least three dimensions."
        )
    elif res_lll.dim() >= 5:
        coeffs, ds = _waverec3d_fold_channels_3d_list(coeffs)
        res_lll = _check_if_tensor(coeffs[0])

    torch_device = res_lll.device
    torch_dtype = res_lll.dtype

    if not _is_dtype_supported(torch_dtype):
        if not _is_dtype_supported(torch_dtype):
            raise ValueError(f"Input dtype {torch_dtype} not supported")

    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    rec_filt = _construct_3d_filt(lo=rec_lo, hi=rec_hi)

    coeff_dicts = cast(Sequence[Dict[str, torch.Tensor]], coeffs[1:])
    for c_pos, coeff_dict in enumerate(coeff_dicts):
        if not isinstance(coeff_dict, dict) or len(coeff_dict) != 7:
            raise ValueError(
                f"Unexpected detail coefficient type: {type(coeff_dict)}. Detail "
                "coefficients must be a dict containing 7 tensors as returned by "
                "wavedec3."
            )
        for coeff in coeff_dict.values():
            if torch_device != coeff.device:
                raise ValueError("coefficients must be on the same device")
            elif torch_dtype != coeff.dtype:
                raise ValueError("coefficients must have the same dtype")
            elif res_lll.shape != coeff.shape:
                raise ValueError(
                    "All coefficients on each level must have the same shape"
                )
        res_lll = torch.stack(
            [
                res_lll,
                coeff_dict["aad"],
                coeff_dict["ada"],
                coeff_dict["add"],
                coeff_dict["daa"],
                coeff_dict["dad"],
                coeff_dict["dda"],
                coeff_dict["ddd"],
            ],
            1,
        )
        res_lll = torch.nn.functional.conv_transpose3d(res_lll, rec_filt, stride=2)
        res_lll = res_lll.squeeze(1)

        # remove the padding
        padfr = (2 * filt_len - 3) // 2
        padba = (2 * filt_len - 3) // 2
        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        padt = (2 * filt_len - 3) // 2
        padb = (2 * filt_len - 3) // 2
        if c_pos + 1 < len(coeff_dicts):
            padr, padl = _adjust_padding_at_reconstruction(
                res_lll.shape[-1], coeff_dicts[c_pos + 1]["aad"].shape[-1], padr, padl
            )
            padb, padt = _adjust_padding_at_reconstruction(
                res_lll.shape[-2], coeff_dicts[c_pos + 1]["aad"].shape[-2], padb, padt
            )
            padba, padfr = _adjust_padding_at_reconstruction(
                res_lll.shape[-3], coeff_dicts[c_pos + 1]["aad"].shape[-3], padba, padfr
            )
        if padt > 0:
            res_lll = res_lll[..., padt:, :]
        if padb > 0:
            res_lll = res_lll[..., :-padb, :]
        if padl > 0:
            res_lll = res_lll[..., padl:]
        if padr > 0:
            res_lll = res_lll[..., :-padr]
        if padfr > 0:
            res_lll = res_lll[..., padfr:, :, :]
        if padba > 0:
            res_lll = res_lll[..., :-padba, :, :]
    res_lll = res_lll.squeeze(1)

    if ds:
        res_lll = _unfold_axes(res_lll, ds, 3)

    if axes != (-3, -2, -1):
        res_lll = _undo_swap_axes(res_lll, list(axes))
    return res_lll
