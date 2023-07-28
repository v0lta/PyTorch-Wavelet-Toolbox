"""This module implements two-dimensional padded wavelet transforms.

The implementation relies on torch.nn.functional.conv2d and
torch.nn.functional.conv_transpose2d under the hood.
"""


from typing import Any, List, Optional, Tuple, Union

import pywt
import torch

from ._util import (
    Wavelet,
    _as_wavelet,
    _fold_channels,
    _get_len,
    _is_dtype_supported,
    _outer,
    _pad_symmetric,
    _unfold_channels,
)
from .conv_transform import (
    _adjust_padding_at_reconstruction,
    _get_filter_tensors,
    _get_pad,
    _translate_boundary_strings,
)


def _construct_2d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Construct two-dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        torch.Tensor: Stacked 2d-filters of dimension

        [filt_no, 1, height, width].

        The four filters are ordered ll, lh, hl, hh.

    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    filt = filt.unsqueeze(1)
    return filt


def _fwt_pad2(
    data: torch.Tensor, wavelet: Union[Wavelet, str], mode: str = "reflect"
) -> torch.Tensor:
    """Pad data for the 2d FWT.

    This function pads along the last two axes.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode (str): The padding mode.
            Supported modes are::

                "reflect", "zero", "constant", "periodic", "symmetric".

            "reflect" is the default mode.

    Returns:
        The padded output tensor.

    """
    mode = _translate_boundary_strings(mode)
    wavelet = _as_wavelet(wavelet)
    padb, padt = _get_pad(data.shape[-2], _get_len(wavelet))
    padr, padl = _get_pad(data.shape[-1], _get_len(wavelet))
    if mode == "symmetric":
        data_pad = _pad_symmetric(data, [(padt, padb), (padl, padr)])
    else:
        data_pad = torch.nn.functional.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad


def _wavedec2d_unfold_channels_2d_list(
    result_list: List[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ],
    ds: List[int],
) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    # unfolds the wavedec2d result lists, restoring the channel dimension.
    unfold_res: List[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = []
    for cres in result_list:
        if isinstance(cres, torch.Tensor):
            unfold_res.append(_unfold_channels(cres, list(ds)))
        else:
            unfold_res.append(
                (
                    _unfold_channels(cres[0], list(ds)),
                    _unfold_channels(cres[1], list(ds)),
                    _unfold_channels(cres[2], list(ds)),
                )
            )
    return unfold_res


def _waverec2d_fold_channels_2d_list(
    coeffs: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> Tuple[
    List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    List[int],
]:
    # fold the input coefficients for processing conv2d_transpose.
    fold_coeffs: List[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = []
    ds = list(_check_if_tensor(coeffs[0]).shape)
    for coeff in coeffs:
        if isinstance(coeff, torch.Tensor):
            fold_coeffs.append(_fold_channels(coeff))
        else:
            fold_coeffs.append(
                (
                    _fold_channels(coeff[0]),
                    _fold_channels(coeff[1]),
                    _fold_channels(coeff[2]),
                )
            )
    return fold_coeffs, ds


def wavedec2(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    mode: str = "reflect",
    level: Optional[int] = None,
) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Non-separated two-dimensional wavelet transform. Only the last two axes change.

    Args:
        data (torch.Tensor): The input data tensor with up to three dimensions.
            2d inputs are interpreted as ``[height, width]``,
            3d inputs are interpreted as ``[batch_size, height, width]``.
            4d inputs are interpreted as ``[batch_size, channels, height, width]``.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet. Refer to the output of
            ``pywt.wavelist(kind="discrete")`` for a list of possible choices.
        mode (str): The padding mode. Options are::

                "reflect", "zero", "constant", "periodic", "symmetric".

            This function defaults to "reflect".
        level (int): The number of desired scales.
            Defaults to None.

    Returns:
        list: A list containing the wavelet coefficients.
        The coefficients are in pywt order. That is::

            [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)] .

        A denotes approximation, H horizontal, V vertical
        and D diagonal coefficients.

    Raises:
        ValueError: If the dimensionality or the dtype of the input data tensor
            is unsupported.

    Example:
        >>> import torch
        >>> import ptwt, pywt
        >>> import numpy as np
        >>> from scipy import datasets
        >>> face = np.transpose(datasets.face(),
        >>>                     [2, 0, 1]).astype(np.float64)
        >>> pytorch_face = torch.tensor(face)
        >>> coefficients = ptwt.wavedec2(pytorch_face, pywt.Wavelet("haar"),
        >>>                              level=2, mode="zero")

    """
    fold = False
    if data.dim() == 2:
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 3:
        # add a channel dimension for torch.
        data = data.unsqueeze(1)
    elif data.dim() == 4:
        # avoid the channel sum, fold the channels into batches.
        fold = True
        ds = data.shape
        data = _fold_channels(data).unsqueeze(1)
    elif data.dim() == 1:
        raise ValueError("Wavedec2 needs more than one input dimension to work.")
    else:
        raise ValueError(
            "Wavedec2 does not support four input dimensions. \
             Optionally-batched two-dimensional inputs work."
        )

    if not _is_dtype_supported(data.dtype):
        raise ValueError(f"Input dtype {data.dtype} not supported")

    wavelet = _as_wavelet(wavelet)
    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    dec_filt = _construct_2d_filt(lo=dec_lo, hi=dec_hi)

    if level is None:
        level = pywt.dwtn_max_level([data.shape[-1], data.shape[-2]], wavelet)

    result_lst: List[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = []
    res_ll = data
    for _ in range(level):
        res_ll = _fwt_pad2(res_ll, wavelet, mode=mode)
        res = torch.nn.functional.conv2d(res_ll, dec_filt, stride=2)
        res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, 1)
        to_append = (res_lh.squeeze(1), res_hl.squeeze(1), res_hh.squeeze(1))
        result_lst.append(to_append)
    result_lst.append(res_ll.squeeze(1))

    if fold:
        result_lst = _wavedec2d_unfold_channels_2d_list(result_lst, list(ds))

    return result_lst[::-1]


def _check_if_tensor(to_check: Any) -> torch.Tensor:
    # Ensuring the first list elements are tensors makes mypy happy :-).
    if not isinstance(to_check, torch.Tensor):
        raise ValueError(
            "First element of coeffs must be the approximation coefficient tensor."
        )
    else:
        return to_check


def waverec2(
    coeffs: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    wavelet: Union[Wavelet, str],
) -> torch.Tensor:
    """Reconstruct a signal from wavelet coefficients.

    Args:
        coeffs (list): The wavelet coefficient list produced by wavedec2.
            The coefficients must be in pywt order. That is::

            [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)] .

            A denotes approximation, H horizontal, V vertical,
            and D diagonal coefficients.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.

    Returns:
        torch.Tensor: The reconstructed signal of shape ``[batch, height, width]`` or
            ``[batch, channel, height, width]`` depending on the input to `wavedec2`.

    Raises:
        ValueError: If coeffs is not in a shape as returned from wavedec2 or
            if the dtype is not supported.

    Example:
        >>> import ptwt, pywt, torch
        >>> import numpy as np
        >>> from scipy import datasets
        >>> face = np.transpose(datasets.face(),
        >>>                     [2, 0, 1]).astype(np.float64)
        >>> pytorch_face = torch.tensor(face)
        >>> coefficients = ptwt.wavedec2(pytorch_face, pywt.Wavelet("haar"),
        >>>                              level=2, mode="constant")
        >>> reconstruction = ptwt.waverec2(coefficients, pywt.Wavelet("haar"))

    """
    wavelet = _as_wavelet(wavelet)

    res_ll = _check_if_tensor(coeffs[0])
    torch_device = res_ll.device
    torch_dtype = res_ll.dtype

    fold = False
    if res_ll.dim() == 4:
        # avoid the channel sum, fold the channels into batches.
        fold = True
        coeffs, ds = _waverec2d_fold_channels_2d_list(coeffs)
        res_ll = _check_if_tensor(coeffs[0])

    if not _is_dtype_supported(torch_dtype):
        raise ValueError(f"Input dtype {torch_dtype} not supported")

    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    rec_filt = _construct_2d_filt(lo=rec_lo, hi=rec_hi)

    for c_pos, coeff_tuple in enumerate(coeffs[1:]):
        if not isinstance(coeff_tuple, tuple) or len(coeff_tuple) != 3:
            raise ValueError(
                f"Unexpected detail coefficient type: {type(coeff_tuple)}. Detail "
                "coefficients must be a 3-tuple of tensors as returned by "
                "wavedec2."
            )

        curr_shape = res_ll.shape
        for coeff in coeff_tuple:
            if torch_device != coeff.device:
                raise ValueError("coefficients must be on the same device")
            elif torch_dtype != coeff.dtype:
                raise ValueError("coefficients must have the same dtype")
            elif coeff.shape != curr_shape:
                raise ValueError(
                    "All coefficients on each level must have the same shape"
                )

        res_lh, res_hl, res_hh = coeff_tuple
        res_ll = torch.stack([res_ll, res_lh, res_hl, res_hh], 1)
        res_ll = torch.nn.functional.conv_transpose2d(
            res_ll, rec_filt, stride=2
        ).squeeze(1)

        # remove the padding
        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        padt = (2 * filt_len - 3) // 2
        padb = (2 * filt_len - 3) // 2
        if c_pos < len(coeffs) - 2:
            padr, padl = _adjust_padding_at_reconstruction(
                res_ll.shape[-1], coeffs[c_pos + 2][0].shape[-1], padr, padl
            )
            padb, padt = _adjust_padding_at_reconstruction(
                res_ll.shape[-2], coeffs[c_pos + 2][0].shape[-2], padb, padt
            )

        if padt > 0:
            res_ll = res_ll[..., padt:, :]
        if padb > 0:
            res_ll = res_ll[..., :-padb, :]
        if padl > 0:
            res_ll = res_ll[..., padl:]
        if padr > 0:
            res_ll = res_ll[..., :-padr]

    if fold:
        res_ll = _unfold_channels(res_ll, list(ds))

    return res_ll
