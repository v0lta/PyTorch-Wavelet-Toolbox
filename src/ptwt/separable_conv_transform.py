"""Implement separable convolution-based transforms.

Under the hood code in this module transforms all dimensions
individually using torch.nn.functional.conv1d and it's
transpose.
"""
from typing import Dict, List, Optional, Union

import numpy as np
import pywt
import torch

from ._util import _as_wavelet, _fold_channels, _unfold_channels
from .conv_transform import wavedec, waverec


def _separable_conv_dwtn_(
    rec_dict: Dict[str, torch.Tensor],
    input: torch.Tensor,
    wavelet: Union[str, pywt.Wavelet],
    mode: str = "reflect",
    key: str = "",
) -> None:
    """Compute a single-level separable fast wavelet transform.

    All but the first axes are transformed.

    Args:
        input (torch.Tensor): Tensor of shape [batch, data_1, ... data_n].
        wavelet (Union[str, pywt.Wavelet]): The Wavelet to work with.
        mode (str): The padding mode. The following methods are supported::

                "reflect", "zero", "constant", "periodic".

            Defaults to "reflect".
        key (str): The filter application path. Defaults to "".
        dict (Dict[str, torch.Tensor]): The result will be stored here
            in place. Defaults to {}.
    """
    axis_total = len(input.shape) - 1
    if len(key) == axis_total:
        rec_dict[key] = input
    if len(key) < axis_total:
        current_axis = len(key) + 1
        transposed = input.transpose(-current_axis, -1)
        flat = transposed.reshape(-1, transposed.shape[-1])
        res_a, res_d = wavedec(flat, wavelet, level=1, mode=mode)
        res_a = res_a.reshape(list(transposed.shape[:-1]) + [res_a.shape[-1]])
        res_d = res_d.reshape(list(transposed.shape[:-1]) + [res_d.shape[-1]])
        res_a = res_a.transpose(-1, -current_axis)
        res_d = res_d.transpose(-1, -current_axis)
        _separable_conv_dwtn_(rec_dict, res_a, wavelet, mode, "a" + key)
        _separable_conv_dwtn_(rec_dict, res_d, wavelet, mode, "d" + key)


def _separable_conv_idwtn(
    in_dict: Dict[str, torch.Tensor], wavelet: Union[str, pywt.Wavelet]
) -> torch.Tensor:
    """Separable single level inverse fast wavelet transform.

    Args:
        in_dict (Dict[str, torch.Tensor]): The dictionary produced
            by _separable_conv_dwtn_ .
        wavelet (Union[str, pywt.Wavelet]): The wavelet used by
            _separable_conv_dwtn_ .

    Returns:
        torch.Tensor: A reconstruction of the original signal.
    """
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
        rec_ad = waverec([flat_a, flat_d], wavelet)
        rec_ad = rec_ad.reshape(list(trans_a.shape[:-1]) + [rec_ad.shape[-1]])
        rec_ad = rec_ad.transpose(-current_axis, -1)
        if a_key[1:]:
            done_dict[a_key[1:]] = rec_ad
        else:
            return rec_ad
    return _separable_conv_idwtn(done_dict, wavelet)


def _separable_conv_wavedecn(
    input: torch.Tensor,
    wavelet: Union[str, pywt.Wavelet],
    mode: str = "reflect",
    level: Optional[int] = None,
) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Compute a multilevel separable padded wavelet analysis transform.

    Args:
        input (torch.Tensor): A tensor of shape [batch, axis_1, ... axis_n].
            Everything but the batch axis will be transformed.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Please consider the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        mode (str): The desired padding mode. Padding extends the signal along
            the edges. Supported methods are::

                "reflect", "zero", "constant", "periodic".

            Defaults to "reflect".
        level (int): The desired decomposition level. If None the
            largest possible decomposition value is used.

    Returns:
        List[Union[torch.Tensor, Dict[str, torch.Tensor]]]: _description_
    """
    result: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = []
    approx = input

    if level is None:
        wlen = len(_as_wavelet(wavelet))
        level = int(
            min([np.log2(axis_len / (wlen - 1)) for axis_len in input.shape[1:]])
        )

    for _ in range(level):
        level_dict: Dict[str, torch.Tensor] = {}
        _separable_conv_dwtn_(level_dict, approx, wavelet, mode, "")
        approx_key = "a" * (len(input.shape) - 1)
        approx = level_dict.pop(approx_key)
        result.append(level_dict)
    result.append(approx)
    return result[::-1]


def _separable_conv_waverecn(
    coeff_list: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    wavelet: Union[str, pywt.Wavelet],
) -> torch.Tensor:
    """Separable n-dimensional wavelet synthesis transform.

    Args:
        coeff_list (List[Union[torch.Tensor, Dict[str, torch.Tensor]]]):
            The output as produced by `_separable_conv_wavedecn`.
        wavelet (Union[str, pywt.Wavelet]):
            The wavelet used by `_separable_conv_wavedecn`.

    Returns:
        torch.Tensor: The reconstruction of the original signal.

    Raises:
        ValueError: If the coeff_list is not structured as expected.
    """
    if not isinstance(coeff_list[0], torch.Tensor):
        raise ValueError("approximation tensor must be first in coefficient list.")
    if not all(map(lambda x: isinstance(x, dict), coeff_list[1:])):
        raise ValueError("All entries after approximation tensor must be dicts.")

    approx: torch.Tensor = coeff_list[0]
    for level_dict in coeff_list[1:]:
        keys = list(level_dict.keys())  # type: ignore
        level_dict["a" * max(map(len, keys))] = approx  # type: ignore
        approx = _separable_conv_idwtn(level_dict, wavelet)  # type: ignore
    return approx


def fswavedec2(
    input: torch.Tensor,
    wavelet: Union[str, pywt.Wavelet],
    mode: str = "reflect",
    level: Optional[int] = None,
) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Compute a fully separable 2D-padded analysis wavelet transform.

    Args:
        input (torch.Tensor): An input signal of shape ``[batch, height, width]``
            or ``[batch, channels, height, width]``.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet. Refer to the output of
            ``pywt.wavelist(kind="discrete")`` for a list of possible choices.
        mode (str): The padding mode. Options are::

                "reflect", "zero", "constant", "periodic".

            This function defaults to "reflect".
        level (int): The number of desired scales.
            Defaults to None.

    Raises:
        ValueError: If the input is not a batched 2D signal.

    Returns:
        List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        A list with the lll coefficients and dictionaries
        with the filter order strings::

        ("ad", "da", "dd")

        as keys. With a for the low pass or approximation filter and
        d for the high-pass or detail filter.


    Example:
        >>> import torch
        >>> import ptwt
        >>> data = torch.randn(5, 10, 10)
        >>> coeff = ptwt.fswavedec2(data, "haar", level=2)

    """
    fold = False
    if len(input.shape) == 2:
        input = input.unsqueeze(0)
    elif input.dim() == 4:
        # fold channels into batches.
        fold = True
        ds = list(input.shape)
        input = _fold_channels(input)
    elif len(input.shape) != 3:
        raise ValueError("Batched 2d inputs required for a 2d transform.")
    res = _separable_conv_wavedecn(input, wavelet, mode, level)

    if fold:
        unfold: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = []
        for resel in res:
            if isinstance(resel, torch.Tensor):
                unfold.append(_unfold_channels(resel, ds))
            else:
                unfold.append(
                    {key: _unfold_channels(value, ds) for key, value in resel.items()}
                )
        res = unfold

    return res


def fswavedec3(
    input: torch.Tensor,
    wavelet: Union[str, pywt.Wavelet],
    mode: str = "reflect",
    level: Optional[int] = None,
) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Compute a fully separable 3D-padded analysis wavelet transform.

    Args:
        input (torch.Tensor): An input signal of shape [batch, depth, height, width].
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet. Refer to the output of
            ``pywt.wavelist(kind="discrete")`` for a list of possible choices.
        mode (str): The padding mode. Options are::

                "reflect", "zero", "constant", "periodic".

            This function defaults to "reflect".
        level (int): The number of desired scales.
            Defaults to None.

    Raises:
        ValueError: If the input is not a batched 3D signal.

    Returns:
        List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        A list with the lll coefficients and dictionaries
        with the filter order strings::

        ("aad", "ada", "add", "daa", "dad", "dda", "ddd")

        as keys. With a for the low pass or approximation filter and
        d for the high-pass or detail filter.

    Example:
        >>> import torch
        >>> import ptwt
        >>> data = torch.randn(5, 10, 10, 10)
        >>> coeff = ptwt.fswavedec3(data, "haar", level=2)
    """
    if len(input.shape) == 3:
        input = input.unsqueeze(0)
    if len(input.shape) != 4:
        raise ValueError("Batched 3d inputs required for a 3d transform.")

    return _separable_conv_wavedecn(input, wavelet, mode, level)


def fswaverec2(
    coeff_list: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    wavelet: Union[str, pywt.Wavelet],
) -> torch.Tensor:
    """Compute a fully separable 2D-padded synthesis wavelet transform.

    Args:
        coeff_list (List[Union[torch.Tensor, Dict[str, torch.Tensor]]]):
            The wavelet coefficients as computed by `fswavedec2`.
        wavelet (Union[str, pywt.Wavelet]): The wavelet to use for the
            synthesis transform.

    Returns:
        torch.Tensor: A reconstruction of the signal encoded in the
            wavelet coefficients.

    Example:
        >>> import torch
        >>> import ptwt
        >>> data = torch.randn(5, 10, 10)
        >>> coeff = ptwt.fswavedec2(data, "haar", level=2)
        >>> rec = ptwt.fswaverec2(coeff, "haar")

    """
    return _separable_conv_waverecn(coeff_list, wavelet)


def fswaverec3(
    coeff_list: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    wavelet: Union[str, pywt.Wavelet],
) -> torch.Tensor:
    """Compute a fully separable 3D-padded synthesis wavelet transform.

    Args:
        coeff_list (List[Union[torch.Tensor, Dict[str, torch.Tensor]]]):
            The wavelet coefficients as computed by `fswavedec3`.
        wavelet (Union[str, pywt.Wavelet]): The wavelet to use for the
            synthesis transform.

    Returns:
        torch.Tensor: A reconstruction of the signal encoded in the
            wavelet coefficients.

    Example:
        >>> import torch
        >>> import ptwt
        >>> data = torch.randn(5, 10, 10, 10)
        >>> coeff = ptwt.fswavedec3(data, "haar", level=2)
        >>> rec = ptwt.fswaverec3(coeff, "haar")

    """
    return _separable_conv_waverecn(coeff_list, wavelet)
