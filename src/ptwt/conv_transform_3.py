"""Code for three dimensional padded transforms."""

from typing import Dict, List, Optional, Union

import pywt
import torch

from ._util import Wavelet, _as_wavelet, _outer
from .conv_transform import _get_pad, _translate_boundary_strings, get_filter_tensors


def _construct_3d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Construct three dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        torch.Tensor: Stacked 3d filters of dimension
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
    data: torch.Tensor, wavelet: Union[Wavelet, str], mode: str
) -> torch.Tensor:
    """Pad data for the 2d FWT.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode (str): The padding mode. Supported modes are "zero".

    Returns:
        The padded output tensor.

    """
    mode = _translate_boundary_strings(mode)

    wavelet = _as_wavelet(wavelet)
    pad_back, pad_front = _get_pad(data.shape[-3], len(wavelet.dec_lo))
    pad_bottom, pad_top = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    pad_right, pad_left = _get_pad(data.shape[-1], len(wavelet.dec_lo))
    data_pad = torch.nn.functional.pad(
        data, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back], mode=mode
    )
    return data_pad


def wavedec3(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    level: Optional[int] = None,
    mode: str = "zero",
) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Compute a three dimensional wavelet transform.

    Args:
        data (torch.Tensor): The input data of shape
            [batch_size, length, height, width]
        wavelet (Union[Wavelet, str]): The wavelet to be used.
        level (Optional[int]): The maximum decomposition level.
            Defaults to None.
        mode (str): The padding mode. Options are
            "zero", "constant" or "periodic".
            Defaults to "zero".

    Returns:
        list: A list with the lll coefficients and dictionaries
            with the filter order strings ("aad", "ada", "add",
            "daa", "dad", "dda", "ddd") as keys.

    Raises:
        ValueError: If the input has fewer than 3 dimensions.

    """
    if data.dim() < 3:
        raise ValueError("Three dimensional inputs required for 3d wavedec.")
    elif data.dim() == 3:
        # add batch dim.
        data = data.unsqueeze(0)

    wavelet = _as_wavelet(wavelet)
    dec_lo, dec_hi, _, _ = get_filter_tensors(
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
        res_lll = _fwt_pad3(res_lll.unsqueeze(1), wavelet, mode=mode)
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
    return result_lst[::-1]


def waverec3(
    coeffs: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    wavelet: Union[Wavelet, str],
) -> torch.Tensor:
    """Reconstruct a signal from wavelet coefficients.

    Args:
        coeffs (list): The wavelet coefficient list produced by wavedec3.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.

    Returns:
        torch.Tensor: The reconstructed signal.
    """
    wavelet = _as_wavelet(wavelet)
    # the Union[tensor, dict] idea is coming from pywt. We don't change it here.
    res_lll: torch.Tensor = coeffs[0]  # type: ignore
    _, _, rec_lo, rec_hi = get_filter_tensors(
        wavelet, flip=False, device=res_lll.device, dtype=res_lll.dtype
    )
    filt_len = rec_lo.shape[-1]
    rec_filt = _construct_3d_filt(lo=rec_lo, hi=rec_hi)

    for c_pos, coeff_dict in enumerate(coeffs[1:]):
        res_lll = torch.stack(
            [
                res_lll,
                coeff_dict["aad"],  # type: ignore
                coeff_dict["ada"],  # type: ignore
                coeff_dict["add"],  # type: ignore
                coeff_dict["daa"],  # type: ignore
                coeff_dict["dad"],  # type: ignore
                coeff_dict["dda"],  # type: ignore
                coeff_dict["ddd"],  # type: ignore
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
        if c_pos < len(coeffs) - 2:
            pred_len = res_lll.shape[-1] - (padl + padr)
            next_len = coeffs[c_pos + 2]["aad"].shape[-1]  # type: ignore
            pred_len2 = res_lll.shape[-2] - (padt + padb)
            next_len2 = coeffs[c_pos + 2]["aad"].shape[-2]  # type: ignore
            pred_len3 = res_lll.shape[-3] - (padfr + padba)
            next_len3 = coeffs[c_pos + 2]["aad"].shape[-3]  # type: ignore
            if next_len != pred_len:
                padr += 1
                pred_len = res_lll.shape[-1] - (padl + padr)
                assert (
                    next_len == pred_len
                ), "padding error, please open an issue on github "
            if next_len2 != pred_len2:
                padb += 1
                pred_len2 = res_lll.shape[-2] - (padt + padb)
                assert (
                    next_len2 == pred_len2
                ), "padding error, please open an issue on github "
            if next_len3 != pred_len3:
                padba += 1
                pred_len3 = res_lll.shape[-3] - (padba + padfr)
                assert (
                    next_len3 == pred_len3
                ), "padding error, please open an issue on github "
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
    return res_lll
