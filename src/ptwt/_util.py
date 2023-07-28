"""Utility methods to compute wavelet decompositions from a dataset."""
from typing import List, Optional, Protocol, Sequence, Tuple, Union

import pywt
import torch


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


def _is_boundary_mode_supported(boundary_mode: Optional[str]) -> bool:
    return boundary_mode in ["qr", "gramschmidt"]


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
            topadl = signal[:padl].flip(0)
            cat_list.insert(0, topadl)
        if padr > 0:
            topadr = signal[-padr::].flip(0)
            cat_list.append(topadr)
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


def _fold_channels(data: torch.Tensor) -> torch.Tensor:
    """Fold [batch, channel, height width] into [batch*channel, height, widht]."""
    ds = data.shape
    fold_data = torch.permute(data, [2, 3, 0, 1])
    fold_data = torch.reshape(fold_data, [ds[2], ds[3], ds[0] * ds[1]])
    return torch.permute(fold_data, [2, 0, 1])


def _unfold_channels(data: torch.Tensor, ds: List[int]) -> torch.Tensor:
    """Unfold [batch*channel, height, widht] into [batch, channel, height, width]."""
    unfold_data = torch.permute(data, [1, 2, 0])
    unfold_data = torch.reshape(
        unfold_data, [data.shape[1], data.shape[2], ds[0], ds[1]]
    )
    return torch.permute(unfold_data, [2, 3, 0, 1])
