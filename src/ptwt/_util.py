"""Utility methods to compute wavelet decompositions from a dataset."""
from typing import Protocol, Sequence, Tuple, Union


import pywt


class Wavelet(Protocol):
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


def _as_wavelet(wavelet: Union[str, pywt.Wavelet]) -> pywt.Wavelet:
    """Ensure the input arguments to be a pywt wavelet object.

    Args:
        wavelet (Union[str, pywt.Wavelet]): The input argument, which is either
            a pywt.Wavelet object or a valid wavelet name string.

    Returns:
        pywt.Wavelet: the input wavelet object or a wavelet object described by the
            input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


def _is_boundary_mode_supported(boundary_mode: str) -> bool:
    return boundary_mode == "qr" or boundary_mode == "gramschmidt"
