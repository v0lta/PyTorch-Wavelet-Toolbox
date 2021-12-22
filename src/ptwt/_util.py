"""Utility methods to compute wavelet decompositions from a dataset."""
from typing import Union


import pywt


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
