"""Test the util methods."""
from typing import Tuple

import pytest
import pywt

from src.ptwt._util import _as_wavelet


class _MyHaarFilterBank(object):
    @property
    def filter_bank(self) -> Tuple[list, list, list, list]:
        """Unscaled Haar wavelet filters."""
        return (
            [1 / 2, 1 / 2.0],
            [-1 / 2.0, 1 / 2.0],
            [1 / 2.0, 1 / 2.0],
            [1 / 2.0, -1 / 2.0],
        )


@pytest.mark.parametrize(
    "wavelet",
    [
        "db4",
        pywt.Wavelet("sym4"),
        pywt.Wavelet("custom_wavelet_object", filter_bank=_MyHaarFilterBank()),
    ],
)
def test_as_wavelet(wavelet: str) -> None:
    """Test return types of _as_wavelet."""
    wavelet_result = _as_wavelet(wavelet)
    assert isinstance(wavelet_result, pywt.Wavelet)


@pytest.mark.parametrize("wavelet", ["invalid_wavelet_name"])
def test_failed_as_wavelet(wavelet: str) -> None:
    """Test expected errors for invalid input to _as_wavelet."""
    with pytest.raises(ValueError):
        wavelet = _as_wavelet(wavelet)
