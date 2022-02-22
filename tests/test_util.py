"""Test the util methods."""
import pytest
import pywt
from src.ptwt._util import _as_wavelet


class _MyHaarFilterBank(object):
    @property
    def filter_bank(self):
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
def test_as_wavelet(wavelet):
    wavelet_result = _as_wavelet(wavelet)
    assert isinstance(wavelet_result, pywt.Wavelet)


@pytest.mark.parametrize("wavelet", ["invalid_wavelet_name"])
def test_failed_as_wavelet(wavelet):
    with pytest.raises(ValueError):
        wavelet = _as_wavelet(wavelet)
