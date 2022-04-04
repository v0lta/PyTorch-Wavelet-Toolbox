"""Test the conv-fwt-3d code."""
# Written by moritz ( @ wolter.tech ) in 2022
import numpy as np
import pytest
import pywt
import torch

from src.ptwt.conv_transform_3 import wavedec3


# TODO: test batch dim!
@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 64),
        (64, 64, 64),
        (31, 64, 64),
        (64, 31, 64),
        (64, 64, 31),
        (31, 31, 31),
        (32, 32, 32),
    ],
)
@pytest.mark.parametrize("wavelet", ["haar", "db2"])
@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("mode", ["zero"])
def test_wavedec3(shape: list, wavelet: str, level: int, mode: str):
    """Test the conv2d-code."""
    data = np.random.randn(*shape)
    pywc = pywt.wavedecn(data, wavelet, level=level, mode=mode)
    ptwc = wavedec3(torch.from_numpy(data), wavelet, level=level, mode=mode)

    for pywc_res, ptwc_res in zip(pywc, ptwc):
        if type(pywc_res) is np.ndarray:
            assert np.allclose(pywc_res, ptwc_res.squeeze(0).numpy())
        else:
            assert type(pywc_res) is dict
            for key in pywc_res.keys():
                assert np.allclose(ptwc_res[key].numpy(), pywc_res[key])


def test_incorrect_dims():
    """Test expected errors for an invalid padding name."""
    data = np.random.randn(64, 64)
    with pytest.raises(ValueError):
        _ = wavedec3(torch.from_numpy(data), "haar")
