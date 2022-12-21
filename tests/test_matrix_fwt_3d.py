from typing import List

import pytest
import numpy as np
import torch
import pywt
import src.ptwt as ptwt


def _cat_batch_list(batch_list: List) -> List:
    cat_list = []
    for batch_el in batch_list:
        cat_list.append(
            
        )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 64, 64),
        (2, 64, 64, 64),
        (3, 31, 64, 64),
        (3, 64, 31, 64),
        (3, 64, 64, 31),
        (3, 31, 31, 31),
        (3, 32, 32, 32),
    ],
)
@pytest.mark.parametrize("wavelet", ["haar", "db2", "db4"])
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("mode", ["zero", "constant", "periodic"])
def test_waverec3(shape: list, wavelet: str, level: int, mode: str):
    """Ensure the 3d analysis transform is invertible."""
    data = np.random.randn(*shape)
    data = torch.from_numpy(data)
    ptwc = ptwt.wavedec3(data, wavelet, level=level, mode=mode)
    batch_list = []
    for _ in range(data.shape[0]):
        pywc = pywt.wavedecn(data.numpy(), wavelet, level=level, mode=mode)
        batch_list.append(pywc)
    

    rec = ptwt.waverec3(ptwc, wavelet)
    assert np.allclose(
        rec.numpy()[..., : shape[1], : shape[2], : shape[3]], data.numpy()
    )
