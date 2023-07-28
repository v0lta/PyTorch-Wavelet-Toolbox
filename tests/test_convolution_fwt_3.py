"""Test our 3d for loop-convolution based fwt code."""

from typing import List

import numpy as np
import pytest
import pywt
import torch

import src.ptwt as ptwt


def _expand_dims(batch_list: List) -> List:
    for pos, bel in enumerate(batch_list):
        if type(bel) is np.ndarray:
            batch_list[pos] = np.expand_dims(bel, 0)
        else:
            for key, item in batch_list[pos].items():
                batch_list[pos][key] = np.expand_dims(item, 0)
    return batch_list


def _cat_batch_list(batch_lists: List) -> List:
    cat_list = None
    for batch_list in batch_lists:
        batch_list = _expand_dims(batch_list)
        if not cat_list:
            cat_list = batch_list
        else:
            for pos, (cat_el, batch_el) in enumerate(zip(cat_list, batch_list)):
                if type(cat_el) == np.ndarray:
                    cat_list[pos] = np.concatenate([cat_el, batch_el])
                elif type(cat_el) == dict:
                    for key, tensor in cat_el.items():
                        cat_el[key] = np.concatenate([tensor, batch_el[key]])
                else:
                    raise NotImplementedError()
    return cat_list


@pytest.mark.parametrize(
    "shape",
    [
        (1, 31, 32, 33),
        (1, 64, 64, 64),
        (2, 64, 64, 64),
        (3, 31, 64, 64),
        (3, 64, 31, 64),
        (3, 64, 64, 31),
        (3, 31, 31, 31),
        (3, 32, 32, 32),
        (3, 31, 32, 33),
    ],
)
@pytest.mark.parametrize("wavelet", ["haar", "db2", "db4"])
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("mode", ["reflect", "zero", "constant", "periodic", "symmetric"])
def test_waverec3(shape: list, wavelet: str, level: int, mode: str) -> None:
    """Ensure the 3d analysis transform is invertible."""
    data = np.random.randn(*shape)
    data = torch.from_numpy(data)
    ptwc = ptwt.wavedec3(data, wavelet, level=level, mode=mode)
    batch_list = []
    for batch_no in range(data.shape[0]):
        pywc = pywt.wavedecn(data[batch_no].numpy(), wavelet, level=level, mode=mode)
        batch_list.append(pywc)
    cat_pywc = _cat_batch_list(batch_list)

    # ensure ptwt and pywt coefficients are identical.
    test_list = []
    for a, b in zip(ptwc, cat_pywc):
        if type(a) is torch.Tensor:
            test_list.append(np.allclose(a, b))
        else:
            test_list.extend([np.allclose(a[key], b[key]) for key in a.keys()])

    assert all(test_list)

    # ensure the transforms are invertible.
    rec = ptwt.waverec3(ptwc, wavelet)
    assert np.allclose(
        rec.numpy()[..., : shape[1], : shape[2], : shape[3]], data.numpy()
    )
