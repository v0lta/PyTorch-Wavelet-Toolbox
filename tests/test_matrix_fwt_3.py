"""Test the 3d matrix-fwt code."""

import numpy as np
import pytest
import pywt
import torch

from src.ptwt.matmul_transform import construct_boundary_a
from src.ptwt.matmul_transform_3 import MatrixWavedec3, MatrixWaverec3
from src.ptwt.sparse_math import _batch_dim_mm


@pytest.mark.parametrize("axis", [1, 2, 3])
@pytest.mark.parametrize(
    "shape", [[32, 32, 32], [64, 32, 32], [32, 64, 32], [32, 32, 64]]
)
def test_single_dim_mm(axis: int, shape: tuple):
    """Test the transposed matrix multiplication approach."""
    test_tensor = torch.rand(4, shape[0], shape[1], shape[2]).type(torch.float64)
    pywt_dec_lo, pywt_dec_hi = pywt.wavedec(
        test_tensor.numpy(), pywt.Wavelet("Haar"), axis=axis, level=1
    )
    haar_mat = construct_boundary_a(pywt.Wavelet("haar"), length=shape[axis - 1])
    mm_res = _batch_dim_mm(haar_mat, test_tensor, dim=axis)
    dec_lo, dec_hi = mm_res.split(shape[axis - 1] // 2, axis)
    assert np.allclose(pywt_dec_lo, dec_lo.numpy())
    assert np.allclose(pywt_dec_hi, dec_hi.numpy())


@pytest.mark.parametrize(
    "shape", [[32, 32, 32], [64, 32, 32], [32, 64, 32], [32, 32, 64]]
)
def test_boundary_wavedec3_level1_haar(shape):
    """Test a separable boundary 3d-transform."""
    batch_size = 1
    test_data = torch.rand(batch_size, shape[0], shape[1], shape[2]).type(torch.float64)

    pywtl, pywth = pywt.wavedec(test_data.numpy(), "haar", level=1, axis=-1)
    pywtll, pywthl = pywt.wavedec(pywtl, "haar", level=1, axis=-2)
    pywtlh, pywthh = pywt.wavedec(pywth, "haar", level=1, axis=-2)

    pylll, pyhll = pywt.wavedec(pywtll, "haar", level=1, axis=-3)
    pyllh, pyhlh = pywt.wavedec(pywtlh, "haar", level=1, axis=-3)
    pylhl, pyhhl = pywt.wavedec(pywthl, "haar", level=1, axis=-3)
    pylhh, pyhhh = pywt.wavedec(pywthh, "haar", level=1, axis=-3)

    pywtres = [
        pylll,
        {
            "aad": pyllh,
            "ada": pylhl,
            "daa": pyhll,
            "add": pylhh,
            "dad": pyhlh,
            "dda": pyhhl,
            "ddd": pyhhh,
        },
    ]
    ptwtres = MatrixWavedec3("haar", 1)(test_data)
    assert len(pywtres) == len(ptwtres)
    test_list = []
    for pywt_el, ptwt_el in zip(pywtres, ptwtres):
        if type(pywt_el) is np.ndarray:
            test_list.append(np.allclose(pywt_el, ptwt_el.numpy()))
        else:
            for key in pywt_el.keys():
                test_list.append(np.allclose(pywt_el[key], ptwt_el[key].numpy()))
    assert all(test_list)


@pytest.mark.slow
@pytest.mark.parametrize("level", [1, 2, 3, None])
@pytest.mark.parametrize(
    "shape", [[31, 32, 33], [63, 35, 32], [32, 62, 31], [32, 32, 64]]
)
def test_boundary_wavedec3_inverse(level, shape):
    """Test the 3d matrix wavedec and the padding for odd axes."""
    batch_size = 1
    test_data = torch.rand(batch_size, shape[0], shape[1], shape[2]).type(torch.float64)
    ptwtres = MatrixWavedec3("haar", level)(test_data)
    rec = MatrixWaverec3("haar")(ptwtres)
    assert np.allclose(
        test_data.numpy(), rec[:, : shape[0], : shape[1], : shape[2]].numpy()
    )
