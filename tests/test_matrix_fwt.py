"""Test the fwt and ifwt matrices."""
# Written by moritz ( @ wolter.tech ) in 2021
import numpy as np
import pytest
import pywt
import torch

from src.ptwt.matmul_transform import (
    MatrixWavedec,
    MatrixWaverec,
    _construct_a,
    _construct_s,
)
from tests._mackey_glass import MackeyGenerator


@pytest.mark.parametrize("size", [8, 16, 24, 32])
def test_analysis_and_synthethis_matrices_db1(size: int) -> None:
    """Ensure the analysis matrix a and the synthesis matrix s invert each other."""
    a_db1 = _construct_a(pywt.Wavelet("db1"), size)
    s_db1 = _construct_s(pywt.Wavelet("db1"), size)
    assert np.allclose(torch.sparse.mm(a_db1, s_db1.to_dense()).numpy(), np.eye(size))


@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize("length", [16, 32, 64, 128])
def test_fwt_ifwt_haar(level: int, length: int) -> None:
    """Test the Haar case."""
    wavelet = pywt.Wavelet("haar")
    data = np.random.uniform(-1, 1, (length))
    coeffs = pywt.wavedec(data, wavelet, level=level)
    matrix_wavedec = MatrixWavedec(wavelet, level)
    coeffs_matfwt = matrix_wavedec(torch.from_numpy(data))
    test_list = [not np.allclose(cmfwt.numpy(), cpywt) for cmfwt, cpywt in zip(coeffs_matfwt, coeffs)]
    assert not any(test_list)


@pytest.mark.slow
def test_fwt_ifwt_mackey_haar_cuda() -> None:
    """Test the Haar case for a long signal on GPU"""
    wavelet = pywt.Wavelet("haar")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = MackeyGenerator(batch_size=2, tmax=512, delta_t=1, device=device)
    pt_data = torch.squeeze(generator()).type(torch.float64)
    # ensure coefficients are equal.
    coeffs = pywt.wavedec(pt_data.cpu().numpy(), wavelet, level=9)
    matrix_wavedec = MatrixWavedec(wavelet, 9)
    coeffs_matfwt = matrix_wavedec(pt_data)
    test_list = [not np.allclose(cmfwt.cpu().numpy(), cpywt) for cmfwt, cpywt in zip(coeffs_matfwt, coeffs)]
    assert not any(test_list)
    # test the inverse fwt.
    matrix_waverec = MatrixWaverec(wavelet)
    reconstructed_data = matrix_waverec(coeffs_matfwt)
    assert np.allclose(pt_data.cpu().numpy(), reconstructed_data.cpu().numpy())


@pytest.mark.slow
@pytest.mark.parametrize("level", [1, 2, 3, 4 ,None])
@pytest.mark.parametrize("wavelet", ["db2", "db3", "db4", "sym5"])
def test_fwt_ifwt_mackey_db2(level: int, wavelet: str) -> None:
    """Test multiple wavelets and levels for a long signal."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavelet = pywt.Wavelet(wavelet)
    generator = MackeyGenerator(batch_size=2, tmax=512, delta_t=1, device=device)
    pt_data = torch.squeeze(generator()).type(torch.float64)
    matrix_wavedec = MatrixWavedec(wavelet, level)
    coeffs_mat_max = matrix_wavedec(pt_data)
    matrix_waverec = MatrixWaverec(wavelet)
    reconstructed_data = matrix_waverec(coeffs_mat_max)
    assert np.allclose(reconstructed_data.cpu().numpy(), pt_data.cpu().numpy())
