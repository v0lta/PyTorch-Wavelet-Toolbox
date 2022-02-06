"""Test the fwt and ifwt matrices."""
# Written by moritz ( @ wolter.tech ) in 2021
import numpy as np
import pytest
import pywt
import torch
from src.ptwt._mackey_glass import MackeyGenerator
from src.ptwt.matmul_transform import (
    MatrixWavedec,
    MatrixWaverec,
    _construct_a,
    _construct_s,
)


def test_analysis_and_synthethis_matrices_db1():
    """Tests the db1 case."""
    a_db1 = _construct_a(pywt.Wavelet("db1"), 8)
    s_db1 = _construct_s(pywt.Wavelet("db1"), 8)
    err = np.mean(np.abs(torch.sparse.mm(a_db1, s_db1.to_dense()).numpy() - np.eye(8)))
    print("db1 8 inverse error", err)
    assert err < 1e-6


def test_fwt_ifwt_level_1():
    """Test the Haar case."""
    wavelet = pywt.Wavelet("haar")
    data2 = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ]
    )

    # level 1
    coeffs = pywt.dwt(data2, wavelet)
    print(coeffs[0], coeffs[1])
    matrix_wavedec = MatrixWavedec(wavelet, 1)
    coeffsmat1 = matrix_wavedec(torch.from_numpy(data2))
    err1 = np.mean(np.abs(coeffs[0] - coeffsmat1[0].squeeze().numpy()))
    err2 = np.mean(np.abs(coeffs[1] - coeffsmat1[1].squeeze().numpy()))
    print(err1 < 0.00001, err2 < 0.00001)
    assert err1 < 1e-6
    assert err2 < 1e-6


def test_fwt_ifwt_level_2():
    """Test the Haar level two case."""
    wavelet = pywt.Wavelet("haar")
    data2 = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
        ]
    )
    coeffs2 = pywt.wavedec(data2, wavelet, level=2, mode="zero")
    matrix_wavedec = MatrixWavedec(wavelet, 2)
    coeffsmat2 = matrix_wavedec(torch.from_numpy(data2))

    err1 = np.mean(np.abs(coeffs2[0] - coeffsmat2[0].squeeze().numpy()))
    err2 = np.mean(np.abs(coeffs2[1] - coeffsmat2[1].squeeze().numpy()))
    err3 = np.mean(np.abs(coeffs2[2] - coeffsmat2[2].squeeze().numpy()))
    print(
        np.mean(np.abs(coeffs2[0] - coeffsmat2[0].squeeze().numpy())) < 1e-6,
        np.mean(np.abs(coeffs2[1] - coeffsmat2[1].squeeze().numpy())) < 1e-6,
        np.mean(np.abs(coeffs2[2] - coeffsmat2[2].squeeze().numpy())) < 1e-6,
    )
    assert err1 < 1e-6
    assert err2 < 1e-6
    assert err3 < 1e-6


def test_fwt_ifwt_level_3():
    """Test the Haar level 3 case."""
    wavelet = pywt.Wavelet("haar")
    data2 = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ]
    )
    coeffs3 = pywt.wavedec(data2, wavelet, level=3)
    matrix_wavedec = MatrixWavedec(wavelet, level=3)
    coeffsmat3 = matrix_wavedec(torch.from_numpy(data2))

    err1 = np.mean(np.abs(coeffs3[0] - coeffsmat3[0].squeeze().numpy()))
    err2 = np.mean(np.abs(coeffs3[1] - coeffsmat3[1].squeeze().numpy()))
    err3 = np.mean(np.abs(coeffs3[2] - coeffsmat3[2].squeeze().numpy()))
    err4 = np.mean(np.abs(coeffs3[3] - coeffsmat3[3].squeeze().numpy()))
    print(err1 < 1e-6, err2 < 1e-6, err3 < 1e-6, err4 < 1e-6)

    assert err1 < 1e-6
    assert err2 < 1e-6
    assert err3 < 1e-6
    assert err4 < 1e-6

    matrix_waverec = MatrixWaverec(wavelet)
    reconstructed_data = matrix_waverec(coeffsmat3)
    err5 = torch.mean(torch.abs(torch.from_numpy(data2) - reconstructed_data))
    print("abs ifwt 3 reconstruction error", err5)
    assert np.allclose(data2, reconstructed_data.numpy())


@pytest.mark.slow
def test_fwt_ifwt_mackey_haar():
    """Test the Haar case for a long signal."""
    wavelet = pywt.Wavelet("haar")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = MackeyGenerator(batch_size=2, tmax=512, delta_t=1, device=device)
    wavelet = pywt.Wavelet("haar")
    pt_data = torch.squeeze(generator())

    coeffs_max = pywt.wavedec(pt_data.cpu().numpy(), wavelet, level=9)

    matrix_wavedec = MatrixWavedec(wavelet, 9)
    coeffs_mat_max = matrix_wavedec(pt_data)

    test_lst = []
    for test_no in range(9):
        test_lst.append(
            np.sum(np.abs(coeffs_max[test_no] - coeffs_mat_max[test_no].T.numpy()))
            < 0.001
        )
    print(test_lst)

    # test the inverse fwt.
    matrix_waverec = MatrixWaverec(wavelet)
    reconstructed_data = matrix_waverec(coeffs_mat_max)
    err1 = torch.mean(torch.abs(pt_data - reconstructed_data))
    print("abs ifwt reconstruction error", err1)
    assert np.allclose(pt_data.numpy(), reconstructed_data.numpy())


@pytest.mark.slow
def test_fwt_ifwt_mackey_db2():
    """Test the db2 case for a long signal."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavelet = pywt.Wavelet("db2")
    generator = MackeyGenerator(batch_size=2, tmax=512, delta_t=1, device=device)
    pt_data = torch.squeeze(generator()).cpu()
    matrix_wavedec = MatrixWavedec(wavelet, 4)
    coeffs_mat_max = matrix_wavedec(pt_data)
    matrix_waverec = MatrixWaverec(wavelet)
    reconstructed_data = matrix_waverec(coeffs_mat_max)
    err = torch.mean(torch.abs(pt_data - reconstructed_data))
    print("reconstruction error:", err)
    assert err < 1e-6
