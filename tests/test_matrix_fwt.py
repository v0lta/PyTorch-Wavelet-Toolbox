import numpy as np
import pywt
import time
import torch
import pytest

from src.ptwt.matmul_transform import (
    construct_a,
    construct_s,
    matrix_wavedec,
    matrix_waverec,
)
from src.ptwt.mackey_glass import MackeyGenerator


# ----------------------- matrix construction tests ----------------------#
def test_analysis_and_synthethis_matrices_db1():
    a_db1 = construct_a(pywt.Wavelet("db1"), 8)
    s_db1 = construct_s(pywt.Wavelet("db1"), 8)
    err = np.mean(np.abs(torch.sparse.mm(a_db1, s_db1.to_dense()).numpy() - np.eye(8)))
    print("db1 8 inverse error", err)
    assert err < 1e-6


def test_analysis_and_synthethis_matrices_db2():
    a_db2 = construct_a(pywt.Wavelet("db2"), 64, wrap=True)
    s_db2 = construct_s(pywt.Wavelet("db2"), 64, wrap=True)
    test_eye = torch.sparse.mm(a_db2, s_db2.to_dense()).numpy()
    err = np.mean(np.abs(test_eye - np.eye(64)))
    print("db2 64 inverse error", err)
    assert err < 1e-6


def test_analysis_and_synthethis_matrices_db2_66():
    a_db2_66 = construct_a(pywt.Wavelet("db2"), 66, wrap=True)
    s_db2_66 = construct_s(pywt.Wavelet("db2"), 66, wrap=True)
    test_eye = torch.sparse.mm(a_db2_66, s_db2_66.to_dense()).numpy()
    err = np.mean(np.abs(test_eye - np.eye(66)))
    print("db2 66 inverse error", err)
    assert err < 1e-6


def test_analysis_and_synthethis_matrices_db8():
    a_db2 = construct_a(pywt.Wavelet("db8"), 128)
    s_db2 = construct_s(pywt.Wavelet("db8"), 128)
    test_eye = torch.sparse.mm(a_db2, s_db2.to_dense()).numpy()
    err = np.mean(np.abs(test_eye - np.eye(128)))
    print("db8 128 inverse error", err)


# ------------------------- Haar fwt-ifwt tests ----------------
def test_fwt_ifwt_level_1():
    wavelet = pywt.Wavelet("haar")
    data2 = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
                      11., 12., 13., 14., 15., 16.])

    # level 1
    coeffs = pywt.dwt(data2, wavelet)
    print(coeffs[0], coeffs[1])
    pt_data = torch.from_numpy(data2).unsqueeze(0)
    coeffsmat1, _ = matrix_wavedec(pt_data, wavelet, 1)
    err1 = np.mean(np.abs(coeffs[0] - coeffsmat1[0].squeeze().numpy()))
    err2 = np.mean(np.abs(coeffs[1] - coeffsmat1[1].squeeze().numpy()))
    print(err1 < 0.00001, err2 < 0.00001)
    assert err1 < 1e-6
    assert err2 < 1e-6


def test_fwt_ifwt_level_2():
    wavelet = pywt.Wavelet("haar")
    data2 = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
                      11., 12., 13., 14., 15., 16.])
    pt_data = torch.from_numpy(data2).unsqueeze(0)

    coeffs2 = pywt.wavedec(data2, wavelet, level=2)
    coeffsmat2, _ = matrix_wavedec(pt_data, wavelet, 2)

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
    wavelet = pywt.Wavelet("haar")
    data2 = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
                      11., 12., 13., 14., 15., 16.])
    pt_data = torch.from_numpy(data2.astype(np.float32)).unsqueeze(0)
    coeffs3 = pywt.wavedec(data2, wavelet, level=3)
    print('stop')
    coeffsmat3, mat_3_lst = matrix_wavedec(pt_data, wavelet, 3)

    err1 = np.mean(np.abs(coeffs3[0] - coeffsmat3[0].squeeze().numpy()))
    err2 = np.mean(np.abs(coeffs3[1] - coeffsmat3[1].squeeze().numpy()))
    err3 = np.mean(np.abs(coeffs3[2] - coeffsmat3[2].squeeze().numpy()))
    err4 = np.mean(np.abs(coeffs3[3] - coeffsmat3[3].squeeze().numpy()))
    print(err1 < 1e-6, err2 < 1e-6, err3 < 1e-6, err4 < 1e-6)

    assert err1 < 1e-6
    assert err2 < 1e-6
    assert err3 < 1e-6
    assert err4 < 1e-6

    reconstructed_data, ifwt_mat_3_lst = matrix_waverec(coeffsmat3, wavelet, 3)
    err5 = torch.mean(torch.abs(pt_data - reconstructed_data))
    print("abs ifwt 3 reconstruction error", err5)
    assert err5 < 1e-6


@pytest.mark.slow
def test_fwt_ifwt_mackey_haar():
    wavelet = pywt.Wavelet("haar")
    generator = MackeyGenerator(
        batch_size=2, tmax=512, delta_t=1
    )
    wavelet = pywt.Wavelet("haar")
    pt_data = torch.squeeze(generator()).cpu()
    numpy_data = pt_data.cpu().numpy()
    pywt_start = time.time()
    coeffs_max = pywt.wavedec(numpy_data, wavelet, level=9)
    time_pywt = time.time() - pywt_start
    sparse_fwt_start = time.time()
    coeffs_mat_max, mat_lst = matrix_wavedec(pt_data, wavelet, 9)
    time_sparse_fwt = time.time() - sparse_fwt_start

    test_lst = []
    for test_no in range(9):
        test_lst.append(
            np.sum(np.abs(coeffs_max[test_no] - coeffs_mat_max[test_no].T.numpy()))
            < 0.001
        )
    print(test_lst)
    print("time pywt", time_pywt)
    print("time_sparse_wt", time_sparse_fwt)

    # test the inverse fwt.
    reconstructed_data, ifwt_mat_lst = matrix_waverec(coeffs_mat_max, wavelet, 9)
    err1 = torch.mean(torch.abs(pt_data - reconstructed_data))
    print("abs ifwt reconstruction error", err1)
    assert err1 < 1e-6

    operator_mat = mat_lst[0].to_dense()
    inv_operator_mat = ifwt_mat_lst[0].to_dense()
    for mat_no in range(1, len(mat_lst)):  # mat_lst[1:]:
        operator_mat = torch.sparse.mm(mat_lst[mat_no], operator_mat)
        inv_operator_mat = torch.sparse.mm(ifwt_mat_lst[mat_no], inv_operator_mat)

    coeff = torch.sparse.mm(operator_mat, pt_data.T)
    data_rec = torch.sparse.mm(inv_operator_mat, coeff).T
    err2 = np.mean(np.abs(data_rec.numpy() - numpy_data))
    print("abs. reconstruction error:", err2)
    assert err2 < 1e-6


# ------------ db2 fwt-ifwt tests --------------------------------------
@pytest.mark.slow
def test_fwt_ifwt_mackey_db2():
    wavelet = pywt.Wavelet("db2")
    generator = MackeyGenerator(
        batch_size=2, tmax=512, delta_t=1
    )
    pt_data = torch.squeeze(generator()).cpu()
    coeffs_mat_max, mat_lst = matrix_wavedec(pt_data, wavelet, 4)

    reconstructed_data, ifwt_mat = matrix_waverec(coeffs_mat_max, wavelet, 4)
    err = torch.mean(torch.abs(pt_data - reconstructed_data))
    print("reconstruction error:", err)
    assert err < 1e-6
