import pywt
import time
import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt

from src.ptwt.matmul_transform import (
    construct_a,
    construct_s,
    construct_boundary_a
)

from src.ptwt.mackey_glass import MackeyGenerator


def test_cyclic_analysis_and_synthethis_matrices_db8():
    a_db8 = construct_a(pywt.Wavelet("db8"), 128)
    s_db8 = construct_s(pywt.Wavelet("db8"), 128)
    
    test_eye_inv = torch.sparse.mm(a_db8, s_db8.to_dense()).numpy()
    test_eye_orth = torch.sparse.mm(a_db8.transpose(1, 0), a_db8.to_dense()).numpy()
    err_inv = np.mean(np.abs(test_eye_inv - np.eye(128)))
    err_orth = np.mean(np.abs(test_eye_orth - np.eye(128)))

    print("db8 128 orthogonal error", err_orth)
    print("db8 128 inverse error", err_inv)
    assert err_orth < 1e-6
    assert err_inv < 1e-6

def test_boundary_filter_analysis_and_synthethis_matrices_db8():
    size = 16
    analysis_matrix = construct_boundary_a(pywt.Wavelet("db4"), size)
    # s_db2 = construct_s(pywt.Wavelet("db8"), size)
    
    # test_eye_inv = torch.sparse.mm(a_db8, s_db2.to_dense()).numpy()
    test_eye_orth = torch.mm(analysis_matrix.transpose(1, 0), analysis_matrix).numpy()
    # err_inv = np.mean(np.abs(test_eye_inv - np.eye(size)))
    err_orth = np.mean(np.abs(test_eye_orth - np.eye(size)))

    print("db4 size orthogonal error", err_orth)
    # print("db8 size inverse error", err_inv)
    assert err_orth < 1e-6
    # assert err_inv < 1e-6



if __name__ == '__main__':
    # test_cyclic_analysis_and_synthethis_matrices_db8()
    test_boundary_filter_analysis_and_synthethis_matrices_db8()
    print('stop')
