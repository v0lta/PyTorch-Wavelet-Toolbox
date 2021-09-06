import pywt
import time
import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt

from src.ptwt.matmul_transform import (
    construct_a,
    construct_s,
    construct_boundary_a,
    construct_boundary_s
)

from src.ptwt.mackey_glass import MackeyGenerator

@pytest.mark.slow
def test_boundary_filter_analysis_and_synthethis_matrices():
    for size in [24, 64, 128, 256]:
        for wavelet in [pywt.Wavelet("db4"), pywt.Wavelet("db6"), pywt.Wavelet("db8")]:
            analysis_matrix = construct_boundary_a(wavelet, size)
            synthesis_matrix = construct_boundary_s(wavelet, size)
            # s_db2 = construct_s(pywt.Wavelet("db8"), size)
            
            # test_eye_inv = torch.sparse.mm(a_db8, s_db2.to_dense()).numpy()
            test_eye_orth = torch.mm(analysis_matrix.transpose(1, 0), analysis_matrix).numpy()
            test_eye_inv = torch.mm(analysis_matrix, synthesis_matrix).numpy()
            err_inv = np.mean(np.abs(test_eye_inv - np.eye(size)))
            err_orth = np.mean(np.abs(test_eye_orth - np.eye(size)))

            print(wavelet.name, "orthogonal error", err_orth, 'size', size)
            print(wavelet.name, "orthogonal error", err_inv,  'size', size)
            assert err_orth < 1e-8
            assert err_inv < 1e-8



if __name__ == '__main__':
    # test_cyclic_analysis_and_synthethis_matrices_db8()
    test_boundary_filter_analysis_and_synthethis_matrices()
    print('stop')
