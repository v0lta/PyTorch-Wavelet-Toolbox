# Created by moritz ( wolter@cs.uni-bonn.de ), 08.09.21
import pywt
import torch
import pytest
import numpy as np
import scipy.signal
from scipy import misc
import matplotlib.pyplot as plt

from src.ptwt.matmul_transform import (
    construct_boundary_a,
    construct_boundary_s
)

from src.ptwt.matmul_transform_2d import construct_conv2d_matrix
from src.ptwt.matmul_transform_2d import construct_strided_conv2d_matrix
from src.ptwt.mackey_glass import MackeyGenerator


@pytest.mark.slow
def test_boundary_filter_analysis_and_synthethis_matrices():
    for size in [24, 64, 128, 256]:
        for wavelet in [pywt.Wavelet("db4"),
                        pywt.Wavelet("db6"), pywt.Wavelet("db8")]:
            analysis_matrix = construct_boundary_a(wavelet, size).to_dense()
            synthesis_matrix = construct_boundary_s(wavelet, size).to_dense()
            # s_db2 = construct_s(pywt.Wavelet("db8"), size)
            # test_eye_inv = torch.sparse.mm(a_db8, s_db2.to_dense()).numpy()
            test_eye_orth = torch.mm(analysis_matrix.transpose(1, 0),
                                     analysis_matrix).numpy()
            test_eye_inv = torch.mm(analysis_matrix, synthesis_matrix).numpy()
            err_inv = np.mean(np.abs(test_eye_inv - np.eye(size)))
            err_orth = np.mean(np.abs(test_eye_orth - np.eye(size)))

            print(wavelet.name, "orthogonal error", err_orth, 'size', size)
            print(wavelet.name, "inverse error", err_inv,  'size', size)
            assert err_orth < 1e-8
            assert err_inv < 1e-8


def test_conv_matrix_2d():
    """ Test the validity of the 2d convolution matrix code.
        It should be equivalent to signal convolve2d as well
        as torch.nn.functional.conv2d .
    """
    for filter_shape in [(3, 3), (3, 2), (2, 3), (5, 3), (3, 5),
                         (2, 5), (5, 2)]:
        for size in [(64, 64), (32, 64), (64, 32), (64, 31), (31, 64),
                     (65, 65)]:
            filter = torch.rand(filter_shape)
            filter = filter.unsqueeze(0).unsqueeze(0)
            face = misc.face()[256:(256+size[0]), 256:(256+size[1])]
            face = np.mean(face, -1)

            res_scipy = scipy.signal.convolve2d(face, filter.squeeze().numpy())
            conv_matrix2d = construct_conv2d_matrix(
                filter.squeeze(), size[0], size[1], torch.float32)

            face = torch.from_numpy(face.astype(np.float32))
            face = face.unsqueeze(0).unsqueeze(0)
            res_flat = torch.sparse.mm(
                conv_matrix2d, face.T.flatten().unsqueeze(-1))
            res_mm = torch.reshape(res_flat,
                                   [filter_shape[1] + size[1] - 1,
                                    filter_shape[0] + size[0] - 1]).T
            res_torch = torch.nn.functional.conv2d(
                face, filter.flip(2, 3),
                padding=(filter_shape[0]-1, filter_shape[1]-1))

            diff_scipy = np.mean(np.abs(res_scipy - res_mm.numpy()))
            diff_torch = np.mean(np.abs(res_torch.numpy() - res_mm.numpy()))

            print(size, filter_shape, 'scipy-error %2.2e' % diff_scipy,
                  np.allclose(res_scipy, res_mm.numpy()),
                  'torch-error %2.2e' % diff_torch, np.allclose(
                      res_torch.numpy(), res_mm.numpy()))
            assert np.allclose(res_scipy, res_mm)
            assert np.allclose(res_torch.numpy(), res_mm.numpy())


def test_strided_conv_matrix_2d():
    for filter_shape in [(3, 3)]:
        for size in [(64, 64)]:
            filter = torch.rand(filter_shape)
            filter = filter.unsqueeze(0).unsqueeze(0)
            face = misc.face()[256:(256+size[0]), 256:(256+size[1])]
            face = np.mean(face, -1)
            face = torch.from_numpy(face.astype(np.float32))
            face = face.unsqueeze(0).unsqueeze(0)

            torch_res = torch.nn.functional.conv2d(
                face, filter.flip(2, 3), padding=2, stride=2)

            strided_matrix = construct_strided_conv2d_matrix(
                filter.squeeze(),
                size[0], size[1], stride=2, dtype=torch.float32)
            res_flat_stride = torch.mm(strided_matrix.to_dense(),
                                       face.flatten().unsqueeze(-1))
            res_mm_stride = np.reshape(
                res_flat_stride,
                [(filter_shape[1] + size[1] - 1) // 2,
                 (filter_shape[0] + size[0] - 1) // 2]).T

            diff_torch = np.mean(np.abs(torch_res.numpy()
                                        - res_mm_stride.numpy()))

            print(size, filter_shape, 'torch-error %2.2e' % diff_torch,
                  np.allclose(torch_res.numpy(), res_mm_stride.numpy()))
            # assert np.allclose(res_torch.numpy(), res_mm.numpy())


if __name__ == '__main__':
    # test_conv_matrix_2d()
    test_strided_conv_matrix_2d()

    filter_shape = [3, 3]
    size = (64, 64)
    filter = torch.rand(filter_shape)
    filter = filter.unsqueeze(0).unsqueeze(0)
    face = misc.face()[256:(256+size[0]), 256:(256+size[1])]
    face = np.mean(face, -1)
    face = torch.from_numpy(face.astype(np.float32))
    face = face.unsqueeze(0).unsqueeze(0)

    torch_res = torch.nn.functional.conv2d(
        face, filter.flip(2, 3), padding=2, stride=2)

    conv_matrix2d = construct_conv2d_matrix(
        filter.squeeze(), size[0], size[1], torch.float32)
    strided_matrix = construct_strided_conv2d_matrix(
        filter.squeeze(),
        size[0], size[1], stride=2, dtype=torch.float32)
    res_flat_stride = torch.mm(strided_matrix.to_dense(),
                               face.flatten().unsqueeze(-1))
    res_mm_stride = np.reshape(res_flat_stride,
                        [(filter_shape[1] + size[1] - 1) // 2,
                         (filter_shape[0] + size[0] - 1) // 2]).T

    diff = torch.abs(torch_res.squeeze() - res_mm_stride) 
    to_plot = torch.cat([torch_res.squeeze(), res_mm_stride, diff], -1)
    print(np.allclose(torch_res.numpy(), res_mm_stride.numpy()))
    plt.imshow(to_plot.numpy())
    plt.show()

    print('stop')

    # test_conv_matrix_2d()

