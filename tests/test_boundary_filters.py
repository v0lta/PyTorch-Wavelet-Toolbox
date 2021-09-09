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



def test_mean_conv_matrix_2d():

    for filter_shape in [(3,3), (3,2), (2,3), (5,3), (3,5), (2,5), (5,2)]:
        for size in [(64, 64), (32, 64), (64, 32), (64, 31), (31, 64), (65, 65)]:
            # size = (256, 256)
            # filter = torch.ones(filter_shape)/9.
            filter = torch.rand(filter_shape)
            filter = filter.unsqueeze(0).unsqueeze(0)
            face = misc.face()[256:(256+size[0]), 256:(256+size[1])]
            face = np.mean(face, -1)

            # res = torch.nn.functional.conv2d(input=face, weight=filter)
            res = scipy.signal.convolve2d(face, filter.squeeze().numpy())

            conv_matrix2d = construct_conv2d_matrix(filter.squeeze(), size[0], size[1], torch.float32)

            # plt.imshow(conv_matrix2d.to_dense().numpy())
            # plt.show()
            face = torch.from_numpy(face.astype(np.float32))
            face = face.unsqueeze(0).unsqueeze(0)
            res_flat = torch.sparse.mm(conv_matrix2d, face.T.flatten().unsqueeze(-1))
            # res_mm = np.reshape(res_flat.numpy(),
            #     [filter_shape[0] + size[0] - 1, filter_shape[1] + size[1] - 1], order='F')
            res_mm = torch.reshape(res_flat,
                  [filter_shape[1] + size[1] - 1, filter_shape[0] + size[0] - 1]).T


            diff = np.mean(np.abs(res - res_mm.numpy()))
            print(size, filter_shape, '%2.2e.'%diff, np.allclose(res, res_mm.numpy()))
            assert np.allclose(res, res_mm)



if __name__ == '__main__':
    import torch
    import scipy.signal

    filter_shape = [3,3]
    size = (5, 5)
    # filter = torch.ones(filter_shape)/9.
    filter = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
    # filter = torch.rand(filter_shape)
    filter = filter.unsqueeze(0).unsqueeze(0)
    face = misc.face()[256:(256+size[0]), 256:(256+size[1])]
    face = np.mean(face, -1)

    # res = torch.nn.functional.conv2d(input=face, weight=filter)
    res = scipy.signal.convolve2d(face, filter.squeeze().numpy())

    conv_matrix2d = construct_conv2d_matrix(filter.squeeze(), size[0], size[1], torch.float32)

    # plt.spy(conv_matrix2d.to_dense().numpy(), marker='.')
    # plt.imshow(conv_matrix2d.to_dense().numpy())
    # plt.show()
    face = torch.from_numpy(face.astype(np.float32))
    face = face.unsqueeze(0).unsqueeze(0)
    res_flat = torch.sparse.mm(conv_matrix2d, face.T.flatten().unsqueeze(-1))
    res_mm = np.reshape(res_flat,
        [filter_shape[0] + size[0] - 1, filter_shape[1] + size[1] - 1]).T

    diff = np.abs(res - res_mm.numpy())
    print(np.mean(diff), np.allclose(res, res_mm.numpy()))
    plot = np.concatenate([res, res_mm.numpy(), diff], -1)
    # plt.imshow(plot)
    # plt.show()

    test_mean_conv_matrix_2d()