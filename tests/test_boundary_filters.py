# Created by moritz ( wolter@cs.uni-bonn.de ), 08.09.21
import pywt
import torch
import pytest
import numpy as np
import scipy.signal
from scipy import misc
import matplotlib.pyplot as plt
from torch._C import dtype
from torch.autograd import backward

from src.ptwt.matmul_transform import (
    construct_boundary_a,
    construct_boundary_s,
    matrix_wavedec,
    matrix_waverec
)

from src.ptwt.matmul_transform_2d import (
    construct_conv_matrix,
    construct_conv2d_matrix,
    construct_strided_conv2d_matrix
)

from src.ptwt.mackey_glass import MackeyGenerator


@pytest.mark.slow
def test_boundary_filter_analysis_and_synthethis_matrices():
    for size in [24, 64, 128, 256]:
        for wavelet in [pywt.Wavelet("db2"), pywt.Wavelet("db4"),
                        pywt.Wavelet("db6"), pywt.Wavelet("db8")]:
            analysis_matrix = construct_boundary_a(wavelet, size,
                boundary='gramschmidt').to_dense()
            synthesis_matrix = construct_boundary_s(wavelet, size,
                boundary='gramschmidt').to_dense()
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


def test_boundary_transform_1d():
    data_list = [np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]),
                 np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])]
    wavelet_list = ['db2', 'haar', 'db3']
    for data in data_list:
        for wavelet_str in wavelet_list:
            for level in [1, 2]:
                for boundary in ['gramschmidt', 'circular']:
                    data_torch = torch.from_numpy(data.astype(np.float64))
                    wavelet = pywt.Wavelet(wavelet_str)
                    coeffs, _ = matrix_wavedec(
                        data_torch, wavelet, level=level, boundary=boundary)
                    rec, _ = matrix_waverec(
                        coeffs, wavelet, level=level, boundary=boundary)
                    rec_pywt = pywt.waverec(
                        pywt.wavedec(data_torch.numpy(), wavelet), wavelet)
                    error = np.sum(np.abs(rec_pywt - rec.numpy()))
                    print('wavelet: {},'.format(wavelet_str),
                          'level: {},'.format(level),
                          'shape: {},'.format(data.shape[-1]),
                          'error {:2.2e}'.format(error))
                    assert np.allclose(rec.numpy(), rec_pywt, atol=1e-05)


def test_boundary_transform_2d():
    pass


def test_conv_matrix():
    # Test the 1d sparse convolution matrix code.

    # test_filters = [torch.tensor([1., 2, 3, 4]), torch.tensor([1., 2, 3])]
    # input_signals = [torch.tensor([1., 2, 3, 4, 5, 6, 7, 8, 9]),
    #                  torch.tensor([1., 2, 3, 4, 5, 6, 7, 8])]
    test_filters = [torch.rand([2]), torch.rand([3]), torch.rand([4])]
    input_signals = [torch.rand([8]), torch.rand([9])]
    for h in test_filters:
        for x in input_signals:

            def test_padding_case(case: str):
                conv_matrix = construct_conv_matrix(h, len(x), case)
                mm_conv_res = torch.sparse.mm(
                    conv_matrix, x.unsqueeze(-1)).squeeze()
                conv_res = scipy.signal.convolve(
                    x.numpy(), h.numpy(), case)
                error = np.sum(
                    np.abs(conv_res - mm_conv_res.numpy()))
                print('1d conv matrix error', case, error, len(h), len(x))
                assert np.allclose(conv_res, mm_conv_res.numpy())

            test_padding_case('full')
            test_padding_case('same')
            test_padding_case('valid')


def test_conv_matrix_2d():
    """ Test the validity of the 2d convolution matrix code.
        It should be equivalent to signal convolve2d.
    """
    for filter_shape in [(2, 2), (3, 3), (3, 2), (2, 3), (5, 3), (3, 5),
                         (2, 5), (5, 2)]:
        for size in [(5, 5), (16, 16), (8, 16), (16, 8), (16, 7), (7, 16),
                     (15, 15)]:
            for mode in ['same', 'full', 'valid']:
                filter = torch.rand(filter_shape)
                filter = filter.unsqueeze(0).unsqueeze(0)
                face = misc.face()[256:(256+size[0]), 256:(256+size[1])]
                face = np.mean(face, -1).astype(np.float32)
                res_scipy = scipy.signal.convolve2d(
                    face, filter.squeeze().numpy(), mode=mode)

                face = torch.from_numpy(face)
                face = face.unsqueeze(0).unsqueeze(0)
                conv_matrix2d = construct_conv2d_matrix(
                    filter.squeeze(), size[0], size[1], mode=mode)
                res_flat = torch.sparse.mm(
                    conv_matrix2d, face.T.flatten().unsqueeze(-1))
                res_mm = torch.reshape(res_flat, [res_scipy.shape[1],
                                                  res_scipy.shape[0]]).T

                diff_scipy = np.mean(np.abs(res_scipy - res_mm.numpy()))
                print(str(size).center(8), filter_shape, mode.center(5),
                      'scipy-error %2.2e' % diff_scipy,
                      np.allclose(res_scipy, res_mm.numpy()))
                assert np.allclose(res_scipy, res_mm)


@pytest.mark.slow
def test_strided_conv_matrix_2d():
    """ Test strided convolution matrices with full and valid padding."""
    for filter_shape in [(3, 3), (2, 2), (4, 4), (3, 2), (2, 3)]:
        for size in [(14, 14), (8, 16), (16, 8),
                     (17, 8), (8, 17), (7, 7), (7, 8), (8, 7)]:
            for mode in ['full', 'valid']:
                filter = torch.rand(filter_shape)
                filter = filter.unsqueeze(0).unsqueeze(0)
                face = misc.face()[256:(256+size[0]), 256:(256+size[1])]
                face = np.mean(face, -1)
                face = torch.from_numpy(face.astype(np.float32))
                face = face.unsqueeze(0).unsqueeze(0)

                if mode == 'full':
                    padding = (filter_shape[0]-1, filter_shape[1]-1)
                elif mode == 'valid':
                    padding = (0, 0)
                torch_res = torch.nn.functional.conv2d(
                    face, filter.flip(2, 3),
                    padding=padding,
                    stride=2).squeeze()

                strided_matrix = construct_strided_conv2d_matrix(
                    filter.squeeze(), size[0], size[1], stride=2, mode=mode)
                res_flat_stride = torch.sparse.mm(
                    strided_matrix, face.T.flatten().unsqueeze(-1))

                if mode == 'full':
                    output_shape = \
                        [int(np.ceil((filter_shape[1] + size[1] - 1) / 2)),
                         int(np.ceil((filter_shape[0] + size[0] - 1) / 2))]
                elif mode == 'valid':
                    output_shape = \
                        [(size[1] - (filter_shape[1])) // 2 + 1,
                         (size[0] - (filter_shape[0])) // 2 + 1]
                res_mm_stride = np.reshape(
                    res_flat_stride, output_shape).T

                diff_torch = np.mean(np.abs(torch_res.numpy()
                                            - res_mm_stride.numpy()))
                print(str(size).center(8), filter_shape,  mode.center(8),
                      'torch-error %2.2e' % diff_torch,
                      np.allclose(torch_res.numpy(), res_mm_stride.numpy()))
                assert np.allclose(torch_res.numpy(),
                                   res_mm_stride.numpy())


def test_strided_conv_matrix_2d_same():
    """ Test strided conv matrix with same padding. """
    for filter_shape in [(3, 3), (4, 4), (4, 3), (3, 4)]:
        for size in [(7, 8), (8, 7), (7, 7), (8, 8), (16, 16),
                     (8, 16), (16, 8)]:
            stride = 2
            filter = torch.rand(filter_shape)
            filter = filter.unsqueeze(0).unsqueeze(0)
            face = misc.face()[256:(256+size[0]), 256:(256+size[1])]
            face = np.mean(face, -1)
            face = torch.from_numpy(face.astype(np.float32))
            face = face.unsqueeze(0).unsqueeze(0)
            padding = get_2d_same_padding(filter_shape, size)
            face_pad = torch.nn.functional.pad(face, padding)
            torch_res = torch.nn.functional.conv2d(
                face_pad, filter.flip(2, 3),
                stride=stride).squeeze()
            strided_matrix = construct_strided_conv2d_matrix(
                filter.squeeze(), face.shape[-2],
                face.shape[-1], stride=stride, mode='same')
            res_flat_stride = torch.sparse.mm(
                strided_matrix, face.T.flatten().unsqueeze(-1))
            output_shape = torch_res.shape
            res_mm_stride = np.reshape(
                res_flat_stride, (output_shape[1], output_shape[0])).T
            diff_torch = np.mean(np.abs(torch_res.numpy()
                                        - res_mm_stride.numpy()))
            print(str(size).center(8), filter_shape, tuple(output_shape),
                  'torch-error %2.2e' % diff_torch,
                  np.allclose(torch_res.numpy(), res_mm_stride.numpy()))

            # diff = np.abs(torch_res - res_mm_stride.numpy())
            # plot = np.concatenate([torch_res.numpy(),
            #                     res_mm_stride.numpy(),
            #                     diff], -1)
            # plt.imshow(plot)
            # plt.show()
            # print('stop')


def get_2d_same_padding(filter_shape, input_size):
    height_offset = input_size[0] % 2
    width_offset = input_size[1] % 2
    padding = (filter_shape[1] // 2,
               filter_shape[1] // 2 - 1 + width_offset,
               filter_shape[0] // 2,
               filter_shape[0] // 2 - 1 + height_offset)
    return padding


if __name__ == '__main__':
    test_boundary_filter_analysis_and_synthethis_matrices()
    test_conv_matrix()
    test_conv_matrix_2d()
    # test_strided_conv_matrix_2d_same()
