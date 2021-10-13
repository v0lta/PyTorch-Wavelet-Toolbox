# Created by moritz ( wolter@cs.uni-bonn.de ), 08.09.21
import pywt
import torch
import pytest
import numpy as np
import scipy.signal
# from scipy import misc


from src.ptwt.matmul_transform import (
    construct_boundary_a,
    construct_boundary_s,
    matrix_wavedec,
    matrix_waverec,
)

from src.ptwt.conv_transform import flatten_2d_coeff_lst

from src.ptwt.matmul_transform_2d import (
    construct_boundary_a2d,
    construct_boundary_s2d,
    MatrixWavedec2d,
    MatrixWaverec2d
)


@pytest.mark.slow
def test_boundary_filter_analysis_and_synthethis_matrices():
    """ check 1d the 1d-fwt matrices for orthogonality and invertability. """
    for size in [24, 64, 128, 256]:
        for wavelet in [pywt.Wavelet("db2"), pywt.Wavelet("db4"),
                        pywt.Wavelet("db6"), pywt.Wavelet("db8")]:
            analysis_matrix = construct_boundary_a(
                wavelet, size,
                boundary='gramschmidt').to_dense()
            synthesis_matrix = construct_boundary_s(
                wavelet, size,
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
    """ ensure matrix fwt reconstructions are pywt compatible. """
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


def test_analysis_synthesis_matrices():
    """ test the 2d analysis and synthesis matrices for various wavelets. """
    for size in [(16, 16), (16, 8), (8, 16)]:
        for wavelet_str in ['db1', 'db2', 'db3', 'db4', 'db5']:
            wavelet = pywt.Wavelet(wavelet_str)
            a = construct_boundary_a2d(wavelet, size[0], size[1],
                                       device=torch.device('cpu'),
                                       dtype=torch.float64)
            s = construct_boundary_s2d(wavelet, size[0], size[1],
                                       device=torch.device('cpu'),
                                       dtype=torch.float64)
            test_inv = torch.sparse.mm(s, a)
            assert test_inv.shape[0] == test_inv.shape[1], \
                'the diagonal matrix must be square.'
            test_eye = torch.eye(test_inv.shape[0])
            err_mat = test_eye - test_inv
            err = torch.sum(torch.abs(err_mat.flatten()))
            print(size, wavelet_str, err.item(),
                  np.allclose(test_inv.to_dense().numpy(), test_eye.numpy()))


@pytest.mark.slow
def test_matrix_analysis_fwt_2d_haar():
    """ Test the fwt-2d matrix-haar transform,
        the coefficients should be equal to the pywt result. """
    for size in ((15, 16), (16, 15), (16, 16)):
        for level in (1, 2, 3):
            face = np.mean(scipy.misc.face()[256:(256+size[0]),
                                             256:(256+size[1])],
                           -1).astype(np.float64)
            pt_face = torch.tensor(face)
            wavelet = pywt.Wavelet("haar")
            matrixfwt = MatrixWavedec2d(wavelet, level=level)
            mat_coeff = matrixfwt(pt_face.unsqueeze(0))
            conv_coeff = pywt.wavedec2(face, wavelet, level=level,
                                       mode='zero')
            flat_mat_coeff = torch.cat(flatten_2d_coeff_lst(mat_coeff), -1)
            flat_conv_coeff = np.concatenate(
                flatten_2d_coeff_lst(conv_coeff), -1)

            err = np.sum(np.abs(flat_mat_coeff.numpy() - flat_conv_coeff))
            test = np.allclose(flat_mat_coeff.numpy(), flat_conv_coeff)
            test2 = np.allclose(mat_coeff[0].numpy(), conv_coeff[0])
            test3 = np.allclose(mat_coeff[1][0].numpy(), conv_coeff[1][0])
            print(size, level, err, test, test2, test3)
            assert test and test2 and test3


@pytest.mark.slow
def test_boundary_matrix_fwt_2d():
    """ Ensure the boundary matrix fwt is invertable."""
    for wavelet_str in ('haar', 'db2', 'db3', 'db4'):
        for level in (1, 2, 3, 4, None):
            for size in ((16, 16), (15, 15), (16, 15), (15, 16)):
                face = np.mean(scipy.misc.face()[256:(256+size[0]),
                                                 256:(256+size[1])],
                               -1).astype(np.float64)
                pt_face = torch.tensor(face)
                wavelet = pywt.Wavelet(wavelet_str)
                matrixfwt = MatrixWavedec2d(wavelet, level=level)
                mat_coeff = matrixfwt(pt_face.unsqueeze(0))
                matrixifwt = MatrixWaverec2d(wavelet)
                reconstruction = matrixifwt(mat_coeff).squeeze(0)
                # remove the padding
                if size[0] % 2 != 0:
                    reconstruction = reconstruction[:-1, :]
                if size[1] % 2 != 0:
                    reconstruction = reconstruction[:, :-1]
                err = np.sum(np.abs(reconstruction.numpy() - face))
                print(size, str(level).center(4),
                      wavelet_str, "error {:3.3e}".format(err),
                      np.allclose(reconstruction.numpy(), face))
                assert np.allclose(reconstruction.numpy(), face)


def test_batched_2d_matrix_fwt_ifwt():
    """ Ensure the batched matrix fwt works properly."""
    wavelet_str = 'db2'
    level = 3
    size = (16, 16)
    face = np.mean(scipy.misc.face()[256:(256+size[0]),
                                     256:(256+size[1])],
                   -1).astype(np.float64)
    pt_face = torch.stack([torch.tensor(face),
                           torch.tensor(face),
                           torch.tensor(face)])
    wavelet = pywt.Wavelet(wavelet_str)
    matrixfwt = MatrixWavedec2d(wavelet, level=level)
    mat_coeff = matrixfwt(pt_face)
    matrixifwt = MatrixWaverec2d(wavelet)
    reconstruction = matrixifwt(mat_coeff)
    err = np.sum(np.abs(reconstruction[0].numpy() - face
                        + reconstruction[1].numpy() - face
                        + reconstruction[2].numpy() - face))
    print(size, str(level).center(4),
          wavelet_str, "error {:3.3e}".format(err),
          np.allclose(reconstruction.numpy(), face))
    assert np.allclose(reconstruction[0].numpy(), face) \
        and np.allclose(reconstruction[1].numpy(), face) \
        and np.allclose(reconstruction[2].numpy(), face)


if __name__ == '__main__':
    # test_matrix_analysis_fwt_2d_haar()
    test_boundary_filter_analysis_and_synthethis_matrices()
    # test_conv_matrix()
    # test_conv_matrix_2d()
    # test_strided_conv_matrix_2d_same()
    # test_batched_2d_matrix_fwt_ifwt()
    test_analysis_synthesis_matrices()
