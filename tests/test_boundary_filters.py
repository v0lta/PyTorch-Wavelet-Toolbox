"""Test code for the boundary wavelets."""
# Created by moritz ( wolter@cs.uni-bonn.de ), 08.09.21
import numpy as np
import pytest
import pywt
import scipy.signal
import torch
from src.ptwt.conv_transform import flatten_2d_coeff_lst
from src.ptwt.matmul_transform import (
    MatrixWavedec,
    MatrixWaverec,
    construct_boundary_a,
    construct_boundary_s,
)
from src.ptwt.matmul_transform_2d import (
    MatrixWavedec2d,
    MatrixWaverec2d,
    construct_boundary_a2d,
    construct_boundary_s2d,
)


@pytest.mark.slow
@pytest.mark.parametrize("size", [24, 64, 128, 256])
@pytest.mark.parametrize(
    "wavelet",
    [
        pywt.Wavelet("db2"),
        pywt.Wavelet("db4"),
        pywt.Wavelet("db6"),
        pywt.Wavelet("db8"),
    ],
)
def test_boundary_filter_analysis_and_synthethis_matrices(size, wavelet):
    """Check 1d the 1d-fwt matrices for orthogonality and invertability."""
    analysis_matrix = construct_boundary_a(
        wavelet, size, boundary="gramschmidt"
    ).to_dense()
    synthesis_matrix = construct_boundary_s(
        wavelet, size, boundary="gramschmidt"
    ).to_dense()
    # s_db2 = construct_s(pywt.Wavelet("db8"), size)
    # test_eye_inv = torch.sparse.mm(a_db8, s_db2.to_dense()).numpy()
    test_eye_orth = torch.mm(analysis_matrix.transpose(1, 0), analysis_matrix).numpy()
    test_eye_inv = torch.mm(analysis_matrix, synthesis_matrix).numpy()
    err_inv = np.mean(np.abs(test_eye_inv - np.eye(size)))
    err_orth = np.mean(np.abs(test_eye_orth - np.eye(size)))
    print(wavelet.name, "orthogonal error", err_orth, "size", size)
    print(wavelet.name, "inverse error", err_inv, "size", size)
    assert err_orth < 1e-8
    assert err_inv < 1e-8


@pytest.mark.parametrize("wavelet_str", ["db2", "db3", "haar"])
@pytest.mark.parametrize(
    "data",
    [
        np.random.randn(32),
        np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]),
        np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]),
        np.random.randn(18),
        np.random.randn(19),
    ],
)
@pytest.mark.parametrize("level", [2, 1])
@pytest.mark.parametrize("boundary", ["gramschmidt", "qr"])
def test_boundary_transform_1d(wavelet_str, data, level, boundary):
    """Ensure matrix fwt reconstructions are pywt compatible."""
    data_torch = torch.from_numpy(data.astype(np.float64))
    wavelet = pywt.Wavelet(wavelet_str)
    matrix_wavedec = MatrixWavedec(wavelet, level=level, boundary=boundary)
    coeffs = matrix_wavedec(data_torch)
    matrix_waverec = MatrixWaverec(wavelet, boundary=boundary)
    rec = matrix_waverec(coeffs)
    rec_pywt = pywt.waverec(
        pywt.wavedec(data_torch.numpy(), wavelet, mode="zero"), wavelet
    )
    error = np.sum(np.abs(rec_pywt - rec.numpy()))
    print(
        "wavelet: {},".format(wavelet_str),
        "level: {},".format(level),
        "shape: {},".format(data.shape[-1]),
        "error {:2.2e}".format(error),
    )
    assert np.allclose(rec.numpy(), rec_pywt)
    # test the operator matrices
    if not matrix_wavedec.padded and not matrix_waverec.padded:
        test_mat = torch.sparse.mm(
            matrix_waverec.sparse_ifwt_operator,
            matrix_wavedec.sparse_fwt_operator,
        )
        assert np.allclose(test_mat.to_dense().numpy(), np.eye(test_mat.shape[0]))


@pytest.mark.parametrize("size", [(16, 16), (16, 8), (8, 16)])
@pytest.mark.parametrize("wavelet_str", ["db1", "db2", "db3", "db4", "db5"])
def test_analysis_synthesis_matrices(size, wavelet_str):
    """Test the 2d analysis and synthesis matrices for various wavelets."""
    wavelet = pywt.Wavelet(wavelet_str)
    a = construct_boundary_a2d(
        wavelet,
        size[0],
        size[1],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    s = construct_boundary_s2d(
        wavelet,
        size[0],
        size[1],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    test_inv = torch.sparse.mm(s, a)
    assert test_inv.shape[0] == test_inv.shape[1], "the diagonal matrix must be square."
    test_eye = torch.eye(test_inv.shape[0])
    err_mat = test_eye - test_inv
    err = torch.sum(torch.abs(err_mat.flatten()))
    print(
        size,
        wavelet_str,
        err.item(),
        np.allclose(test_inv.to_dense().numpy(), test_eye.numpy()),
    )


@pytest.mark.slow
@pytest.mark.parametrize("size", [(15, 16), (16, 15), (16, 16)])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_matrix_analysis_fwt_2d_haar(size, level):
    """Test the fwt-2d matrix-haar transform, should be equal to the pywt."""
    face = np.mean(
        scipy.misc.face()[256 : (256 + size[0]), 256 : (256 + size[1])], -1
    ).astype(np.float64)
    pt_face = torch.tensor(face)
    wavelet = pywt.Wavelet("haar")
    matrixfwt = MatrixWavedec2d(wavelet, level=level)
    mat_coeff = matrixfwt(pt_face.unsqueeze(0))
    conv_coeff = pywt.wavedec2(face, wavelet, level=level, mode="zero")
    flat_mat_coeff = torch.cat(flatten_2d_coeff_lst(mat_coeff), -1)
    flat_conv_coeff = np.concatenate(flatten_2d_coeff_lst(conv_coeff), -1)

    err = np.sum(np.abs(flat_mat_coeff.numpy() - flat_conv_coeff))
    test = np.allclose(flat_mat_coeff.numpy(), flat_conv_coeff)
    test2 = np.allclose(mat_coeff[0].numpy(), conv_coeff[0])
    test3 = np.allclose(mat_coeff[1][0].numpy(), conv_coeff[1][0])
    print(size, level, err, test, test2, test3)
    assert test and test2 and test3


@pytest.mark.slow
@pytest.mark.parametrize("wavelet_str", ["haar", "db2", "db3", "db4"])
@pytest.mark.parametrize(
    "size",
    [
        (25, 26),
        (26, 25),
        (25, 25),
        (16, 16),
        (15, 15),
        (16, 15),
        (15, 16),
    ],
)
@pytest.mark.parametrize("level", [4, 3, 2, 1, None])
@pytest.mark.parametrize("separable", [False, True])
def test_boundary_matrix_fwt_2d(wavelet_str, size, level, separable):
    """Ensure the boundary matrix fwt is invertable."""
    face = np.mean(
        scipy.misc.face()[256 : (256 + size[0]), 256 : (256 + size[1])], -1
    ).astype(np.float64)
    pt_face = torch.tensor(face)
    wavelet = pywt.Wavelet(wavelet_str)
    matrixfwt = MatrixWavedec2d(wavelet, level=level, separable=separable)
    mat_coeff = matrixfwt(pt_face.unsqueeze(0))
    matrixifwt = MatrixWaverec2d(wavelet, separable=separable)
    reconstruction = matrixifwt(mat_coeff).squeeze(0)
    # remove the padding
    if size[0] % 2 != 0:
        reconstruction = reconstruction[:-1, :]
    if size[1] % 2 != 0:
        reconstruction = reconstruction[:, :-1]
    err = np.sum(np.abs(reconstruction.numpy() - face))
    print(
        size,
        str(level).center(4),
        wavelet_str,
        "error {:3.3e}".format(err),
        np.allclose(reconstruction.numpy(), face),
    )
    assert np.allclose(reconstruction.numpy(), face)
    # test the operator matrices
    if not separable and not matrixfwt.padded and not matrixifwt.padded:
        test_mat = torch.sparse.mm(
            matrixifwt.sparse_ifwt_operator, matrixfwt.sparse_fwt_operator
        )
        assert np.allclose(test_mat.to_dense().numpy(), np.eye(test_mat.shape[0]))


@pytest.mark.parametrize("wavelet_str", ["db2"])
@pytest.mark.parametrize("level", [3])
@pytest.mark.parametrize("size", [(16, 16)])
@pytest.mark.parametrize("separable", [False, True])
def test_batched_2d_matrix_fwt_ifwt(wavelet_str, level, size, separable):
    """Ensure the batched matrix fwt works properly."""
    face = scipy.misc.face()[256 : (256 + size[0]), 256 : (256 + size[1])].astype(
        np.float64
    )
    pt_face = torch.tensor(face).permute([2, 0, 1])
    wavelet = pywt.Wavelet(wavelet_str)
    matrixfwt = MatrixWavedec2d(wavelet, level=level, separable=separable)
    mat_coeff = matrixfwt(pt_face)
    matrixifwt = MatrixWaverec2d(wavelet, separable=separable)
    reconstruction = matrixifwt(mat_coeff)
    err = np.sum(
        np.abs(
            reconstruction[0].numpy()
            - face[:, :, 0]
            + reconstruction[1].numpy()
            - face[:, :, 1]
            + reconstruction[2].numpy()
            - face[:, :, 2]
        )
    )
    print(
        size,
        str(level).center(4),
        wavelet_str,
        "error {:3.3e}".format(err),
        np.allclose(reconstruction.permute(1, 2, 0).numpy(), face),
    )
    assert (
        np.allclose(reconstruction[0].numpy(), face[:, :, 0])
        and np.allclose(reconstruction[1].numpy(), face[:, :, 1])
        and np.allclose(reconstruction[2].numpy(), face[:, :, 2])
    )


@pytest.mark.parametrize("wavelet_str", ["db2", "db3", "haar"])
@pytest.mark.parametrize("boundary", ["qr", "gramschmidt"])
def test_matrix_transform_1d_rebuild(wavelet_str, boundary):
    """Ensure matrix fwt reconstructions are pywt compatible."""
    data_list = [np.random.randn(15), np.random.randn(19)]
    wavelet = pywt.Wavelet(wavelet_str)
    matrix_waverec = MatrixWaverec(wavelet, boundary=boundary)
    for level in [2, 1]:
        matrix_wavedec = MatrixWavedec(wavelet, level=level, boundary=boundary)
        for data in data_list:
            data_torch = torch.from_numpy(data.astype(np.float64))
            coeffs = matrix_wavedec(data_torch)
            rec = matrix_waverec(coeffs)
            rec_pywt = pywt.waverec(
                pywt.wavedec(data_torch.numpy(), wavelet, mode="zero"), wavelet
            )
            error = np.sum(np.abs(rec_pywt - rec.numpy()))
            print(
                "wavelet: {},".format(wavelet_str),
                "level: {},".format(level),
                "shape: {},".format(data.shape[-1]),
                "error {:2.2e}".format(error),
            )
            assert np.allclose(rec.numpy(), rec_pywt)
            # test the operator matrices
            if not matrix_wavedec.padded and not matrix_waverec.padded:
                test_mat = torch.sparse.mm(
                    matrix_waverec.sparse_ifwt_operator,
                    matrix_wavedec.sparse_fwt_operator,
                )
                assert np.allclose(
                    test_mat.to_dense().numpy(), np.eye(test_mat.shape[0])
                )


@pytest.mark.slow
@pytest.mark.parametrize("wavelet_str", ["haar", "db4"])
@pytest.mark.parametrize("separable", [False, True])
def test_matrix_transform_2d_rebuild(wavelet_str, separable):
    """Ensure the boundary matrix fwt is invertable."""
    wavelet = pywt.Wavelet(wavelet_str)
    matrixifwt = MatrixWaverec2d(wavelet, separable=separable)
    for level in [4, 1, None]:
        matrixfwt = MatrixWavedec2d(wavelet, level=level, separable=separable)
        for size in [[16, 16], [11, 11]]:
            face = np.mean(
                scipy.misc.face()[256 : (256 + size[0]), 256 : (256 + size[1])], -1
            ).astype(np.float64)
            pt_face = torch.tensor(face)
            mat_coeff = matrixfwt(pt_face.unsqueeze(0))
            reconstruction = matrixifwt(mat_coeff).squeeze(0)
            # remove the padding
            if size[0] % 2 != 0:
                reconstruction = reconstruction[:-1, :]
            if size[1] % 2 != 0:
                reconstruction = reconstruction[:, :-1]
            # err = np.sum(np.abs(reconstruction.numpy() - face))
            assert np.allclose(reconstruction.numpy(), face)
            # test the operator matrices
            if not separable and not matrixfwt.padded and not matrixifwt.padded:
                test_mat = torch.sparse.mm(
                    matrixifwt.sparse_ifwt_operator, matrixfwt.sparse_fwt_operator
                )
                assert np.allclose(
                    test_mat.to_dense().numpy(), np.eye(test_mat.shape[0])
                )
