"""Implement matrix based fwt and ifwt.

The implementation is based on the description
in Strang Nguyen (p. 32), as well as the description
of boundary filters in "Ripples in Mathematics" section 10.3 .
"""
# Created by moritz (wolter@cs.uni-bonn.de) at 14.04.20
import numpy as np
import torch
from .sparse_math import (
    _orth_by_qr,
    _orth_by_gram_schmidt,
    construct_strided_conv_matrix
)

from .conv_transform import get_filter_tensors


def cat_sparse_identity_matrix(
        sparse_matrix: torch.Tensor, new_length: int) -> torch.Tensor:
    """Concatenate a sparse input matrix and a sparse identity matrix.

    Args:
        sparse_matrix (torch.Tensor): The input matrix.
        new_length (int):
            The length up to which the diagonal should be elongated.

    Returns:
        torch.Tensor: Square [input, eye] matrix
            of size [new_length, new_length]
    """
    # assert square matrix.
    assert (
        sparse_matrix.shape[0] == sparse_matrix.shape[1]
    ), "Input matrices must be square. Odd input images lead to non-square matrices."
    assert new_length > sparse_matrix.shape[0],\
        "cant add negatively many entries."
    x = torch.arange(sparse_matrix.shape[0], new_length,
                     dtype=sparse_matrix.dtype,
                     device=sparse_matrix.device)
    y = torch.arange(sparse_matrix.shape[0], new_length,
                     dtype=sparse_matrix.dtype,
                     device=sparse_matrix.device)
    extra_indices = torch.stack([x, y])
    extra_values = torch.ones(
        [new_length - sparse_matrix.shape[0]], dtype=sparse_matrix.dtype,
        device=sparse_matrix.device)
    new_indices = torch.cat(
        [sparse_matrix.coalesce().indices(), extra_indices], -1)
    new_values = torch.cat(
        [sparse_matrix.coalesce().values(), extra_values], -1)
    new_matrix = torch.sparse_coo_tensor(new_indices, new_values)
    return new_matrix


def _construct_a(wavelet, length: int,
                 device: torch.device = torch.device("cpu"),
                 dtype=torch.float64) -> torch.tensor:
    """Construct a raw analysis matrix.

    The resulting matrix will only be orthogonal in the Haar case,
    in most cases you will want to use construct_boundary_a instead.

    Args:
        wavelet (pywt.Wavelet): The wavelet filter to use.
        length (int): The length of the input signal to transfrom.
        device (torch.device, optional): Where to create the matrix.
            Choose cpu or GPU Defaults to torch.device("cpu").
        dtype (optional): The desired torch datatype. Choose torch.float32
            or torch.float64. Defaults to torch.float64.

    Returns:
        torch.tensor: The sparse raw analysis matrix.
    """
    dec_lo, dec_hi, _, _ = get_filter_tensors(
        wavelet, flip=False, device=device, dtype=dtype)
    analysis_lo = construct_strided_conv_matrix(
        dec_lo.squeeze(), length, 2, 'sameshift')
    analysis_hi = construct_strided_conv_matrix(
        dec_hi.squeeze(), length, 2, 'sameshift')
    analysis = torch.cat([analysis_lo, analysis_hi])
    return analysis


def _construct_s(wavelet, length: int,
                 device: torch.device = torch.device("cpu"),
                 dtype=torch.float64) -> torch.tensor:
    """Create a raw synthesis matrix.

    The construced matrix is NOT necessary orthogonal.
    In most cases construct_boundary_s should be used instead.

    Args:
        wavelet (pywt.Wavelet): The wavelet object to use.
        length (int): The lenght of the originally transformed signal.
        device (torch.device, optional): Choose cuda or cpu.
            Defaults to torch.device("cpu").
        dtype ([type], optional): The desired data type. Choose torch.float32
            or torch.float64. Defaults to torch.float64.

    Returns:
        torch.tensor: The raw sparse synthesis matrix.
    """
    _, _, rec_lo, rec_hi = get_filter_tensors(
        wavelet, flip=True, device=device, dtype=dtype)
    synthesis_lo = construct_strided_conv_matrix(
        rec_lo.squeeze(), length, 2, 'sameshift')
    synthesis_hi = construct_strided_conv_matrix(
        rec_hi.squeeze(), length, 2, 'sameshift')
    synthesis = torch.cat([synthesis_lo, synthesis_hi])
    return synthesis.transpose(0, 1)


def _get_to_orthogonalize(
        matrix: torch.Tensor, filt_len: int) -> torch.Tensor:
    """Find matrix rows with fewer entries than filt_len.

    The returned rows will need to be orthogonalized.

    Args:
        matrix (torch.Tensor): The wavelet matrix under consideration.
        filt_len (int): The number of entries we would expect per row.

    Returns:
        torch.Tensor: The row indices with too few entries.
    """
    unique, count = torch.unique_consecutive(
        matrix.coalesce().indices()[0, :], return_counts=True)
    return unique[count != filt_len]


def orthogonalize(matrix: torch.Tensor, filt_len: int,
                  method: str = 'qr') -> torch.Tensor:
    """Orthogonalization for sparse filter matrices.

    Args:
        matrix (torch.Tensor): The sparse filter matrix to orthogonalize.
        filt_len (int): The length of the wavelet filter coefficients.
        method (str): The orthogonalization method to use. Choose qr
            or gramschmidt. The dense qr code will run much faster
            than sparse gramschidt. Choose gramschmidt if qr fails.
            Defaults to qr.

    Returns:
        torch.Tensor: Orthogonal sparse transformation matrix.
    """
    to_orthogonalize = _get_to_orthogonalize(matrix, filt_len)
    if len(to_orthogonalize) > 0:
        if method == 'qr':
            matrix = _orth_by_qr(matrix, to_orthogonalize)
        else:
            matrix = _orth_by_gram_schmidt(matrix, to_orthogonalize)

    return matrix


class MatrixWavedec(object):
    """Compute the sparse matrix fast wavelet transform.

    Example:
        >>> import ptwt, torch, pywt
        >>> import numpy as np
        >>> # generate an input of even length.
        >>> data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> data_torch = torch.from_numpy(data.astype(np.float32))
        >>> matrix_wavedec = ptwt.MatrixWavedec(
                pywt.Wavelet('haar'), level=2)
        >>> coefficients = matrix_wavedec(data_torch)
    """

    def __init__(self, wavelet, level: int = None,
                 boundary: str = 'qr'):
        """Create a matrix-fwt object.

        Args:
            wavelet: A wavelet object.
            level: The desired level up to which to compute the fwt.
            boundary: The desired approach to boundary value treatment.
                Choose qr or gramschmidt. Defaults to qr.
        """
        self.wavelet = wavelet
        self.level = level
        self.boundary = boundary
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
        assert len(dec_lo) == len(dec_hi),\
            "All filters must have the same length."
        assert len(dec_hi) == len(rec_lo),\
            "All filters must have the same length."
        assert len(rec_lo) == len(rec_hi),\
            "All filters must have the same length."
        assert self.level > 0, "level must be a positive integer."

        self.fwt_matrix = None
        self.split_list = []
        self.input_length = None

    def __call__(self, data) -> list:
        """Compute the matrix fwt.

        Args:
            data: Batched input data [batch_size, time],
                  should be of even length.
                  WARNING: If the input length is odd it will be padded on the
                  right to make it even.

        Returns:
            list: A list with the coefficients for each scale.
        """
        if len(data.shape) == 1:
            # assume time series
            data = data.unsqueeze(0)
        if data.shape[-1] % 2 != 0:
            # odd length input
            # print('input length odd, padding a zero on the right')
            data = torch.nn.functional.pad(data, [0, 1])

        filt_len = len(self.wavelet)
        length = data.shape[1]
        split_list = [length]
        fwt_mat_list = []

        re_build = False
        if self.level is None:
            self.level = int(np.log2(length))
            re_build = True

        if self.input_length != length:
            re_build = True

        if self.fwt_matrix is None or re_build:
            for s in range(1, self.level + 1):
                if split_list[-1] < filt_len:
                    break
                an = construct_boundary_a(
                    self.wavelet, split_list[-1],
                    dtype=data.dtype, boundary=self.boundary,
                    device=data.device)
                if s > 1:
                    an = cat_sparse_identity_matrix(an, length)
                fwt_mat_list.append(an)
                new_split_size = length // np.power(2, s)
                split_list.append(new_split_size)

            self.fwt_matrix = fwt_mat_list[0]
            for fwt_mat in fwt_mat_list[1:]:
                self.fwt_matrix = torch.sparse.mm(
                    fwt_mat, self.fwt_matrix)
            split_list.append(length // np.power(2, self.level))
            self.split_list = split_list

        coefficients = torch.sparse.mm(
            self.fwt_matrix, data.T)
        return torch.split(coefficients, self.split_list[1:][::-1])


def construct_boundary_a(wavelet, length: int,
                         device: torch.device = torch.device("cpu"),
                         boundary: str = 'qr',
                         dtype: torch.dtype = torch.float64):
    """Construct a boundary-wavelet filter 1d-analysis matrix.

    Args:
        wavelet : The wavelet filter object to use.
        length (int):  The number of entries in the input signal.
        boundary (str): A string indicating the desired boundary treatment.
            Possible options are qr and gramschmidt. Defaults to
            qr.
        device: Where to place the matrix. Choose cpu or cuda.
            Defaults to cpu.
        dtype: Choose float32 or float64.

    Returns:
        torch.Tensor: The sparse analysis matrix.
    """
    a_full = _construct_a(wavelet, length, dtype=dtype, device=device)
    a_orth = orthogonalize(a_full, len(wavelet), method=boundary)
    return a_orth


def construct_boundary_s(wavelet, length,
                         device: torch.device = torch.device('cpu'),
                         boundary: str = 'qr',
                         dtype=torch.float64) -> torch.Tensor:
    """Construct a boundary-wavelet filter 1d-synthesis matarix.

    Args:
        wavelet : The wavelet filter object to use.
        length (int):  The number of entries in the input signal.
        device (torch.device): Where to place the matrix.
            Choose cpu or cuda. Defaults to cpu.
        boundary (str): A string indicating the desired boundary treatment.
            Possible options are qr and gramschmidt. Defaults to qr.
        dtype: Choose torch.float32 or torch.float64.
            Defaults to torch.float32.

    Returns:
        torch.Tensor: The sparse synthesis matrix.
    """
    s_full = _construct_s(wavelet, length, dtype=dtype, device=device)
    s_orth = orthogonalize(
        s_full.transpose(1, 0), len(wavelet), method=boundary)
    return s_orth.transpose(1, 0)


class MatrixWaverec(object):
    """Matrix based inverse fast wavelet transform.

    Example:
        >>> import ptwt, torch, pywt
        >>> import numpy as np
        >>> # generate an input of even length.
        >>> data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> data_torch = torch.from_numpy(data.astype(np.float32))
        >>> matrix_wavedec = ptwt.MatrixWavedec(
                pywt.Wavelet('haar'), level=2)
        >>> coefficients = matrix_wavedec(data_torch)
        >>> matrix_waverec = ptwt.MatrixWaverec(
                pywt.Wavelet('haar'), level=2)
        >>> reconstruction = matrix_waverec(coefficients)
    """

    def __init__(self, wavelet, level: int = None,
                 boundary: str = 'qr'):
        """Create an analysis transformation object.

        Args:
            wavelet (pywt.Wavelet):  The wavelet used to compute
                the forward transform.
            level (int): The level up to which the coefficients
                have been computed. Defaults to None.
            boundary (str): The boundary treatment method.
                Choose 'gramschmidt' or 'qr'. Defaults to 'qr'.
        """
        self.wavelet = wavelet
        self.level = level
        self.boundary = boundary
        self.ifwt_matrix = None
        assert self.level > 0, "level must be a positive integer."

    def __call__(self, coefficients: list) -> torch.Tensor:
        """Run the synthesis or inverse matrix fwt.

        Args:
            coefficients: The coefficients produced by the forward transform.

        Returns:
            torch.Tensor: The input signal reconstruction.
        """
        # if the coefficients come in a tuple or list concatenate!
        if (type(coefficients) is tuple) or (type(coefficients) is list):
            coefficients = torch.cat(coefficients, 0)

        filt_len = len(self.wavelet)
        length = coefficients.shape[0]

        re_build = False
        if self.level is None:
            self.level = int(np.log2(length))
        else:
            if self.level != int(np.log2(length)):
                re_build = True

        if self.ifwt_matrix is None or re_build:
            ifwt_mat_lst = []
            split_lst = [length]
            for s in range(1, self.level + 1):
                if split_lst[-1] < filt_len:
                    break
                sn = construct_boundary_s(
                    self.wavelet, split_lst[-1], dtype=coefficients.dtype,
                    boundary=self.boundary, device=coefficients.device)
                if s > 1:
                    sn = cat_sparse_identity_matrix(sn, length)
                ifwt_mat_lst.append(sn)
                new_split_size = length // np.power(2, s)
                split_lst.append(new_split_size)

            self.ifwt_matrix = ifwt_mat_lst[-1]
            for ifwt_mat in ifwt_mat_lst[:-1][::-1]:
                self.ifwt_matrix = torch.sparse.mm(ifwt_mat, self.ifwt_matrix)

        reconstruction = torch.sparse.mm(self.ifwt_matrix, coefficients)
        return reconstruction.T


if __name__ == '__main__':
    import pywt
    import torch
    import matplotlib.pyplot as plt
    a = _construct_a(pywt.Wavelet("haar"), 20,
                     torch.device('cpu'))
    s = _construct_s(pywt.Wavelet("haar"), 20,
                     torch.device('cpu'))
    plt.spy(torch.sparse.mm(s, a).to_dense(), marker='.')
    plt.show()
