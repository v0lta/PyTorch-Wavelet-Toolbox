"""Implement matrix based fwt and ifwt.

This module uses boundary filters instead of padding.

The implementation is based on the description
in Strang Nguyen (p. 32), as well as the description
of boundary filters in "Ripples in Mathematics" section 10.3 .
"""
# Created by moritz (wolter@cs.uni-bonn.de) at 14.04.20
import numpy as np
import torch

from .conv_transform import get_filter_tensors
from .sparse_math import (
    _orth_by_gram_schmidt,
    _orth_by_qr,
    construct_strided_conv_matrix,
)

cpu = torch.device("cpu")


def cat_sparse_identity_matrix(
    sparse_matrix: torch.Tensor, new_length: int
) -> torch.Tensor:
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
    ), "Matrices must be square. Odd inputs can cause to non-square matrices."
    assert new_length > sparse_matrix.shape[0], "cant add negatively many entries."
    x = torch.arange(
        sparse_matrix.shape[0],
        new_length,
        dtype=sparse_matrix.dtype,
        device=sparse_matrix.device,
    )
    y = torch.arange(
        sparse_matrix.shape[0],
        new_length,
        dtype=sparse_matrix.dtype,
        device=sparse_matrix.device,
    )
    extra_indices = torch.stack([x, y])
    extra_values = torch.ones(
        [new_length - sparse_matrix.shape[0]],
        dtype=sparse_matrix.dtype,
        device=sparse_matrix.device,
    )
    new_indices = torch.cat([sparse_matrix.coalesce().indices(), extra_indices], -1)
    new_values = torch.cat([sparse_matrix.coalesce().values(), extra_values], -1)
    new_matrix = torch.sparse_coo_tensor(new_indices, new_values)
    return new_matrix


def _construct_a(
    wavelet,
    length: int,
    device: torch.device = cpu,
    dtype=torch.float64,
) -> torch.tensor:
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
        wavelet, flip=False, device=device, dtype=dtype
    )
    analysis_lo = construct_strided_conv_matrix(
        dec_lo.squeeze(), length, 2, "sameshift"
    )
    analysis_hi = construct_strided_conv_matrix(
        dec_hi.squeeze(), length, 2, "sameshift"
    )
    analysis = torch.cat([analysis_lo, analysis_hi])
    return analysis


def _construct_s(
    wavelet,
    length: int,
    device: torch.device = cpu,
    dtype=torch.float64,
) -> torch.tensor:
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
        wavelet, flip=True, device=device, dtype=dtype
    )
    synthesis_lo = construct_strided_conv_matrix(
        rec_lo.squeeze(), length, 2, "sameshift"
    )
    synthesis_hi = construct_strided_conv_matrix(
        rec_hi.squeeze(), length, 2, "sameshift"
    )
    synthesis = torch.cat([synthesis_lo, synthesis_hi])
    return synthesis.transpose(0, 1)


def _get_to_orthogonalize(matrix: torch.Tensor, filt_len: int) -> torch.Tensor:
    """Find matrix rows with fewer entries than filt_len.

    The returned rows will need to be orthogonalized.

    Args:
        matrix (torch.Tensor): The wavelet matrix under consideration.
        filt_len (int): The number of entries we would expect per row.

    Returns:
        torch.Tensor: The row indices with too few entries.
    """
    unique, count = torch.unique_consecutive(
        matrix.coalesce().indices()[0, :], return_counts=True
    )
    return unique[count != filt_len]


def orthogonalize(
    matrix: torch.Tensor, filt_len: int, method: str = "qr"
) -> torch.Tensor:
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
        if method == "qr":
            matrix = _orth_by_qr(matrix, to_orthogonalize)
        else:
            matrix = _orth_by_gram_schmidt(matrix, to_orthogonalize)

    return matrix


class MatrixWavedec(object):
    """Compute the sparse matrix fast wavelet transform.

    Intermediate scale results must be divisible
    by two. A working third level transform
    could use and input length of 128.
    This would lead to intermediate resolutions
    of 64 and 32. All are divisible by two.

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

    def __init__(self, wavelet, level: int = None, boundary: str = "qr"):
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
        assert len(dec_lo) == len(dec_hi), "All filters must have the same length."
        assert len(dec_hi) == len(rec_lo), "All filters must have the same length."
        assert len(rec_lo) == len(rec_hi), "All filters must have the same length."
        assert self.level > 0, "level must be a positive integer."

        self.fwt_matrix_list = []
        self.split_list = []
        self.input_length = None
        self.padded = False

    @property
    def sparse_fwt_operator(self) -> torch.Tensor:
        """Return the sparse transformation operator.

        If the input signal at all levels is divisible by two,
        the whole operation is padding-free and can be expressed
        as a single matrix multiply.

        With the operator torch.sparse.mm(sparse_fwt_operator, data.T)
        to computes a batched fwt.

        This property exists to make the operator matrix transparent.
        Calling the object will handle odd-length inputs properly.

        Returns:
            torch.Tensor: The sparse operator matrix.
        """
        if len(self.fwt_matrix_list) == 1:
            return self.fwt_matrix_list[0]
        elif len(self.fwt_matrix_list) > 1 and self.padded is False:
            fwt_matrix = self.fwt_matrix_list[0]
            for scale_mat in self.fwt_matrix_list[1:]:
                scale_mat = cat_sparse_identity_matrix(scale_mat, fwt_matrix.shape[0])
                fwt_matrix = torch.sparse.mm(scale_mat, fwt_matrix)
            return fwt_matrix
        else:
            return None

    def __call__(self, data) -> list:
        """Compute the matrix fwt.

        Matrix fwt are used to avoid padding.

        Args:
            data: Batched input data [batch_size, time],
                  should be of even length.

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

        re_build = False
        if self.level is None:
            self.level = int(np.log2(length))
            re_build = True

        if self.input_length != length:
            re_build = True

        if not self.fwt_matrix_list or re_build:
            self.ifwt_matrix_list = []
            self.padded = False
            for s in range(1, self.level + 1):
                if split_list[-1] < filt_len:
                    break
                an = construct_boundary_a(
                    self.wavelet,
                    split_list[-1],
                    dtype=data.dtype,
                    boundary=self.boundary,
                    device=data.device,
                )
                self.fwt_matrix_list.append(an)
                new_split_size = length // np.power(2, s)
                if new_split_size % 2 != 0:
                    # padding
                    new_split_size += 1
                    self.padded = True
                split_list.append(new_split_size)
            split_list.append(length // np.power(2, self.level))
            self.split_list = split_list

        lo = data.T
        result_list = []
        for fwt_matrix in self.fwt_matrix_list:
            if lo.shape[0] % 2 != 0:
                # fix odd coefficients lengths for the conv matrix to work.
                lo = torch.nn.functional.pad(lo.T.unsqueeze(1), [0, 1]).squeeze(1).T
            coefficients = torch.sparse.mm(fwt_matrix, lo)
            lo, hi = torch.split(coefficients, coefficients.shape[0] // 2, dim=0)
            result_list.append(hi)
        result_list.append(lo)
        return result_list[::-1]


def construct_boundary_a(
    wavelet,
    length: int,
    device: torch.device = cpu,
    boundary: str = "qr",
    dtype: torch.dtype = torch.float64,
):
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


def construct_boundary_s(
    wavelet,
    length,
    device: torch.device = cpu,
    boundary: str = "qr",
    dtype=torch.float64,
) -> torch.Tensor:
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
    s_orth = orthogonalize(s_full.transpose(1, 0), len(wavelet), method=boundary)
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

    def __init__(self, wavelet, level: int = None, boundary: str = "qr"):
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
        self.ifwt_matrix_list = []
        self.padded = False
        assert self.level > 0, "level must be a positive integer."

    @property
    def sparse_ifwt_operator(self) -> torch.Tensor:
        """Return the sparse transformation operator.

        If the input signal at all levels is divisible by two,
        the whole operation is padding-free and can be expressed
        as a single matrix multiply.

        Having concatenated the analysis coefficients,
        torch.sparse.mm(sparse_ifwt_operator, coefficients.T)
        to computes a batched ifwt.

        This functionality is manly here to make the operator-matrix
        transparent. Calling the object handles padding for odd inputs.

        Returns:
            torch.Tensor: The sparse operator matrix.
        """
        if len(self.ifwt_matrix_list) == 1:
            return self.ifwt_matrix_list[0]
        elif len(self.ifwt_matrix_list) > 1 and self.padded is False:
            ifwt_matrix = self.ifwt_matrix_list[-1]
            for scale_matrix in self.ifwt_matrix_list[:-1][::-1]:
                ifwt_matrix = cat_sparse_identity_matrix(
                    ifwt_matrix, scale_matrix.shape[0]
                )
                ifwt_matrix = torch.sparse.mm(scale_matrix, ifwt_matrix)
            return ifwt_matrix
        else:
            return None

    def __call__(self, coefficients: list) -> torch.Tensor:
        """Run the synthesis or inverse matrix fwt.

        Args:
            coefficients: The coefficients produced by the forward transform.

        Returns:
            torch.Tensor: The input signal reconstruction.
        """
        filt_len = len(self.wavelet)
        length = coefficients[-1].shape[0] * 2

        re_build = False
        if self.level is None:
            self.level = int(np.log2(length))
        else:
            if self.level != int(np.log2(length)):
                re_build = True

        if not self.ifwt_matrix_list or re_build:
            self.ifwt_matrix_list = []
            split_lst = [length]
            for s in range(1, self.level + 1):
                if split_lst[-1] < filt_len:
                    break
                sn = construct_boundary_s(
                    self.wavelet,
                    split_lst[-1],
                    dtype=coefficients[-1].dtype,
                    boundary=self.boundary,
                    device=coefficients[-1].device,
                )
                self.ifwt_matrix_list.append(sn)
                new_split_size = length // np.power(2, s)
                if new_split_size % 2 != 0:
                    # padding
                    new_split_size += 1
                    self.padded = True
                split_lst.append(new_split_size)

        lo = coefficients[0]
        for c_pos, hi in enumerate(coefficients[1:]):
            lo = torch.cat([lo, hi], 0)
            lo = torch.sparse.mm(self.ifwt_matrix_list[::-1][c_pos], lo)

            # remove padding
            if c_pos < len(coefficients) - 2:
                pred_len = lo.shape[0]
                next_len = coefficients[c_pos + 2].shape[0]
                if next_len != pred_len:
                    lo = lo[:-1, :]
                    pred_len = lo.shape[0]
                    assert (
                        next_len == pred_len
                    ), "padding error, please open an issue on github "
        return lo.T


if __name__ == "__main__":
    import pywt
    import torch
    import matplotlib.pyplot as plt

    a = _construct_a(pywt.Wavelet("haar"), 20, torch.device("cpu"))
    s = _construct_s(pywt.Wavelet("haar"), 20, torch.device("cpu"))
    plt.spy(torch.sparse.mm(s, a).to_dense(), marker=".")
    plt.show()
