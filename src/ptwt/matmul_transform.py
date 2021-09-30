# Created by moritz (wolter@cs.uni-bonn.de) at 14.04.20
"""
This module implements matrix based fwt and ifwt
based on the description in Strang Nguyen (p. 32).
As well as the description of boundary filters in
"Ripples in Mathematics" section 10.3 .
"""
import numpy as np
import torch
from .sparse_math import (
    _orth_by_qr,
    _orth_by_gram_schmidt
)


def cat_sparse_identity_matrix(sparse_matrix, new_length):
    """Concatenate a sparse input matrix and a sparse identity matrix.
    Args:
        sparse_matrix: The input matrix.
        new_length: The length up to which the diagonal should be elongated.

    Returns:
        Square [input, eye] matrix of size [new_length, new_length]
    """
    # assert square matrix.
    assert (
        sparse_matrix.shape[0] == sparse_matrix.shape[1]
    ), "wavelet matrices are square"
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


# construct the FWT analysis matrix.
def construct_a(wavelet, length, wrap=True, dtype=torch.float64):
    """Constructs the sparse analysis matrix to compute a matrix based fwt.
    Following page 31 of the Strang Nguyen Wavelets and Filter Banks book.
    Args:
        wavelet: The wavelet coefficients stored in a wavelet object.
        length: The number of entries in the input signal.
        wrap: Filter wrap around produces square matrices.
        dtype: The datatype to use. Defaults to torch.float64.
    Returns:
        The sparse fwt matrix A.
    """
    dec_lo, dec_hi, _, _ = wavelet.filter_bank
    filt_len = len(dec_lo)
    # right hand filtering and decimation matrix
    # set up the indices.

    h = length // 2
    w = length

    xl = np.stack([np.arange(0, h)] * filt_len).T.flatten()
    yl = np.concatenate([np.arange(0, filt_len)] * h) + 2 * xl
    xb = xl + h
    yb = yl
    x = np.concatenate([xl, xb])
    y = np.concatenate([yl, yb])
    if wrap:
        y = y % w
    a_indices = torch.from_numpy(np.stack([x, y]).astype(int))
    al_entries = torch.tensor([dec_lo[::-1]] * h).flatten().type(dtype)
    ab_entries = torch.tensor([dec_hi[::-1]] * h).flatten().type(dtype)
    a_entries = torch.cat([al_entries, ab_entries])
    a_ten = torch.sparse.FloatTensor(a_indices, a_entries).coalesce()
    # left hand filtering and decimation matrix
    return a_ten


def _get_to_orthogonalize(
        matrix: torch.Tensor, filt_len: int) -> torch.Tensor:
    """Find matrix rows with fewer entries than filt_len.
       These rows will need to be orthogonalized.

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
    """ Orthogonalization for sparse filter matrices.

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


def matrix_wavedec(data, wavelet, level: int = None,
                   boundary: str = 'circular'):
    """Experimental computation of the sparse matrix fast wavelet transform.
    Args:
        wavelet: A wavelet object.
        data: Batched input data [batch_size, time], should be of even length.
              WARNING: If the input length is odd a zero will be padded on the
              right to make it even.
        level: The desired level up to which to compute the fwt.
        boundary: The desired approach to boundary value treatment.
            Choose circular or gramschmidt. Defaults to circular.
    Returns: The wavelet coefficients in a single vector.
             As well as the transformation matrices.
    """
    if len(data.shape) == 1:
        # assume time series
        data = data.unsqueeze(0)
    if data.shape[-1] % 2 != 0:
        # odd length input
        # print('input length odd, padding a zero on the right')
        data = torch.nn.functional.pad(data, [0, 1])

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    assert len(dec_lo) == len(dec_hi), "All filters must have the same length."
    assert len(dec_hi) == len(rec_lo), "All filters must have the same length."
    assert len(rec_lo) == len(rec_hi), "All filters must have the same length."
    filt_len = len(dec_lo)

    length = data.shape[1]
    split_list = [length]
    fwt_mat_list = []

    if level is None:
        level = int(np.log2(length))
    else:
        assert level > 0, "level must be a positive integer."

    for s in range(1, level + 1):
        if split_list[-1] < filt_len:
            break
        an = construct_boundary_a(
            wavelet, split_list[-1], dtype=data.dtype, boundary=boundary)
        if s > 1:
            an = cat_sparse_identity_matrix(an, length)
        fwt_mat_list.append(an)
        new_split_size = length // np.power(2, s)
        split_list.append(new_split_size)
    coefficients = data.T

    for fwt_mat in fwt_mat_list:
        coefficients = torch.sparse.mm(fwt_mat, coefficients)
    split_list.append(length // np.power(2, level))
    return torch.split(coefficients, split_list[1:][::-1]), fwt_mat_list


def construct_s(wavelet, length, wrap=True, dtype=torch.float64):
    """Construct the sparse synthesis matrix used to invert the
        fwt.
    Args:
        wavelet: The wavelet coefficients stored in a wavelet object.
        length: The number of entries in the input signal.
        wrap: Filter wrap around produces square matrices.
    Returns:
        The signal reconstruction.
    """
    # construct the FWT synthesis matrix.
    _, _, rec_lo, rec_hi = wavelet.filter_bank
    filt_len = len(rec_lo)
    # right hand filtering and decimation matrix
    # set up the indices.
    h = length // 2
    w = length
    yl = np.stack([np.arange(0, h)] * filt_len).T.flatten()
    xl = np.concatenate([np.arange(0, filt_len)] * h) + 2 * yl
    xb = xl
    yb = yl + h
    x = np.concatenate([xl, xb])
    y = np.concatenate([yl, yb])
    if wrap:
        x = x % w
    s_indices = torch.from_numpy(np.stack([x, y]).astype(int))
    sl_entries = torch.tensor([rec_lo] * h).flatten().type(dtype)
    sb_entries = torch.tensor([rec_hi] * h).flatten().type(dtype)
    s_entries = torch.cat([sl_entries, sb_entries])
    s_ten = torch.sparse.FloatTensor(s_indices, s_entries)
    # left hand filtering and decimation matrix
    return s_ten


def clip_and_orthogonalize(matrix, wavelet):
    filt_len = len(wavelet)

    if filt_len > 2:
        clipl = (filt_len - 2) // 2
        clipr = (filt_len - 2) // 2
        dense = matrix.to_dense()
        clip = dense[:, (clipl):-(clipr)].to_sparse()
        orth = orthogonalize(clip, filt_len)
        return orth
    else:
        return matrix


def construct_boundary_a(wavelet, length: int,
                         boundary: str = 'circular',
                         dtype=torch.float64):
    """ Construct a boundary-wavelet filter 1d-analysis matrix.

    Args:
        wavelet : The wavelet filter object to use.
        length (int):  The number of entries in the input signal.
        boundary (str): A string indicating the desired boundary treatment.
            Possible options are circular and gramschmidt. Defaults to
            circular.

    Returns:
        [torch.sparse.FloatTensor]: The analysis matrix.
    """
    if boundary == 'circular':
        return construct_a(wavelet, length, wrap=True, dtype=dtype)
    elif boundary == 'gramschmidt':
        a_full = construct_a(wavelet, length, wrap=False, dtype=dtype)
        a_orth = clip_and_orthogonalize(a_full, wavelet)
        return a_orth
    else:
        raise ValueError("Unknown boundary treatment")


def construct_boundary_s(wavelet, length,
                         boundary: str = 'circular',
                         dtype=torch.float64):
    """ Construct a boundary-wavelet filter 1d-synthesis matarix.

    Args:
        wavelet : The wavelet filter object to use.
        length (int):  The number of entries in the input signal.
        boundary (str): A string indicating the desired boundary treatment.
            Possible options are circular and gramschmidt. Defaults to
            circular.

    Returns:
        [torch.sparse.FloatTensor]: The synthesis matrix.
    """
    if boundary == 'circular':
        return construct_s(wavelet, length, wrap=True, dtype=dtype)
    elif boundary == 'gramschmidt':
        s_full = construct_s(wavelet, length, wrap=False, dtype=dtype)
        s_orth = clip_and_orthogonalize(
            s_full.transpose(1, 0), wavelet)
        return s_orth.transpose(1, 0)
    else:
        raise ValueError("Unknown boundary treatment")


def matrix_waverec(
        coefficients, wavelet, level: int = None,
        boundary: str = 'circular'):
    """Experimental matrix based inverse fast wavelet transform.

    Args:
        coefficients: The coefficients produced by the forward transform.
        wavelet: The wavelet used to compute the forward transform.
        level (int, optional): The level up to which the coefficients
            have been computed.

    Returns:
        The input signal reconstruction.
    """
    # if the coefficients come in a list concatenate!
    if type(coefficients) is tuple:
        coefficients = torch.cat(coefficients, 0)

    filt_len = len(wavelet)
    length = coefficients.shape[0]

    if level is None:
        level = int(np.log2(length))
    else:
        assert level > 0, "level must be a positive integer."

    ifwt_mat_lst = []
    split_lst = [length]
    for s in range(1, level + 1):
        if split_lst[-1] < filt_len:
            break
        sn = construct_boundary_s(
            wavelet, split_lst[-1], dtype=coefficients.dtype,
            boundary=boundary)
        if s > 1:
            sn = cat_sparse_identity_matrix(sn, length)
        ifwt_mat_lst.append(sn)
        new_split_size = length // np.power(2, s)
        split_lst.append(new_split_size)
    reconstruction = coefficients
    for ifwt_mat in ifwt_mat_lst[::-1]:
        reconstruction = torch.sparse.mm(ifwt_mat, reconstruction)
    return reconstruction.T, ifwt_mat_lst[::-1]
