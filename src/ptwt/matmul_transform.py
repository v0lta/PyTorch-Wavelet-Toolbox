# Created by moritz (wolter@cs.uni-bonn.de) at 14.04.20
"""
This module implements matrix based fwt and ifwt 
based on the description in Strang Nguyen (p. 32).
"""
import numpy as np
import torch


def cat_sparse_identity_matrix(sparse_matrix, new_length):
    """Concatenate a sparse input matrix and a sparse identity matrix.
    :param sparse_matrix: The input matrix.
    :param new_length: The length up to which the diagonal should be elongated.
    :return: Square [input, eye] matrix of size [new_length, new_length]
    """
    # assert square matrix.
    assert (
        sparse_matrix.shape[0] == sparse_matrix.shape[1]
    ), "wavelet matrices are square"
    assert new_length > sparse_matrix.shape[0], "cant add negatively many entries."
    x = torch.arange(sparse_matrix.shape[0], new_length)
    y = torch.arange(sparse_matrix.shape[0], new_length)
    extra_indices = torch.stack([x, y])
    extra_values = torch.ones([new_length - sparse_matrix.shape[0]])
    new_indices = torch.cat([sparse_matrix.coalesce().indices(), extra_indices], -1)
    new_values = torch.cat([sparse_matrix.coalesce().values(), extra_values], -1)
    new_matrix = torch.sparse.FloatTensor(new_indices, new_values)
    return new_matrix


# construct the FWT analysis matrix.
def construct_a(wavelet, length, wrap=True):
    """Constructs the sparse analysis matrix to compute a matrix based fwt.
    Following page 31 of the Strang Nguyen Wavelets and Filter Banks book.
    Args:
        wavelet: The wavelet coefficients stored in a wavelet object.
        length: The number of entries in the input signal.
        wrap: Filter wrap around produces square matrices.
    Returns:
        The sparse fwt matrix A.
    """
    dec_lo, dec_hi, _, _ = wavelet.filter_bank
    filt_len = len(dec_lo)
    # right hand filtering and decimation matrix
    # set up the indices.

    h = length // 2
    w = length

    # x = []; y = []
    # for i in range(0, h):
    #     for j in range(filt_len):
    #         x.append(i)
    #         y.append((j+2*i) % w)
    # for k in range(0, h):
    #     for j in range(filt_len):
    #         x.append(k + h)
    #         y.append((j+2*k) % w)

    xl = np.stack([np.arange(0, h)] * filt_len).T.flatten()
    yl = np.concatenate([np.arange(0, filt_len)] * h) + 2 * xl
    xb = xl + h
    yb = yl
    x = np.concatenate([xl, xb])
    y = np.concatenate([yl, yb])
    if wrap:
        y = y % w
    a_indices = torch.from_numpy(np.stack([x, y]).astype(np.int))
    al_entries = torch.tensor([dec_lo[::-1]] * h).flatten()
    ab_entries = torch.tensor([dec_hi[::-1]] * h).flatten()
    a_entries = torch.cat([al_entries, ab_entries])
    a_ten = torch.sparse.FloatTensor(a_indices, a_entries)
    # left hand filtering and decimation matrix
    return a_ten


def matrix_wavedec(data, wavelet, level: int = None):
    """Experimental computation of the sparse matrix fast wavelet transform.
    Args:
        wavelet: A wavelet object.
        data: Batched input data [batch_size, time]
        level: The desired level up to which to compute the fwt.
    Returns: The wavelet coefficients in a single vector.
             As well as the transformation matrices.
    """

    if len(data.shape) == 1:
        # assume time series
        data = data.unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    assert len(dec_lo) == len(dec_hi), "All filters hast have the same length."
    assert len(dec_hi) == len(rec_lo), "All filters hast have the same length."
    assert len(rec_lo) == len(rec_hi), "All filters hast have the same length."
    filt_len = len(dec_lo)

    length = data.shape[1]

    if level is None:
        level = int(np.log2(length))
    else:
        assert level > 0, "level must be a positive integer."
    ar = construct_a(wavelet, length)
    if level == 1:
        coefficients = torch.sparse.mm(ar, data.T)
        return torch.split(coefficients, coefficients.shape[0] // 2), [ar]
    al2 = construct_a(wavelet, length // 2)
    al2 = cat_sparse_identity_matrix(al2, length)
    if level == 2:
        coefficients = torch.sparse.mm(al2, torch.sparse.mm(ar, data.T))
        return torch.split(coefficients, [length // 4, length // 4, length // 2]), [
            ar,
            al2,
        ]
    ar3 = construct_a(wavelet, length // 4)
    ar3 = cat_sparse_identity_matrix(ar3, length)
    if level == 3:
        coefficients = torch.sparse.mm(
            ar3, torch.sparse.mm(al2, torch.sparse.mm(ar, data.T))
        )
        return (
            torch.split(
                coefficients, [length // 8, length // 8, length // 4, length // 2]
            ),
            [ar, al2, ar3],
        )
    fwt_mat_lst = [ar, al2, ar3]
    split_lst = [length // 2, length // 4, length // 8]
    for s in range(4, level + 1):
        if split_lst[-1] < filt_len:
            break
        an = construct_a(wavelet, split_lst[-1])
        an = cat_sparse_identity_matrix(an, length)
        fwt_mat_lst.append(an)
        new_split_size = length // np.power(2, s)
        split_lst.append(new_split_size)
    coefficients = data.T
    for fwt_mat in fwt_mat_lst:
        coefficients = torch.sparse.mm(fwt_mat, coefficients)
    split_lst.append(length // np.power(2, level))
    return torch.split(coefficients, split_lst[::-1]), fwt_mat_lst


def construct_s(wavelet, length, wrap=True):
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
    s_indices = torch.from_numpy(np.stack([x, y]).astype(np.int))
    sl_entries = torch.tensor([rec_lo] * h).flatten()
    sb_entries = torch.tensor([rec_hi] * h).flatten()
    s_entries = torch.cat([sl_entries, sb_entries])
    s_ten = torch.sparse.FloatTensor(s_indices, s_entries)
    # left hand filtering and decimation matrix
    return s_ten


def matrix_waverec(coefficients, wavelet, level: int = None):
    """Experimental matrix based inverse fast wavelet transform.

    Args:
        coefficients: The coefficients produced by the forward transform.
        wavelet: The wavelet used to compute the forward transform.
        level (int, optional): The level up to which the coefficients
            have been computed.

    Returns:
        The input signal reconstruction.
    """
    _, _, rec_lo, rec_hi = wavelet.filter_bank

    # if the coefficients come in a list concatenate!
    if type(coefficients) is tuple:
        coefficients = torch.cat(coefficients, 0)

    filt_len = len(rec_lo)
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
        sn = construct_s(wavelet, split_lst[-1])
        if s > 1:
            sn = cat_sparse_identity_matrix(sn, length)
        ifwt_mat_lst.append(sn)
        new_split_size = length // np.power(2, s)
        split_lst.append(new_split_size)
    reconstruction = coefficients
    for ifwt_mat in ifwt_mat_lst[::-1]:
        reconstruction = torch.sparse.mm(ifwt_mat, reconstruction)
    return reconstruction.T, ifwt_mat_lst[::-1]
