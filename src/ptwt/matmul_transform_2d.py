# Written by moritz ( @ wolter.tech ) in 2021

import torch
import numpy as np

from src.ptwt.sparse_math import (
    sparse_kron,
    sparse_diag
)
from src.ptwt.conv_transform import (
    flatten_2d_coeff_lst,
    construct_2d_filt,
    get_filter_tensors
)
from src.ptwt.matmul_transform import orth_via_gram_schmidt

import matplotlib.pyplot as plt


def construct_conv_matrix(filter: torch.tensor,
                          input_columns: int,
                          mode: str = 'valid') -> torch.Tensor:
    """Constructs a convolution matrix,
       full and valid padding are supported.

    Args:
        filter (torch.tensor): The 1d-filter to convolve with.
        input_columns (int): The number of columns in the input.
        mode (str): String indetifier for the desired padding.
            Defaults to valid.

    Returns:
        torch.Tensor: The sparse convolution tensor.

    For reference see:
    https://github.com/RoyiAvital/StackExchangeCodes/blob/\
        master/StackOverflow/Q2080835/CreateConvMtxSparse.m
    """
    filter_length = len(filter)

    if mode == 'full':
        start_row = 0
        stop_row = input_columns + filter_length - 1
    elif mode == 'same':
        filter_offset = filter_length % 2
        # signal_offset = input_columns % 2
        start_row = filter_length // 2 - 1 + filter_offset
        stop_row = start_row + input_columns - 1
    elif mode == 'valid':
        start_row = filter_length - 1
        stop_row = input_columns - 1
    else:
        raise ValueError('unkown padding type.')

    row_indices = []
    column_indices = []
    values = []
    for column in range(0, input_columns):
        for row in range(0, filter_length):
            check_row = column + row
            if (check_row >= start_row) and (check_row <= stop_row):
                row_indices.append(row + column - start_row)
                column_indices.append(column)
                values.append(filter[row])
    indices = np.stack([row_indices, column_indices])
    values = torch.stack(values)
    return torch.sparse_coo_tensor(indices, values, dtype=filter.dtype)


def construct_conv2d_matrix(filter: torch.tensor,
                            input_rows: int,
                            input_columns: int,
                            mode: str = 'valid') -> torch.Tensor:
    """ Create a two dimensional convolution matrix.
        Convolving with this matrix should be equivalent to
        a call to scipy.signal.convolve2d and a reshape.

    Args:
        filter (torch.tensor): A filter of shape [height, width]
            to convolve with.
        input_rows (int): The number of rows in the input matrix.
        input_columns (int): The number of columns in the input matrix.
        mode: [str] = The desired padding method. Defaults to 'valid'
            or no padding.
    Returns:
        [torch.sparse.FloatTensor]: A sparse convolution matrix.
    """
    kernel_column_number = filter.shape[-1]
    matrix_block_number = kernel_column_number

    block_matrix_list = []
    for i in range(matrix_block_number):
        block_matrix_list.append(construct_conv_matrix(
            filter[:, i], input_rows, mode))

    if mode == 'full':
        diag_index = 0
        kronecker_rows = input_columns + kernel_column_number - 1
    elif mode == 'same':
        filter_offset = kernel_column_number % 2
        diag_index = kernel_column_number // 2 - 1 + filter_offset
        kronecker_rows = input_columns
    elif mode == 'valid':
        diag_index = kernel_column_number - 1
        kronecker_rows = input_columns - kernel_column_number + 1
    else:
        raise ValueError('unknown conv type.')

    diag_values = torch.ones([int(np.min([kronecker_rows, input_columns]))],
                             dtype=filter.dtype)
    diag = sparse_diag(diag_values, diag_index, kronecker_rows, input_columns)
    sparse_conv_matrix = sparse_kron(diag, block_matrix_list[0])

    for block_matrix in block_matrix_list[1:]:
        diag_index -= 1
        diag = sparse_diag(diag_values, diag_index,
                           kronecker_rows, input_columns)
        sparse_conv_matrix += sparse_kron(diag, block_matrix)

    return sparse_conv_matrix


def construct_strided_conv2d_matrix(
        filter: torch.tensor,
        input_rows: int,
        input_columns: int,
        stride: int = 2,
        mode='full'):
    filter_shape = filter.shape
    convolution_matrix = construct_conv2d_matrix(
        filter,
        input_rows, input_columns, mode=mode)

    if mode == 'full':
        output_rows = filter_shape[0] + input_rows - 1
        output_columns = filter_shape[1] + input_columns - 1
    elif mode == 'valid':
        output_rows = input_rows - filter_shape[0] + 1
        output_columns = input_columns - filter_shape[1] + 1
    elif mode == 'same':
        output_rows = input_rows
        output_columns = input_columns
    else:
        raise ValueError("Padding mode not accepted.")

    output_elements = output_rows * output_columns
    element_numbers = np.arange(output_elements).reshape(
        output_columns, output_rows)

    strided_rows = element_numbers[::stride, ::stride]
    strided_rows = strided_rows.flatten()

    indices = convolution_matrix.coalesce().indices().numpy()
    values = convolution_matrix.coalesce().values().numpy()
    mask = []
    strided_row_indices = []
    non_zero_row_entries = indices[0, :]
    index_counter = 0
    previous_entry = 0
    for entry in non_zero_row_entries:
        next_hits = strided_rows[index_counter:(index_counter+2)]
        if entry in next_hits:
            mask.append(True)
            if previous_entry != entry:
                index_counter += 1
            strided_row_indices.append(index_counter)
        else:
            mask.append(False)
        previous_entry = entry
    mask = np.array(mask)

    strided_row_indices = np.array(strided_row_indices)
    strided_col_indices = indices[1, mask]
    strided_indices = np.stack([strided_row_indices, strided_col_indices], 0)
    strided_values = values[mask]
    size = (np.max(strided_row_indices) + 1,
            np.max(indices[1, :]) + 1)
    strided_matrix = torch.sparse_coo_tensor(
        strided_indices, strided_values,
        size=size, dtype=filter.dtype).coalesce()

    # strided_matrix_2 = convolution_matrix.to_dense()[
    #    strided_rows, :].to_sparse()
    # diff = np.abs(
    #       strided_matrix.to_dense().numpy()
    #       - strided_matrix_2.to_dense().numpy())
    # to_plot = np.concatenate(
    #    [strided_matrix.to_dense(), strided_matrix_2.to_dense(), diff], 1)
    # plt.imshow(to_plot)
    # plt.show()
    return strided_matrix


def construct_a_2d(wavelet, height: int, width: int,
                   device, dtype=torch.float64):
    dec_lo, dec_hi, _, _ = get_filter_tensors(
        wavelet, flip=False, device=device, dtype=dtype)
    dec_filt = construct_2d_filt(lo=dec_lo, hi=dec_hi)
    ll, lh, hl, hh = dec_filt.squeeze(1)
    analysis_ll = construct_strided_conv2d_matrix(
        ll, height, width, mode='valid')
    analysis_lh = construct_strided_conv2d_matrix(
        lh, height, width, mode='valid')
    analysis_hl = construct_strided_conv2d_matrix(
        hl, height, width, mode='valid')
    analysis_hh = construct_strided_conv2d_matrix(
        hh, height, width, mode='valid')
    analysis = torch.cat([analysis_ll, analysis_hl,
                          analysis_lh, analysis_hh], 0)

    # a_ll_valid = construct_strided_conv2d_matrix(
    #    ll, height, width, mode='valid')

    return analysis


def construct_s_2d(wavelet, height: int, width: int,
                   device, dtype=torch.float64):
    _, _, rec_lo, rec_hi = get_filter_tensors(
        wavelet, flip=True, device=device, dtype=dtype)
    dec_filt = construct_2d_filt(lo=rec_lo, hi=rec_hi)
    ll, lh, hl, hh = dec_filt.squeeze(1)
    synthesis_ll = construct_strided_conv2d_matrix(
        ll, height, width, mode='valid')
    synthesis_lh = construct_strided_conv2d_matrix(
        lh, height, width, mode='valid')
    synthesis_hl = construct_strided_conv2d_matrix(
        hl, height, width, mode='valid')
    synthesis_hh = construct_strided_conv2d_matrix(
        hh, height, width, mode='valid')
    synthesis = torch.cat([synthesis_ll, synthesis_hl,
                           synthesis_lh, synthesis_hh], 0).coalesce()
    indices = synthesis.indices()
    shape = synthesis.shape
    transpose_indices = torch.stack([indices[1, :], indices[0, :]])
    transpose_synthesis = torch.sparse_coo_tensor(transpose_indices,
                                                  synthesis.values(),
                                                  size=(shape[1], shape[0]))
    return transpose_synthesis


def construct_boundary_a2d(
        wavelet, height: int, width: int,
        device, dtype=torch.float64):

    a = construct_a_2d(
        wavelet, height, width, device, dtype=dtype)

    orth_a = orth_via_gram_schmidt(a.to_dense(), len(wavelet))
    return orth_a


def split_coeff_vector(coeff_vector):
    vector_length = len(coeff_vector)
    split_coeffs = torch.split(coeff_vector, vector_length // 4)
    return split_coeffs


def split_and_reshape_coeff_vector(coeff_vector, input_shape, filter_shape):
    split_coeffs = split_coeff_vector(coeff_vector)
    reshape_list = []
    for split in split_coeffs:
        reshape_list.append(torch.reshape(
             split, [(input_shape[1] - (filter_shape[1])) // 2 + 1,
                     (input_shape[0] - (filter_shape[0])) // 2 + 1])
        )
    return reshape_list


def matrix_fwt_2d(input_signal_2d, wavelet):
    height, width = input_signal_2d.shape
    a = construct_boundary_a2d(wavelet, height, width,
                               device=input_signal_2d.device,
                               dtype=input_signal_2d.dtype)
    res_mm = torch.sparse.mm(a, input_signal_2d.flatten().unsqueeze(-1))
    res_mm_split = split_and_reshape_coeff_vector(
        res_mm, [height, width], [len(wavelet), len(wavelet)])
    return res_mm_split


if __name__ == '__main__':
    import scipy.misc
    import pywt
    from src.ptwt.conv_transform import wavedec2

    # single level db2 - 2d
    face = np.mean(scipy.misc.face()[256:(256+8), 256:(256+8)],
                   -1).astype(np.float64)
    pt_face = torch.tensor(face)
    wavelet = pywt.Wavelet("haar")
    a = construct_a_2d(wavelet, pt_face.shape[0], pt_face.shape[1],
                       device=pt_face.device, dtype=pt_face.dtype)
    s = construct_s_2d(wavelet, pt_face.shape[0], pt_face.shape[1],
                       device=pt_face.device, dtype=pt_face.dtype)
    test_eye = torch.sparse.mm(a,s)
    plt.imshow(test_eye.to_dense()); plt.show()
    res_mm = torch.sparse.mm(a, pt_face.flatten().unsqueeze(-1))
    res_mm_split = matrix_fwt_2d(pt_face, wavelet)
    res_coeff = wavedec2(pt_face.unsqueeze(0).unsqueeze(0), wavelet, level=1)
    flat_coeff = torch.cat(flatten_2d_coeff_lst(res_coeff), -1)

    plt.plot(res_mm, '.')
    plt.plot(flat_coeff, '.')
    plt.show()
    print('stop')
