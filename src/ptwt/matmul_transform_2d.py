# Written by moritz ( @ wolter.tech )

import torch
import numpy as np
from src.ptwt.sparse_math import (
    sparse_kron,
    sparse_diag
)

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
        start_row = filter_length // 2
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
        filter (torch.tensor): The filter to convolve with.
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
        diag_index = kernel_column_number // 2
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
        output_elements = output_rows * output_columns
    elif mode == 'valid':
        output_rows = input_rows - filter_shape[0] + 1
        output_columns = input_columns - filter_shape[1] + 1
        output_elements = output_rows * output_columns
    else:
        raise ValueError("Padding mode not accepted.")

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
    strided_matrix = torch.sparse_coo_tensor(
        strided_indices, strided_values, dtype=filter.dtype).coalesce()

    # strided_matrix_2 = convolution_matrix.to_dense()[strided_rows, :].to_sparse()

    # diff = np.abs(
    #      strided_matrix.to_dense().numpy() - strided_matrix_2.to_dense().numpy())
    # to_plot = np.concatenate([strided_matrix.to_dense(), strided_matrix_2.to_dense(), diff], 1)
    # plt.imshow(to_plot)
    # plt.show()

    return strided_matrix
