import torch
import numpy as np
import matplotlib.pyplot as plt


def construct_conv_matrix(filter: torch.tensor,
                          input_columns: int,
                          conv_type: str = 'valid') -> torch.Tensor:
    """Constructs a convolution matrix,
       full and valid padding are supported.

    Args:
        filter (torch.tensor): The 1d-filter to convolve with.
        input_columns (int): The number of columns in the input.
        conv_type (str): String indetifier for the desired padding.
            Defaults to valid.

    Returns:
        torch.Tensor: The sparse convolution tensor.

    For reference see:
    https://github.com/RoyiAvital/StackExchangeCodes/blob/\
        master/StackOverflow/Q2080835/CreateConvMtxSparse.m
    """
    filter_length = len(filter)

    if conv_type == 'full':
        start_row = 0
        stop_row = input_columns + filter_length - 1
    elif conv_type == 'same':
        start_row = filter_length // 2
        stop_row = start_row + input_columns - 1
    elif conv_type == 'valid':
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

    return torch.sparse_coo_tensor(indices, values)


def construct_conv_2d_matrix(filter: torch.tensor,
                             input_rows: int,
                             input_columns: int,
                             conv_type: str = 'valid') -> torch.Tensor:
    """


    Based on the matlab code at:
    https://github.com/RoyiAvital/StackExchangeCodes/blob/master/\
        StackOverflow/Q2080835/CreateConvMtx2DSparse.m
    """

    kernel_column_number = filter.shape[-1]
    matrix_block_number = kernel_column_number

    block_matrix_list = []
    for i in range(block_matrix_list):
        pass

    if conv_type == 'full':
        diag_index = 0
        kronecker_rows = input_columns + kernel_column_number - 1
    elif conv_type == 'same':
        diag_index = kernel_column_number // 2
        kronecker_rows = input_columns
    elif conv_type == 'valid':
        diag_index = kernel_column_number - 1
        kronecker_rows = input_columns - kernel_column_number + 1
    else:
        raise ValueError('unknown conv type.')

    diag_values = np.ones(np.min([kronecker_rows, input_columns], 1))
    sparse_matrix = None

    return None




def construct_conv2d_matrix(filter: torch.tensor, input_rows: int,
                            input_columns: int, dtype=torch.float64):
    """ Create a two dimensional convolution matrix.
        Convolving with this matrix should be equivalent to
        a call to scipy.signal.convolve2d and a reshape.

    Args:
        filter (torch.tensor): The filter to convolve with.
        input_rows (int): The number of rows in the input matrix.
        input_columns (int): The number of columns in the input matrix.
        dtype (optional): Input data type. Defaults to torch.float64.

    Returns:
        [torch.sparse.FloatTensor]: A sparse convolution matrix.
    """
    filter_rows, filter_columns = filter.shape

    block_height = input_rows + filter_rows - 1
    # block_width = input_columns
    block_entries = input_rows*filter_rows

    all_entries = filter_columns*input_columns*block_entries

    sparse_columns = np.zeros([all_entries])
    sparse_rows = np.zeros([all_entries])
    sparse_entries = torch.zeros([all_entries])

    # matrix_height = (input_columns + filter_columns - 1)*block_height
    # matrix_width = input_columns*block_width

    col = np.stack([np.arange(0, input_rows)]*filter_rows)
    row = col + np.arange(0, filter_rows)[:, np.newaxis]
    col = col.flatten()
    row = row.flatten()
    row = np.stack([row]*input_columns, -1)
    col = np.stack([col]*input_columns, -1)

    column_offset = np.arange(0, input_columns)*input_rows
    column_offset = np.stack([column_offset]*(input_rows*filter_rows))
    column_offset = column_offset + col
    column_offset = column_offset.T.flatten()
    row_offset = np.arange(0, input_columns)*block_height
    row_offset = np.stack([row_offset]*(input_rows*filter_rows))
    row_offset = row_offset + row
    row_offset = row_offset.T.flatten()

    for col in range(0, filter_columns):
        entries = filter[:, col]
        entries = torch.stack([entries]*input_rows).T.flatten()
        entries = torch.stack([entries]*input_columns).flatten()
        start = col*input_columns*block_entries
        stop = start + input_columns*block_entries

        sparse_rows[start:stop] = row_offset
        sparse_columns[start:stop] = column_offset
        sparse_entries[start:stop] = entries
        row_offset += block_height

    sparse_indices = np.stack([sparse_rows, sparse_columns])
    # sparse_indices = np.stack([sparse_columns, sparse_rows])
    matrix = torch.sparse_coo_tensor(sparse_indices,
                                     sparse_entries, dtype=dtype)
    # assert (matrix_height, matrix_width) == matrix.shape
    # plt.imshow(matrix.to_dense()); plt.show()
    return matrix


def construct_strided_conv2d_matrix(
        filter: torch.tensor,
        input_rows: int,
        input_columns: int,
        stride: int = 2,
        dtype=torch.float64,
        no_padding=False):
    filter_shape = filter.shape
    convolution_matrix = construct_conv2d_matrix(
        filter,
        input_rows, input_columns, dtype=dtype)

    output_rows = filter_shape[0] + input_rows - 1
    output_columns = filter_shape[1] + input_columns - 1
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
    strided_matrix = torch.sparse_coo_tensor(
        strided_indices, strided_values, dtype=dtype).coalesce()

    # strided_matrix_2 = convolution_matrix.to_dense()[strided_rows, :].to_sparse()

    # diff = np.abs(
    #      strided_matrix.to_dense().numpy() - strided_matrix_2.to_dense().numpy())
    # to_plot = np.concatenate([strided_matrix.to_dense(), strided_matrix_2.to_dense(), diff], 1)
    # plt.imshow(to_plot)
    # plt.show()

    return strided_matrix
