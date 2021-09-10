import torch
import numpy as np
import matplotlib.pyplot as plt


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
        dtype=torch.float64):
    filter_shape = filter.shape
    convolution_matrix = construct_conv2d_matrix(
        filter,
        input_rows, input_columns, dtype=dtype)

    output_rows = filter_shape[0] + input_rows - 1
    output_columns = filter_shape[1] + input_columns - 1
    output_elements = output_rows * output_columns

    element_numbers = np.arange(output_elements).reshape(
        output_columns, output_rows)

    strided_rows = element_numbers[::stride, ::stride].flatten()

    # TODO: finish me!
    indices = convolution_matrix.coalesce().indices().numpy()
    values = convolution_matrix.coalesce().values().numpy()
    mask = []
    non_zero_row_entries = indices[0, :]
    # lambda ?
    for entry in non_zero_row_entries:
        if entry in strided_rows:
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask)

    strided_indices = indices[:, mask]
    strided_values = values[mask]
    strided_matrix_2 = torch.sparse_coo_tensor(
        strided_indices, strided_values, dtype=dtype)

    strided_matrix = convolution_matrix.to_dense()[strided_rows, :]

    return strided_matrix.to_sparse()