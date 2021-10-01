# Written by moritz ( @ wolter.tech ) 17.09.21
import torch
import numpy as np
import pytest


def _dense_kron(sparse_tensor_a: torch.Tensor,
                sparse_tensor_b: torch.Tensor) -> torch.Tensor:
    """ Faster than sparse_kron, but limited to resolutions
        of approximately 128x128 pixels by memory on my machine."""
    return torch.kron(sparse_tensor_a.to_dense(),
                      sparse_tensor_b.to_dense()).to_sparse()


def sparse_kron(sparse_tensor_a: torch.Tensor,
                sparse_tensor_b: torch.Tensor) -> torch.Tensor:
    """ A sparse kronecker product. As defined at:
        https://en.wikipedia.org/wiki/Kronecker_product
        Adapted from:
        https://github.com/scipy/scipy/blob/v1.7.1/scipy/sparse/construct.py#L274-L357

    Args:
        sparse_tensor_a (torch.Tensor): Sparse 2d-Tensor a of shape [m, n].
        sparse_tensor_b (torch.Tensor): Sparse 2d-Tensor b of shape [p, q].

    Returns:
        torch.Tensor: The resulting [mp, nq] tensor.
    """
    if not sparse_tensor_a.is_coalesced():
        sparse_tensor_a = sparse_tensor_a.coalesce()
    if not sparse_tensor_b.is_coalesced():
        sparse_tensor_b = sparse_tensor_b.coalesce()
    output_shape = (sparse_tensor_a.shape[0]
                    * sparse_tensor_b.shape[0],
                    sparse_tensor_a.shape[1]
                    * sparse_tensor_b.shape[1])
    nzz_a = len(sparse_tensor_a.values())
    nzz_b = len(sparse_tensor_b.values())

    # take care of the zero case.
    if nzz_a == 0 or nzz_b == 0:
        return torch.sparse_coo_tensor(
            torch.zeros([2, 1]), torch.zeros([1]),
            size=output_shape)

    # expand A's entries into blocks
    row = sparse_tensor_a.indices()[0, :].repeat_interleave(nzz_b)
    col = sparse_tensor_a.indices()[1, :].repeat_interleave(nzz_b)
    data = sparse_tensor_a.values().repeat_interleave(nzz_b)
    row *= sparse_tensor_b.shape[0]
    col *= sparse_tensor_b.shape[1]

    # increment block indices
    row, col = row.reshape(-1, nzz_b), col.reshape(-1, nzz_b)
    row += sparse_tensor_b.indices()[0, :]
    col += sparse_tensor_b.indices()[1, :]
    row, col = row.reshape(-1), col.reshape(-1)

    # compute block entries
    data = data.reshape(-1, nzz_b) * sparse_tensor_b.values()
    data = data.reshape(-1)
    result = torch.sparse_coo_tensor(
        torch.stack([row, col], 0),
        data, size=output_shape)

    return result


def sparse_diag(diagonal: torch.Tensor,
                col_offset: int,
                rows: int, cols: int) -> torch.Tensor:
    """ creates an rows-by-cols sparse matrix
        S by taking the columns of Bin and
        placing them along the diagonal specified by col_offset"""
    diag_indices = torch.stack(
        [torch.arange(len(diagonal), device=diagonal.device),
         torch.arange(len(diagonal), device=diagonal.device)])
    if col_offset > 0:
        diag_indices[1] += col_offset
    if col_offset < 0:
        diag_indices[0] += abs(col_offset)

    if torch.max(diag_indices[0]) >= rows:
        mask = diag_indices[0] < rows
        diag_indices = diag_indices[:, mask]
        diagonal = diagonal[mask]
    if torch.max(diag_indices[1]) >= cols:
        mask = diag_indices[1] < cols
        diag_indices = diag_indices[:, mask]
        diagonal = diagonal[mask]

    diag = torch.sparse_coo_tensor(diag_indices, diagonal,
                                   size=(rows, cols),
                                   dtype=diagonal.dtype)

    return diag


def sparse_matmul_select(matrix, row):
    selection_matrix = torch.sparse_coo_tensor(
        torch.stack([torch.tensor(0, device=row.device), row]).unsqueeze(-1),
        torch.tensor(1.), device=matrix.device,
        dtype=matrix.dtype, size=(1, matrix.shape[0])
    )
    return torch.sparse.mm(selection_matrix, matrix)


def sparse_replace_row(matrix: torch.Tensor, row_index: int,
                       row: torch.Tensor) -> torch.Tensor:
    """Replace a row within a sparse [rows, cols] matrix,
       I.e. matrix[row_no, :] = row.

    Args:
        matrix (torch.Tensor): A sparse two dimensional matrix.
        row_index (int): The row to replace.
        row (torch.Tensor): The row to insert into the sparse matrix.

    Returns:
        [torch.Tensor]: A sparse matrix, with the new row inserted at
        row_index.
    """
    if not matrix.is_coalesced():
        matrix = matrix.coalesce()
    assert matrix.shape[-1] == row.shape[-1], \
        "matrix and replacement-row must share the same column number."

    # delete existing indices we dont want
    diag_indices = torch.arange(matrix.shape[0])
    diag = torch.ones_like(diag_indices)
    diag[row_index] = 0
    removal_matrix = torch.sparse_coo_tensor(
        torch.stack([diag_indices]*2, 0), diag,
        size=matrix.shape, device=matrix.device,
        dtype=matrix.dtype
    )
    if not row.is_coalesced():
        row = row.coalesce()

    addition_matrix = torch.sparse_coo_tensor(
        torch.stack((row.indices()[0, :] + row_index,
                     row.indices()[1, :]), 0),
        row.values(),
        size=matrix.shape, device=matrix.device,
        dtype=matrix.dtype
    )
    result = torch.sparse.mm(removal_matrix, matrix) \
        + addition_matrix
    return result


def _orth_by_qr(matrix: torch.Tensor,
                rows_to_orthogonalize: torch.Tensor) -> torch.Tensor:
    """ Orthogonalize a wavelet matrix through qr decomposition.
    A dense qr-decomposition is used for gpu-efficiency reasons.
    If memory becomes a constraint choose _orth_by_gram_schmidt
    instead, which is implemented using only sparse function calls.

    Args:
        matrix (torch.Tensor): The matrix to orthogonalize.
        rows_to_orthogonalize (torch.Tensor): The matrix rows, which need work.

    Returns:
        torch.Tensor: The corrected sparse matrix.
    """
    selection_indices = torch.stack(
        [torch.arange(len(rows_to_orthogonalize), device=matrix.device),
         rows_to_orthogonalize], 0)
    selection_matrix = torch.sparse_coo_tensor(
        selection_indices,
        torch.ones_like(rows_to_orthogonalize),
        dtype=matrix.dtype, device=matrix.device
    )

    sel = torch.sparse.mm(selection_matrix, matrix)
    q, _ = torch.linalg.qr(sel.to_dense().T)
    q_rows = q.T.to_sparse()

    diag_indices = torch.arange(matrix.shape[0])
    diag = torch.ones_like(diag_indices)
    diag[rows_to_orthogonalize] = 0
    removal_matrix = torch.sparse_coo_tensor(
        torch.stack([diag_indices]*2, 0), diag,
        size=matrix.shape, device=matrix.device,
        dtype=matrix.dtype
    )
    result = torch.sparse.mm(removal_matrix, matrix)
    for pos, row in enumerate(q_rows):
        row = row.unsqueeze(0).coalesce()
        addition_matrix = torch.sparse_coo_tensor(
            torch.stack((row.indices()[0, :] + rows_to_orthogonalize[pos],
                         row.indices()[1, :]), 0),
            row.values(),
            size=matrix.shape, device=matrix.device,
            dtype=matrix.dtype
        )
        result += addition_matrix
    return result.coalesce()


def _orth_by_gram_schmidt(
        matrix: torch.Tensor, to_orthogonalize: torch.Tensor) -> torch.Tensor:
    """ Orthogonalize by using a sparse implementation of the Gram-Schmidt
        method. This function is very memory efficient and very slow.

    Args:
        matrix (torch.Tensor): The sparse matrix to work on.
        to_orthogonalize (torch.Tensor): The matrix rows, which need work.

    Returns:
        torch.Tensor: The orthogonalized sparse matrix.
    """
    done = []
    # loop over the rows we want to orthogonalize
    for row_no_to_ortho in to_orthogonalize:
        current_row = matrix.select(
            0, row_no_to_ortho).unsqueeze(0)
        sum = torch.zeros_like(current_row)
        for done_row_no in done:
            done_row = matrix.select(0, done_row_no).unsqueeze(0)
            non_orthogonal = torch.sparse.mm(current_row,
                                             done_row.transpose(1, 0))
            non_orthogonal_values = non_orthogonal.coalesce().values()
            if len(non_orthogonal_values) == 0:
                non_orthogonal_item = 0
            else:
                non_orthogonal_item = non_orthogonal_values.item()
            sum += non_orthogonal_item*done_row
        orthogonal_row = current_row - sum
        length = torch.native_norm(orthogonal_row)
        orthonormal_row = orthogonal_row / length
        matrix = sparse_replace_row(
            matrix, row_no_to_ortho,
            orthonormal_row)
        done.append(row_no_to_ortho)
    return matrix



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
    elif mode == 'same' or mode == 'sameshift':
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
                            mode: str = 'valid',
                            fully_sparse: bool = True) -> torch.Tensor:
    """ Create a two dimensional sparse convolution matrix.
        Convolving with this matrix should be equivalent to
        a call to scipy.signal.convolve2d and a reshape.

    Args:
        filter (torch.tensor): A filter of shape [height, width]
            to convolve with.
        input_rows (int): The number of rows in the input matrix.
        input_columns (int): The number of columns in the input matrix.
        mode: (str) = The desired padding method. Options are
            full, same and valid. Defaults to 'valid' or no padding.
        fully_sparse (bool): Use a sparse implementation of the Kronecker
            to save memory. Defaults to True.
    Returns:
        [torch.sparse.FloatTensor]: A sparse convolution matrix.
    """
    if fully_sparse:
        kron = sparse_kron
    else:
        kron = _dense_kron

    kernel_column_number = filter.shape[-1]
    matrix_block_number = kernel_column_number

    block_matrix_list = []
    for i in range(matrix_block_number):
        block_matrix_list.append(construct_conv_matrix(
            filter[:, i], input_rows, mode))

    if mode == 'full':
        diag_index = 0
        kronecker_rows = input_columns + kernel_column_number - 1
    elif mode == 'same' or mode == 'sameshift':
        filter_offset = kernel_column_number % 2
        diag_index = kernel_column_number // 2 - 1 + filter_offset
        kronecker_rows = input_columns
    elif mode == 'valid':
        diag_index = kernel_column_number - 1
        kronecker_rows = input_columns - kernel_column_number + 1
    else:
        raise ValueError('unknown conv type.')

    diag_values = torch.ones([int(np.min([kronecker_rows, input_columns]))],
                             dtype=filter.dtype, device=filter.device)
    diag = sparse_diag(diag_values, diag_index, kronecker_rows, input_columns)
    sparse_conv_matrix = kron(diag, block_matrix_list[0])

    for block_matrix in block_matrix_list[1:]:
        diag_index -= 1
        diag = sparse_diag(diag_values, diag_index,
                           kronecker_rows, input_columns)
        sparse_conv_matrix += kron(diag, block_matrix)

    return sparse_conv_matrix


def construct_strided_conv2d_matrix(
        filter: torch.Tensor,
        input_rows: int,
        input_columns: int,
        stride: int = 2,
        mode='full') -> torch.Tensor:
    """ Create a strided sparse two dimensional convolution
       matrix.

    Args:
        filter (torch.tensor): The two dimensional convolution filter.
        input_rows (int): The number of rows in the 2d-input matrix.
        input_columns (int): The number of columns in the 2d- input matrix.
        stride (int, optional): The stride between the filter positions.
            Defaults to 2.
        mode (str, optional): The convolution type.
            Options are 'full', 'valid', 'same' and 'sameshift'.
            Defaults to 'full'.

    Raises:
        ValueError: Raised if an unknown convolution string is
            provided.

    Returns:
        [torch.Tensor]: The sparse convolution tensor.
    """
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
    elif mode == 'same' or mode == 'sameshift':
        output_rows = input_rows
        output_columns = input_columns
    else:
        raise ValueError("Padding mode not accepted.")

    output_elements = output_rows * output_columns
    element_numbers = torch.arange(output_elements,
                                   device=filter.device).reshape(
        output_columns, output_rows)

    start = 0
    if mode == 'sameshift':
        start += 1

    strided_rows = element_numbers[start::stride, start::stride]
    strided_rows = strided_rows.flatten()
    selection_eye = torch.sparse_coo_tensor(
        torch.stack([torch.arange(len(strided_rows),
                                  device=convolution_matrix.device),
                     strided_rows],
                    0),
        torch.ones(len(strided_rows)),
        dtype=convolution_matrix.dtype,
        device=convolution_matrix.device,
        size=[len(strided_rows), convolution_matrix.shape[0]])
    # return convolution_matrix.index_select(0, strided_rows) 
    return torch.sparse.mm(selection_eye, convolution_matrix)



if __name__ == '__main__':
    test_matrix = torch.ones([4, 4]).to_sparse()
    new_matrix = sparse_replace_row(
        test_matrix, 1,
        torch.tensor([1., 2, 3, 4]).unsqueeze(0).to_sparse())
    print(new_matrix.to_dense())
    print('stop')
