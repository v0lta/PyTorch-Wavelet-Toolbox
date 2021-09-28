# Written by moritz ( @ wolter.tech ) 17.09.21
from numpy.core.numeric import indices
import torch
import numpy as np
import matplotlib.pyplot as plt


def sparse_kron(sparse_tensor_a: torch.Tensor,
                sparse_tensor_b: torch.Tensor) -> torch.Tensor:
    """ A sparse kronecker product. As defined at:
        https://en.wikipedia.org/wiki/Kronecker_product

    Args:
        sparse_tensor_a (torch.Tensor): Sparse 2d-Tensor a of shape [m, n].
        sparse_tensor_b (torch.Tensor): Sparse 2d-Tensor b of shape [p, q].

    Returns:
        torch.Tensor: The resulting [mp, nq] tensor.
    """
    sparse_tensor_ac = sparse_tensor_a.coalesce()
    sparse_tensor_bc = sparse_tensor_b.coalesce()
    kron_result = []
    for row in range(sparse_tensor_a.shape[0]):
        new_kron_col = []
        for col in range(sparse_tensor_a.shape[1]):
            if sparse_tensor_ac[row, col] != 0:
                new_kron_col.append(
                    sparse_tensor_bc * sparse_tensor_ac[row, col])
            else:
                new_kron_col.append(
                    torch.zeros_like(sparse_tensor_bc))
        kron_result.append(torch.cat(new_kron_col, -1))
    kron_result = torch.cat(kron_result)
    return kron_result


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
        method. This function is memory efficient but slow.

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


if __name__ == '__main__':
    a = torch.tensor([[1, 2], [3, 2], [5, 6]]).to_sparse().cuda()
    b = torch.tensor([[7, 8], [9, 0]]).to_sparse().cuda()
    sparse_result = sparse_kron(a, b)
    err = torch.sum(torch.abs(sparse_result.to_dense() -
                    torch.kron(a.to_dense(), b.to_dense())))
    print(err)
    print(sparse_result.to_dense())
    new_matrix = sparse_replace_row(sparse_result, 1,
                                    torch.tensor([1., 2, 3, 4]).cuda())
    print(new_matrix.to_dense())
    print('stop')
