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
    assert matrix.shape[-1] == row.shape[0], \
        "matrix and replacement-row must share the same column number."
    row = row.unsqueeze(0)
    if not row.is_sparse:
        row = row.to_sparse()
    if not row.is_coalesced():
        row = row.coalesce()

    # delete existing indices we dont want

    new_indices = matrix.indices()[
        :, matrix.indices()[0, :] != row_index]
    new_values = matrix.values()[
        matrix.indices()[0, :] != row_index]

    replacement_row_indices = torch.stack(
        [torch.tensor(row_index, device=matrix.device)]*len(row.values()))
    replacement_indices = torch.stack([replacement_row_indices,
                                       row.indices()[1, :]])
    new_indices = torch.cat([new_indices, replacement_indices], -1)
    new_values = torch.cat([new_values, row.values()], -1)
    new_matrix = torch.sparse_coo_tensor(
        new_indices, new_values, size=matrix.shape,
        dtype=matrix.dtype, device=matrix.device)
    return new_matrix


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
