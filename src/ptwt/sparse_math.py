# Written by moritz ( @ wolter.tech ) 17.09.21
import torch
import matplotlib.pyplot as plt


def sparse_kron(sparse_tensor_a: torch.Tensor,
                sparse_tensor_b: torch.Tensor) -> torch.Tensor:
    """ A sparse kronecker product."""
    sparse_tensor_ac = sparse_tensor_a.coalesce()
    sparse_tensor_bc = sparse_tensor_b.coalesce()
    kron_result = []
    for row in range(sparse_tensor_a.shape[0]):
        new_kron_col = []
        for col in range(sparse_tensor_a.shape[1]):
            new_kron_col.append(
                sparse_tensor_bc * sparse_tensor_ac[row, col])
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
        [torch.arange(len(diagonal)),
         torch.arange(len(diagonal))])
    if col_offset > 0:
        diag_indices[1] += col_offset
    if col_offset < 0:
        diag_indices[0] += col_offset
    diag = torch.sparse_coo_tensor(diag_indices, diagonal,
                                   size=(rows, cols))
    return diag


if __name__ == '__main__':
    a = torch.tensor([[1, 2], [3, 2], [5, 6]]).to_sparse()
    b = torch.tensor([[7, 8], [9, 0]]).to_sparse()

    print(torch.kron(a.to_dense(), b.to_dense()))
    err = torch.sum(torch.abs(sparse_kron(a, b).to_dense() -
                    torch.kron(a.to_dense(), b.to_dense())))
    print(err)

    import matplotlib.pyplot as plt
    plt.imshow(sparse_diag(torch.ones([5]), 0, 7, 5).to_dense())
    plt.show()
