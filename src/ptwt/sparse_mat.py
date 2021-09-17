# Written by moritz ( @ wolter.tech ) 17.09.21
import torch
import matplotlib.pyplot as plt


def sparse_kron(sparse_tensor_a, sparse_tensor_b):
    """ """
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


if __name__ == '__main__':
    a = torch.tensor([[1, 2], [3, 2], [5, 6]]).to_sparse()
    b = torch.tensor([[7, 8], [9, 0]]).to_sparse()

    print(torch.kron(a.to_dense(), b.to_dense()))
    err = torch.sum(torch.abs(sparse_kron(a, b).to_dense() -
                    torch.kron(a.to_dense(), b.to_dense())))
    print(err)


