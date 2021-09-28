# Written by moritz ( @ wolter.tech ) in 2021
import torch
import numpy as np
from src.ptwt.sparse_math import sparse_kron


def test_kron():
    # test the implementation by evaluating the example from
    # https://de.wikipedia.org/wiki/Kronecker-Produkt
    a = torch.tensor([[1, 2], [3, 2], [5, 6]]).to_sparse()
    b = torch.tensor([[7, 8], [9, 0]]).to_sparse()
    sparse_result = sparse_kron(a, b)
    dense_result = torch.kron(a.to_dense(), b.to_dense())
    err = torch.sum(torch.abs(sparse_result.to_dense() -
                    dense_result))
    condition = np.allclose(sparse_result.to_dense().numpy(),
                            dense_result.numpy())
    print('error {:2.2f}'.format(err), condition)
    assert condition


if __name__ == '__main__':
    test_kron()
