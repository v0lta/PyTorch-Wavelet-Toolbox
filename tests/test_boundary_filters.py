# Created by moritz ( wolter@cs.uni-bonn.de ), 08.09.21
import pywt
import torch
import pytest
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from src.ptwt.matmul_transform import (
    construct_boundary_a,
    construct_boundary_s
)

from src.ptwt.mackey_glass import MackeyGenerator

@pytest.mark.slow
def test_boundary_filter_analysis_and_synthethis_matrices():
    for size in [24, 64, 128, 256]:
        for wavelet in [pywt.Wavelet("db4"), pywt.Wavelet("db6"), pywt.Wavelet("db8")]:
            analysis_matrix = construct_boundary_a(wavelet, size).to_dense()
            synthesis_matrix = construct_boundary_s(wavelet, size).to_dense()
            # s_db2 = construct_s(pywt.Wavelet("db8"), size)
            
            # test_eye_inv = torch.sparse.mm(a_db8, s_db2.to_dense()).numpy()
            test_eye_orth = torch.mm(analysis_matrix.transpose(1, 0), analysis_matrix).numpy()
            test_eye_inv = torch.mm(analysis_matrix, synthesis_matrix).numpy()
            err_inv = np.mean(np.abs(test_eye_inv - np.eye(size)))
            err_orth = np.mean(np.abs(test_eye_orth - np.eye(size)))

            print(wavelet.name, "orthogonal error", err_orth, 'size', size)
            print(wavelet.name, "inverse error", err_inv,  'size', size)
            assert err_orth < 1e-8
            assert err_inv < 1e-8



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
    block_width = input_columns
    block_entries = input_rows*filter_rows

    all_entries = filter_columns*input_columns*block_entries

    sparse_columns = np.zeros([all_entries])
    sparse_rows = np.zeros([all_entries])
    sparse_entries = torch.zeros([all_entries])

    matrix_height = (input_columns + filter_columns - 1)*block_height 
    matrix_width = input_columns*block_width

    col = np.stack([np.arange(0, input_rows)]*filter_rows) # + 1
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
        entries = torch.stack([entries]*input_rows).flatten() # double check T?
        entries = torch.stack([entries]*input_columns).flatten()
        start = col*input_columns*block_entries
        stop = start + input_columns*block_entries

        sparse_rows[start:stop] = row_offset 
        sparse_columns[start:stop] = column_offset
        sparse_entries[start:stop] = entries
        row_offset += block_height

    sparse_indices = np.stack([sparse_rows, sparse_columns])
    # sparse_indices = np.stack([sparse_columns, sparse_columns])
    matrix = torch.sparse_coo_tensor(sparse_indices, sparse_entries, dtype=dtype)
    # assert (matrix_height, matrix_width) == matrix.shape
    # plt.imshow(matrix.to_dense()); plt.show()
    return matrix


def test_mean_conv_matrix_2d():

    for filter_shape in [(3,3), (3,2), (2,3), (5,3), (3,5), (2,5), (5,2)]:
        for size in [(64, 64), (32, 64), (64, 32), (64, 31), (31, 64), (65, 65)]:
            # size = (256, 256)
            filter = torch.ones(filter_shape)/9.
            filter = filter.unsqueeze(0).unsqueeze(0)
            face = misc.face()[256:(256+size[0]), 256:(256+size[1])]
            face = np.mean(face, -1)

            # res = torch.nn.functional.conv2d(input=face, weight=filter)
            res = scipy.signal.convolve2d(face, filter.squeeze().numpy())

            conv_matrix2d = construct_conv2d_matrix(filter.squeeze(), size[0], size[1], torch.float32)

            # plt.imshow(conv_matrix2d.to_dense().numpy())
            # plt.show()
            face = torch.from_numpy(face.astype(np.float32))
            face = face.unsqueeze(0).unsqueeze(0)
            res_flat = torch.sparse.mm(conv_matrix2d, face.T.flatten().unsqueeze(-1))
            res_mm = np.reshape(res_flat.numpy(),
                [filter_shape[0] + size[0] - 1, filter_shape[1] + size[1] - 1], order='F')

            diff = np.mean(np.abs(res - res_mm))
            print(size, filter_shape, '%2.2e.'%diff, np.allclose(res, res_mm))
            # assert np.allclose(res, res_mm.numpy(), atol=1e-6)



if __name__ == '__main__':
    import torch
    import scipy.signal

    test_mean_conv_matrix_2d()

    filter_shape = [3,2]
    size = (5, 5)
    filter = torch.ones(filter_shape)/9.
    filter = filter.unsqueeze(0).unsqueeze(0)
    face = misc.face()[256:(256+size[0]), 256:(256+size[1])]
    face = np.mean(face, -1)

    # res = torch.nn.functional.conv2d(input=face, weight=filter)
    res = scipy.signal.convolve2d(face, filter.squeeze().numpy())

    conv_matrix2d = construct_conv2d_matrix(filter.squeeze(), size[0], size[1], torch.float32)

    # plt.spy(conv_matrix2d.to_dense().numpy(), marker='.')
    # plt.show()
    face = torch.from_numpy(face.astype(np.float32))
    face = face.unsqueeze(0).unsqueeze(0)
    res_flat = torch.sparse.mm(conv_matrix2d, face.T.flatten().unsqueeze(-1))
    res_mm = np.reshape(res_flat,
        [filter_shape[0] + size[0] - 1, filter_shape[1] + size[1] - 1], order='F')

    diff = np.abs(res - res_mm.numpy())
    print(np.mean(diff), np.allclose(res, res_mm.numpy()))
    plot = np.concatenate([res, res_mm.numpy(), diff], -1)
    plt.imshow(plot)
    plt.show()
