# Written by moritz ( @ wolter.tech ) in 2021

import torch
import numpy as np


from .sparse_math import (
    sparse_kron,
    sparse_diag,
    _dense_kron
)
from .conv_transform import (
    flatten_2d_coeff_lst,
    construct_2d_filt,
    get_filter_tensors
)
from .matmul_transform import (
    orthogonalize,
    cat_sparse_identity_matrix
)


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
    # return convolution_matrix.index_select(0, strided_rows)  # slow
    return torch.sparse.mm(selection_eye, convolution_matrix)  # fast


def construct_a_2d(wavelet, height: int, width: int,
                   device, dtype=torch.float64):
    dec_lo, dec_hi, _, _ = get_filter_tensors(
        wavelet, flip=False, device=device, dtype=dtype)
    dec_filt = construct_2d_filt(lo=dec_lo, hi=dec_hi)
    ll, lh, hl, hh = dec_filt.squeeze(1)
    analysis_ll = construct_strided_conv2d_matrix(
        ll, height, width, mode='sameshift')
    analysis_lh = construct_strided_conv2d_matrix(
        lh, height, width, mode='sameshift')
    analysis_hl = construct_strided_conv2d_matrix(
        hl, height, width, mode='sameshift')
    analysis_hh = construct_strided_conv2d_matrix(
        hh, height, width, mode='sameshift')
    analysis = torch.cat([analysis_ll, analysis_hl,
                          analysis_lh, analysis_hh], 0)
    return analysis


def construct_s_2d(wavelet, height: int, width: int,
                   device, dtype=torch.float64):
    _, _, rec_lo, rec_hi = get_filter_tensors(
        wavelet, flip=True, device=device, dtype=dtype)
    dec_filt = construct_2d_filt(lo=rec_lo, hi=rec_hi)
    ll, lh, hl, hh = dec_filt.squeeze(1)
    synthesis_ll = construct_strided_conv2d_matrix(
        ll, height, width, mode='sameshift')
    synthesis_lh = construct_strided_conv2d_matrix(
        lh, height, width, mode='sameshift')
    synthesis_hl = construct_strided_conv2d_matrix(
        hl, height, width, mode='sameshift')
    synthesis_hh = construct_strided_conv2d_matrix(
        hh, height, width, mode='sameshift')
    synthesis = torch.cat([synthesis_ll, synthesis_hl,
                           synthesis_lh, synthesis_hh], 0).coalesce()
    indices = synthesis.indices()
    shape = synthesis.shape
    transpose_indices = torch.stack([indices[1, :], indices[0, :]])
    transpose_synthesis = torch.sparse_coo_tensor(transpose_indices,
                                                  synthesis.values(),
                                                  size=(shape[1], shape[0]))
    return transpose_synthesis


def construct_boundary_a2d(
        wavelet, height: int, width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """ Construct a boundary fwt matrix for the input wavelet.

    Args:
        wavelet: The input wavelet, either a
            pywt.Wavelet or a ptwt.WaveletFilter.
        height (int): The height of the input matrix.
            Should be divisible by two.
        width (int): The width of the input matrix.
            Should be divisible by two.
        device (torch.device): Where to place the matrix. Either on
            the CPU or GPU.
        dtype ([type], optional): The desired data-type for the matrix.
            Defaults to torch.float64.

    Returns:
        [torch.Tensor]: A sparse fwt matrix, with orthogonalized boundary
            wavelets.
    """
    a = construct_a_2d(
        wavelet, height, width, device, dtype=dtype)
    orth_a = orthogonalize(
        a, len(wavelet)*len(wavelet))
    return orth_a


def construct_boundary_s2d(
        wavelet, height: int, width: int,
        device, dtype=torch.float64) -> torch.Tensor:
    """ Construct a 2d-fwt matrix, with boundary wavelets.

    Args:
        wavelet: A pywt wavelet.
        height (int): The original height of the input matrix.
        width (int): The width of the original input matrix.
        device (torch.device): Choose CPU or GPU.
        dtype (torch.dtype, optional): The data-type of the
            sparse matrix, choose float32 or 64.
            Defaults to torch.float64.

    Returns:
        torch.Tensor: The synthesis matrix, used to compute the
            inverse fast wavelet transform.
    """
    s = construct_s_2d(
        wavelet, height, width, device, dtype=dtype)
    orth_s = orthogonalize(
        s.transpose(1, 0), len(wavelet)*len(wavelet)).transpose(1, 0)
    return orth_s


class MatrixWavedec2d(object):
    """ Sparse matrix 2d wavelet transform.
        Constructing the sparse fwt-matrix is expensive.
        The matrix is therefore constructed only once and
        stored in this object for future use.
    """
    def __init__(self, wavelet, level: int):
        """ Creates a new matrix fwt object.

        Args:
            wavelet: A pywt wavelet.
            level (int): The level up to which to compute the fwt.
        """
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
        assert len(dec_lo) == len(dec_hi),\
            "All filters must have the same length."
        assert len(dec_hi) == len(rec_lo),\
            "All filters must have the same length."
        assert len(rec_lo) == len(rec_hi),\
            "All filters must have the same length."
        self.level = level
        self.wavelet = wavelet
        self.fwt_matrix = None
        self.input_signal_shape = None

    def __call__(self, input_signal: torch.Tensor) -> list:
        """ Compute the fwt for the given input signal.
            The fwt matrix is set up during the first call
            and stored for future use.

        Args:
            input_signal (torch.Tensor): An input signal of shape
                [height, width]

        Returns:
            [list]: The resulting coefficients per level stored in
            a pywt style list.
        """
        filt_len = len(self.wavelet)

        if input_signal.shape[-1] % 2 != 0:
            # odd length input
            # print('input length odd, padding a zero on the right')
            input_signal = torch.nn.functional.pad(input_signal, [0, 1])
        if input_signal.shape[-2] % 2 != 0:
            # odd length input
            # print('input length odd, padding a zero on the right')
            input_signal = torch.nn.functional.pad(
                input_signal, [0, 0, 0, 1])
        height, width = input_signal.shape

        re_build = False
        if self.input_signal_shape is None:
            self.input_signal_shape = input_signal.shape
        else:
            # if the input shape changed the matrix has to be
            # constructed again.
            if self.input_signal_shape[0] != input_signal.shape[0]:
                re_build = True
            if self.input_signal_shape[1] != input_signal.shape[1]:
                re_build = True

        if self.fwt_matrix is None or re_build:
            size_list = [(height, width)]
            fwt_mat_list = []
            if self.level is None:
                self.level = int(np.min(
                    [np.log2(height), np.log2(width)]))
            else:
                assert self.level > 0, "level must be a positive integer."

            for s in range(1, self.level + 1):
                current_height, current_width = size_list[-1]
                if current_height < filt_len or current_width < filt_len:
                    break
                analysis_matrix_2d = construct_boundary_a2d(
                    self.wavelet, current_height, current_width,
                    dtype=input_signal.dtype, device=input_signal.device)
                if s > 1:
                    analysis_matrix_2d = cat_sparse_identity_matrix(
                        analysis_matrix_2d, height*width)
                fwt_mat_list.append(analysis_matrix_2d)
                size_list.append((height // np.power(2, s),
                                  width // np.power(2, s)))

            self.fwt_matrix = fwt_mat_list[0]
            for fwt_mat in fwt_mat_list[1:]:
                self.fwt_matrix = torch.sparse.mm(fwt_mat, self.fwt_matrix)
            self.size_list = size_list

        coefficients = torch.sparse.mm(
            self.fwt_matrix, input_signal.flatten().unsqueeze(-1))

        split_list = []
        next_to_split = coefficients
        for size in self.size_list[1:]:
            split_size = int(np.prod(size))
            four_split = torch.split(next_to_split, split_size)
            next_to_split = four_split[0]
            reshaped = tuple(torch.reshape(el, size) for el in four_split[1:])
            split_list.append(reshaped)
        split_list.append(torch.reshape(next_to_split, size))

        return split_list[::-1]


class MatrixWaverec2d(object):
    """ Synthesis or inverse matrix based-fast wavelet
        transformation operation object.
        Constructing the fwt matrix is expensive.
        The matrix is, therefore, constructed only
        once and stored for later use.
    """

    def __init__(self, wavelet):
        """ Create the inverse matrix based fast wavelet
           transformation.

        Args:
            wavelet: A pywt.Wavelet compatible wavelet object.
        """
        self.wavelet = wavelet
        self.ifwt_matrix = None
        self.level = None

    def __call__(self, coefficients: list) -> torch.Tensor:
        """ Compute the inverse matrix 2d fast wavelet transform.

        Args:
            coefficients (list): The coefficient list as returned
            by the MatrixWavedec2d-Object.

        Returns:
            torch.Tensor: The  original signal reconstruction of
            shape [height, width].
        """
        level = len(coefficients) - 1
        re_build = False
        if self.level is None:
            self.level = level
        else:
            if self.level != level:
                self.level = level
                re_build = True

        coeff_vec = torch.cat(flatten_2d_coeff_lst(coefficients))
        shape = tuple(c*2 for c in coefficients[-1][0].shape)
        current_height, current_width = shape
        ifwt_mat_list = []
        if self.ifwt_matrix is None or re_build:
            for s in range(0, self.level):
                synthesis_matrix_2d = construct_boundary_s2d(
                    self.wavelet, current_height, current_width,
                    dtype=coefficients[-1][0].dtype,
                    device=coefficients[-1][0].device)
                if s >= 1:
                    synthesis_matrix_2d = cat_sparse_identity_matrix(
                        synthesis_matrix_2d, len(coeff_vec))
                current_height = current_height // 2
                current_width = current_width // 2
                ifwt_mat_list.append(synthesis_matrix_2d)

            self.ifwt_matrix = ifwt_mat_list[-1]
            for ifwt_mat in ifwt_mat_list[:-1][::-1]:
                self.ifwt_matrix = torch.sparse.mm(ifwt_mat, self.ifwt_matrix)

        reconstruction = torch.sparse.mm(
            self.ifwt_matrix, coeff_vec.unsqueeze(-1))

        return reconstruction.reshape(shape)


if __name__ == '__main__':
    import scipy
    import scipy.misc
    import pywt
    import time
    size = 128, 128
    level = 3
    wavelet_str = 'db2'
    face = np.mean(scipy.misc.face()[:size[0],
                                     :size[1]],
                   -1).astype(np.float64)
    pt_face = torch.tensor(face).cuda()
    wavelet = pywt.Wavelet(wavelet_str)
    matrixfwt = MatrixWavedec2d(wavelet, level=level)
    start_time = time.time()
    mat_coeff = matrixfwt(pt_face)
    total = time.time() - start_time
    print("runtime: {:2.2f}".format(total))
    start_time_2 = time.time()
    mat_coeff2 = matrixfwt(pt_face)
    total_2 = time.time() - start_time_2
    print("runtime: {:2.2f}".format(total_2))
    matrixifwt = MatrixWaverec2d(wavelet)
    reconstruction = matrixifwt(mat_coeff)
    reconstruction2 = matrixifwt(mat_coeff)
    # remove the padding
    if size[0] % 2 != 0:
        reconstruction = reconstruction[:-1, :]
    if size[1] % 2 != 0:
        reconstruction = reconstruction[:, :-1]
    err = np.sum(np.abs(reconstruction.cpu().numpy() - face))
    print(size, str(level).center(4),
          wavelet_str, "error {:3.3e}".format(err),
          np.allclose(reconstruction.cpu().numpy(), face))
    assert np.allclose(reconstruction.cpu().numpy(), face)
