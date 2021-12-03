"""Two dimensional matrix based fast wavelet transform implementations.

This module uses boundary filters to minimize padding.
"""
# Written by moritz ( @ wolter.tech ) in 2021
import numpy as np
import pywt
import torch

from .conv_transform import construct_2d_filt, get_filter_tensors
from .matmul_transform import cat_sparse_identity_matrix, orthogonalize
from .sparse_math import construct_strided_conv2d_matrix


def _construct_a_2d(
    wavelet: pywt.Wavelet,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.tensor:
    """Construct a raw two dimensional analysis wavelet transformation matrix.

    Args:
        wavelet (pywt.Wavelet): The wavelet to use.
        height (int): The Height of the input image.
        width (int): The Width of the input image.
        device (torch.device): Where to place to matrix, either cpu or gpu.
        dtype (torch.dtype, optional): Desired matrix data-type.
            Defaults to torch.float64.

    Returns:
        torch.tensor: A sparse fwt analysis matrix.

    Note:
        The construced matrix is NOT necessary orthogonal.
        In most cases construct_boundary_a2d should be used instead.

    """
    dec_lo, dec_hi, _, _ = get_filter_tensors(
        wavelet, flip=False, device=device, dtype=dtype
    )
    dec_filt = construct_2d_filt(lo=dec_lo, hi=dec_hi)
    ll, lh, hl, hh = dec_filt.squeeze(1)
    analysis_ll = construct_strided_conv2d_matrix(ll, height, width, mode="sameshift")
    analysis_lh = construct_strided_conv2d_matrix(lh, height, width, mode="sameshift")
    analysis_hl = construct_strided_conv2d_matrix(hl, height, width, mode="sameshift")
    analysis_hh = construct_strided_conv2d_matrix(hh, height, width, mode="sameshift")
    analysis = torch.cat([analysis_ll, analysis_hl, analysis_lh, analysis_hh], 0)
    return analysis


def _construct_s_2d(
    wavelet: pywt.Wavelet,
    height: int,
    width: int,
    device: torch.device,
    dtype=torch.float64,
) -> torch.tensor:
    """Construct a raw fast wavelet transformation synthesis matrix.

    Note:
        The construced matrix is NOT necessary orthogonal.
        In most cases construct_boundary_s2d should be used instead.

    Args:
        wavelet (pywt.Wavelet): The wavelet to use.
        height (int): The height of the input image, which was originally
            transformed.
        width (int): The width of the input image, which was originally
            transformed.
        device ([type]): Where to place the synthesis matrix,
            usually cpu or gpu.
        dtype ([type], optional): The data-type the matrix should have.
            Defaults to torch.float64.

    Returns:
        [torch.tensor]: The generated fast wavelet synthesis matrix.
    """
    _, _, rec_lo, rec_hi = get_filter_tensors(
        wavelet, flip=True, device=device, dtype=dtype
    )
    dec_filt = construct_2d_filt(lo=rec_lo, hi=rec_hi)
    ll, lh, hl, hh = dec_filt.squeeze(1)
    synthesis_ll = construct_strided_conv2d_matrix(ll, height, width, mode="sameshift")
    synthesis_lh = construct_strided_conv2d_matrix(lh, height, width, mode="sameshift")
    synthesis_hl = construct_strided_conv2d_matrix(hl, height, width, mode="sameshift")
    synthesis_hh = construct_strided_conv2d_matrix(hh, height, width, mode="sameshift")
    synthesis = torch.cat(
        [synthesis_ll, synthesis_hl, synthesis_lh, synthesis_hh], 0
    ).coalesce()
    indices = synthesis.indices()
    shape = synthesis.shape
    transpose_indices = torch.stack([indices[1, :], indices[0, :]])
    transpose_synthesis = torch.sparse_coo_tensor(
        transpose_indices, synthesis.values(), size=(shape[1], shape[0])
    )
    return transpose_synthesis


def construct_boundary_a2d(
    wavelet,
    height: int,
    width: int,
    device: torch.device,
    boundary: str = "qr",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a boundary fwt matrix for the input wavelet.

    Args:
        wavelet: The input wavelet, either a
            pywt.Wavelet or a ptwt.WaveletFilter.
        height (int): The height of the input matrix.
            Should be divisible by two.
        width (int): The width of the input matrix.
            Should be divisible by two.
        device (torch.device): Where to place the matrix. Either on
            the CPU or GPU.
        boundary (str): The method to use for matrix orthogonalization.
            Choose qr or gramschmidt. Defaults to qr.
        dtype (torch.dtype, optional): The desired data-type for the matrix.
            Defaults to torch.float64.

    Returns:
        torch.Tensor: A sparse fwt matrix, with orthogonalized boundary
            wavelets.
    """
    a = _construct_a_2d(wavelet, height, width, device, dtype=dtype)
    orth_a = orthogonalize(a, len(wavelet) * len(wavelet), method=boundary)
    return orth_a


def construct_boundary_s2d(
    wavelet,
    height: int,
    width: int,
    device: torch.device,
    boundary: str = "qr",
    dtype=torch.float64,
) -> torch.Tensor:
    """Construct a 2d-fwt matrix, with boundary wavelets.

    Args:
        wavelet: A pywt wavelet.
        height (int): The original height of the input matrix.
        width (int): The width of the original input matrix.
        device (torch.device): Choose CPU or GPU.
        boundary (str): The method to use for matrix orthogonalization.
            Choose qr or gramschmidt. Defaults to qr.
        dtype (torch.dtype, optional): The data-type of the
            sparse matrix, choose float32 or 64.
            Defaults to torch.float64.

    Returns:
        torch.Tensor: The synthesis matrix, used to compute the
            inverse fast wavelet transform.
    """
    s = _construct_s_2d(wavelet, height, width, device, dtype=dtype)
    orth_s = orthogonalize(
        s.transpose(1, 0), len(wavelet) * len(wavelet), method=boundary
    ).transpose(1, 0)
    return orth_s


def _matrix_pad_2d(height: int, width: int) -> tuple:
    pad_tuple = (False, False)
    if height % 2 != 0:
        height += 1
        pad_tuple = (pad_tuple[0], True)
    if width % 2 != 0:
        width += 1
        pad_tuple = (True, pad_tuple[1])
    return height, width, pad_tuple


class MatrixWavedec2d(object):
    """Experimental sparse matrix 2d wavelet transform.

       For a completely pad free transform,
       input images are expected to be divisible by two.
       For multiscale transforms all intermediate
       scale dimensions should be divisible
       by two, i.e. 128, 128 -> 64, 64 -> 32, 32 would work
       well for a level three transform.
       In this case multiplication with the `sparse_fwt_operator`
       property is equivalent.

    Note:
        Constructing the sparse fwt-matrix is expensive.
        For longer wavelets, high level transforms, and large
        input images this may take a while.
        The matrix is therefore constructed only once.
        It can be accessed via the sparse_fwt_operator property.

    Example:
        >>> import ptwt, torch, pywt
        >>> import numpy as np
        >>> import scipy.misc
        >>> face = scipy.misc.face()[:256, :256, :].astype(np.float32)
        >>> pt_face = torch.tensor(face).permute([2, 0, 1])
        >>> matrixfwt = ptwt.MatrixWavedec2d(pywt.Wavelet("haar"), level=2)
        >>> mat_coeff = matrixfwt(pt_face)

    """

    def __init__(self, wavelet: pywt.Wavelet, level: int, boundary: str = "qr"):
        """Create a new matrix fwt object.

        Args:
            wavelet: A pywt wavelet.
            level (int): The level up to which to compute the fwt.
            boundary (str): The method used for boundary filter treatment.
                Choose 'qr' or 'gramschmidt'. 'qr' relies on pytorch's
                dense qr implementation, it is fast but memory hungry.
                The 'gramschmidt' option is sparse, memory efficient,
                and slow.
                Choose 'gramschmidt' if 'qr' runs out of memory.
                Defaults to 'qr'.
        """
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
        assert len(dec_lo) == len(dec_hi), "All filters must have the same length."
        assert len(dec_hi) == len(rec_lo), "All filters must have the same length."
        assert len(rec_lo) == len(rec_hi), "All filters must have the same length."
        self.level = level
        self.wavelet = wavelet
        self.input_signal_shape = None
        self.boundary = boundary
        self.fwt_matrix_list = []
        self.pad_list = []
        self.padded = False

    @property
    def sparse_fwt_operator(self) -> torch.Tensor:
        """Compute the operator matrix for padding-free cases.

           This property exists to make the transformation matrix available.
           To benefit from code handling odd-length levels call the object.

        Returns:
            torch.Tensor: The sparse 2d-fwt operator matrix.
        """
        if len(self.fwt_matrix_list) == 1:
            return self.fwt_matrix_list[0]
        elif len(self.fwt_matrix_list) > 1 and self.padded is False:
            fwt_matrix = self.fwt_matrix_list[0]
            for scale_mat in self.fwt_matrix_list[1:]:
                scale_mat = cat_sparse_identity_matrix(scale_mat, fwt_matrix.shape[0])
                fwt_matrix = torch.sparse.mm(scale_mat, fwt_matrix)
            return fwt_matrix
        else:
            return None

    def __call__(self, input_signal: torch.Tensor) -> list:
        """Compute the fwt for the given input signal.

        The fwt matrix is set up during the first call
        and stored for future use.

        Args:
            input_signal (torch.Tensor): An input signal of shape
                [batch_size, height, width]

        Returns:
            (list): The resulting coefficients per level stored in
            a pywt style list.
        """
        filt_len = len(self.wavelet)

        if input_signal.shape[1] == 1:
            input_signal = input_signal.squeeze(1)
        batch_size, height, width = input_signal.shape

        re_build = False
        if self.input_signal_shape is None:
            self.input_signal_shape = input_signal.shape[-2:]
        else:
            # the input shape changed rebuild the operator-matrices
            if self.input_signal_shape[0] != height:
                re_build = True
            if self.input_signal_shape[1] != width:
                re_build = True

        if not self.fwt_matrix_list or re_build:
            self.size_list = []
            self.fwt_matrix_list = []
            self.pad_list = []
            self.padded = False
            if self.level is None:
                self.level = int(np.min([np.log2(height), np.log2(width)]))
            else:
                assert self.level > 0, "level must be a positive integer."

            current_height, current_width = height, width
            for _ in range(1, self.level + 1):
                if current_height < filt_len or current_width < filt_len:
                    # we have reached the max decomposition depth.
                    break
                # the conv matrices require even length inputs.
                current_height, current_width, pad_tuple = _matrix_pad_2d(
                    current_height, current_width
                )
                if any(pad_tuple):
                    self.padded = True
                self.pad_list.append(pad_tuple)
                self.size_list.append((current_height, current_width))
                analysis_matrix_2d = construct_boundary_a2d(
                    self.wavelet,
                    current_height,
                    current_width,
                    dtype=input_signal.dtype,
                    device=input_signal.device,
                    boundary=self.boundary,
                )
                self.fwt_matrix_list.append(analysis_matrix_2d)
                current_height = current_height // 2
                current_width = current_width // 2
            self.size_list.append((current_height, current_width))

        split_list = []
        ll = input_signal.reshape([batch_size, -1]).T
        for scale, fwt_matrix in enumerate(self.fwt_matrix_list):
            pad = self.pad_list[scale]
            size = self.size_list[scale]
            if pad[0] or pad[1]:
                if pad[0] and not pad[1]:
                    ll_reshape = ll.T.reshape(batch_size, size[0], size[1] - 1)
                    ll = torch.nn.functional.pad(ll_reshape, [0, 1])
                elif pad[1] and not pad[0]:
                    ll_reshape = ll.T.reshape(batch_size, size[0] - 1, size[1])
                    ll = torch.nn.functional.pad(ll_reshape, [0, 0, 0, 1])
                elif pad[0] and pad[1]:
                    ll_reshape = ll.T.reshape(batch_size, size[0] - 1, size[1] - 1)
                    ll = torch.nn.functional.pad(ll_reshape, [0, 1, 0, 1])
                ll = ll.reshape([batch_size, -1]).T
            coefficients = torch.sparse.mm(fwt_matrix, ll)
            four_split = torch.split(
                coefficients, int(np.prod((size[0] // 2, size[1] // 2)))
            )
            reshaped = tuple(
                (el.T.reshape(batch_size, size[0] // 2, size[1] // 2))
                for el in four_split[1:]
            )
            split_list.append(reshaped)
            ll = four_split[0]
        split_list.append(ll.T.reshape(batch_size, size[0] // 2, size[1] // 2))
        return split_list[::-1]


class MatrixWaverec2d(object):
    """Synthesis or inverse matrix based-wavelet transformation object.

    Note:
        Constructing the fwt matrix is expensive.
        The matrix is, therefore, constructed only
        once and stored for later use.

    Example:
        >>> import ptwt, torch, pywt
        >>> import numpy as np
        >>> import scipy.misc
        >>> face = scipy.misc.face()[:256, :256, :].astype(np.float32)
        >>> pt_face = torch.tensor(face).permute([2, 0, 1])
        >>> matrixfwt = ptwt.MatrixWavedec2d(pywt.Wavelet("haar"), level=2)
        >>> mat_coeff = matrixfwt(pt_face)
        >>> matrixifwt = ptwt.MatrixWaverec2d(pywt.Wavelet("haar"))
        >>> reconstruction = matrixifwt(mat_coeff)
    """

    def __init__(self, wavelet: pywt.Wavelet, boundary: str = "qr"):
        """Create the inverse matrix based fast wavelet transformation.

        Args:
            wavelet: A pywt.Wavelet compatible wavelet object.
            boundary: The method used to treat the boundary cases.
                Choose 'qr' or 'gramschmidt'. Defaults to 'qr'.
        """
        self.wavelet = wavelet
        self.ifwt_matrix_list = []
        self.level = None
        self.boundary = boundary
        self.padded = False

    @property
    def sparse_ifwt_operator(self):
        """Compute the ifwt operator matrix for pad-free cases.

        Returns:
            torch.Tensor: The sparse 2d-ifwt operator matrix.
        """
        if len(self.ifwt_matrix_list) == 1:
            return self.ifwt_matrix_list[0]
        elif len(self.ifwt_matrix_list) > 1 and self.padded is False:
            ifwt_matrix = self.ifwt_matrix_list[-1]
            for scale_mat in self.ifwt_matrix_list[:-1][::-1]:
                ifwt_matrix = cat_sparse_identity_matrix(
                    ifwt_matrix, scale_mat.shape[0]
                )
                ifwt_matrix = torch.sparse.mm(scale_mat, ifwt_matrix)
            return ifwt_matrix
        else:
            return None

    def __call__(self, coefficients: list) -> torch.Tensor:
        """Compute the inverse matrix 2d fast wavelet transform.

        Args:
            coefficients (list): The coefficient list as returned
                                 by the MatrixWavedec2d-Object.

        Returns:
            torch.Tensor: The  original signal reconstruction of
            shape [batch_size, height, width].
        """
        level = len(coefficients) - 1
        re_build = False
        if self.level is None:
            self.level = level
        else:
            if self.level != level:
                self.level = level
                re_build = True

        height, width = tuple(c * 2 for c in coefficients[-1][0].shape[-2:])
        current_height, current_width = height, width

        if not self.ifwt_matrix_list or re_build:
            self.ifwt_matrix_list = []
            self.padded = False
            for _ in range(0, self.level):
                current_height, current_width, pad_tuple = _matrix_pad_2d(
                    current_height, current_width
                )
                if any(pad_tuple):
                    self.padded = True
                synthesis_matrix_2d = construct_boundary_s2d(
                    self.wavelet,
                    current_height,
                    current_width,
                    dtype=coefficients[-1][0].dtype,
                    device=coefficients[-1][0].device,
                    boundary=self.boundary,
                )
                current_height = current_height // 2
                current_width = current_width // 2
                self.ifwt_matrix_list.append(synthesis_matrix_2d)

        batch_size = coefficients[-1][0].shape[0]
        reconstruction = torch.reshape(coefficients[0], [batch_size, -1]).T
        for c_pos, lh_hl_hh in enumerate(coefficients[1:]):
            reconstruction = torch.cat(
                [
                    reconstruction.T,
                    lh_hl_hh[0].reshape([batch_size, -1]),
                    lh_hl_hh[1].reshape([batch_size, -1]),
                    lh_hl_hh[2].reshape([batch_size, -1]),
                ],
                -1,
            )
            ifwt_mat = self.ifwt_matrix_list[::-1][c_pos]
            reconstruction = torch.sparse.mm(ifwt_mat, reconstruction.T)

            # remove the padding
            if c_pos < len(coefficients) - 2:
                pred_len = [s * 2 for s in lh_hl_hh[0].shape[-2:]]
                next_len = list(coefficients[c_pos + 2][0].shape[-2:])
                if pred_len != next_len:
                    reconstruction = reconstruction.T.reshape([batch_size] + pred_len)
                    if pred_len[0] != next_len[0]:
                        reconstruction = reconstruction[:, :-1, :]
                    if pred_len[1] != next_len[1]:
                        reconstruction = reconstruction[:, :, :-1]
                    reconstruction = reconstruction.reshape(batch_size, -1).T
        return reconstruction.T.reshape((batch_size, height, width))


if __name__ == "__main__":
    import scipy
    import scipy.misc
    import pywt
    import time

    size = 32, 32
    level = 3
    wavelet_str = "db2"
    face = np.mean(scipy.misc.face()[: size[0], : size[1]], -1).astype(np.float64)
    pt_face = torch.tensor(face).cuda()
    wavelet = pywt.Wavelet(wavelet_str)
    matrixfwt = MatrixWavedec2d(wavelet, level=level)
    start_time = time.time()
    mat_coeff = matrixfwt(pt_face.unsqueeze(0))
    total = time.time() - start_time
    print("runtime: {:2.2f}".format(total))
    start_time_2 = time.time()
    mat_coeff2 = matrixfwt(pt_face.unsqueeze(0))
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
    print(
        size,
        str(level).center(4),
        wavelet_str,
        "error {:3.3e}".format(err),
        np.allclose(reconstruction.cpu().numpy(), face),
    )
    assert np.allclose(reconstruction.cpu().numpy(), face)
