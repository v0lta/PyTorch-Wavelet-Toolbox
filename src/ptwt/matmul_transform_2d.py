"""Two dimensional matrix based fast wavelet transform implementations.

This module uses boundary filters to minimize padding.
"""
# Written by moritz ( @ wolter.tech ) in 2021
import sys
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import torch

from ._util import Wavelet, _as_wavelet, _is_boundary_mode_supported
from .conv_transform import construct_2d_filt, get_filter_tensors
from .matmul_transform import (
    cat_sparse_identity_matrix,
    construct_boundary_a,
    construct_boundary_s,
    orthogonalize,
)
from .sparse_math import batch_mm, construct_strided_conv2d_matrix


def _construct_a_2d(
    wavelet: Union[Wavelet, str],
    height: int,
    width: int,
    device: Union[torch.device, str],
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a raw two dimensional analysis wavelet transformation matrix.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        height (int): The height of the input image.
        width (int): The width of the input image.
        device (torch.device or str): Where to place the matrix.
        dtype (torch.dtype, optional): Desired matrix data-type.
            Defaults to torch.float64.

    Returns:
        torch.Tensor: A sparse fwt analysis matrix.

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
    wavelet: Union[Wavelet, str],
    height: int,
    width: int,
    device: Union[torch.device, str],
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a raw fast wavelet transformation synthesis matrix.

    Note:
        The construced matrix is NOT necessary orthogonal.
        In most cases construct_boundary_s2d should be used instead.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        height (int): The height of the input image, which was originally
            transformed.
        width (int): The width of the input image, which was originally
            transformed.
        device ([type]): Where to place the synthesis matrix,
            usually cpu or gpu.
        dtype ([type], optional): The data-type the matrix should have.
            Defaults to torch.float64.

    Returns:
        [torch.Tensor]: The generated fast wavelet synthesis matrix.
    """
    wavelet = _as_wavelet(wavelet)
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
    wavelet: Union[Wavelet, str],
    height: int,
    width: int,
    device: Union[torch.device, str],
    boundary: str = "qr",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a boundary fwt matrix for the input wavelet.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
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
    wavelet = _as_wavelet(wavelet)
    a = _construct_a_2d(wavelet, height, width, device, dtype=dtype)
    orth_a = orthogonalize(a, wavelet.dec_len ** 2, method=boundary)  # noqa: BLK100
    return orth_a


def construct_boundary_s2d(
    wavelet: Union[Wavelet, str],
    height: int,
    width: int,
    device: Union[torch.device, str],
    boundary: str = "qr",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a 2d-fwt matrix, with boundary wavelets.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
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
    wavelet = _as_wavelet(wavelet)
    s = _construct_s_2d(wavelet, height, width, device, dtype=dtype)
    orth_s = orthogonalize(
        s.transpose(1, 0), wavelet.rec_len ** 2, method=boundary  # noqa: BLK100
    ).transpose(1, 0)
    return orth_s


def _matrix_pad_2d(height: int, width: int) -> Tuple[int, int, Tuple[bool, bool]]:
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

    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        level: Optional[int] = None,
        boundary: str = "qr",
        separable: bool = False,
    ):
        """Create a new matrix fwt object.

        Args:
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
            level (int, optional): The level up to which to compute the fwt. If None,
                the maximum level based on the signal length is chosen. Defaults to
                None.
            boundary (str): The method used for boundary filter treatment.
                Choose 'qr' or 'gramschmidt'. 'qr' relies on pytorch's
                dense qr implementation, it is fast but memory hungry.
                The 'gramschmidt' option is sparse, memory efficient,
                and slow.
                Choose 'gramschmidt' if 'qr' runs out of memory.
                Defaults to 'qr'.
            separable (bool): If this flag is set, a separable transformation
                is used, i.e. a 1d transformation along each axis. This is significantly
                faster than a non-separable transformation since only a small constant-
                size part of the matrices must be orthogonalized. Defaults to False.

        Raises:
            NotImplementedError: If the selected `boundary` mode is not supported.
            ValueError: If the wavelet filters have different lenghts.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.level = level
        self.boundary = boundary
        self.separable = separable
        self.input_signal_shape: Optional[Tuple[int, int]] = None
        self.fwt_matrix_list: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = []
        self.pad_list: List[Tuple[bool, bool]] = []
        self.padded = False

        # TODO: Allow separate wavelets and lengths for each axis in the separable case

        if not _is_boundary_mode_supported(self.boundary):
            raise NotImplementedError

        if self.wavelet.dec_len != self.wavelet.rec_len:
            raise ValueError("All filters must have the same length")

    @property
    def sparse_fwt_operator(self) -> torch.Tensor:
        """Compute the operator matrix for padding-free cases.

            This property exists to make the transformation matrix available.
            To benefit from code handling odd-length levels call the object.

        Returns:
            torch.Tensor: The sparse 2d-fwt operator matrix.

        Raises:
            NotImplementedError: if a separable transformation was used or if padding
                had to be used in the creation of the transformation matrices.
            ValueError: If no level transformation matrices are stored (most likely
                since the object was not called yet).
        """
        if self.separable:
            raise NotImplementedError

        # in the non-separable case the list entries are tensors
        fwt_matrix_list = cast(List[torch.Tensor], self.fwt_matrix_list)

        if len(fwt_matrix_list) == 1:
            return fwt_matrix_list[0]
        elif len(fwt_matrix_list) > 1:
            if self.padded:
                raise NotImplementedError

            fwt_matrix = fwt_matrix_list[0]
            for scale_mat in fwt_matrix_list[1:]:
                scale_mat = cat_sparse_identity_matrix(scale_mat, fwt_matrix.shape[0])
                fwt_matrix = torch.sparse.mm(scale_mat, fwt_matrix)
            return fwt_matrix
        else:
            raise ValueError(
                "Call this object first to create the transformation matrices for each "
                "level."
            )

    def _construct_analysis_matrices(
        self,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> None:
        if self.level is None or self.input_signal_shape is None:
            raise AssertionError
        self.fwt_matrix_list = []
        self.size_list = []
        self.pad_list = []
        self.padded = False

        filt_len = self.wavelet.dec_len
        current_height, current_width = self.input_signal_shape
        for curr_level in range(1, self.level + 1):
            if current_height < filt_len or current_width < filt_len:
                # we have reached the max decomposition depth.
                sys.stderr.write(
                    f"Warning: The selected number of decomposition levels {self.level}"
                    f" is too large for the given input shape {self.input_signal_shape}"
                    f". At level {curr_level}, at least one of the current signal "
                    f"height and width ({current_height}, {current_width}) is smaller "
                    f"than the filter length {filt_len}. Therefore, the transformation "
                    f"is only computed up to the decomposition level {curr_level-1}.\n"
                )
                break
            # the conv matrices require even length inputs.
            current_height, current_width, pad_tuple = _matrix_pad_2d(
                current_height, current_width
            )
            if any(pad_tuple):
                self.padded = True
            self.pad_list.append(pad_tuple)
            self.size_list.append((current_height, current_width))
            if self.separable:
                analysis_matrix_rows = construct_boundary_a(
                    wavelet=self.wavelet,
                    length=current_height,
                    boundary=self.boundary,
                    device=device,
                    dtype=dtype,
                )
                analysis_matrix_cols = construct_boundary_a(
                    wavelet=self.wavelet,
                    length=current_width,
                    boundary=self.boundary,
                    device=device,
                    dtype=dtype,
                )
                self.fwt_matrix_list.append(
                    (analysis_matrix_rows, analysis_matrix_cols)
                )
            else:
                analysis_matrix_2d = construct_boundary_a2d(
                    self.wavelet,
                    current_height,
                    current_width,
                    boundary=self.boundary,
                    device=device,
                    dtype=dtype,
                )
                self.fwt_matrix_list.append(analysis_matrix_2d)
            current_height = current_height // 2
            current_width = current_width // 2
        self.size_list.append((current_height, current_width))

    def __call__(
        self, input_signal: torch.Tensor
    ) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Compute the fwt for the given input signal.

        The fwt matrix is set up during the first call
        and stored for future use.

        Args:
            input_signal (torch.Tensor): An input signal of shape
                [batch_size, height, width]

        Returns:
            (list): The resulting coefficients per level stored in
            a pywt style list.

        Raises:
            ValueError: If the decomposition level is not a positive integer.
        """
        if input_signal.shape[1] == 1:
            input_signal = input_signal.squeeze(1)

        if len(input_signal.shape) == 2:
            input_signal = input_signal.unsqueeze(0)

        batch_size, height, width = input_signal.shape

        re_build = False
        if (
            self.input_signal_shape is None
            or self.input_signal_shape[0] != height
            or self.input_signal_shape[1] != width
        ):
            self.input_signal_shape = height, width
            re_build = True

        if self.level is None:
            self.level = int(np.min([np.log2(height), np.log2(width)]))
            re_build = True
        elif self.level <= 0:
            raise ValueError("level must be a positive integer.")

        if not self.fwt_matrix_list or re_build:
            self._construct_analysis_matrices(
                device=input_signal.device, dtype=input_signal.dtype
            )

        split_list: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []
        if self.separable:
            ll = input_signal
            for scale, fwt_mats in enumerate(self.fwt_matrix_list):
                fwt_row_matrix, fwt_col_matrix = fwt_mats
                pad = self.pad_list[scale]
                current_height, current_width = self.size_list[scale]
                if pad[0] or pad[1]:
                    if pad[0] and not pad[1]:
                        ll = torch.nn.functional.pad(ll, [0, 1])
                    elif pad[1] and not pad[0]:
                        ll = torch.nn.functional.pad(ll, [0, 0, 0, 1])
                    elif pad[0] and pad[1]:
                        ll = torch.nn.functional.pad(ll, [0, 1, 0, 1])

                ll = batch_mm(fwt_col_matrix, ll.transpose(-2, -1)).transpose(-2, -1)
                ll = batch_mm(fwt_row_matrix, ll)

                a_coeffs, d_coeffs = torch.split(ll, current_height // 2, dim=-2)
                ll, lh = torch.split(a_coeffs, current_width // 2, dim=-1)
                hl, hh = torch.split(d_coeffs, current_width // 2, dim=-1)

                # TODO: Is the order consistent with the non-separable case?
                split_list.append((lh, hl, hh))
            split_list.append(ll)
        else:
            ll = input_signal.reshape([batch_size, -1]).T
            for scale, fwt_matrix in enumerate(self.fwt_matrix_list):
                fwt_matrix = cast(torch.Tensor, fwt_matrix)
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
                reshaped = cast(
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    tuple(
                        (el.T.reshape(batch_size, size[0] // 2, size[1] // 2))
                        for el in four_split[1:]
                    ),
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

    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        boundary: str = "qr",
        separable: bool = False,
    ):
        """Create the inverse matrix based fast wavelet transformation.

        Args:
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
            boundary (str): The method used for boundary filter treatment.
                Choose 'qr' or 'gramschmidt'. 'qr' relies on pytorch's dense qr
                implementation, it is fast but memory hungry. The 'gramschmidt' option
                is sparse, memory efficient, and slow. Choose 'gramschmidt' if 'qr' runs
                out of memory. Defaults to 'qr'.
            separable (bool): If this flag is set, a separable transformation
                is used, i.e. a 1d transformation along each axis. This is significantly
                faster than a non-separable transformation since only a small constant-
                size part of the matrices must be orthogonalized. Defaults to False.

        Raises:
            NotImplementedError: If the selected `boundary` mode is not supported.
            ValueError: If the wavelet filters have different lenghts.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.boundary = boundary
        self.separable = separable

        self.ifwt_matrix_list: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = []
        self.level: Optional[int] = None
        self.input_signal_shape: Optional[Tuple[int, int]] = None

        self.padded = False

        # TODO: Allow separate wavelets and lengths for each axis in the separable case

        if not _is_boundary_mode_supported(self.boundary):
            raise NotImplementedError

        if self.wavelet.dec_len != self.wavelet.rec_len:
            raise ValueError("All filters must have the same length")

    @property
    def sparse_ifwt_operator(self) -> torch.Tensor:
        """Compute the ifwt operator matrix for pad-free cases.

        Returns:
            torch.Tensor: The sparse 2d-ifwt operator matrix.

        Raises:
            NotImplementedError: if a separable transformation was used or if padding
                had to be used in the creation of the transformation matrices.
            ValueError: If no level transformation matrices are stored (most likely
                since the object was not called yet).
        """
        if self.separable:
            raise NotImplementedError

        # in the non-separable case the list entries are tensors
        ifwt_matrix_list = cast(List[torch.Tensor], self.ifwt_matrix_list)

        if len(ifwt_matrix_list) == 1:
            return ifwt_matrix_list[0]
        elif len(ifwt_matrix_list) > 1:
            if self.padded:
                raise NotImplementedError

            ifwt_matrix = ifwt_matrix_list[-1]
            for scale_mat in ifwt_matrix_list[:-1][::-1]:
                ifwt_matrix = cat_sparse_identity_matrix(
                    ifwt_matrix, scale_mat.shape[0]
                )
                ifwt_matrix = torch.sparse.mm(scale_mat, ifwt_matrix)
            return ifwt_matrix
        else:
            raise ValueError(
                "Call this object first to create the transformation matrices for each "
                "level."
            )

    def _construct_synthesis_matrices(
        self,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> None:
        self.ifwt_matrix_list = []
        self.padded = False
        if self.level is None or self.input_signal_shape is None:
            raise AssertionError

        current_height, current_width = self.input_signal_shape
        filt_len = self.wavelet.rec_len

        for curr_level in range(1, self.level + 1):
            if current_height < filt_len or current_width < filt_len:
                sys.stderr.write(
                    f"Warning: The selected number of decomposition levels {self.level}"
                    f" is too large for the given input shape {self.input_signal_shape}"
                    f". At level {curr_level}, at least one of the current signal "
                    f"height and width ({current_height}, {current_width}) is smaller "
                    f"than the filter length {filt_len}. Therefore, the transformation "
                    f"is only computed up to the decomposition level {curr_level-1}.\n"
                )
                break
            current_height, current_width, pad_tuple = _matrix_pad_2d(
                current_height, current_width
            )
            if any(pad_tuple):
                self.padded = True
            if self.separable:
                synthesis_matrix_rows = construct_boundary_s(
                    wavelet=self.wavelet,
                    length=current_height,
                    boundary=self.boundary,
                    device=device,
                    dtype=dtype,
                )
                synthesis_matrix_cols = construct_boundary_s(
                    wavelet=self.wavelet,
                    length=current_width,
                    boundary=self.boundary,
                    device=device,
                    dtype=dtype,
                )
                self.ifwt_matrix_list.append(
                    (synthesis_matrix_rows, synthesis_matrix_cols)
                )
            else:
                synthesis_matrix_2d = construct_boundary_s2d(
                    self.wavelet,
                    current_height,
                    current_width,
                    boundary=self.boundary,
                    device=device,
                    dtype=dtype,
                )
                self.ifwt_matrix_list.append(synthesis_matrix_2d)
            current_height = current_height // 2
            current_width = current_width // 2

    def _process_coeffs(
        self,
        ll: torch.Tensor,
        lh_hl_hh: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Size,
        torch.device,
        torch.dtype,
    ]:
        if len(lh_hl_hh) != 3:
            raise ValueError(
                "Detail coefficients must be a 3-tuple of tensors as returned by "
                "MatrixWavedec2."
            )

        lh, hl, hh = lh_hl_hh

        torch_device = None
        curr_shape = None
        torch_dtype = None
        for coeff in [ll, lh, hl, hh]:
            if coeff is not None:
                if curr_shape is None:
                    curr_shape = coeff.shape
                    torch_device = coeff.device
                    torch_dtype = coeff.dtype
                elif curr_shape != coeff.shape:
                    # TODO: Add check that coeffs are on the same device
                    raise ValueError("coeffs must have the same shape")

        if torch_device is None or curr_shape is None or torch_dtype is None:
            raise ValueError("At least one coefficient parameter must be specified.")

        if ll is None:
            ll = torch.zeros(curr_shape, device=torch_device, dtype=torch_dtype)
        if hl is None:
            hl = torch.zeros(curr_shape, device=torch_device, dtype=torch_dtype)
        if lh is None:
            lh = torch.zeros(curr_shape, device=torch_device, dtype=torch_dtype)
        if hh is None:
            hh = torch.zeros(curr_shape, device=torch_device, dtype=torch_dtype)
        return (ll, lh, hl, hh), curr_shape, torch_device, torch_dtype

    def __call__(
        self,
        coefficients: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ],
    ) -> torch.Tensor:
        """Compute the inverse matrix 2d fast wavelet transform.

        Args:
            coefficients (list): The coefficient list as returned
                by the `MatrixWavedec2d`-Object.

        Returns:
            torch.Tensor: The original signal reconstruction of
                shape [batch_size, height, width].

        Raises:
            ValueError: If the decomposition level is not a positive integer or if the
                coefficients are not in the shape as it is returned from a
                `MatrixWavedec2` object.
        """
        level = len(coefficients) - 1
        height, width = tuple(c * 2 for c in coefficients[-1][0].shape[-2:])

        re_build = False
        if (
            self.input_signal_shape is None
            or self.input_signal_shape[0] != height
            or self.input_signal_shape[1] != width
        ):
            self.input_signal_shape = height, width
            re_build = True

        if self.level != level:
            self.level = level
            re_build = True

        # TODO: handle coefficients[-1][0] == None
        if not self.ifwt_matrix_list or re_build:
            self._construct_synthesis_matrices(
                device=coefficients[-1][0].device,
                dtype=coefficients[-1][0].dtype,
            )

        batch_size = coefficients[-1][0].shape[0]
        ll = coefficients[0]
        if not isinstance(ll, torch.Tensor):
            raise ValueError(
                "First element of coeffs must be the approximation coefficient tensor."
            )

        for c_pos, coeff_tuple in enumerate(coefficients[1:]):
            if not isinstance(coeff_tuple, tuple) or len(coeff_tuple) != 3:
                raise ValueError(
                    (
                        "Unexpected detail coefficient type: {}. Detail coefficients "
                        "must be a 3-tuple of tensors as returned by MatrixWavedec2."
                    ).format(type(coeff_tuple))
                )
            (ll, lh, hl, hh), curr_shape, _, _ = self._process_coeffs(ll, coeff_tuple)

            if self.separable:
                synthesis_matrix_rows, synthesis_matrix_cols = self.ifwt_matrix_list[
                    ::-1
                ][c_pos]
                a_coeffs = torch.cat((ll, lh), -1)
                d_coeffs = torch.cat((hl, hh), -1)
                coeff_tensor = torch.cat((a_coeffs, d_coeffs), -2)
                if len(curr_shape) == 2:
                    coeff_tensor = coeff_tensor.unsqueeze(0)
                ll = batch_mm(
                    synthesis_matrix_cols, coeff_tensor.transpose(-2, -1)
                ).transpose(-2, -1)
                ll = batch_mm(synthesis_matrix_rows, ll)
            else:
                ll = torch.cat(
                    [
                        ll.reshape([batch_size, -1]),
                        lh.reshape([batch_size, -1]),
                        hl.reshape([batch_size, -1]),
                        hh.reshape([batch_size, -1]),
                    ],
                    -1,
                )
                ifwt_mat = cast(torch.Tensor, self.ifwt_matrix_list[::-1][c_pos])
                ll = torch.sparse.mm(ifwt_mat, ll.T)

            pred_len = [s * 2 for s in curr_shape[-2:]]
            if not self.separable:
                ll = ll.T.reshape([batch_size] + pred_len)
            # remove the padding
            if c_pos < len(coefficients) - 2:
                next_len = list(coefficients[c_pos + 2][0].shape[-2:])
                if pred_len != next_len:
                    if pred_len[0] != next_len[0]:
                        ll = ll[:, :-1, :]
                    if pred_len[1] != next_len[1]:
                        ll = ll[:, :, :-1]

        return ll


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
