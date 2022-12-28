"""Implement 3D seperable boundary transforms."""
import sys
from functools import partial
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch

from ._util import Wavelet, _as_wavelet, _is_boundary_mode_supported

# from .conv_transform import get_filter_tensors
from .matmul_transform import construct_boundary_a, construct_boundary_s
from .sparse_math import _batch_dim_mm


class PadTuple(NamedTuple):
    """Replaces PadTuple = namedtuple("PadTuple", ("depth", "height", "width"))."""

    depth: bool
    height: bool
    width: bool


def _matrix_pad_3(
    depth: int, height: int, width: int
) -> Tuple[int, int, int, PadTuple]:
    pad_depth, pad_height, pad_width = (False, False, False)
    if height % 2 != 0:
        height += 1
        pad_height = True
    if width % 2 != 0:
        width += 1
        pad_width = True
    if depth % 2 != 0:
        depth += 1
        pad_depth = True
    return depth, height, width, PadTuple(pad_depth, pad_height, pad_width)


class MatrixWavedec3(object):
    """Compute 3d seperable transforms."""

    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        level: Optional[int] = None,
        boundary: Optional[str] = "qr",
    ):
        """Create a *seperable* three dimensional fast boundary wavelet transform.

        Args:
            wavelet (Union[Wavelet, str]): The wavelet to use.
            level (Optional[int]): The desired decompositon level.
                Defaults to None.
            boundary (Optional[str]): The matrix orthogonalization method.
                Defaults to "qr".

        Raises:
            NotImplementedError: If the chosen orthogonalization method
                is not implemented.
            ValueError: If the analysis and synthesis filters do not have
                the same length.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.level = level
        self.boundary = boundary
        self.input_signal_shape: Optional[Tuple[int, int, int]] = None
        self.fwt_matrix_list: List[List[torch.Tensor]] = []

        if not _is_boundary_mode_supported(self.boundary):
            raise NotImplementedError
        if self.wavelet.dec_len != self.wavelet.rec_len:
            raise ValueError("All filters must have the same length")

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
        current_depth, current_height, current_width = self.input_signal_shape
        for curr_level in range(1, self.level + 1):
            if (
                current_height < filt_len
                or current_width < filt_len
                or current_depth < filt_len
            ):
                # we have reached the max decomposition depth.
                sys.stderr.write(
                    f"Warning: The selected number of decomposition levels {self.level}"
                    f" is too large for the given input shape {self.input_signal_shape}"
                    f". At level {curr_level}, at least one of the current signal "
                    f"depth, height, and width ({current_depth}, {current_height},"
                    f"{current_width}) is smaller "
                    f"than the filter length {filt_len}. Therefore, the transformation "
                    f"is only computed up to the decomposition level {curr_level-1}.\n"
                )
                break
            # the conv matrices require even length inputs.
            current_depth, current_height, current_width, pad_tuple = _matrix_pad_3(
                depth=current_depth, height=current_height, width=current_width
            )
            if any(pad_tuple):
                self.padded = True
            self.pad_list.append(pad_tuple)
            self.size_list.append((current_depth, current_height, current_width))

            matrix_construction_fun = partial(
                construct_boundary_a,
                wavelet=self.wavelet,
                boundary=self.boundary,
                device=device,
                dtype=dtype,
            )
            analysis_matrics = [
                matrix_construction_fun(length=dimension_length)
                for dimension_length in (current_depth, current_height, current_width)
            ]

            self.fwt_matrix_list.append(analysis_matrics)

            current_depth, current_height, current_width = (
                current_depth // 2,
                current_height // 2,
                current_width // 2,
            )
        self.size_list.append((current_depth, current_height, current_width))

    def __call__(
        self, input_signal: torch.Tensor
    ) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute a seperable 3d-boundary wavelet transform.

        Args:
            input_signal (torch.Tensor): An input signal of shape
                [batch_size, depth, height, width].

        Raises:
            ValueError: If the input dimensions dont work.

        Returns:
            List[Union[torch.Tensor, TypedDict[str, torch.Tensor]]]:
                A list with the approximation coefficients,
                and a coefficeint dict for each scale.
        """
        if input_signal.dim() == 3:
            # add batch dim to unbatched input
            input_signal = input_signal.unsqueeze(0)
        elif input_signal.dim() != 4:
            raise ValueError(
                f"Invalid input tensor shape {input_signal.size()}. "
                "The input signal is expected to be of the form "
                "[batch_size, depth, height, width]."
            )

        _, depth, height, width = input_signal.shape

        re_build = False
        if (
            self.input_signal_shape is None
            or self.input_signal_shape[0] != depth
            or self.input_signal_shape[1] != height
            or self.input_signal_shape[2] != width
        ):
            self.input_signal_shape = depth, height, width
            re_build = True

        if self.level is None:
            self.level = int(np.min([np.log2(depth), np.log2(height), np.log2(width)]))
            re_build = True
        elif self.level <= 0:
            raise ValueError("level must be a positive integer.")

        if not self.fwt_matrix_list or re_build:
            self._construct_analysis_matrices(
                device=input_signal.device, dtype=input_signal.dtype
            )

        split_list: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = []
        ll = input_signal
        for scale, fwt_mats in enumerate(self.fwt_matrix_list):
            # fwt_depth_matrix, fwt_row_matrix, fwt_col_matrix = fwt_mats
            pad_tuple = self.pad_list[scale]
            # current_depth, current_height, current_width = self.size_list[scale]
            if pad_tuple.width:
                ll = torch.nn.functional.pad(ll, [0, 1, 0, 0, 0, 0])
            if pad_tuple.height:
                ll = torch.nn.functional.pad(ll, [0, 0, 0, 1, 0, 0])
            if pad_tuple.depth:
                ll = torch.nn.functional.pad(ll, [0, 0, 0, 0, 0, 1])

            for dim, mat in enumerate(fwt_mats[::-1]):
                ll = _batch_dim_mm(mat, ll, dim=(-1) * (dim + 1))

            def _split_rec(
                tensor: torch.Tensor,
                key: str,
                depth: int,
                dict: Dict[str, torch.Tensor],
            ) -> None:
                if key:
                    dict[key] = tensor
                if len(key) < depth:
                    dim = len(key) + 1
                    ca, cd = torch.split(tensor, tensor.shape[-dim] // 2, dim=-dim)
                    _split_rec(ca, "a" + key, depth, dict)
                    _split_rec(cd, "d" + key, depth, dict)

            coeff_dict: Dict[str, torch.Tensor] = {}
            _split_rec(ll, "", 3, coeff_dict)
            ll = coeff_dict["aaa"]
            result_keys = list(
                filter(lambda x: len(x) == 3 and not x == "aaa", coeff_dict.keys())
            )
            coeff_dict = {
                key: tensor for key, tensor in coeff_dict.items() if key in result_keys
            }
            split_list.append(coeff_dict)
        split_list.append(ll)
        return split_list[::-1]


class MatrixWaverec3(object):
    """Reconstruct a signal from 3d-seperable-fwt coefficients."""

    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        boundary: str = "qr",
    ):
        """Compute a three dimensional seperable boundary wavlet synthesis transform.

        Args:
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
            boundary (str): The method used for boundary filter treatment.
                Choose 'qr' or 'gramschmidt'. 'qr' relies on pytorch's dense qr
                implementation, it is fast but memory hungry. The 'gramschmidt' option
                is sparse, memory efficient, and slow. Choose 'gramschmidt' if 'qr' runs
                out of memory. Defaults to 'qr'.

        Raises:
            NotImplementedError: If the selected `boundary` mode is not supported.
            ValueError: If the wavelet filters have different lenghts.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.boundary = boundary
        self.ifwt_matrix_list: List[List[torch.Tensor]] = []
        self.input_signal_shape: Optional[Tuple[int, int, int]] = None
        self.level: Optional[int] = None

        if not _is_boundary_mode_supported(self.boundary):
            raise NotImplementedError

        if self.wavelet.dec_len != self.wavelet.rec_len:
            raise ValueError("All filters must have the same length")

    def _construct_synthesis_matrices(
        self,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> None:
        self.ifwt_matrix_list = []
        self.padded = False
        if self.level is None or self.input_signal_shape is None:
            raise AssertionError

        current_depth, current_height, current_width = self.input_signal_shape
        filt_len = self.wavelet.rec_len

        for curr_level in range(1, self.level + 1):
            if (
                current_depth < filt_len
                or current_height < filt_len
                or current_width < filt_len
            ):
                sys.stderr.write(
                    f"Warning: The selected number of decomposition levels {self.level}"
                    f" is too large for the given input shape {self.input_signal_shape}"
                    f". At level {curr_level}, at least one of the current signal "
                    f" depth, height and width ({current_depth}, {current_height}, "
                    f"{current_width}) is smaller than the filter length {filt_len}."
                    f" Therefore, the transformation "
                    f"is only computed up to the decomposition level {curr_level-1}.\n"
                )
                break
            # the conv matrices require even length inputs.
            current_depth, current_height, current_width, pad_tuple = _matrix_pad_3(
                depth=current_depth, height=current_height, width=current_width
            )
            if any(pad_tuple):
                self.padded = True

            matrix_construction_fun = partial(
                construct_boundary_s,
                wavelet=self.wavelet,
                boundary=self.boundary,
                device=device,
                dtype=dtype,
            )
            synthesis_matrices = [
                matrix_construction_fun(length=dimension_length)
                for dimension_length in (current_depth, current_height, current_width)
            ]

            self.ifwt_matrix_list.append(synthesis_matrices)
            current_depth, current_height, current_width = (
                current_depth // 2,
                current_height // 2,
                current_width // 2,
            )

    def __call__(
        self, coefficients: List[Union[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        """Recustruct a batched 3d-signal from it's coefficients.

        Args:
            coefficients (List[Union[torch.Tensor, Dict[str, torch.Tensor]]]):
                The output from MatrixWavedec3.

        Returns:
            torch.Tensor: A reconstruction of the original signal.

        Raises:
            ValueError: If the data structure is inconsistent.
        """
        level = len(coefficients) - 1
        if type(coefficients[-1]) is dict:
            depth, height, width = tuple(
                c * 2 for c in coefficients[-1]["ddd"].shape[-3:]
            )
        else:
            raise ValueError("Waverec3 expects dicts of tensors.")

        re_build = False
        if (
            self.input_signal_shape is None
            or self.input_signal_shape[0] != depth
            or self.input_signal_shape[1] != height
            or self.input_signal_shape[2] != width
        ):
            self.input_signal_shape = depth, height, width
            re_build = True

        if self.level != level:
            self.level = level
            re_build = True

        if not self.ifwt_matrix_list or re_build:
            self._construct_synthesis_matrices(
                device=coefficients[-1]["ddd"].device,
                dtype=coefficients[-1]["ddd"].dtype,
            )

        ll: torch.Tensor = coefficients[0]  # type: ignore
        if not isinstance(ll, torch.Tensor):
            raise ValueError(
                "First element of coeffs must be the approximation coefficient tensor."
            )

        for c_pos, coeff_dict in enumerate(coefficients[1:]):
            if not isinstance(coeff_dict, dict):
                raise ValueError(
                    (
                        "Unexpected detail coefficient type: {}. Detail coefficients "
                        "must be a 3-tuple of tensors as returned by MatrixWavedec2."
                    ).format(type(coeff_dict))
                )

            def _cat_coeff_recursive(dict: Dict[str, torch.Tensor]) -> torch.Tensor:
                done_dict = {}
                a_initial_keys = list(filter(lambda x: x[0] == "a", dict.keys()))
                for a_key in a_initial_keys:
                    d_key = "d" + a_key[1:]
                    cat_d = dict[d_key]
                    d_shape = cat_d.shape
                    # undo any analysis padding.
                    cat_a = dict[a_key][:, : d_shape[1], : d_shape[2], : d_shape[3]]
                    cat_tensor = torch.cat([cat_a, cat_d], dim=-len(a_key))
                    if a_key[1:]:
                        done_dict[a_key[1:]] = cat_tensor
                    else:
                        return cat_tensor

                return _cat_coeff_recursive(done_dict)

            coeff_dict["a" * len(list(coeff_dict.keys())[-1])] = ll
            ll = _cat_coeff_recursive(coeff_dict)

            for dim, mat in enumerate(self.ifwt_matrix_list[level - 1 - c_pos][::-1]):
                ll = _batch_dim_mm(mat, ll, dim=(-1) * (dim + 1))

        return ll
