"""Compute analysis wavelet packet representations."""
# Created on Fri Apr 6 2021 by moritz (wolter@cs.uni-bonn.de)

import collections
from functools import partial
from itertools import product
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import pywt
import torch


from ._util import Wavelet, _as_wavelet
from .conv_transform import wavedec, wavedec2
from .matmul_transform import MatrixWavedec
from .matmul_transform_2d import MatrixWavedec2d

if TYPE_CHECKING:
    BaseDict = collections.UserDict[str, torch.Tensor]
else:
    BaseDict = collections.UserDict


class WaveletPacket(BaseDict):
    """One dimensional wavelet packets."""

    def __init__(
        self,
        data: Optional[torch.Tensor],
        wavelet: Union[Wavelet, str],
        mode: str = "reflect",
        boundary_orthogonalization: str = "qr",
    ) -> None:
        """Create a wavelet packet decomposition object.

        The decompositions will rely on padded fast wavelet transforms.

        Args:
            data (torch.Tensor, optional): The input data array of shape [time]
                or [batch_size, time]. If None, the object is initialized without
                performing a decomposition.
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
            mode (str): The desired padding method. If you select 'boundary',
                the sparse matrix backend will be used. Defaults to 'reflect'.
            boundary_orthogonalization (str): The orthogonalization method
                to use. Only used if `mode` equals 'boundary'. Choose from
                'qr' or 'gramschmidt'. Defaults to 'qr'.
        """
        self.wavelet = _as_wavelet(wavelet)
        if mode == "zero":
            # translate pywt to PyTorch
            self.mode = "constant"
        else:
            self.mode = mode
        self.boundary = boundary_orthogonalization
        self._matrix_wavedec_dict: Dict[int, MatrixWavedec] = {}
        if data is not None:
            self.transform(data)
        else:
            self.data = {}

    def transform(
        self, data: torch.Tensor, max_level: Optional[int] = None
    ) -> "WaveletPacket":
        """Calculate the 1d wavelet packet transform for the input data.

        Args:
            data (torch.Tensor): The input data array of shape [time]
                or [batch_size, time].
            max_level (int, optional): The highest decomposition level to compute.
                If None, the maximum level is determined from the input data shape.
                Defaults to None.
        """
        self.data = {}
        if max_level is None:
            max_level = pywt.dwt_max_level(data.shape[-1], self.wavelet.dec_len)
        self.max_level = max_level
        self._recursive_dwt(data, level=0, path="")
        return self

    def _get_wavedec(
        self,
        length: int,
    ) -> Callable[[torch.Tensor], List[torch.Tensor]]:
        if self.mode == "boundary":
            if length not in self._matrix_wavedec_dict.keys():
                self._matrix_wavedec_dict[length] = MatrixWavedec(
                    self.wavelet, level=1, boundary=self.boundary
                )
            return self._matrix_wavedec_dict[length]
        else:
            return partial(wavedec, wavelet=self.wavelet, level=1, mode=self.mode)

    def get_level(self, level: int) -> List[str]:
        """Return the graycode ordered paths to the filter tree nodes.

        Args:
            level (int): The depth of the tree.

        Returns:
            list: A list with the paths to each node.
        """
        return self._get_graycode_order(level)

    def _get_graycode_order(self, level: int, x: str = "a", y: str = "d") -> List[str]:
        graycode_order = [x, y]
        for _ in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [
                y + path for path in graycode_order[::-1]
            ]
        return graycode_order

    # ignoring missing return type, as recursive nesting is currently not supported
    # see https://github.com/python/mypy/issues/731
    def _recursive_dwt(  # type: ignore[no-untyped-def]
        self, data: torch.Tensor, level: int, path: str
    ):
        if not self.max_level:
            raise AssertionError
        self.data[path] = torch.squeeze(data)
        if level < self.max_level:
            res_lo, res_hi = self._get_wavedec(data.shape[-1])(data)
            return (
                self._recursive_dwt(res_lo, level + 1, path + "a"),
                self._recursive_dwt(res_hi, level + 1, path + "d"),
            )
        else:
            self.data[path] = torch.squeeze(data)


class WaveletPacket2D(BaseDict):
    """Two dimensional wavelet packets."""

    def __init__(
        self,
        data: Optional[torch.Tensor],
        wavelet: Union[Wavelet, str],
        mode: str = "reflect",
        boundary_orthogonalization: str = "qr",
        separable: bool = False,
    ) -> None:
        """Create a 2D-Wavelet packet tree.

        Args:
            data (torch.tensor, optional): The input data tensor
                of shape [batch_size, height, width].  If None, the object
                is initialized without performing a decomposition.
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
            mode (str): A string indicating the desired padding mode.
                If you select 'boundary', the sparse matrix backend is used.
                Defaults to 'reflect'
            boundary_orthogonalization (str): The orthogonalization method
                to use in the sparse matrix backend. Only used if `mode`
                equals 'boundary'. Choose from 'qr' or 'gramschmidt'.
                Defaults to 'qr'.
            separable (bool): If true and the sparse matrix backend is selected,
                a separable transform is performed, i.e. each image axis is
                transformed separately. Defaults to False.
        """
        self.wavelet = _as_wavelet(wavelet)

        # translate pywt to PyTorch
        self.mode = "constant" if mode == "zero" else mode

        self.boundary = boundary_orthogonalization
        self.separable = separable
        self.matrix_wavedec2_dict: Dict[Tuple[int, ...], MatrixWavedec2d] = {}

        self.max_level: Optional[int] = None
        if data is not None:
            self.transform(data)
        else:
            self.data = {}

    def transform(
        self, data: torch.Tensor, max_level: Optional[int] = None
    ) -> "WaveletPacket2D":
        """Calculate the 2d wavelet packet transform for the input data.

           The transform function allows reusing the same object.

        Args:
            data (torch.tensor): The input data tensor
                of shape [batch_size, height, width]
            max_level (int, optional): The highest decomposition level to compute.
                If None, the maximum level is determined from the input data shape.
                Defaults to None.
        """
        self.data = {}
        if max_level is None:
            max_level = pywt.dwt_max_level(min(data.shape[-2:]), self.wavelet.dec_len)
        self.max_level = max_level

        if data.dim() == 3:
            data = torch.unsqueeze(data, 1)

        self._recursive_dwt2d(data, level=0, path="")
        return self

    def _get_wavedec(
        self, shape: Tuple[int, ...]
    ) -> Callable[
        [torch.Tensor],
        List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    ]:
        if self.mode == "boundary":
            shape = tuple(shape)
            if shape not in self.matrix_wavedec2_dict.keys():
                self.matrix_wavedec2_dict[shape] = MatrixWavedec2d(
                    self.wavelet,
                    level=1,
                    boundary=self.boundary,
                    separable=self.separable,
                )
            fun = self.matrix_wavedec2_dict[shape]
            return fun
        else:
            return partial(wavedec2, wavelet=self.wavelet, level=1, mode=self.mode)

    # ignoring missing return type, as recursive nesting is currently not supported
    # see https://github.com/python/mypy/issues/731
    def _recursive_dwt2d(  # type: ignore[no-untyped-def]
        self, data: torch.Tensor, level: int, path: str
    ):
        if not self.max_level:
            raise AssertionError
        self.data[path] = data
        if level < self.max_level:
            # resa, (resh, resv, resd) = self.wavedec2(
            #    data, self.wavelet, level=1, mode=self.mode
            # )
            result_a, (result_h, result_v, result_d) = self._get_wavedec(
                data.shape[-2:]
            )(data)
            # assert for type checking
            assert not isinstance(result_a, tuple)
            return (
                self._recursive_dwt2d(result_a, level + 1, path + "a"),
                self._recursive_dwt2d(result_h, level + 1, path + "h"),
                self._recursive_dwt2d(result_v, level + 1, path + "v"),
                self._recursive_dwt2d(result_d, level + 1, path + "d"),
            )
        else:
            self.data[path] = torch.squeeze(data)


def get_freq_order(level: int) -> List[List[Tuple[str, ...]]]:
    """Get the frequency order for a given packet decomposition level.

    Args:
        level (int): The number of decomposition scales.

    Returns:
        list: A list with the tree nodes in frequency order.

    Note:
        Adapted from:
        https://github.com/PyWavelets/pywt/blob/master/pywt/_wavelet_packets.py

        The code elements denote the filter application order. The filters
        are named following the pywt convention as:
        a - LL, low-low coefficients
        h - LH, low-high coefficients
        v - HL, high-low coefficients
        d - HH, high-high coefficients
    """
    wp_natural_path = list(product(["a", "h", "v", "d"], repeat=level))

    def _get_graycode_order(level: int, x: str = "a", y: str = "d") -> List[str]:
        graycode_order = [x, y]
        for _ in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [
                y + path for path in graycode_order[::-1]
            ]
        return graycode_order

    def expand_2d_path(path: Tuple[str, ...]) -> Tuple[str, str]:
        expanded_paths = {"d": "hh", "h": "hl", "v": "lh", "a": "ll"}
        return (
            "".join([expanded_paths[p][0] for p in path]),
            "".join([expanded_paths[p][1] for p in path]),
        )

    nodes_dict: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for (row_path, col_path), node in [
        (expand_2d_path(node), node) for node in wp_natural_path
    ]:
        nodes_dict.setdefault(row_path, {})[col_path] = node
    graycode_order = _get_graycode_order(level, x="l", y="h")
    nodes = [nodes_dict[path] for path in graycode_order if path in nodes_dict]
    result = []
    for row in nodes:
        result.append([row[path] for path in graycode_order if path in row])
    return result


if __name__ == "__main__":
    import numpy as np

    # import matplotlib.pyplot as plt
    # import scipy.signal as signal
    # from itertools import product

    from scipy import misc

    face = misc.face()[:128, :128]
    wavelet = pywt.Wavelet("haar")
    wp_tree = pywt.WaveletPacket2D(
        data=np.mean(face, axis=-1).astype(np.float32), wavelet=wavelet, mode="zero"
    )

    # Get the full decomposition
    max_lev = 5
    wp_keys = list(product(["a", "d", "h", "v"], repeat=max_lev))
    count = 0
    img_rows = None
    img = []
    for node in wp_keys:
        packet = np.squeeze(wp_tree["".join(node)].data)
        if img_rows is not None:
            img_rows = np.concatenate([img_rows, packet], axis=1)
        else:
            img_rows = packet
        count += 1
        if count > 31:
            count = 0
            img.append(img_rows)
            img_rows = None

    img_pywt = np.concatenate(img, axis=0)

    pt_data = torch.unsqueeze(
        torch.from_numpy(np.mean(face, axis=-1).astype(np.float32)), 0
    )
    pt_data = torch.cat([pt_data, pt_data], 0)
    ptwt_wp_tree = WaveletPacket2D(data=pt_data, wavelet=wavelet, mode="boundary")

    # get the PyTorch decomposition
    count = 0
    img_pt = []
    img_rows_pt = None
    for node in wp_keys:
        packet = torch.squeeze(ptwt_wp_tree["".join(node)][0])
        if img_rows_pt is not None:
            img_rows_pt = torch.cat([img_rows_pt, packet], axis=1)
        else:
            img_rows_pt = packet
        count += 1
        if count > 31:
            count = 0
            img_pt.append(img_rows_pt)
            img_rows_pt = None

    img_pt = torch.cat(img_pt, dim=0).numpy()
    abs = np.abs(img_pt - img_pywt)

    err = np.mean(abs)
    print("total error", err, ["ok" if err < 1e-4 else "failed!"])
    assert err < 1e-4

    print(
        "a",
        np.mean(
            np.abs(wp_tree["a"].data - torch.squeeze(ptwt_wp_tree["a"][0]).numpy())
        ),
    )
    print(
        "h",
        np.mean(
            np.abs(wp_tree["h"].data - torch.squeeze(ptwt_wp_tree["h"][0]).numpy())
        ),
    )
    print(
        "v",
        np.mean(
            np.abs(wp_tree["v"].data - torch.squeeze(ptwt_wp_tree["v"][0]).numpy())
        ),
    )
    print(
        "d",
        np.mean(
            np.abs(wp_tree["d"].data - torch.squeeze(ptwt_wp_tree["d"][0]).numpy())
        ),
    )
