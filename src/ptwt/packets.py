"""Compute analysis wavelet packet representations."""
# Created on Fri Apr 6 2021 by moritz (wolter@cs.uni-bonn.de)

import collections
from functools import partial
from itertools import product
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import pywt
import torch

from ._util import Wavelet, _as_wavelet
from .conv_transform import wavedec, waverec
from .conv_transform_2 import wavedec2, waverec2
from .matmul_transform import MatrixWavedec, MatrixWaverec
from .matmul_transform_2 import MatrixWavedec2, MatrixWaverec2

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
        maxlevel: Optional[int] = None,
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
            maxlevel (int, optional): Value is passed on to `transform`.
                The highest decomposition level to compute. If None, the maximum level
                is determined from the input data shape. Defaults to None.

        Example:
            >>> import torch, pywt, ptwt
            >>> import numpy as np
            >>> import scipy.signal
            >>> import matplotlib.pyplot as plt
            >>> t = np.linspace(0, 10, 1500)
            >>> w = scipy.signal.chirp(t, f0=1, f1=50, t1=10, method="linear")
            >>> wp = ptwt.WaveletPacket(data=torch.from_numpy(w.astype(np.float32)),
            >>>     wavelet=pywt.Wavelet("db3"), mode="reflect")
            >>> np_lst = []
            >>> for node in wp.get_level(5):
            >>>     np_lst.append(wp[node])
            >>> viz = np.stack(np_lst).squeeze()
            >>> plt.imshow(np.abs(viz))
            >>> plt.show()

        """
        self.wavelet = _as_wavelet(wavelet)
        self.mode = mode
        self.boundary = boundary_orthogonalization
        self._matrix_wavedec_dict: Dict[int, MatrixWavedec] = {}
        self._matrix_waverec_dict: Dict[int, MatrixWaverec] = {}
        self.maxlevel: Optional[int] = None
        if data is not None:
            if len(data.shape) == 1:
                # add a batch dimension.
                data = data.unsqueeze(0)
            self.transform(data, maxlevel)
        else:
            self.data = {}

    def transform(
        self, data: torch.Tensor, maxlevel: Optional[int] = None
    ) -> "WaveletPacket":
        """Calculate the 1d wavelet packet transform for the input data.

        Args:
            data (torch.Tensor): The input data array of shape [time]
                or [batch_size, time].
            maxlevel (int, optional): The highest decomposition level to compute.
                If None, the maximum level is determined from the input data shape.
                Defaults to None.
        """
        self.data = {}
        if maxlevel is None:
            maxlevel = pywt.dwt_max_level(data.shape[-1], self.wavelet.dec_len)
        self.maxlevel = maxlevel
        self._recursive_dwt(data, level=0, path="")
        return self

    def reconstruct(self) -> "WaveletPacket":
        """Recursively reconstruct the input starting from the leaf nodes.

        Reconstruction replaces the input-data originally assigned to this object.

        Note:
           Only changes to leaf node data impacts the results,
           since changes in all other nodes will be replaced with
           a reconstruction from the leafs.

        Example:
            >>> import numpy as np
            >>> import ptwt, torch
            >>> signal = np.random.randn(1, 16)
            >>> ptwp = ptwt.WaveletPacket(torch.from_numpy(signal), "haar",
            >>>     mode="boundary", maxlevel=2)
            >>> ptwp["aa"].data *= 0
            >>> ptwp.reconstruct()
            >>> print(ptwp[""])
        """
        if self.maxlevel is None:
            self.maxlevel = pywt.dwt_max_level(self[""].shape[-1], self.wavelet.dec_len)

        for level in reversed(range(self.maxlevel)):
            for node in self.get_level(level):
                data_a = self[node + "a"]
                data_b = self[node + "d"]
                rec = self._get_waverec(data_a.shape[-1])([data_a, data_b])
                self[node] = rec
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

    def _get_waverec(
        self,
        length: int,
    ) -> Callable[[List[torch.Tensor]], torch.Tensor]:
        if self.mode == "boundary":
            if length not in self._matrix_waverec_dict.keys():
                self._matrix_waverec_dict[length] = MatrixWaverec(
                    self.wavelet, boundary=self.boundary
                )
            return self._matrix_waverec_dict[length]
        else:
            return partial(waverec, wavelet=self.wavelet)

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
        if level == 0:
            return [""]
        else:
            return graycode_order

    def _recursive_dwt(self, data: torch.Tensor, level: int, path: str) -> None:
        if not self.maxlevel:
            raise AssertionError

        # TODO: This is a workaround since the convolutional transforms insert a
        #       squeezable dimension. We should adapt the wavedec code instead.
        if data.dim() == 3:
            data = data.squeeze(1)

        self.data[path] = data
        if level < self.maxlevel:
            res_lo, res_hi = self._get_wavedec(data.shape[-1])(data)
            self._recursive_dwt(res_lo, level + 1, path + "a")
            self._recursive_dwt(res_hi, level + 1, path + "d")

    def __getitem__(self, key: str) -> torch.Tensor:
        """Access the coefficients in the wavelet packets tree.

        Args:
            key (str): The key of the accessed coefficents. The string may only consist
                of the following chars: 'a', 'd'.

        Returns:
            torch.Tensor: The accessed wavelet packet coefficients.

        Raises:
            ValueError: If the wavelet packet tree is not initialized.
            KeyError: If no wavelet coefficients are indexed by the specified key.
        """
        if self.maxlevel is None:
            raise ValueError(
                "The wavelet packet tree must be initialized via 'transform' before "
                "its values can be accessed!"
            )
        if key not in self and len(key) > self.maxlevel:
            raise KeyError(
                f"The requested level {len(key)} with key '{key}' is too large and "
                "cannot be accessed! This wavelet packet tree is initialized with "
                f"maximum level {self.maxlevel}."
            )
        return super().__getitem__(key)


class WaveletPacket2D(BaseDict):
    """Two dimensional wavelet packets.

    Example code illustrating the use of this class is available at:
    https://github.com/v0lta/PyTorch-Wavelet-Toolbox/tree/main/examples/deepfake_analysis
    """

    def __init__(
        self,
        data: Optional[torch.Tensor],
        wavelet: Union[Wavelet, str],
        mode: str = "reflect",
        boundary_orthogonalization: str = "qr",
        separable: bool = False,
        maxlevel: Optional[int] = None,
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
            maxlevel (int, optional): Value is passed on to `transform`.
                The highest decomposition level to compute. If None, the maximum level
                is determined from the input data shape. Defaults to None.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.mode = mode
        self.boundary = boundary_orthogonalization
        self.separable = separable
        self.matrix_wavedec2_dict: Dict[Tuple[int, ...], MatrixWavedec2] = {}
        self.matrix_waverec2_dict: Dict[Tuple[int, ...], MatrixWaverec2] = {}

        self.maxlevel: Optional[int] = None
        if data is not None:
            self.transform(data, maxlevel)
        else:
            self.data = {}

    def transform(
        self, data: torch.Tensor, maxlevel: Optional[int] = None
    ) -> "WaveletPacket2D":
        """Calculate the 2d wavelet packet transform for the input data.

           The transform function allows reusing the same object.

        Args:
            data (torch.tensor): The input data tensor
                of shape [batch_size, height, width]
            maxlevel (int, optional): The highest decomposition level to compute.
                If None, the maximum level is determined from the input data shape.
                Defaults to None.
        """
        self.data = {}
        if maxlevel is None:
            maxlevel = pywt.dwt_max_level(min(data.shape[-2:]), self.wavelet.dec_len)
        self.maxlevel = maxlevel

        self._recursive_dwt2d(data, level=0, path="")
        return self

    def reconstruct(self) -> "WaveletPacket2D":
        """Recursively reconstruct the input starting from the leaf nodes.

        Note:
           Only changes to leaf node data impacts the results,
           since changes in all other nodes will be replaced with
           a reconstruction from the leafs.
        """
        if self.maxlevel is None:
            self.maxlevel = pywt.dwt_max_level(
                min(self[""].shape[-2:]), self.wavelet.dec_len
            )

        for level in reversed(range(self.maxlevel)):
            for node in self.get_natural_order(level):
                if self.mode == "boundary":
                    data_a = self[node + "a"]
                    data_h = self[node + "h"]
                    data_v = self[node + "v"]
                    data_d = self[node + "d"]
                    rec = self._get_waverec(data_a.shape[-2:])(
                        (data_a, (data_h, data_v, data_d))
                    )
                    self[node] = rec
                else:
                    data_a = self[node + "a"].unsqueeze(1)
                    data_h = self[node + "h"].unsqueeze(1)
                    data_v = self[node + "v"].unsqueeze(1)
                    data_d = self[node + "d"].unsqueeze(1)
                    rec = self._get_waverec(data_a.shape[-2:])(
                        (data_a, (data_h, data_v, data_d))
                    )
                    self[node] = rec.squeeze(1)
        return self

    def get_natural_order(self, level: int) -> List[str]:
        """Get the natural ordering for a given decomposition level.

        Args:
            level (int): The decomposition level.

        Returns:
            list: A list with the filter order strings.
        """
        return ["".join(p) for p in list(product(["a", "h", "v", "d"], repeat=level))]

    def _get_wavedec(
        self, shape: Tuple[int, ...]
    ) -> Callable[
        [torch.Tensor],
        List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    ]:
        if self.mode == "boundary":
            shape = tuple(shape)
            if shape not in self.matrix_wavedec2_dict.keys():
                self.matrix_wavedec2_dict[shape] = MatrixWavedec2(
                    self.wavelet,
                    level=1,
                    boundary=self.boundary,
                    separable=self.separable,
                )
            fun = self.matrix_wavedec2_dict[shape]
            return fun
        else:
            return partial(wavedec2, wavelet=self.wavelet, level=1, mode=self.mode)

    def _get_waverec(
        self, shape: Tuple[int, ...]
    ) -> Callable[
        [Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        torch.Tensor,
    ]:
        if self.mode == "boundary":
            shape = tuple(shape)
            if shape not in self.matrix_waverec2_dict.keys():
                self.matrix_waverec2_dict[shape] = MatrixWaverec2(
                    self.wavelet,
                    boundary=self.boundary,
                    separable=self.separable,
                )
            fun = self.matrix_waverec2_dict[shape]
            return fun  # type: ignore
        else:
            return partial(waverec2, wavelet=self.wavelet)

    def _recursive_dwt2d(self, data: torch.Tensor, level: int, path: str) -> None:
        if not self.maxlevel:
            raise AssertionError

        # TODO: This is a workaround since the convolutional transforms insert a
        #       squeezable dimension. We should adapt the wavedec2 code instead.
        if data.dim() == 4:
            data = data.squeeze(1)

        self.data[path] = data
        if level < self.maxlevel:
            result_a, (result_h, result_v, result_d) = self._get_wavedec(
                data.shape[-2:]
            )(data)
            # assert for type checking
            assert not isinstance(result_a, tuple)
            self._recursive_dwt2d(result_a, level + 1, path + "a")
            self._recursive_dwt2d(result_h, level + 1, path + "h")
            self._recursive_dwt2d(result_v, level + 1, path + "v")
            self._recursive_dwt2d(result_d, level + 1, path + "d")

    def __getitem__(self, key: str) -> torch.Tensor:
        """Access the coefficients in the wavelet packets tree.

        Args:
            key (str): The key of the accessed coefficents. The string may only consist
                of the following chars: 'a', 'h', 'v', 'd'.

        Returns:
            torch.Tensor: The accessed wavelet packet coefficients.

        Raises:
            ValueError: If the wavelet packet tree is not initialized.
            KeyError: If no wavelet coefficients are indexed by the specified key.
        """
        if self.maxlevel is None:
            raise ValueError(
                "The wavelet packet tree must be initialized via 'transform' before "
                "its values can be accessed!"
            )
        if key not in self and len(key) > self.maxlevel:
            raise KeyError(
                f"The requested level {len(key)} with key '{key}' is too large and "
                "cannot be accessed! This wavelet packet tree is initialized with "
                f"maximum level {self.maxlevel}."
            )
        return super().__getitem__(key)


def get_freq_order(level: int) -> List[List[Tuple[str, ...]]]:
    """Get the frequency order for a given packet decomposition level.

    Use this code to create two-dimensional frequency orderings.

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

    def _expand_2d_path(path: Tuple[str, ...]) -> Tuple[str, str]:
        expanded_paths = {"d": "hh", "h": "hl", "v": "lh", "a": "ll"}
        return (
            "".join([expanded_paths[p][0] for p in path]),
            "".join([expanded_paths[p][1] for p in path]),
        )

    nodes_dict: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for (row_path, col_path), node in [
        (_expand_2d_path(node), node) for node in wp_natural_path
    ]:
        nodes_dict.setdefault(row_path, {})[col_path] = node
    graycode_order = _get_graycode_order(level, x="l", y="h")
    nodes = [nodes_dict[path] for path in graycode_order if path in nodes_dict]
    result = []
    for row in nodes:
        result.append([row[path] for path in graycode_order if path in row])
    return result
