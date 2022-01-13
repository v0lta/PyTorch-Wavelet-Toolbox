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
from .matmul_transform_2d import MatrixWavedec2d


if TYPE_CHECKING:
    BaseDict = collections.UserDict[str, torch.Tensor]
else:
    BaseDict = collections.UserDict


class WaveletPacket(BaseDict):
    """One dimensional wavelet packets."""

    def __init__(
        self, data: torch.Tensor, wavelet: Union[Wavelet, str], mode: str = "reflect"
    ) -> None:
        """Create a wavelet packet decomposition object.

        The decompositions will rely on padded fast wavelet transforms.

        Args:
            data (torch.Tensor): The input data array of shape [time].
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
            mode (str): The desired padding method. Defaults to 'reflect'.
        """
        self.input_data = data
        self.wavelet = wavelet
        self.mode = mode
        self._wavepacketdec(self.input_data, wavelet)

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
        self, data: torch.Tensor, level: int, max_level: int, path: str
    ):
        self.data[path] = torch.squeeze(data)
        if level < max_level:
            res_lo, res_hi = wavedec(data, self.wavelet, level=1, mode=self.mode)
            return (
                self._recursive_dwt(res_lo, level + 1, max_level, path + "a"),
                self._recursive_dwt(res_hi, level + 1, max_level, path + "d"),
            )
        else:
            self.data[path] = torch.squeeze(data)

    # ignoring missing return type, as recursive nesting is currently not supported
    # see https://github.com/python/mypy/issues/731
    def _wavepacketdec(  # type: ignore[no-untyped-def]
        self,
        data: torch.Tensor,
        wavelet: Union[Wavelet, str],
        max_level: Optional[int] = None,
    ):
        wavelet = _as_wavelet(wavelet)
        self.data = {}
        filter_len = len(wavelet.dec_lo)
        if max_level is None:
            max_level = pywt.dwt_max_level(data.shape[-1], filter_len)
        self._recursive_dwt(data, level=0, max_level=max_level, path="")


class WaveletPacket2D(BaseDict):
    """Two dimensional wavelet packets."""

    def __init__(
        self,
        data: torch.Tensor,
        wavelet: Union[Wavelet, str],
        mode: str,
        max_level: Optional[int] = None,
    ) -> None:
        """Create a 2D-Wavelet packet tree.

        Args:
            data (torch.tensor): The input data array
                of shape [batch_size, height, width]
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
            mode (str): A string indicating the desired padding mode,
                choose zero or reflect.
            max_level (int, optional): The highest decomposition level.
        """
        self.input_data = torch.unsqueeze(data, 1)
        self.wavelet = _as_wavelet(wavelet)
        self.mode = mode
        if mode == "zero":
            # translate pywt to PyTorch
            self.mode = "constant"

        if max_level is None:
            self.max_level = pywt.dwt_max_level(
                min(self.input_data.shape[2:]), self.wavelet
            )
        else:
            self.max_level = max_level

        self.matrix_wavedec2_dict: Dict[Tuple[int, ...], MatrixWavedec2d] = {}
        self.data = {}
        self._recursive_dwt2d(self.input_data, level=0, path="")

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
                    self.wavelet, level=1
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
        self.data[path] = data
        if level < self.max_level:
            # resa, (resh, resv, resd) = self.wavedec2(
            #    data, self.wavelet, level=1, mode=self.mode
            # )
            result_a, (result_h, result_v, result_d) = self._get_wavedec(data.shape)(
                data
            )
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
