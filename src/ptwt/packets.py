#
# Created on Fri Apr 6 2021 by moritz (wolter@cs.uni-bonn.de)
#
import collections
import pywt
import torch
from functools import partial

from .conv_transform import wavedec, wavedec2
from .matmul_transform_2d import MatrixWavedec2d


class WaveletPacket(collections.UserDict):
    def __init__(self, data: torch.tensor, wavelet, mode: str = "reflect"):
        """Create a wavelet packet decomposition object.
           The decompositions will rely on padded fast wavelet transforms.
        Args:
            data (np.array): The input data array of shape [time].
            wavelet (pywt.Wavelet or WaveletFilter): The wavelet to use.
            mode ([str]): The desired padding method.
        """
        self.input_data = data
        self.wavelet = wavelet
        self.mode = mode
        self.data = None
        self._wavepacketdec(self.input_data, wavelet)

    def get_level(self, level):
        return self.get_graycode_order(level)

    def get_graycode_order(self, level, x="a", y="d"):
        graycode_order = [x, y]
        for i in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [
                y + path for path in graycode_order[::-1]
            ]
        return graycode_order

    def _recursive_dwt(self, data, level, max_level, path):
        self.data[path] = torch.squeeze(data)
        if level < max_level:
            res_lo, res_hi = wavedec(
                data, self.wavelet, level=1, mode=self.mode)
            return (
                self._recursive_dwt(res_lo, level + 1, max_level, path + "a"),
                self._recursive_dwt(res_hi, level + 1, max_level, path + "d"),
            )
        else:
            self.data[path] = torch.squeeze(data)

    def _wavepacketdec(self, data, wavelet, max_level=None):
        self.data = {}
        filter_len = len(wavelet.dec_lo)
        if max_level is None:
            max_level = pywt.dwt_max_level(data.shape[-1], filter_len)
        self._recursive_dwt(data, level=0, max_level=max_level, path="")


class WaveletPacket2D(collections.UserDict):
    """Two dimensional wavelet packets."""

    def __init__(self, data, wavelet, mode, max_level=None):
        """Create a 2D-Wavelet packet tree.

        Args:
            data (torch.tensor): The input data array
                                 of shape [batch_size, height, width]
            wavelet (Wavelet Object): A named wavelet tuple.
            mode (str): A string indicating the desired padding mode,
            choose zero or reflect.
        """
        self.input_data = torch.unsqueeze(data, 1)
        self.wavelet = wavelet
        self.mode = mode
        if mode == "zero":
            # translate pywt to PyTorch 
            self.mode = "constant"

        self.max_level = max_level
        if self.max_level is None:
            self.max_level = pywt.dwt_max_level(
                min(self.input_data.shape[2:]), self.wavelet)

        self.matrix_wavedec2_dict = {}
        self.data = {}
        self._recursive_dwt2d(
            self.input_data, level=0, path="")


    def _get_wavedec(self, shape):
        if self.mode == 'boundary':
            shape = tuple(shape)
            if not shape in self.matrix_wavedec2_dict.keys():
                self.matrix_wavedec2_dict[shape] = MatrixWavedec2d(
                    self.wavelet, level=1
                )
            fun = self.matrix_wavedec2_dict[shape]
            return fun
        else:
            return partial(wavedec2, wavelet=self.wavelet,
                           level=1, mode=self.mode)


    def _recursive_dwt2d(self, data, level, path):
        self.data[path] = data
        if level < self.max_level:
            # resa, (resh, resv, resd) = self.wavedec2(
            #    data, self.wavelet, level=1, mode=self.mode
            # )
            result_a, (result_h, result_v, result_d) = self._get_wavedec(data.shape)(data)
            return (
                self._recursive_dwt2d(result_a, level + 1, path + "a"),
                self._recursive_dwt2d(result_h, level + 1, path + "h"),
                self._recursive_dwt2d(result_v, level + 1, path + "v"),
                self._recursive_dwt2d(result_d, level + 1, path + "d"),
            )
        else:
            self.data[path] = torch.squeeze(data)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.signal as signal
    from itertools import product

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

    img_pt = torch.cat(img_pt, axis=0).numpy()
    abs = np.abs(img_pt - img_pywt)

    err = np.mean(abs)
    print("total error", err, ["ok" if err < 1e-4 else "failed!"])
    assert err < 1e-4

    print(
        "a",
        np.mean(np.abs(wp_tree["a"].data - torch.squeeze(ptwt_wp_tree["a"][0]).numpy())),
    )
    print(
        "h",
        np.mean(np.abs(wp_tree["h"].data - torch.squeeze(ptwt_wp_tree["h"][0]).numpy())),
    )
    print(
        "v",
        np.mean(np.abs(wp_tree["v"].data - torch.squeeze(ptwt_wp_tree["v"][0]).numpy())),
    )
    print(
        "d",
        np.mean(np.abs(wp_tree["d"].data - torch.squeeze(ptwt_wp_tree["d"][0]).numpy())),
    )
