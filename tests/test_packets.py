#
# Created on Fri Apr 6 2021 by moritz (wolter@cs.uni-bonn.de)
#
import torch
import pywt
import numpy as np
from scipy import misc
from itertools import product
from src.ptwt.packets import WaveletPacket, WaveletPacket2D


def test_packet_harbo_lvl3():
    # From Jensen, La Cour-Harbo,
    # Rippels in Mathematics, Chapter 8 (page 89).
    w = [56., 40., 8., 24., 48., 48., 40., 16.]

    class MyHaarFilterBank(object):
        @property
        def filter_bank(self):
            return ([1/2, 1/2.], [-1/2., 1/2.],
                    [1/2., 1/2.], [1/2., -1/2.])

    wavelet = pywt.Wavelet('unscaled Haar Wavelet',
                           filter_bank=MyHaarFilterBank())
    data = torch.tensor(w)
    twp = WaveletPacket(data, wavelet, mode='reflect')
    nodes = twp.get_level(3)
    twp_lst = []
    for node in nodes:
        twp_lst.append(torch.squeeze(twp[node]))
    res = torch.stack(twp_lst).numpy()

    wp = pywt.WaveletPacket(data=np.array(w), wavelet=wavelet,
                            mode='reflect')
    nodes = [node.path for node in wp.get_level(3, 'freq')]
    np_lst = []
    for node in nodes:
        np_lst.append(wp[node].data)
    viz = np.concatenate(np_lst)

    err = np.mean(np.abs(res - viz))
    assert err < 1e-8


def test_2d_packets():
    for max_lev in [2, 3, 4, 5]:
        for wavelet_str in ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8']:
            face = misc.face()
            wavelet = pywt.Wavelet(wavelet_str)
            wp_tree = pywt.WaveletPacket2D(
                data=np.mean(face, axis=-1).astype(np.float32),
                wavelet=wavelet, mode='reflect')
            # Get the full decomposition
            wp_keys = list(product(['a', 'd', 'h', 'v'], repeat=max_lev))
            count = 0
            img_rows = None
            img = []
            for node in wp_keys:
                packet = np.squeeze(wp_tree[''.join(node)].data)
                if img_rows is not None:
                    img_rows = np.concatenate([img_rows, packet], axis=1)
                else:
                    img_rows = packet
                count += 1
                if count >= np.sqrt(len(wp_keys)):
                    count = 0
                    img.append(img_rows)
                    img_rows = None

            img_pywt = np.concatenate(img, axis=0)
            pt_data = torch.unsqueeze(
                torch.from_numpy(np.mean(face, axis=-1).astype(np.float32)), 0)
            ptwt_wp_tree = WaveletPacket2D(
                data=pt_data, wavelet=wavelet, mode='reflect')

            # get the pytorch decomposition
            count = 0
            img_pt = []
            img_rows_pt = None
            for node in wp_keys:
                packet = torch.squeeze(ptwt_wp_tree[''.join(node)])
                if img_rows_pt is not None:
                    img_rows_pt = torch.cat([img_rows_pt, packet], axis=1)
                else:
                    img_rows_pt = packet
                count += 1
                if count >= np.sqrt(len(wp_keys)):
                    count = 0
                    img_pt.append(img_rows_pt)
                    img_rows_pt = None

            img_pt = torch.cat(img_pt, axis=0).numpy()
            abs_err_img = np.abs(img_pt - img_pywt)
            abs_err = np.mean(abs_err_img)
            print(wavelet_str, max_lev, 'total error', abs_err, ['ok' if abs_err < 1e-4 else 'failed!'])
            assert abs_err < 1e-4


if __name__ == '__main__':
    test_2d_packets()