#
# Created on Fri Apr 6 2021 by moritz (wolter@cs.uni-bonn.de)
#
import torch
import pywt
import numpy as np
from src.ptwt.packets import WaveletPacket


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
