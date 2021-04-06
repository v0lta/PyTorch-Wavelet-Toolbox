#
# Created on Fri Apr 6 2021 by moritz (wolter@cs.uni-bonn.de)
#
import torch
import pywt
import collections
from src.ptwt.conv_transform import wavedec


class WaveletPacket(collections.UserDict):

    def __init__(self, data: torch.tensor, wavelet, mode: str = 'reflect'):
        """Create a wavelet packet decomposition object
        Args:
            data (np.array): The input data array of shape [time].
            wavelet (pywt.Wavelet or WaveletFilter): The wavelet to use.
            mode ([str]): The desired padding method
        """
        self.input_data = data
        self.wavelet = wavelet
        self.mode = mode
        self.nodes = {}
        self.data = None
        self._wavepacketdec(self.input_data, wavelet, mode=mode)

    def get_level(self, level):
        return self.get_graycode_order(level)

    def get_graycode_order(self, level, x='a', y='d'):
        graycode_order = [x, y]
        for i in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + \
                            [y + path for path in graycode_order[::-1]]
        return graycode_order

    def recursive_dwt(self, data, level, max_level, path):
        self.data[path] = torch.squeeze(data)
        if level < max_level:
            res_lo, res_hi = wavedec(data, self.wavelet, level=1,
                                     mode=self.mode)
            return self.recursive_dwt(res_lo, level+1, max_level, path + 'a'),\
                self.recursive_dwt(res_hi, level+1, max_level, path + 'd')
        else:
            self.data[path] = torch.squeeze(data)

    def _wavepacketdec(self, data, wavelet, level=None, mode='reflect'):
        self.data = {}
        filt_len = len(wavelet.dec_lo)
        if level is None:
            level = pywt.dwt_max_level(data.shape[-1], filt_len)
        self.recursive_dwt(data, level=0, max_level=level, path='')


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.signal as signal
    import os
    os.environ["DISPLAY"] = ":1"
    import matplotlib

    t = np.linspace(0, 10, 5001)
    wavelet = pywt.Wavelet('db4')
    w = signal.chirp(t, f0=.00001, f1=20, t1=10, method='linear')

    plt.plot(t, w)
    plt.title("Linear Chirp, f(0)=6, f(10)=1")
    plt.xlabel('t (sec)')
    plt.show()

    wp = WaveletPacket(data=torch.tensor(w.astype(np.float32)),
                       wavelet=wavelet,
                       mode='reflect')
    nodes = wp.get_level(7)
    np_lst = []
    for node in nodes:
        np_lst.append(np.squeeze(wp[node]))
    viz = np.stack(np_lst)
    plt.imshow(viz[:20, :])
    plt.show()


    # wp = pywt.WaveletPacket(data=w, wavelet=wavelet,
    #                         mode='reflect')
    # nodes = [node.path for node in wp.get_level(7, 'freq')]
    # np_lst = []
    # for node in nodes:
    #     np_lst.append(wp[node].data)
    # viz = np.stack(np_lst)
    # plt.imshow(viz[:20, :])
    # plt.show()
    print('stop')


    # plt.imshow(np.log(np.abs(viz)+0.01))
    # plt.show()
    # print(wp['aa'].data)
    # print(wp['ad'].data)
    # print(wp['dd'].data)
    # print(wp['da'].data)


    # x = np.linspace(0, 1, num=512)
    # data = np.sin(250 * np.pi * x**2)

    # wavelet = 'db2'
    # level = 4
    # order = "freq"  # other option is "normal"
    # interpolation = 'nearest'

    # # Construct wavelet packet
    # wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
    # nodes = wp.get_level(level, order=order)
    # labels = [n.path for n in nodes]
    # values = np.array([n.data for n in nodes], 'd')
    # values = abs(values)

    # # Show signal and wavelet packet coefficients
    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.2, bottom=.03, left=.07, right=.97, top=.92)
    # ax = fig.add_subplot(2, 1, 1)
    # ax.set_title("linchirp signal")
    # ax.plot(x, data, 'b')
    # ax.set_xlim(0, x[-1])

    # ax = fig.add_subplot(2, 1, 2)
    # ax.set_title("Wavelet packet coefficients at level %d" % level)
    # ax.imshow(values, interpolation=interpolation,  aspect="auto",
    #         origin="lower", extent=[0, 1, 0, len(values)])
    # ax.set_yticks(np.arange(0.5, len(labels) + 0.5), labels)

    # # Show spectrogram and wavelet packet coefficients
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(211)
    # ax2.specgram(data, NFFT=64, noverlap=32, Fs=2,
    #             interpolation='bilinear')
    # ax2.set_title("Spectrogram of signal")
    # ax3 = fig2.add_subplot(212)
    # ax3.imshow(values, origin='upper', extent=[-1, 1, -1, 1],
    #         interpolation='nearest')
    # ax3.set_title("Wavelet packet coefficients")
    # plt.show()