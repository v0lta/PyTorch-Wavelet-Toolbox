import pywt
import scipy.signal
import numpy as np


def test_cwt1d():
    t = np.linspace(0, 10, 250)
    chirp = scipy.signal.chirp(t, f0=0.00001, f1=1, t1=10, method="linear")
    t = np.linspace(-1, 1, 250)
    sig = np.cos(2 * np.pi * 7 * t) + scipy.signal.gausspulse(t - 0.4, fc=2)
    plt.figure()
    plt.plot(chirp)
    wavelet = pywt.ContinuousWavelet("morl")
    trans, freq = pywt.cwt(chirp, scales=np.arange(1, 75), wavelet=wavelet)
    plt.figure()
    plt.imshow(trans)
    plt.show()
    print("done")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_cwt1d()
    print("done")
