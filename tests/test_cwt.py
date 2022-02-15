import numpy as np
from scipy import signal

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t = np.linspace(-2, 2, 800, endpoint=False)
    sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
    sig = signal.chirp(t, f0=1, f1=50, t1=10, method="linear")
    widths = np.arange(1, 31)
    cwtmatr = signal.cwt(sig, signal.ricker, widths)
    fig, axs = plt.subplots(2)
    axs[0].plot(t, sig)
    axs[1].imshow(cwtmatr, cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.show()