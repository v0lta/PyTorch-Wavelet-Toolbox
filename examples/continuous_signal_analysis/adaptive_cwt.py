import torch
import numpy as np
import src.ptwt as ptwt
import pywt
import matplotlib.pyplot as plt
import scipy.signal as signal
from src.ptwt.continuous_transform import ShannonWavelet

if __name__ == "__main__":
    t = np.linspace(-2, 2, 800, endpoint=False)
    # sig = signal.chirp(t, f0=1, f1=12, t1=2, method="linear")
    sig = np.sin(t*3*np.pi)
    sig += np.sin(t*10*np.pi)
    widths = np.arange(1, 31)
    wavelet = ShannonWavelet(name='shan10-10')
    # wavelet = pywt.ContinuousWavelet(name='shan12-9')
    # wavelet = pywt.ContinuousWavelet(name='mexh')
    optimizer = torch.optim.Adam(wavelet.parameters(), lr=1e-3)
    sig = torch.from_numpy(sig)
    iterations = 150
    for it in range(iterations):

        cwtmatr_pt, freqs = ptwt.cwt(
            sig, widths, wavelet, sampling_period=(4 / 800) * np.pi
        )
        norm = torch.linalg.norm(torch.abs(cwtmatr_pt))
        # probably not a good cost. TODO: Find something better.
        cost = torch.abs(torch.sum(sig*sig) - torch.sum(cwtmatr_pt*cwtmatr_pt)) + 1e-5*norm
        cost.backward()
        optimizer.step()
        if it % 2 == 0:
            print("iteration {}, cost {:2.2f}, cwt_norm {:2.2f}".format(it, cost.item(), norm.item()))
        
        if it % 10 == 0:
            print('Bandwidth {:2.2f}'.format(wavelet.bandwidth.item()))
            print('Center {:2.2f}'.format(wavelet.center.item()))

    cwtmatr = np.abs(cwtmatr_pt.detach().numpy())            
    fig, axs = plt.subplots(2)
    axs[0].plot(t, sig)
    axs[0].set_ylabel("magnitude")
    axs[1].imshow(
        cwtmatr,
        cmap="PRGn",
        aspect="auto",
        vmax=abs(cwtmatr).max(),
        vmin=0,
        extent=[min(t), max(t), min(freqs), max(freqs)],
    )
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("frequency")
    plt.show()

