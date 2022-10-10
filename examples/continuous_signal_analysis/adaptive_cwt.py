import torch
import numpy as np
import src.ptwt as ptwt
import matplotlib.pyplot as plt
import pywt
from src.ptwt.continuous_transform import _ShannonWavelet, _ComplexMorletWavelet

if __name__ == "__main__":
    t1 = np.linspace(-2, 0, 400, endpoint=False)
    t2 = np.linspace(0, 2, 400, endpoint=False)
    # sig = signal.chirp(t, f0=1, f1=12, t1=2, method="linear")
    sig1 = np.sin(t1*10*np.pi)
    sig2 = np.sin(t2*20*np.pi)
    sig = np.concatenate([sig1, sig2])
    t = np.concatenate([t1, t2])
    widths = np.arange(1, 40)
    sig = torch.from_numpy(sig)
    wavelet = _ComplexMorletWavelet(name='cmor0.5-0.5')
    # wavelet = pywt.ContinuousWavelet("morl")
    optimizer = torch.optim.Adam(wavelet.parameters(), lr=1e-3)
    
    iterations = 500
    for it in range(iterations):
    
        cwtmatr_pt, freqs = ptwt.cwt(
                sig, widths, wavelet, sampling_period=(4 / 800) * np.pi
        )
        norm = torch.mean(torch.abs(cwtmatr_pt**2))
        # probably not a good cost. TODO: Find something better.
        cost = torch.abs(torch.sum(torch.abs(sig)**2) - torch.sum(torch.abs(cwtmatr_pt)**2)) + norm
        cost.backward()
        optimizer.step()
        if it % 2 == 0:
            print("iteration {}, cost {:2.2f}, cwt_norm {:2.2f}, bandwidth {:2.4f}, center {:2.4f}".format(
                it, cost.item(), norm.item(), wavelet.bandwidth.item(), wavelet.center.item()))

    cwtmatr_pt, freqs = ptwt.cwt(
                 sig, widths, wavelet, sampling_period=(4 / 800) * np.pi
    )


    cwtmatr = np.abs(cwtmatr_pt.detach().numpy())            
    print(np.max(cwtmatr), np.mean(cwtmatr))
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
    # pass
