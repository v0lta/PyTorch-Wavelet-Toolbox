import torch
import pywt
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# use from src.ptwt.packets if you cloned the repo instead of using pip.
from ptwt import WaveletPacket

fs = 1000
t = np.linspace(0, 2, int(2//(1/fs)))
w = np.sin(256*np.pi*t**2)

wavelet = pywt.Wavelet("sym6")
wp = WaveletPacket(
    data=torch.from_numpy(w.astype(np.float32)), wavelet=wavelet, mode="boundary"
)
nodes = wp.get_level(5)
np_lst = []
for node in nodes:
    np_lst.append(wp[node])
viz = np.stack(np_lst).squeeze()

fig, axs = plt.subplots(2)
axs[0].plot(t, w)
axs[0].set_title("Analyzed signal")
axs[0].set_xlabel("t [s]")

axs[1].set_title("Wavelet analysis")
axs[1].imshow(np.abs(viz))
axs[1].set_xlabel("time")
axs[1].set_ylabel("frequency")
axs[1].invert_yaxis()
plt.show()
