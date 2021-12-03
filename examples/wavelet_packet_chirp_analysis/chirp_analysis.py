import torch
import pywt
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# use from src.ptwt.packets if you cloned the repo instead of using pip.
from ptwt import WaveletPacket

t = np.linspace(0, 10, 1500)
w = scipy.signal.chirp(t, f0=1, f1=50, t1=10, method="linear")

wavelet = pywt.Wavelet("db3")
wp = WaveletPacket(
    data=torch.from_numpy(w.astype(np.float32)), wavelet=wavelet, mode="reflect"
)
nodes = wp.get_level(5)
np_lst = []
for node in nodes:
    np_lst.append(wp[node])
viz = np.stack(np_lst)

fig, axs = plt.subplots(2)
axs[0].plot(t, w)
axs[0].set_title("Linear Chirp, f(0)=1, f(10)=50")
axs[0].set_xlabel("t [s]")

axs[1].set_title("Wavelet analysis")
axs[1].imshow(np.abs(viz))
axs[1].set_xlabel("time")
axs[1].set_ylabel("frequency")
axs[1].invert_yaxis()
plt.show()
