import pywt
import ptwt
import torch
import numpy as np
import time
from typing import NamedTuple

import matplotlib.pyplot as plt
import tikzplotlib

class WaveletTuple(NamedTuple):
    """Replaces namedtuple("Wavelet", ("dec_lo", "dec_hi", "rec_lo", "rec_hi"))."""

    dec_lo: torch.Tensor
    dec_hi: torch.Tensor
    rec_lo: torch.Tensor
    rec_hi: torch.Tensor


def _set_up_wavelet_tuple(wavelet, dtype):
    return WaveletTuple(
        torch.tensor(wavelet.dec_lo).type(dtype),
        torch.tensor(wavelet.dec_hi).type(dtype),
        torch.tensor(wavelet.rec_lo).type(dtype),
        torch.tensor(wavelet.rec_hi).type(dtype),
    )

def _jit_wavedec_fun(data, wavelet):
    return ptwt.wavedec(data, wavelet, "reflect", level=10)


if __name__ == '__main__':
    length = 1e6
    repetitions = 100

    pywt_time_cpu = []
    ptwt_time_cpu = []
    ptwt_time_gpu = []
    ptwt_time_cpu_jit = []
    ptwt_time_gpu_jit = []

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        start = time.perf_counter()
        res = pywt.wavedec(data, "db5", level=10)
        end = time.perf_counter()
        pywt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data)
        start = time.perf_counter()
        res = ptwt.wavedec(data, "db5", level=10)
        end = time.perf_counter()
        ptwt_time_cpu.append(end - start)

    wavelet = _set_up_wavelet_tuple(pywt.Wavelet("db5"), torch.float32)
    jit_wavedec = torch.jit.trace(
            _jit_wavedec_fun,
            (data, wavelet),
            strict=False,
        )

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data)
        start = time.perf_counter()
        res = jit_wavedec(data, wavelet)
        end = time.perf_counter()
        ptwt_time_cpu_jit.append(end - start)


    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = ptwt.wavedec(data, "db5", level=10)
        end = time.perf_counter()
        ptwt_time_gpu.append(end - start)

    wavelet = _set_up_wavelet_tuple(pywt.Wavelet("db5"), torch.float32)
    jit_wavedec = torch.jit.trace(
            _jit_wavedec_fun,
            (data, wavelet),
            strict=False,
        )

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = jit_wavedec(data, wavelet)
        end = time.perf_counter()
        ptwt_time_gpu_jit.append(end - start)

    print(f"1d-pywt-cpu:{np.mean(pywt_time_cpu):5.5f} +- {np.std(pywt_time_cpu):5.5f}")
    print(f"1d-ptwt-cpu:{np.mean(ptwt_time_cpu):5.5f} +- {np.std(ptwt_time_cpu):5.5f}")    
    print(f"1d-ptwt-cpu-jit:{np.mean(ptwt_time_cpu_jit):5.5f} +- {np.std(ptwt_time_cpu_jit):5.5f}")
    print(f"1d-ptwt-gpu:{np.mean(ptwt_time_gpu):5.5f} +- {np.std(ptwt_time_gpu):5.5f}")
    print(f"1d-ptwt-jit:{np.mean(ptwt_time_gpu_jit):5.5f} +- {np.std(ptwt_time_gpu_jit):5.5f}")
    plt.semilogy(pywt_time_cpu, label='pywt-cpu')
    plt.semilogy(ptwt_time_cpu, label='ptwt-cpu')
    plt.semilogy(ptwt_time_cpu_jit, label='ptwt-cpu-jit')
    plt.semilogy(ptwt_time_gpu, label='ptwt-gpu')
    plt.semilogy(ptwt_time_gpu_jit, label='ptwt-gpu-jit')
    plt.legend()
    plt.xlabel('repetition')
    plt.ylabel('runtime [s]')
    # tikzplotlib.save("timeitconv1d.tex", standalone=True)
    # plt.savefig('timeitconv1d.pdf')
    plt.show()

    print("done")