import torch
import ptwt
import pywt
import time
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


from ptwt.continuous_transform import _ShannonWavelet


def _to_jit_cwt(sig):
    widths = torch.arange(1, 31)
    wavelet = _ShannonWavelet("shan0.1-0.4")
    sampling_period = (4 / 800) * np.pi
    cwtmatr, _ = ptwt.cwt(sig, widths, wavelet, sampling_period=sampling_period)
    return cwtmatr


if __name__ == '__main__':
    length = 1e4
    repetitions = 100

    pywt_time_cpu = []
    ptwt_time_cpu = []
    ptwt_time_gpu = []
    ptwt_time_gpu_jit = []

    for _ in range(repetitions):
        sig = np.random.randn(32, int(length)).astype(np.float32)
        start = time.perf_counter()
        cwtmatr, _ = pywt.cwt(
            sig,
            np.arange(1, 31),
            pywt.ContinuousWavelet("shan0.1-0.4"),
            sampling_period=(4 / 800) * np.pi,
        )
        end = time.perf_counter()
        pywt_time_cpu.append(end - start)

    for _ in range(repetitions):
        sig = np.random.randn(32, int(length)).astype(np.float32)
        sig = torch.from_numpy(sig)
        start = time.perf_counter()
        cwtmatr, _ = ptwt.cwt(
            sig,
            torch.arange(1, 31),
            pywt.ContinuousWavelet("shan0.1-0.4"),
            sampling_period=(4 / 800) * np.pi,
        )
        end = time.perf_counter()
        ptwt_time_cpu.append(end - start)


    for _ in range(repetitions):
        sig = np.random.randn(32, int(length)).astype(np.float32)
        sig = torch.from_numpy(sig).cuda()
        start = time.perf_counter()
        cwtmatr, _ = ptwt.cwt(
            sig,
            torch.arange(1, 31),
            pywt.ContinuousWavelet("shan0.1-0.4"),
            sampling_period=(4 / 800) * np.pi,
        )
        end = time.perf_counter()
        ptwt_time_gpu.append(end - start)

   

    jit_cwt = torch.jit.trace(_to_jit_cwt, (sig), strict=False)    


    for _ in range(repetitions):
        sig = np.random.randn(32, int(length)).astype(np.float32)
        sig = torch.from_numpy(sig).cuda()
        start = time.perf_counter()
        cwtmatr = jit_cwt(sig)
        end = time.perf_counter()
        ptwt_time_gpu_jit.append(end - start)
    
    print(f"cwt-pywt-cpu:{np.mean(pywt_time_cpu):5.5f} +- {np.std(pywt_time_cpu):5.5f}")
    print(f"cwt-ptwt-cpu:{np.mean(ptwt_time_cpu):5.5f} +- {np.std(ptwt_time_cpu):5.5f}")    
    print(f"cwt-ptwt-gpu:{np.mean(ptwt_time_gpu):5.5f} +- {np.std(ptwt_time_gpu):5.5f}")
    print(f"cwt-ptwt-jit:{np.mean(ptwt_time_gpu_jit):5.5f} +- {np.std(ptwt_time_gpu_jit):5.5f}")
    plt.semilogy(pywt_time_cpu, label='pywt-cpu')
    plt.semilogy(ptwt_time_cpu, label='ptwt-cpu')
    plt.semilogy(ptwt_time_gpu, label='ptwt-gpu')
    plt.semilogy(ptwt_time_gpu_jit, label='ptwt-gpu-jit')
    plt.legend()
    plt.xlabel('repetition')
    plt.ylabel('runtime [s]')
    # tikzplotlib.save("timeitcwt.tex", standalone=True)
    # plt.savefig('timeitcwt.pdf')
    plt.show()
