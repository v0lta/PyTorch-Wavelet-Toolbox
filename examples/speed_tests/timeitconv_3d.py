from typing import NamedTuple

import pywt
import ptwt
import torch
import numpy as np
import time

import matplotlib.pyplot as plt
# import tikzplotlib


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



def _to_jit_wavedec_3(data, wavelet):
    """Ensure uniform datatypes in lists for the tracer.

    Going from List[Union[torch.Tensor, Dict[str, torch.Tensor]]] to List[torch.Tensor]
    means we have to stack the lists in the output.
    """
    assert data.shape == (32, 100, 100, 100), "Changing the chape requires re-tracing."
    coeff = ptwt.wavedec3(data, wavelet, "reflect", level=3)
    coeff2 = []
    keys = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")
    for c in coeff:
        if isinstance(c, torch.Tensor):
            coeff2.append(c)
        else:
            coeff2.append(torch.stack([c[key] for key in keys]))
    return coeff2


if __name__ == '__main__':
    length = 1e2
    repetitions = int(1e2)

    pywt_time_cpu = []
    ptwt_time_cpu = []
    ptwt_time_gpu = []
    ptwt_time_jit = []

    for _ in range(repetitions):
        data = np.random.randn(
            32, int(length), int(length), int(length)).astype(np.float32)
        start = time.perf_counter()
        res = pywt.wavedecn(data, "db5", level=3, axes=range(1, 3))
        end = time.perf_counter()

        pywt_time_cpu.append(end - start)


    for _ in range(repetitions):
        data = np.random.randn(
            32, int(length), int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data)
        start = time.perf_counter()
        res = ptwt.wavedec3(data, "db5", level=3)
        end = time.perf_counter()
        ptwt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(
            32, int(length), int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = ptwt.wavedec3(data, "db5", level=3)
        end = time.perf_counter()
        ptwt_time_gpu.append(end - start)

    wavelet = _set_up_wavelet_tuple(pywt.Wavelet("db5"), torch.float32)
    jit_wavedec = torch.jit.trace(
            _to_jit_wavedec_3,
            (data, wavelet),
            strict=False,
        )
    for _ in range(repetitions):
        data = np.random.randn(
            32, int(length), int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = jit_wavedec(data, wavelet)
        end = time.perf_counter()
        ptwt_time_jit.append(end - start)

    print(f"3d-pywt-cpu:{np.mean(pywt_time_cpu):5.5f} +- {np.std(pywt_time_cpu):5.5f}")
    print(f"3d-ptwt-cpu:{np.mean(ptwt_time_cpu):5.5f} +- {np.std(ptwt_time_cpu):5.5f}")
    print(f"3d-ptwt-gpu:{np.mean(ptwt_time_gpu):5.5f} +- {np.std(ptwt_time_gpu):5.5f}")
    print(f"3d-ptwt-jit:{np.mean(ptwt_time_jit):5.5f} +- {np.std(ptwt_time_jit):5.5f}")
    plt.semilogy(pywt_time_cpu, label='pywt-cpu')
    plt.semilogy(ptwt_time_cpu, label='ptwt-cpu')
    plt.semilogy(ptwt_time_gpu, label='ptwt-gpu')
    plt.semilogy(ptwt_time_jit, label='ptwt-jit')
    plt.legend()
    plt.xlabel('repetition')
    plt.ylabel('runtime [s]')
    # tikzplotlib.save("timeitconv3dmat.tex", standalone=True)
    # plt.savefig('timeitconv3dmat.pdf')
    plt.show()