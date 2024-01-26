import time
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
from pytorch_wavelets import DWTForward

import ptwt


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


def _to_jit_wavedec_2(data, wavelet):
    """Ensure uniform datatypes in lists for the tracer.
    Going from List[Union[torch.Tensor, List[torch.Tensor]]] to List[torch.Tensor]
    means we have to stack the lists in the output.
    """
    assert data.shape == (32, 1e3, 1e3), "Changing the chape requires re-tracing."
    coeff = ptwt.fswavedec2(data, wavelet, "reflect", level=5)
    coeff2 = []
    for c in coeff:
        if isinstance(c, torch.Tensor):
            coeff2.append(c)
        else:
            coeff2.append(torch.stack([cel for _, cel in c.items()]))
    return coeff2


if __name__ == "__main__":
    repetitions = 100
    length = 1e3

    pywt_time_cpu = []

    cotter_time_cpu = []
    cotter_time_gpu = []

    ptwt_time_cpu = []
    ptwt_time_gpu = []
    ptwt_time_jit = []
    ptwt_time_jit_cpu = []

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        start = time.perf_counter()
        pywt_res = pywt.fswavedecn(
            data, "db5", levels=5, mode="periodic", axes=(-2, -1)
        )
        end = time.perf_counter()
        pywt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data)
        start = time.perf_counter()
        wavedec2 = DWTForward(5, "db5", mode="periodic")
        cres = wavedec2(data.unsqueeze(1))
        end = time.perf_counter()
        cotter_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data)
        start = time.perf_counter()
        res = ptwt.fswavedec2(data, "db5", mode="periodic", level=5)
        end = time.perf_counter()
        ptwt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        wavedec2 = DWTForward(5, "db5", mode="periodic").cuda()
        cres = wavedec2(data.unsqueeze(1))
        torch.cuda.synchronize()
        end = time.perf_counter()
        cotter_time_gpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = ptwt.fswavedec2(data, "db5", mode="periodic", level=5)
        torch.cuda.synchronize()
        end = time.perf_counter()
        ptwt_time_gpu.append(end - start)

    wavelet = _set_up_wavelet_tuple(pywt.Wavelet("db5"), torch.float32)
    jit_wavedec = torch.jit.trace(
        _to_jit_wavedec_2,
        (data.cuda(), wavelet),
        strict=False,
    )

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()

        pc_start = time.perf_counter()
        res = jit_wavedec(data, wavelet)
        torch.cuda.synchronize()
        pc_end = time.perf_counter()
        ptwt_time_jit.append(pc_end - pc_start)

    jit_wavedec = torch.jit.trace(
        _to_jit_wavedec_2,
        (data.cpu(), wavelet),
        strict=False,
    )

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data)

        pc_start = time.perf_counter()
        res = jit_wavedec(data, wavelet)
        pc_end = time.perf_counter()
        ptwt_time_jit_cpu.append(pc_end - pc_start)

    print("2d separable fwt results")
    print(f"2d-pywt-cpu:{np.mean(pywt_time_cpu):5.5f} +- {np.std(pywt_time_cpu):5.5f}")

    print(
        f"2d-pytorch_wavelets-cpu:{np.mean(cotter_time_cpu):5.5f} +- {np.std(cotter_time_cpu):5.5f}"
    )
    print(
        f"2d-pytorch_wavelets-gpu:{np.mean(cotter_time_gpu):5.5f} +- {np.std(cotter_time_gpu):5.5f}"
    )

    print(
        f"2d-ptwt-cpu    :{np.mean(ptwt_time_cpu):5.5f} +- {np.std(ptwt_time_cpu):5.5f}"
    )
    print(
        f"2d-ptwt-cpu-jit:{np.mean(ptwt_time_jit_cpu):5.5f} +- {np.std(ptwt_time_jit_cpu):5.5f}"
    )

    print(
        f"2d-ptwt-gpu    :{np.mean(ptwt_time_gpu):5.5f} +- {np.std(ptwt_time_gpu):5.5f}"
    )
    print(
        f"2d-ptwt-gpu-jit:{np.mean(ptwt_time_jit):5.5f} +- {np.std(ptwt_time_jit):5.5f}"
    )
    # plt.semilogy(pywt_time_cpu, label='pywt-cpu')
    # plt.semilogy(ptwt_time_cpu, label='ptwt-cpu')
    # plt.semilogy(ptwt_time_jit_cpu, label='ptwt-cpu-jit')
    # plt.semilogy(ptwt_time_gpu, label='ptwt-gpu')
    # plt.semilogy(ptwt_time_jit, label='ptwt-gpu-jit')
    # plt.semilogy(cotter_time_cpu, label='pytorch_wavelets-cpu')
    # plt.semilogy(cotter_time_gpu, label='pytorch_wavelets-gpu')
    # plt.legend()
    # plt.xlabel('repetition')
    # plt.ylabel('runtime [s]')
    # plt.show()

    time_stack = np.stack(
        [
            pywt_time_cpu,
            cotter_time_cpu,
            ptwt_time_cpu,
            cotter_time_gpu,
            ptwt_time_gpu,
            ptwt_time_jit,
        ],
        -1,
    )
    plt.boxplot(time_stack)
    plt.yscale("log")
    plt.xticks(
        [1, 2, 3, 4, 5, 6],
        [
            "pywt-cpu",
            "pytorch-wavelets-cpu",
            "ptwt-cpu",
            "pytorch-wavelets-gpu",
            "ptwt-gpu",
            "ptwt-gpu-jit",
        ],
    )
    plt.xticks(rotation=20)
    plt.ylabel("runtime [s]")
    plt.title("DWT-2D-separable")
    plt.savefig("figs/timeitconv2d_sep.png")
