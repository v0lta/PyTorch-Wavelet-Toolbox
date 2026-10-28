"""Test the continuous transformation code."""

from typing import Any

import numpy as np
import pytest
import pywt
import torch
from scipy import signal

from ptwt.continuous_transform import _ShannonWavelet, cwt, icwt

continuous_wavelets = [
    "cgau1",
    "cgau2",
    "cgau3",
    "cgau4",
    "cgau5",
    "cgau6",
    "cgau7",
    "cgau8",
    "gaus1",
    "gaus2",
    "gaus3",
    "gaus4",
    "gaus5",
    "gaus6",
    "gaus7",
    "gaus8",
    "mexh",
    "morl",
]

cmor_wavelets = [
    "cmor0.5-0.5",
    "cmor1.0-1.0",
    "cmor1.5-1.0",
    "cmor2.0-1.5",
]


@pytest.mark.parametrize("scales", [np.arange(1, 16), 5.0, torch.arange(1, 15)])
@pytest.mark.parametrize("samples", [31, 32])
@pytest.mark.parametrize("wavelet", continuous_wavelets)
def test_cwt(wavelet: str, samples: int, scales: Any) -> None:
    """Test the cwt implementation for various wavelets."""
    t = np.linspace(-1, 1, samples, endpoint=False)
    sig = signal.chirp(t, f0=1, f1=50, t1=10, method="linear")
    scales_np = scales.numpy() if type(scales) is torch.Tensor else scales
    cwtmatr, freqs = pywt.cwt(data=sig, scales=scales_np, wavelet=wavelet)
    sig = torch.from_numpy(sig)
    cwtmatr_pt, freqs_pt = cwt(sig, scales, wavelet)
    assert np.allclose(cwtmatr_pt.numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)


@pytest.mark.slow
@pytest.mark.parametrize("cuda", [False, True])
def test_cwt_cuda(cuda: bool, wavelet: str = "cgau6") -> None:
    """Test the cwt implementation with a GPU."""
    t = np.linspace(-1, 1, 400, endpoint=False)
    sig = signal.chirp(t, f0=0, f1=20, t1=1, method="linear")
    cwtmatr, freqs = pywt.cwt(data=sig, scales=np.arange(1, 64), wavelet=wavelet)
    sig = torch.from_numpy(sig)
    if cuda and torch.cuda.is_available():
        sig = sig.cuda()
    cwtmatr_pt, freqs_pt = cwt(sig, np.arange(1, 64), wavelet)
    if cuda:
        cwtmatr_pt = cwtmatr_pt.cpu()
    assert np.allclose(cwtmatr_pt.numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)


@pytest.mark.parametrize("wavelet", continuous_wavelets)
def test_cwt_batched(wavelet: str) -> None:
    """Test batched transforms."""
    sig = np.random.randn(10, 200)
    widths = np.arange(1, 30)
    cwtmatr, freqs = pywt.cwt(sig, widths, wavelet)
    sig_np = torch.from_numpy(sig)
    cwtmatr_pt, freqs_pt = cwt(sig_np, widths, wavelet)
    assert np.allclose(cwtmatr_pt.numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)


@pytest.mark.parametrize("type", ["shan1-1"])
@pytest.mark.parametrize("grid_size", [8, 9, 10])
def test_nn_schannon_wavefun(type: str, grid_size: int) -> None:
    """Test the wavelet sampling for the differentiable shannon example."""
    pywt_shannon = pywt.ContinuousWavelet(type)
    ptwt_shannon = _ShannonWavelet(type)

    pywt_sample = pywt_shannon.wavefun(grid_size)
    ptwt_sample = ptwt_shannon.wavefun(grid_size)

    assert np.allclose(pywt_sample[0], ptwt_sample[0].detach().numpy())
    assert np.allclose(pywt_sample[1], ptwt_sample[1].detach().numpy())


@pytest.mark.parametrize("scales", [np.arange(1, 16), 5.0, torch.arange(1, 15)])
@pytest.mark.parametrize("samples", [31, 32])
@pytest.mark.parametrize("cuda", [False, True])
def test_nn_cwt(samples: int, scales: Any, cuda: bool) -> None:
    """Test the cwt using a differentiable continuous wavelet."""
    pywt_shannon = pywt.ContinuousWavelet("shan1-1")
    ptwt_shannon = _ShannonWavelet("shan1-1")
    t = np.linspace(-1, 1, samples, endpoint=False)
    sig = signal.chirp(t, f0=1, f1=50, t1=10, method="linear")
    scales_np = scales.numpy() if type(scales) is torch.Tensor else scales
    cwtmatr, freqs = pywt.cwt(data=sig, scales=scales_np, wavelet=pywt_shannon)
    sig = torch.from_numpy(sig)
    if cuda and torch.cuda.is_available():
        sig = sig.cuda()
    cwtmatr_pt, freqs_pt = cwt(data=sig, scales=scales, wavelet=ptwt_shannon)
    assert np.allclose(cwtmatr_pt.detach().cpu().numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)


@pytest.mark.parametrize("scales", [np.arange(1, 16), 5.0, torch.arange(1, 15)])
@pytest.mark.parametrize("samples", [31, 32])
@pytest.mark.parametrize("wavelet", cmor_wavelets)
def test_cwt_cmor(wavelet: str, samples: int, scales: Any) -> None:
    """Test cwt with complex Morlet wavelets (cmor{B}-{C})."""
    t = np.linspace(-1, 1, samples, endpoint=False)
    sig = signal.chirp(t, f0=1, f1=50, t1=10, method="linear")
    scales_np = scales.numpy() if type(scales) is torch.Tensor else scales
    cwtmatr, freqs = pywt.cwt(data=sig, scales=scales_np, wavelet=wavelet)
    sig_pt = torch.from_numpy(sig)
    cwtmatr_pt, freqs_pt = cwt(sig_pt, scales, wavelet)
    assert np.allclose(cwtmatr_pt.numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)


@pytest.mark.slow
@pytest.mark.parametrize("cuda", [False, True])
def test_icwt_cuda(cuda: bool, wavelet: str = "cmor1.5-1.0") -> None:
    """Test that icwt output lives on the same device as the input coefficients."""
    if cuda and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    Fs = 500
    t = np.linspace(0, 1, Fs, endpoint=False)
    sig_np = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)
    scales_samples = np.geomspace(4, 100, 64)
    sig_pt = torch.from_numpy(sig_np).float()
    if cuda:
        sig_pt = sig_pt.cuda()
    cwtmatr_pt, _ = cwt(sig_pt, scales_samples, wavelet, sampling_period=1 / Fs)
    x_rec, _ = icwt(
        time_array=t,
        scales=scales_samples / Fs,
        coefficients=cwtmatr_pt,
        wavelet=wavelet,
    )
    assert x_rec.device == cwtmatr_pt.device


def test_icwt_accepts_wavelet_object() -> None:
    """icwt should accept a pywt.ContinuousWavelet as well as a string."""
    Fs = 500
    t = np.linspace(0, 1, Fs, endpoint=False)
    scales_pywt = np.geomspace(2, 64, 32) * Fs
    scales_dim = scales_pywt / Fs
    sig = torch.randn(1, Fs, dtype=torch.float32)
    wavelet_str = "cmor1.5-1.0"
    wavelet_obj = pywt.ContinuousWavelet("cmor1.5-1.0")
    coefficients, _ = cwt(sig, scales_pywt, wavelet_str, sampling_period=1 / Fs)
    x_rec_str, reconstruction_constant_str = icwt(
        time_array=t, scales=scales_dim, coefficients=coefficients, wavelet=wavelet_str
    )
    x_rec_obj, reconstruction_constant_obj = icwt(
        time_array=t, scales=scales_dim, coefficients=coefficients, wavelet=wavelet_obj
    )
    assert torch.allclose(x_rec_str, x_rec_obj)
    assert np.isclose(reconstruction_constant_str, reconstruction_constant_obj)


@pytest.mark.parametrize("wavelet", ["mexh", "morl", "cgau1", "gaus1"])
def test_icwt_rejects_non_cmor(wavelet: str) -> None:
    """icwt should raise ValueError for non-cmor wavelet families."""
    Fs = 200
    t = np.linspace(0, 1, Fs, endpoint=False)
    sig = torch.randn(1, Fs, dtype=torch.float32)
    scales_samples = np.geomspace(2, 32, 16)
    coefficients, _ = cwt(sig, scales_samples, wavelet, sampling_period=1 / Fs)
    with pytest.raises(NotImplementedError, match="cmor"):
        icwt(
            time_array=t,
            scales=scales_samples / Fs,
            coefficients=coefficients,
            wavelet=wavelet,
        )


@pytest.mark.parametrize("wavelet", ["cmor1.5-1.0", "cmor0.5-1.0", "cmor3.0-2.0"])
def test_icwt_cdelta_valid(wavelet: str) -> None:
    """reconstruction_constant must be finite and positive for all valid cmor wavelets."""
    Fs = 500
    t = np.linspace(0, 1, Fs, endpoint=False)
    scales_samples = np.geomspace(4, 100, 64)
    sig_np = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)
    sig = torch.from_numpy(sig_np).float()
    coefficients, _ = cwt(sig, scales_samples, wavelet, sampling_period=1 / Fs)
    x_rec, reconstruction_constant = icwt(
        time_array=t,
        scales=scales_samples / Fs,
        coefficients=coefficients,
        wavelet=wavelet,
    )
    assert np.isfinite(reconstruction_constant)
    assert reconstruction_constant > 0
    mae = np.abs(x_rec.numpy() - sig.numpy()).mean()
    assert mae < 0.05, f"mean absolute error: {mae}"


@pytest.mark.parametrize("scales", [np.geomspace(8, 400, 64), np.geomspace(8, 200, 32)])
@pytest.mark.parametrize("wavelet", ["cmor1.5-1.0", "cmor1.0-1.0", "cmor2.0-1.5"])
def test_icwt(wavelet: str, scales: np.ndarray) -> None:
    """Test that icwt reconstructs a band-limited signal."""
    Fs = 1000
    t = np.arange(0, 2, 1 / Fs)
    sig = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)
    sig_pt = torch.from_numpy(sig).float()
    coefficients, _ = cwt(sig_pt, scales, wavelet, sampling_period=1 / Fs)
    x_rec, reconstruction_constant = icwt(
        time_array=t,
        scales=scales / Fs,
        coefficients=coefficients,
        wavelet=wavelet,
    )
    x_rec_np = x_rec.numpy()
    assert np.all(np.isfinite(x_rec_np))
    assert np.isfinite(reconstruction_constant)
    assert reconstruction_constant > 0
    mae = np.abs(x_rec_np - sig).mean()
    assert mae < 0.05, f"mean absolute error: {mae}"
