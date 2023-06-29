"""Ensure pytorch's torch.jit.trace feature works properly."""
from typing import NamedTuple

import numpy as np
import pytest
import pywt
import torch
from scipy import signal

import src.ptwt as ptwt
from src.ptwt.continuous_transform import _ShannonWavelet
from tests._mackey_glass import MackeyGenerator


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


def _to_jit_wavedec_fun(data, wavelet, level):
    return ptwt.wavedec(data, wavelet, "reflect", level)


@pytest.mark.slow
@pytest.mark.parametrize("wavelet_string", ["db1", "db3", "db4", "sym5"])
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("length", [64, 65])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conv_fwt_jit(wavelet_string, level, length, batch_size, dtype):
    """Test jitting a convolution fwt, for various levels and padding options."""
    generator = MackeyGenerator(
        batch_size=batch_size, tmax=length, delta_t=1, device="cpu"
    )

    mackey_data_1 = torch.squeeze(generator(), -1).type(dtype)
    wavelet = pywt.Wavelet(wavelet_string)
    wavelet = _set_up_wavelet_tuple(wavelet, dtype)

    with pytest.warns(Warning):
        jit_wavedec = torch.jit.trace(
            _to_jit_wavedec_fun,
            (mackey_data_1, wavelet, torch.tensor(level)),
            strict=False,
        )
        ptcoeff = jit_wavedec(mackey_data_1, wavelet, level=torch.tensor(level))
        jit_waverec = torch.jit.trace(ptwt.waverec, (ptcoeff, wavelet))
        res = jit_waverec(ptcoeff, wavelet)
    assert np.allclose(mackey_data_1.numpy(), res.numpy()[:, : mackey_data_1.shape[-1]])


def _to_jit_wavedec_2(data, wavelet):
    """Ensure uniform datatypes in lists for the tracer.

    Going from List[Union[torch.Tensor, Tuple[torch.Tensor]]] to List[torch.Tensor]
    means we have to stack the lists in the output.
    """
    assert data.shape == (10, 20, 20), "Changing the chape requires re-tracing."
    coeff = ptwt.wavedec2(data, wavelet, "reflect", 2)
    coeff2 = []
    for c in coeff:
        if isinstance(c, torch.Tensor):
            coeff2.append(c)
        else:
            coeff2.append(torch.stack(c))
    return coeff2


def _to_jit_waverec_2(data, wavelet):
    """Undo the stacking from the jit wavedec2 wrapper."""
    d_unstack = [data[0]]
    for c in data[1:]:
        d_unstack.append(tuple(sc.squeeze(0) for sc in torch.split(c, 1, dim=0)))
    rec = ptwt.waverec2(d_unstack, wavelet)
    return rec


def test_conv_fwt_jit_2d():
    """Test the jit compilation feature for the wavedec2 function."""
    data = torch.randn(10, 20, 20).type(torch.float64)
    wavelet = pywt.Wavelet("db4")
    coeff = _to_jit_wavedec_2(data, wavelet)
    rec = _to_jit_waverec_2(coeff, wavelet)
    assert np.allclose(rec.squeeze(1).numpy(), data.numpy())

    wavelet = _set_up_wavelet_tuple(wavelet, dtype=torch.float64)
    with pytest.warns(Warning):
        jit_wavedec2 = torch.jit.trace(
            _to_jit_wavedec_2,
            (data, wavelet),
            strict=False,
        )
        jit_ptcoeff = jit_wavedec2(data, wavelet)
        # unstack the lists.
        jit_waverec = torch.jit.trace(_to_jit_waverec_2, (jit_ptcoeff, wavelet))
        rec = jit_waverec(jit_ptcoeff, wavelet)
    assert np.allclose(rec.squeeze(1).numpy(), data.numpy(), atol=1e-7)


def _to_jit_wavedec_3(data, wavelet):
    """Ensure uniform datatypes in lists for the tracer.

    Going from List[Union[torch.Tensor, Dict[str, torch.Tensor]]] to List[torch.Tensor]
    means we have to stack the lists in the output.
    """
    assert data.shape == (10, 20, 20, 20), "Changing the chape requires re-tracing."
    coeff = ptwt.wavedec3(data, wavelet, "reflect", 2)
    coeff2 = []
    keys = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")
    for c in coeff:
        if isinstance(c, torch.Tensor):
            coeff2.append(c)
        else:
            coeff2.append(torch.stack([c[key] for key in keys]))
    return coeff2


def _to_jit_waverec_3(data, wavelet):
    """Undo the stacking from the jit wavedec3 wrapper."""
    d_unstack = [data[0]]
    keys = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")
    for c in data[1:]:
        d_unstack.append(
            {key: sc.squeeze(0) for sc, key in zip(torch.split(c, 1, dim=0), keys)}
        )
    rec = ptwt.waverec3(d_unstack, wavelet)
    return rec


def test_conv_fwt_jit_3d():
    """Test the jit compilation feature for the wavedec3 function."""
    data = torch.randn(10, 20, 20, 20).type(torch.float64)
    wavelet = pywt.Wavelet("db4")
    coeff = _to_jit_wavedec_3(data, wavelet)
    rec = _to_jit_waverec_3(coeff, wavelet)
    assert np.allclose(rec.squeeze(1).numpy(), data.numpy())

    wavelet = _set_up_wavelet_tuple(wavelet, dtype=torch.float64)
    with pytest.warns(Warning):
        jit_wavedec3 = torch.jit.trace(
            _to_jit_wavedec_3,
            (data, wavelet),
            strict=False,
        )
        jit_ptcoeff = jit_wavedec3(data, wavelet)
        # unstack the lists.
        jit_waverec = torch.jit.trace(_to_jit_waverec_3, (jit_ptcoeff, wavelet))
        rec = jit_waverec(jit_ptcoeff, wavelet)
    assert np.allclose(rec.squeeze(1).numpy(), data.numpy(), atol=1e-7)


def _to_jit_cwt(sig):
    widths = torch.arange(1, 31)
    wavelet = _ShannonWavelet("shan0.1-0.4")
    sampling_period = (4 / 800) * np.pi
    cwtmatr, _ = ptwt.cwt(sig, widths, wavelet, sampling_period=sampling_period)
    return cwtmatr


def test_cwt_jit():
    """Test cwt jitting."""
    t = np.linspace(-2, 2, 800, endpoint=False)
    sig = torch.from_numpy(signal.chirp(t, f0=1, f1=12, t1=2, method="linear"))
    with pytest.warns(Warning):
        jit_cwt = torch.jit.trace(_to_jit_cwt, (sig), strict=False)
    jitcwtmatr = jit_cwt(sig)

    cwtmatr, _ = ptwt.cwt(
        sig,
        torch.arange(1, 31),
        pywt.ContinuousWavelet("shan0.1-0.4"),
        sampling_period=(4 / 800) * np.pi,
    )
    assert np.allclose(jitcwtmatr.numpy(), cwtmatr.numpy())
