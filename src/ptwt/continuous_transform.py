"""PyTorch compatible cwt code.

Based on https://github.com/PyWavelets/pywt/blob/master/pywt/_cwt.py
"""
from typing import Tuple, Union

import numpy as np
import torch
from pywt import ContinuousWavelet, DiscreteContinuousWavelet, Wavelet
from pywt._functions import scale2frequency
from torch.fft import fft, ifft
import warnings


def _next_fast_len(n: int) -> int:
    """Round up size to the nearest power of two.

    Given a number of samples `n`, returns the next power of two
    following this number to take advantage of FFT speedup.
    This fallback is less efficient than `scipy.fftpack.next_fast_len`
    """
    return int(2 ** np.ceil(np.log2(n)))


def cwt(
    data: torch.Tensor,
    scales: Union[np.ndarray, torch.Tensor],  # type: ignore
    wavelet: Union[ContinuousWavelet, str],
    sampling_period: float = 1.0,
) -> Tuple[torch.Tensor, np.ndarray]:  # type: ignore
    """Compute the single dimensional continuous wavelet transform.

    This function is a PyTorch port of pywt.cwt as found at:
    https://github.com/PyWavelets/pywt/blob/master/pywt/_cwt.py

    Args:
        data (torch.Tensor): The input tensor of shape [batch_size, time].
        scales (torch.Tensor or np.array):
            The wavelet scales to use. One can use
            ``f = pywt.scale2frequency(wavelet, scale)/sampling_period`` to determine
            what physical frequency, ``f``. Here, ``f`` is in hertz when the
            ``sampling_period`` is given in seconds.
            wavelet (str or Wavelet of ContinuousWavelet): The wavelet to work with.
        wavelet (ContinuousWavelet or str): The continuous wavelet to work with.
        sampling_period (float): Sampling period for the frequencies output (optional).
            The values computed for ``coefs`` are independent of the choice of
            ``sampling_period`` (i.e. ``scales`` is not scaled by the sampling
            period).

    Raises:
        ValueError: If a scale is too small for the input signal.

    Returns:
        Tuple[torch.Tensor, np.ndarray]: A tuple with the transformation matrix
            and frequencies in this order.

    Example:
        >>> import torch, ptwt
        >>> import numpy as np
        >>> import scipy.signal as signal
        >>> t = np.linspace(-2, 2, 800, endpoint=False)
        >>> sig = signal.chirp(t, f0=1, f1=12, t1=2, method="linear")
        >>> widths = np.arange(1, 31)
        >>> cwtmatr, freqs = ptwt.cwt(
        >>>     torch.from_numpy(sig), widths, "mexh", sampling_period=(4 / 800) * np.pi
        >>> )
    """
    # accept array_like input; make a copy to ensure a contiguous array
    if not isinstance(wavelet, (ContinuousWavelet, Wavelet, DifferentiableWavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)
    if type(scales) is torch.Tensor:
        scales = scales.numpy()
    elif np.isscalar(scales):
        scales = np.array([scales])
    # if not np.isscalar(axis):
    #    raise np.AxisError("axis must be a scalar.")

    precision = 10
    int_psi, x = _integrate_wavelet(wavelet, precision=precision)
    if type(wavelet) is ContinuousWavelet:
        int_psi = np.conj(int_psi) if wavelet.complex_cwt else int_psi
    int_psi = torch.tensor(int_psi, device=data.device)

    # convert int_psi, x to the same precision as the data
    x = np.asarray(x, dtype=data.cpu().numpy().real.dtype)

    size_scale0 = -1
    fft_data = None

    out = []
    for scale in scales:
        step = x[1] - x[0]
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
        j = j.astype(int)  # floor
        if j[-1] >= len(int_psi):
            j = np.extract(j < len(int_psi), j)
        int_psi_scale = int_psi[j].flip(0)

        # The padding is selected for:
        # - optimal FFT complexity
        # - to be larger than the two signals length to avoid circular
        #   convolution
        size_scale = _next_fast_len(data.shape[-1] + len(int_psi_scale) - 1)
        if size_scale != size_scale0:
            # Must recompute fft_data when the padding size changes.
            fft_data = fft(data, size_scale, dim=-1)
        size_scale0 = size_scale
        fft_wav = fft(int_psi_scale, size_scale, dim=-1)
        conv = ifft(fft_wav * fft_data, dim=-1)
        conv = conv[..., : data.shape[-1] + len(int_psi_scale) - 1]

        coef = -np.sqrt(scale) * torch.diff(conv, dim=-1)

        # transform axis is always -1
        d = (coef.shape[-1] - data.shape[-1]) / 2.0
        if d > 0:
            coef = coef[..., int(np.floor(d)) : -int(np.ceil(d))]
        elif d < 0:
            raise ValueError("Selected scale of {} too small.".format(scale))

        out.append(coef)
    out_tensor = torch.stack(out)
    if type(wavelet) is Wavelet:
        out_tensor = out_tensor.real
    elif isinstance(wavelet, DifferentiableWavelet):
        out_tensor = out_tensor # TODO: fixme
    else:
        out_tensor = out_tensor if wavelet.complex_cwt else out_tensor.real

    frequencies = scale2frequency(wavelet, scales, precision)
    if np.isscalar(frequencies):
        frequencies = np.array([frequencies])
    frequencies /= sampling_period
    return out_tensor, frequencies


def _integrate_wavelet(wavelet, precision=8):
    """
    Ported from:
    https://github.com/PyWavelets/pywt/blob/cef09e7f419aaf4c39b9f778bdc2d54b32fd7337/pywt/_functions.py#L60

    Modified to enable gradient flow through the cwt.

    Integrate `psi` wavelet function from -Inf to x using the rectangle
    integration method.
    Parameters
    ----------
    wavelet : Wavelet instance or str
        Wavelet to integrate.  If a string, should be the name of a wavelet.
    precision : int, optional
        Precision that will be used for wavelet function
        approximation computed with the wavefun(level=precision)
        Wavelet's method (default: 8).
    Returns
    -------
    [int_psi, x] :
        for orthogonal wavelets
    [int_psi_d, int_psi_r, x] :
        for other wavelets
    Examples
    --------
    >>> from pywt import Wavelet, _integrate_wavelet
    >>> wavelet1 = Wavelet('db2')
    >>> [int_psi, x] = _integrate_wavelet(wavelet1, precision=5)
    >>> wavelet2 = Wavelet('bior1.3')
    >>> [int_psi_d, int_psi_r, x] = _integrate_wavelet(wavelet2, precision=5)
    """

    def _integrate(arr, step):
        if type(arr) is np.ndarray:
            integral = np.cumsum(arr)
        else:
            integral = torch.cumsum(arr, -1)
        integral *= step
        return integral

    if type(wavelet) in (tuple, list):
        msg = ("Integration of a general signal is deprecated "
               "and will be removed in a future version of pywt.")
        warnings.warn(msg, DeprecationWarning)
    elif not isinstance(wavelet, (Wavelet, ContinuousWavelet, DifferentiableWavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)

    if type(wavelet) in (tuple, list):
        psi, x = np.asarray(wavelet[0]), np.asarray(wavelet[1])
        step = x[1] - x[0]
        return _integrate(psi, step), x

    functions_approximations = wavelet.wavefun(precision)

    if len(functions_approximations) == 2:      # continuous wavelet
        psi, x = functions_approximations
        step = x[1] - x[0]
        return _integrate(psi, step), x

    elif len(functions_approximations) == 3:    # orthogonal wavelet
        _, psi, x = functions_approximations
        step = x[1] - x[0]
        return _integrate(psi, step), x

    else:                                       # biorthogonal wavelet
        _, psi_d, _, psi_r, x = functions_approximations
        step = x[1] - x[0]
        return _integrate(psi_d, step), _integrate(psi_r, step), x


class DifferentiableWavelet(torch.nn.Module, ContinuousWavelet):
    pass

class Shannon_Wavelet(DifferentiableWavelet):
    """A differentiable Shannon wavelet."""

    def __init__(self, name='shan1-1'):
        """Create a trainable shannon wavelet.

        Args:
            bandwidth (int): _description_
            center (int): _description_
        """
        super().__init__()
        self.bandwidth = torch.tensor(self.bandwidth_frequency)
        self.center = torch.tensor(self.center_frequency)

    def __call__(self, grid_values: torch.Tensor) -> torch.Tensor:
        shannon = \
            torch.sqrt(self.bandwidth) \
                * (torch.sin(torch.pi*self.bandwidth*grid_values)
                   / (torch.pi*self.bandwidth*grid_values) ) \
                * torch.exp(1j*2*torch.pi*self.center*grid_values)
        return shannon

    def wavefun(self, precision: int, dtype=torch.float64) -> torch.Tensor:
        length = 2**precision
        grid = torch.linspace(self.lower_bound,
                              self.upper_bound,
                              length,
                              dtype=dtype)
        return self(grid), grid
