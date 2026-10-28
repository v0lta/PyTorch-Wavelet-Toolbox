"""PyTorch compatible cwt code.

This module is based on pywt's cwt implementation.
"""

from typing import Any, overload

import numpy as np
import torch
from pywt import ContinuousWavelet, DiscreteContinuousWavelet, Wavelet
from pywt._functions import scale2frequency
from torch.fft import fft, ifft

__all__ = [
    "complex_morlet_fourier_transform",
    "complex_morlet_reconstruction",
    "cwt",
    "icwt",
]


def _next_fast_len(n: int) -> int:
    """Round up to the nearest power of two.

    Args:
        n: the number of samples

    Returns:
        The next power of two following this number to take advantage of FFT speedup

    .. warning::

        This fallback is less efficient than ``scipy.fftpack.next_fast_len``
    """
    return int(2 ** np.ceil(np.log2(n)))


def cwt(
    data: torch.Tensor,
    scales: np.ndarray | torch.Tensor,
    wavelet: ContinuousWavelet | str,
    sampling_period: float = 1.0,
    precision: int = 12,
) -> tuple[torch.Tensor, np.ndarray]:
    """Compute the single-dimensional continuous wavelet transform.

    This function is a PyTorch port of pywt.cwt as found at:
    https://github.com/PyWavelets/pywt/blob/master/pywt/_cwt.py

    Args:
        data:
            The input tensor of shape ``[batch_size, time]``.

        scales: The wavelet scales to use. One can use ``f =
            pywt.scale2frequency(wavelet, scale)/sampling_period`` to determine what
            physical frequency, ``f``. Here, ``f`` is in hertz when the ``sampling_period``
            is given in seconds.

        wavelet:
            The continuous wavelet to work with.

        sampling_period: The optional sampling period
            for the frequencies output. The values computed for ``coefs`` are
            independent of the choice of ``sampling_period`` (i.e. ``scales`` is not
            scaled by the sampling period)

        precision:
            Length of the wavelet used for the CWT

    Returns:
        A pair of out_tensor and frequencies. The first tuple-element contains the
        transformation matrix of shape ``[scales, batch, time]``. The second element
        contains an array with frequency information.

    Raises:
        ValueError: If a scale is too small for the input signal.

    Example:
        >>> import torch, ptwt
        >>> import numpy as np
        >>> import scipy.signal as signal
        >>> t = np.linspace(-2, 2, 800, endpoint=False)
        >>> sig = signal.chirp(t, f0=1, f1=12, t1=2, method="linear")
        >>> widths = np.arange(1, 31)
        >>> cwtmatr, freqs = ptwt.cwt(
        ...     torch.from_numpy(sig), widths, "mexh", sampling_period=(4 / 800) * np.pi
        ... )
    """
    # accept array_like input; make a copy to ensure a contiguous array
    if not isinstance(
        wavelet, (ContinuousWavelet, Wavelet, _DifferentiableContinuousWavelet)
    ):
        wavelet = DiscreteContinuousWavelet(wavelet)
    if type(scales) is torch.Tensor:
        scales = scales.numpy()
    elif np.isscalar(scales):
        scales = np.array([scales])

    if isinstance(wavelet, torch.nn.Module) and data.is_cuda:
        wavelet.cuda()

    int_psi, x = _integrate_wavelet(wavelet, precision=precision)
    if type(wavelet) is ContinuousWavelet:
        int_psi = np.conj(int_psi) if wavelet.complex_cwt else int_psi
        int_psi = torch.tensor(int_psi, device=data.device)
    elif isinstance(wavelet, torch.nn.Module):
        int_psi = torch.conj(int_psi) if wavelet.complex_cwt else int_psi
    else:
        int_psi = torch.tensor(int_psi, device=data.device)
        x = torch.tensor(x, device=data.device)

    # convert int_psi, x to the same precision as the data
    # x = np.asarray(x, dtype=data.cpu().numpy().real.dtype)

    size_scale0 = -1
    fft_data = None

    out = []
    for scale in scales:
        step = x[1] - x[0]
        j = torch.arange(
            scale * (x[-1] - x[0]) + 1, device=data.device, dtype=data.dtype
        ) / (scale * step)
        j = torch.floor(j).type(torch.long)
        if j[-1] >= len(int_psi):
            # j = np.extract(j < len(int_psi), j)
            j = torch.masked_select(j, j < len(int_psi))
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
            raise ValueError(f"Selected scale of {scale} too small.")

        out.append(coef)
    out_tensor = torch.stack(out)
    if type(wavelet) is Wavelet:
        out_tensor = out_tensor.real
    elif isinstance(wavelet, _DifferentiableContinuousWavelet):
        out_tensor = out_tensor if wavelet.complex_cwt else out_tensor.real
        wavelet.cpu()
    else:
        out_tensor = out_tensor if wavelet.complex_cwt else out_tensor.real

    with torch.no_grad():
        frequencies = scale2frequency(wavelet, scales, precision)
        if np.isscalar(frequencies):
            frequencies = np.array([frequencies])
    frequencies /= sampling_period

    if isinstance(wavelet, _DifferentiableContinuousWavelet) and data.is_cuda:
        wavelet.cuda()

    return out_tensor, frequencies


def _integrate_wavelet(wavelet: ContinuousWavelet | str, precision: int = 8) -> Any:
    """Integrate `psi` wavelet function from -Inf to x using rectangle integration.

    Modified to enable gradient flow through the cwt.

    Ported from:
    https://github.com/PyWavelets/pywt/blob/cef09e7f419aaf4c39b9f778bdc2d54b32fd7337/pywt/_functions.py#L60

    Args:
        wavelet: Wavelet to integrate.
            If a string, should be the name of a wavelet.

        precision: Precision that will be used for wavelet function
            approximation computed with the wavefun(level=precision) Wavelet's method.
            Defaults to 8.

    Returns:
        A tuple (int_psi, x) for orthogonal wavelets; for other wavelets, a tuple
        (int_psi_d, int_psi_r, x) is returned instead.

    Example:
        >>> from pywt import Wavelet
        >>> from ptwt.continuous_transform import _integrate_wavelet
        >>> wavelet1 = Wavelet("db2")
        >>> [int_psi, x] = _integrate_wavelet(wavelet1, precision=5)
        >>> wavelet2 = Wavelet("bior1.3")
        >>> [int_psi_d, int_psi_r, x] = _integrate_wavelet(wavelet2, precision=5)
    """

    @overload
    def _integrate(
        arr: torch.Tensor, step: np.float64 | torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def _integrate(arr: np.ndarray, step: np.float64 | torch.Tensor) -> np.ndarray: ...

    def _integrate(
        arr: np.ndarray | torch.Tensor, step: np.float64 | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(arr, np.ndarray):
            np_integral = np.cumsum(arr)
            np_integral *= step
            return np_integral
        elif isinstance(arr, torch.Tensor):
            torch_integral: torch.Tensor = torch.cumsum(arr, -1)
            torch_integral *= step
            return torch_integral
        else:
            raise TypeError("Only ndarrays or tensors are integratable.")

    if type(wavelet) is str or not isinstance(
        wavelet, (Wavelet, ContinuousWavelet, _DifferentiableContinuousWavelet)
    ):
        wavelet = DiscreteContinuousWavelet(wavelet)

    functions_approximations = wavelet.wavefun(precision)

    if len(functions_approximations) == 2:  # continuous wavelet
        psi, x = functions_approximations
        step = x[1] - x[0]
        return _integrate(psi, step), x

    elif len(functions_approximations) == 3:  # orthogonal wavelet
        _, psi, x = functions_approximations  # type: ignore
        step = x[1] - x[0]
        return _integrate(psi, step), x

    else:  # biorthogonal wavelet
        _, psi_d, _, psi_r, x = functions_approximations  # type: ignore
        step = x[1] - x[0]
        return _integrate(psi_d, step), _integrate(psi_r, step), x


def icwt(
    time_array: np.ndarray,
    scales: np.ndarray,
    coefficients: np.ndarray | torch.Tensor,
    wavelet: ContinuousWavelet | str,
) -> tuple[torch.Tensor, float]:
    r"""Reconstruct a real-valued signal from its CWT coefficients.

    This inverts :func:`ptwt.cwt` using the :cite:`torrence1998` reconstruction formula
    (their Eq. 11):

    .. math::

        x(t) = \frac{\delta j \,\sqrt{\delta t}}{\psi(0)\, C_{\delta}}
               \sum_{j=0}^{J} \frac{\mathrm{Re}[W(s_j, t)]}{\sqrt{s_j}}

    where

    - :math:`s_j` is the :math:`j`-th analysis scale **in seconds**, with :math:`j = 0,
      \dots, J` and :math:`J = S - 1` for :math:`S` scales (see the note below for how
      to obtain it from the scales given to :func:`ptwt.cwt`)
    - :math:`\delta j = \log_2(\frac{s_1}{s_0})` is the logarithmic scale spacing
    - :math:`\delta t = \frac{1}{F_s}` is the sampling interval in seconds, computed as
      ``time_array[1] - time_array[0]``
    - :math:`\psi(0) = \frac{1}{\sqrt{\pi B}}` is the mother wavelet evaluated at the
      origin, where :math:`B` is the bandwidth of the wavelet
      ``cmor{bandwidth}-{center_frequency}``
    - :math:`C_{\delta}` is the wavelet-dependent reconstruction constant, which depends
      on :math:`B` and on the center frequency :math:`C` (see
      :func:`~ptwt.continuous_transform.complex_morlet_reconstruction` for how it is
      computed)
    - :math:`W(s_j, t)` are the CWT coefficients at scale :math:`s_j`

    .. warning::

        This function currently **only supports complex Morlet (cmor) wavelets** (e.g.
        ``"cmor1.5-1.0"`` or ``pywt.ContinuousWavelet("cmor1.5-1.0")``). Any other
        wavelet family raises a ``NotImplementedError``.

    .. note::

        **Scales and the sampling frequency** ``Fs``. ``Fs`` is the sampling frequency
        of the signal in Hz (samples per second), so ``Fs = 1 / (time_array[1] -
        time_array[0])``. It is not an argument of this function; it only appears here
        to convert scales.

        :func:`ptwt.cwt` interprets its ``scales`` in units of samples and only uses
        ``sampling_period`` to compute the returned frequencies. This function expects
        scales in seconds. Convert them with

        ``scales = cwt_scales / Fs``

        where ``cwt_scales`` are the scales passed to :func:`ptwt.cwt` (equivalently,
        ``cwt_scales * sampling_period``).

    Args:
        time_array:
            1-D time array of length *T* (seconds). Must be uniformly sampled.

        scales:
            1-D array of analysis scales of length *S*, in seconds, i.e. ``cwt_scales /
            Fs`` where ``cwt_scales`` are the scales (in samples) passed to
            :func:`ptwt.cwt`. Must be logarithmically spaced, since :math:`\delta j` is
            derived from ``scales[1] / scales[0]``.

        coefficients:
            CWT coefficient matrix of shape ``[S, batch, T]`` or ``[S, T]`` as returned
            by :func:`ptwt.cwt`. Accepts both :class:`torch.Tensor` and
            :class:`numpy.ndarray`; the output dtype/device matches the input.

        wavelet:
            The wavelet used for the forward CWT. Must be a cmor wavelet, specified
            either as a :class:`pywt.ContinuousWavelet` instance or a string of the form
            ``"cmor{bandwidth}-{center_frequency}"`` (e.g. ``"cmor1.5-1.0"``).

    Returns:
        A tuple ``(reconstructed_signal, reconstruction_constant)`` where

        - ``reconstructed_signal`` is a :class:`torch.Tensor` with the same shape as the
          time dimension of ``coefficients`` (i.e. ``[batch, T]`` or ``[T]``).
        - ``reconstruction_constant`` is the scalar reconstruction constant
          :math:`C_{\delta}` that was used, based on the :math:`B` and :math:`C`
          parameters of the complex Morlet wavelet.

    Raises:
        TypeError: If ``wavelet`` is neither a string nor a
            :class:`pywt.ContinuousWavelet`.

        NotImplementedError: If ``wavelet`` is not a complex Morlet (cmor)
            family wavelet.

        ValueError: If the bandwidth and center frequency cannot be parsed
            from the wavelet name.

        ValueError: If ``scales`` contains fewer than two elements
            (:math:`\delta j` cannot be computed).

    Example:
        >>> import torch, numpy as np, ptwt, pywt
        >>> Fs = 1000  # sampling frequency in Hz, so sampling_period = 1 / Fs
        >>> t_np = np.linspace(0, 1, Fs, endpoint=False)
        >>> x = torch.sin(2 * np.pi * 50 * torch.from_numpy(t_np)).unsqueeze(0)
        >>> cwt_scales = np.geomspace(2, 256, 64)  # in samples, as ptwt.cwt expects
        >>> wavelet = pywt.ContinuousWavelet("cmor1.5-1.0")
        >>> coeffs, _ = ptwt.cwt(x, cwt_scales, wavelet, sampling_period=1 / Fs)
        >>> x_rec, reconstruction_constant = ptwt.icwt(
        ...     t_np, cwt_scales / Fs, coeffs, wavelet  # scales in seconds
        ... )
    """
    if isinstance(wavelet, str):
        wavelet_name = wavelet
    elif isinstance(wavelet, ContinuousWavelet):
        wavelet_name = wavelet.name
    else:
        raise TypeError(
            f"wavelet must be a pywt.ContinuousWavelet or str, got {type(wavelet)}"
        )

    if not wavelet_name.lower().startswith("cmor"):
        raise NotImplementedError(
            f"icwt_torrence currently only supports complex Morlet (cmor) wavelets; "
            f"got '{wavelet_name}'.  Other wavelet families require separate "
            f"derivations of psi(0) and psi_hat and are not yet implemented."
        )

    try:
        parameter_string = wavelet_name[4:]
        bandwidth_str, center_frequency_str = parameter_string.split("-")
        bandwidth = float(bandwidth_str)
        center_frequency = float(center_frequency_str)
    except (ValueError, AttributeError) as error:
        raise ValueError(
            f"Could not parse bandwidth and center_frequency from wavelet name '{wavelet_name}'. "
            f"Expected format: 'cmor{{bandwidth}}-{{center_frequency}}' (e.g. 'cmor1.5-1.0')."
        ) from error

    if len(scales) < 2:
        raise ValueError("Scale array 'scales' must contain at least 2 elements.")

    sampling_interval = time_array[1] - time_array[0]

    scale_spacing = np.log2(scales[1] / scales[0])

    wavelet_at_origin = 1.0 / np.sqrt(np.pi * bandwidth)

    reconstruction_constant = complex_morlet_reconstruction(bandwidth, center_frequency)

    if isinstance(coefficients, np.ndarray):
        coefficients = torch.from_numpy(coefficients)

    scales_tensor = torch.tensor(
        scales, dtype=coefficients.real.dtype, device=coefficients.device
    )

    if coefficients.dim() == 3:
        # [S, batch, T] -> broadcast scales over batch and time dimensions
        inverse_sqrt_scales = scales_tensor[:, None, None] ** (-0.5)
    else:
        # [S, T] -> broadcast scales over time dimension only
        inverse_sqrt_scales = scales_tensor[:, None] ** (-0.5)

    weighted_scale_sum = torch.sum(coefficients.real * inverse_sqrt_scales, dim=0)

    amplitude_factor = (scale_spacing * np.sqrt(sampling_interval)) / (
        wavelet_at_origin * reconstruction_constant
    )
    reconstructed_signal = amplitude_factor * weighted_scale_sum

    return reconstructed_signal, reconstruction_constant


def complex_morlet_reconstruction(
    bandwidth: float,
    center_frequency: float,
    fft_length: int = 4096,
    n_scales: int = 1000,
    min_scale: float = 0.5,
    max_scale: float = 10000.0,
) -> float:
    r"""Compute the reconstruction constant :math:`C_{\delta}` for a complex Morlet wavelet.

    This is the constant for the wavelet ``cmor{bandwidth}-{center_frequency}``. It is
    defined by :cite:`torrence1998` Eq. 13 as

    .. math::

        C_{\delta} = \frac{\delta j}{\psi(0)}
            \sum_{j=0}^{J} \frac{\mathrm{Re}\left[W_{\delta}(s_j)\right]}{\sqrt{s_j}}

    and is needed to normalize the inverse CWT in :func:`ptwt.icwt`. It depends only on
    the wavelet parameters and **not** on the signal being reconstructed.

    The quantities in this equation are:

    - :math:`s_j` is the :math:`j`-th scale. Here scales are dimensionless, i.e.
      measured in units of the sampling interval (:math:`\delta t = 1`), so the constant
      does not depend on the sampling frequency. The scales are logarithmically spaced
      between ``min_scale`` and ``max_scale``, with :math:`j = 0, \dots, J` and :math:`J
      = \text{n\_scales} - 1`. They are internal scales used only to evaluate the
      constant and are independent of the ``scales`` passed to :func:`ptwt.icwt`.
    - :math:`\delta j = \log_2(\frac{s_1}{s_0})` is the logarithmic scale spacing
      (:cite:`torrence1998` Eq. 9).
    - :math:`\psi(0) = \frac{1}{\sqrt{\pi B}}` is the mother wavelet evaluated at the
      origin (:cite:`torrence1998` Eq. 4), where :math:`B` is ``bandwidth``.
    - :math:`W_{\delta}(s_j)` is the wavelet transform at scale :math:`s_j` of a
      discrete delta function at :math:`n = 0` (:cite:`torrence1998` Eq. 12). The FFT of
      such a signal is constant, so the transform reduces to a sum over the :math:`N`
      FFT bins (``fft_length``):

      .. math::

          W_{\delta}(s_j) = \frac{\sqrt{s_j}}{N}
              \sum_{k=0}^{N-1} \hat{\psi}^{*}(s_j \omega_k)

      where :math:`\omega_k = 2\pi f_k` is the angular frequency of the :math:`k`-th FFT
      bin, :math:`f_k` is the :math:`k`-th value returned by
      ``np.fft.fftfreq(fft_length)`` (cycles per sample), and :math:`^{*}` denotes the
      complex conjugate.

      .. note::

          The prefactor :math:`\frac{\sqrt{s_j}}{N}` uses the :math:`\sqrt{s}`
          normalization of :func:`ptwt.cwt` with :math:`\delta t = 1`. It therefore
          differs from the prefactor :math:`\sqrt{\frac{2\pi s}{\delta t}}` printed in
          Eq. 12 of the paper.

    The Fourier transform of the complex Morlet wavelet used in :math:`W_{\delta}` is

    .. math::

        \hat{\psi}(s\omega) =
            \exp\!\left(-\frac{B(s\omega - \omega_0)^2}{4}\right)
            \cdot H(s\omega)

    where :math:`s` is the (dimensionless) scale, :math:`\omega` is the angular
    frequency in radians per sample, :math:`B` is ``bandwidth``, :math:`\omega_0 = 2\pi
    C` is the center angular frequency with :math:`C` = ``center_frequency``, and
    :math:`H` is the Heaviside step function that keeps only positive frequencies, which
    makes the wavelet analytic (one-sided).

    Args:
        bandwidth: Bandwidth parameter :math:`B` of the complex Morlet wavelet
            (``cmor{bandwidth}-{center_frequency}``).

        center_frequency: Center-frequency parameter :math:`C` of the complex
            Morlet wavelet (``cmor{bandwidth}-{center_frequency}``).

        fft_length: Number of FFT points :math:`N` used for the internal
            numerical integration. Larger values give a more accurate result, but
            increase runtime. Defaults to 4,096.

        n_scales: Number of scales :math:`J + 1` to sum over when evaluating
            Eq. 13. Should span several decades; defaults to 1,000.

        min_scale: Smallest scale :math:`s_0` in the internal sum
            (dimensionless). Defaults to 0.5.

        max_scale: Largest scale :math:`s_J` in the internal sum
            (dimensionless). Defaults to 10,000.

    Returns:
        The scalar reconstruction constant :math:`C_{\delta}`.
    """
    center_angular_frequency = 2.0 * np.pi * center_frequency

    angular_frequencies = np.fft.fftfreq(fft_length) * 2.0 * np.pi

    scales = np.geomspace(min_scale, max_scale, n_scales)

    scale_spacing = np.log2(scales[1] / scales[0])

    wavelet_at_origin = 1.0 / np.sqrt(np.pi * bandwidth)

    # W_delta(s_j) from the docstring; the 1/N is the FFT normalization.
    wavelet_coefficients = np.array(
        [
            (np.sqrt(scale) / fft_length)
            * np.sum(
                np.conj(
                    complex_morlet_fourier_transform(
                        scale * angular_frequencies, bandwidth, center_angular_frequency
                    )
                )
            )
            for scale in scales
        ]
    )

    reconstruction_constant = (scale_spacing / wavelet_at_origin) * np.sum(
        np.real(wavelet_coefficients) / np.sqrt(scales)
    )

    return reconstruction_constant


def complex_morlet_fourier_transform(
    scaled_angular_frequencies: np.ndarray,
    bandwidth: float,
    center_angular_frequency: float,
) -> np.ndarray:
    r"""Normalized Fourier transform of the cmor mother wavelet.

    Implements the analytic (one-sided) complex Morlet transfer function. The Heaviside
    factor ensures the wavelet is strictly analytic.

    Args:
        scaled_angular_frequencies:
            The product :math:`s\omega` of the scale :math:`s` and the angular
            frequencies :math:`\omega` (in radians per sample).

        bandwidth:
            Bandwidth parameter :math:`B` of the complex Morlet wavelet.

        center_angular_frequency:
            Center angular frequency :math:`\omega_0 = 2\pi C`, where :math:`C` is the
            center frequency of the complex Morlet wavelet.

    Returns:
        Complex array of wavelet Fourier coefficients, with the same shape as
        ``scaled_angular_frequencies``.
    """
    positive_frequency_mask = (scaled_angular_frequencies > 0).astype(float)
    return (
        np.exp(
            -bandwidth
            * (scaled_angular_frequencies - center_angular_frequency) ** 2
            / 4.0
        )
        * positive_frequency_mask
    )


class _WaveletParameter(torch.nn.Parameter):
    pass


class _DifferentiableContinuousWavelet(
    torch.nn.Module,
    ContinuousWavelet,  # type: ignore
):
    """A base class for learnable Continuous Wavelets."""

    def __init__(self, name: str):
        """Create a trainable shannon wavelet."""
        super().__init__()
        super(ContinuousWavelet, self).__init__()

        self.dtype = torch.float64
        # Use torch nn parameter
        self.bandwidth_par = _WaveletParameter(
            torch.sqrt(torch.tensor(self.bandwidth_frequency, dtype=self.dtype)),
            requires_grad=True,
        )
        self.center_par = _WaveletParameter(
            torch.sqrt(torch.tensor(self.center_frequency, dtype=self.dtype)),
            requires_grad=True,
        )

    def __call__(self, grid_values: torch.Tensor) -> torch.Tensor:
        """Return numerical values for the wavelet on a grid.

        Raises:
            NotImplementedError: If this call is not overwritten by a child.
        """
        raise NotImplementedError

    @property
    def bandwidth(self) -> torch.Tensor:
        """Square the bandwith parameter to ensure positive values."""
        return self.bandwidth_par * self.bandwidth_par

    @property
    def center(self) -> torch.Tensor:
        """Square the bandwith parameter to ensure positive values."""
        return self.center_par * self.center_par

    def wavefun(
        self, precision: int, dtype: torch.dtype = torch.float64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Define a grid and evaluate the wavelet on it."""
        length = 2**precision
        # load the bounds from untyped pywt code.
        lower_bound: float = float(self.lower_bound)  # type: ignore
        upper_bound: float = float(self.upper_bound)  # type: ignore
        grid = torch.linspace(
            lower_bound,
            upper_bound,
            length,
            dtype=dtype,
            device=self.bandwidth_par.device,
        )
        return self(grid), grid


class _ShannonWavelet(_DifferentiableContinuousWavelet):
    """A differentiable Shannon wavelet."""

    def __call__(self, grid_values: torch.Tensor) -> torch.Tensor:
        """Return numerical values for the wavelet on a grid."""
        shannon = (
            torch.sqrt(self.bandwidth)
            * (
                torch.sin(torch.pi * self.bandwidth * grid_values)
                / (torch.pi * self.bandwidth * grid_values)
            )
            * torch.exp(1j * 2 * torch.pi * self.center * grid_values)
        )
        return shannon


class _ComplexMorletWavelet(_DifferentiableContinuousWavelet):
    """A differentiable Shannon wavelet."""

    def __call__(self, grid_values: torch.Tensor) -> torch.Tensor:
        """Return numerical values for the wavelet on a grid."""
        morlet = (
            1.0
            / torch.sqrt(torch.pi * self.bandwidth)
            * torch.exp(-(grid_values**2) / self.bandwidth)
            * torch.exp(1j * 2 * torch.pi * self.center * grid_values)
        )
        return morlet
