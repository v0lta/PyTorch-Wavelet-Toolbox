"""PyTorch compatible cwt code.

This module is based on pywt's cwt implementation.
"""

from typing import Any, overload

import numpy as np
import torch
from pywt import ContinuousWavelet, DiscreteContinuousWavelet, Wavelet
from pywt._functions import scale2frequency
from torch.fft import fft, ifft

__all__ = ["cwt", "icwt"]


def _next_fast_len(n: int) -> int:
    """Round up size to the nearest power of two.

    Given a number of samples `n`, returns the next power of two
    following this number to take advantage of FFT speedup.
    This fallback is less efficient than `scipy.fftpack.next_fast_len`
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
        data (torch.Tensor): The input tensor of shape ``[batch_size, time]``.
        scales (torch.Tensor or np.array):
            The wavelet scales to use. One can use
            ``f = pywt.scale2frequency(wavelet, scale)/sampling_period`` to determine
            what physical frequency, ``f``. Here, ``f`` is in hertz when the
            ``sampling_period`` is given in seconds.
        wavelet (ContinuousWavelet or str): The continuous wavelet to work with.
        sampling_period (float): Sampling period for the frequencies output (optional).
            The values computed for ``coefs`` are independent of the choice of
            ``sampling_period`` (i.e. ``scales`` is not scaled by the sampling
            period).
        precision (int): Length of the wavelet used for the CWT.

    Raises:
        ValueError: If a scale is too small for the input signal.

    Returns:
        A tuple (out_tensor, frequencies). The first tuple-element contains
        the transformation matrix of shape ``[scales, batch, time]``.
        The second element contains an array with frequency information.

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
    """
    Integrate `psi` wavelet function from -Inf to x using rectangle integration.

    Modified to enable gradient flow through the cwt.

    Ported from:
    https://github.com/PyWavelets/pywt/blob/cef09e7f419aaf4c39b9f778bdc2d54b32fd7337/pywt/_functions.py#L60


    Args:
        wavelet (Wavelet instance or str): Wavelet to integrate.
            If a string, should be the name of a wavelet.
        precision (int): Precision that will be used for wavelet function
            approximation computed with the wavefun(level=precision)
            Wavelet's method. Defaults to 8.

    Returns:
        A tuple (int_psi, x) for orthogonal wavelets;
        for other wavelets, a tuple (int_psi_d, int_psi_r, x) is returned instead.

    Example:
        >>> from pywt import Wavelet
        >>> from ptwt.continuous_transform import _integrate_wavelet
        >>> wavelet1 = Wavelet('db2')
        >>> [int_psi, x] = _integrate_wavelet(wavelet1, precision=5)
        >>> wavelet2 = Wavelet('bior1.3')
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


def cdelta_cmor(
    bandwidth: float,
    center_frequency: float,
    fft_length: int = 4096,
    n_scales: int = 1000,
    min_scale: float = 0.5,
    max_scale: float = 10000.0,
) -> float:
    """Compute the Torrence & Compo reconstruction constant *C_delta* for a
    complex Morlet wavelet ``cmor{B}-{C}``.

    The constant is defined in Torrence & Compo (1998) Eq. 13 and is required
    to correctly normalise the inverse CWT sum.  It depends only on the wavelet
    family / parameters and **not** on the signal being reconstructed.

    The complex Morlet (cmor) wavelet used here has the Fourier transform

    .. math::

        \\hat{\\psi}(s\\omega) =
            \\exp\\!\\left(-\\frac{B(s\\omega - \\omega_0)^2}{4}\\right)
            \\cdot H(s\\omega)

    where :math:`\\omega_0 = 2\\pi C` is the centre (angular) frequency,
    :math:`B` is the bandwidth parameter, and :math:`H` is the Heaviside
    step function that enforces the analytic (one-sided) nature of the wavelet.

    The value at the origin is

    .. math::

        \\psi(0) = \\frac{1}{\\sqrt{\\pi B}}

    as required by Torrence & Compo Eq. 4.

    Args:
        bandwidth: Bandwidth parameter of the cmor wavelet (``cmor{bandwidth}-{center_frequency}``).
        center_frequency: Centre-frequency parameter of the cmor wavelet (``cmor{bandwidth}-{center_frequency}``).
        fft_length: Number of FFT points used for the internal numerical integration.
            Larger values give a more accurate result but increase runtime.
            Defaults to 4096.
        n_scales: Number of scales to sum over when evaluating Eq. 13.
            Should span several decades; defaults to 1000.
        min_scale: Smallest scale in the internal sum (dimensionless).
            Defaults to 0.5.
        max_scale: Largest scale in the internal sum (dimensionless).
            Defaults to 10000.

    Returns:
        The scalar reconstruction constant *reconstruction_constant*.

    References:
        Torrence & Compo (1998), Eq. 12–13.
    """
    # Angular centre frequency of the cmor wavelet
    center_angular_frequency = 2.0 * np.pi * center_frequency

    # Discrete angular frequencies for an N-point FFT
    angular_frequencies = np.fft.fftfreq(fft_length) * 2.0 * np.pi

    # Logarithmically spaced scale array (same spacing as the analysis scales)
    scales = np.geomspace(min_scale, max_scale, n_scales)

    # Logarithmic voice spacing  dj  (Torrence & Compo Eq. 9)
    scale_spacing = np.log2(scales[1] / scales[0])

    # Value of the cmor mother wavelet at the origin: psi(0) = 1 / sqrt(pi * B)
    # (Torrence & Compo Eq. 4, analytic morlet variant)
    wavelet_at_origin = 1.0 / np.sqrt(np.pi * bandwidth)

    def wavelet_fourier_transform(scaled_angular_frequencies: np.ndarray) -> np.ndarray:
        """Normalised Fourier transform of the cmor mother wavelet.

        Implements the analytic (one-sided) complex Morlet transfer function.
        The Heaviside factor ensures the wavelet is strictly analytic.

        Args:
            scaled_angular_frequencies: Product of scale *s* and angular
                frequency array *angular_frequencies*.

        Returns:
            Complex array of wavelet Fourier coefficients.
        """
        # Heaviside: keep only positive frequencies (analytic signal)
        positive_frequency_mask = (scaled_angular_frequencies > 0).astype(float)
        return (
            np.exp(
                -bandwidth
                * (scaled_angular_frequencies - center_angular_frequency) ** 2
                / 4.0
            )
            * positive_frequency_mask
        )

    # -------------------------------------------------------------------
    # Torrence & Compo Eq. 12:
    #
    #   W_n(s) = sqrt(2 * pi * s / delta_t) * sum_k { Psi*(s*omega_k) }
    #                                               N
    #
    # For a delta function at n=0  the FFT of the input is all-ones, so the
    # convolution in frequency space becomes a plain sum of psi_hat values.
    # The 1/N factor from the IFFT convention is absorbed into the 1/N
    # normalisation below.
    # -------------------------------------------------------------------
    wavelet_coefficients = np.array(
        [
            (np.sqrt(scale) / fft_length)
            * np.sum(np.conj(wavelet_fourier_transform(scale * angular_frequencies)))
            for scale in scales
        ]
    )

    # -------------------------------------------------------------------
    # Torrence & Compo Eq. 13:
    #
    #   C_delta = (dj / psi(0)) * sum_j { Re[W_j(s_j)] / sqrt(s_j) }
    #
    # This represents the reconstruction of a delta function; C_delta
    # normalises the amplitude so that the inverse CWT exactly recovers
    # the original signal.
    # -------------------------------------------------------------------
    reconstruction_constant = float(
        (scale_spacing / wavelet_at_origin)
        * np.sum(np.real(wavelet_coefficients) / np.sqrt(scales))
    )

    return reconstruction_constant


def icwt(
    time_array: np.ndarray,
    scales: np.ndarray,
    coefficients: np.ndarray | torch.Tensor,
    wavelet: ContinuousWavelet | str,
) -> tuple[torch.Tensor, float]:
    """Reconstruct a real-valued signal from its CWT coefficients using the
     Torrence & Compo (1998) inversion formula (their Eq. 11).

     The reconstruction formula is

     .. math::

         x(t) = \\frac{\\delta j \\,\\sqrt{\\delta t}}{\\psi(0)\\, C_\\delta}
                \\sum_{j} \\frac{\\mathrm{Re}[W(a_j, t)]}{\\sqrt{a_j}}

     where

     * :math:`\\delta j` is the logarithmic scale spacing,
     * :math:`\\delta t` is the sampling interval,
     * :math:`\\psi(0)` is the mother wavelet evaluated at the origin,
     * :math:`C_\\delta` is the wavelet-dependent reconstruction constant
       (see :func:`cdelta_cmor`), and
     * :math:`W(a_j, t)` are the CWT coefficients at scale :math:`a_j`.

     .. note::
         This function currently **only supports complex Morlet (cmor) wavelets**
         (e.g. ``"cmor1.5-1.0"`` or ``pywt.ContinuousWavelet("cmor1.5-1.0")``).
         Attempting to use any other wavelet family will raise a ``ValueError``.

    .. note::
         The scale array ``scales`` must correspond to the scales used in the
         forward :func:`ptwt.cwt` call that produced ``cwt_coefficients``.
         :func:`ptwt.cwt` uses its ``scales`` argument directly (in samples) and
         only applies ``sampling_period`` to the returned frequencies, so divide
         the scales you gave to ``cwt`` by ``Fs`` (equivalently, multiply them by
         ``sampling_period``) before passing them here.

     Args:
         time_array: 1-D time array of length *T* (seconds).  Must be uniformly sampled.
         scales: 1-D array of analysis scales of length *S* (dimensionless, i.e.
             ``pywt_scales / Fs``).  Must be logarithmically spaced (the
             logarithmic voice spacing :math:`\\delta j` is derived from
             ``scales[1] / scales[0]``).
         coefficients: CWT coefficient matrix of shape ``[S, batch, T]`` or ``[S, T]``
             as returned by :func:`ptwt.cwt`.  Accepts both
             :class:`torch.Tensor` and :class:`numpy.ndarray`; the output
             dtype/device matches the input.
         wavelet: The wavelet used for the forward CWT.  Must be a cmor wavelet,
             specified either as a :class:`pywt.ContinuousWavelet` instance or
             a string of the form ``"cmor{}-{C}"`` (e.g. ``"cmor1.5-1.0"``).

     Returns:
         A tuple ``(reconstructed_signal, reconstruction_constant)`` where

         * ``reconstructed_signal`` is a :class:`torch.Tensor` with the same shape
           as the time dimension of ``cwt_coefficients`` (i.e. ``[batch, T]`` or ``[T]``).
         * ``reconstruction_constant`` is the scalar Torrence & Compo reconstruction constant
           that was used.

     Raises:
         ValueError: If ``wavelet`` is not a complex Morlet (cmor) family wavelet.
         ValueError: If ``scales`` contains fewer than two elements (dj cannot be
             computed).

     References:
         Torrence & Compo (1998), Eq. 11.

     Example::

         >>> import torch, numpy as np, ptwt, pywt
         >>> Fs = 1000
         >>> t_np = np.linspace(0, 1, Fs, endpoint=False)
         >>> t_pt = torch.from_numpy(t_np)
         >>> x = torch.sin(2 * np.pi * 50 * t_pt).unsqueeze(0)
         >>> scales_pywt = np.geomspace(1, 256, 64) * Fs  # pywt convention
         >>> wavelet = pywt.ContinuousWavelet("cmor1.5-1.0")
         >>> coeffs, _ = ptwt.cwt(x, scales_pywt, wavelet, sampling_period=1/Fs)
         >>> x_rec, reconstruction_constant = icwt(
         ...     t_np, scales_pywt / Fs, coeffs, wavelet
         ... )
    """
    # ------------------------------------------------------------------
    # Validate wavelet family – only cmor is currently supported
    # ------------------------------------------------------------------
    if isinstance(wavelet, str):
        wavelet_name = wavelet
    elif isinstance(wavelet, ContinuousWavelet):
        wavelet_name = wavelet.name
    else:
        raise TypeError(
            f"wavelet must be a pywt.ContinuousWavelet or str, got {type(wavelet)}"
        )

    if not wavelet_name.lower().startswith("cmor"):
        raise ValueError(
            f"icwt_torrence currently only supports complex Morlet (cmor) wavelets; "
            f"got '{wavelet_name}'.  Other wavelet families require separate "
            f"derivations of psi(0) and psi_hat and are not yet implemented."
        )

    # Parse B and C from the wavelet name string "cmor{B}-{C}"
    try:
        parameter_string = wavelet_name[4:]  # strip "cmor"
        bandwidth_str, center_frequency_str = parameter_string.split("-")
        bandwidth = float(bandwidth_str)
        center_frequency = float(center_frequency_str)
    except (ValueError, AttributeError) as error:
        raise ValueError(
            f"Could not parse bandwidth and center_frequency from wavelet name '{wavelet_name}'. "
            f"Expected format: 'cmor{{bandwidth}}-{{enter_frequency}}' (e.g. 'cmor1.5-1.0')."
        ) from error

    if len(scales) < 2:
        raise ValueError("Scale array 'scales' must contain at least 2 elements.")

    # ------------------------------------------------------------------
    # Derive transform parameters
    # ------------------------------------------------------------------

    # Sampling interval (seconds per sample)
    sampling_interval = float(time_array[1] - time_array[0])

    # Logarithmic voice spacing dj (Torrence & Compo Eq. 9)
    scale_spacing = float(np.log2(scales[1] / scales[0]))

    # Value of the cmor mother wavelet at the origin: psi(0) = 1 / sqrt(pi * B)
    wavelet_at_origin = float(1.0 / np.sqrt(np.pi * bandwidth))

    # Reconstruction constant C_delta (Torrence & Compo Eq. 13)
    reconstruction_constant = cdelta_cmor(bandwidth, center_frequency)

    # ------------------------------------------------------------------
    # Convert inputs to torch tensors, preserving device / dtype
    # ------------------------------------------------------------------
    if isinstance(coefficients, np.ndarray):
        coefficients_tensor = torch.from_numpy(coefficients)
    else:
        coefficients_tensor = (
            coefficients  # already a tensor; keeps device & grad info
        )

    # Scale array as a tensor on the same device as the coefficients
    scales_tensor = torch.tensor(
        scales, dtype=coefficients_tensor.real.dtype, device=coefficients_tensor.device
    )

    # ------------------------------------------------------------------
    # Torrence & Compo Eq. 11:
    #
    #   x_n  =  (dj * sqrt(dt)) / (psi(0) * C_delta)
    #           * sum_j { Re[W_j(n)] / sqrt(a_j) }
    #
    # coefficients_tensor shape:  [S, batch, T]  or  [S, T]
    # scales_tensor shape:        [S]
    # ------------------------------------------------------------------

    # Normalise each scale slice by 1/sqrt(a_j) then sum over the scale axis
    if coefficients_tensor.dim() == 3:
        # [S, batch, T] -> broadcast scales over batch and time dimensions
        inverse_sqrt_scales = scales_tensor[:, None, None] ** (-0.5)
    else:
        # [S, T] -> broadcast scales over time dimension only
        inverse_sqrt_scales = scales_tensor[:, None] ** (-0.5)

    # Sum of Re[W(s, t)] / sqrt(s)  over scale axis (dim=0)
    weighted_scale_sum = torch.sum(
        coefficients_tensor.real * inverse_sqrt_scales, dim=0
    )

    # Apply the Torrence & Compo amplitude factor
    amplitude_factor = (scale_spacing * np.sqrt(sampling_interval)) / (
        wavelet_at_origin * reconstruction_constant
    )
    reconstructed_signal = amplitude_factor * weighted_scale_sum

    return reconstructed_signal, reconstruction_constant


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
