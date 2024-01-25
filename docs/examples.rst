Wavelet transforms by example
=============================


The continuous wavelet transform
-------------------------------------------------
The provided function :py:meth:`ptwt.continuous_transform.cwt` implements the Continuous Wavelet Transform (CWT) for analyzing signals for usage in PyTorch. It computes the CWT matrix for a given input signal using specified wavelet scales and wavelet functions.

Inputs:

* ``data``: Input signal tensor.

* ``scales``: Array of desired wavelet scales.

* ``wavelet``: Continuous wavelet function or string representing a wavelet.

* ``sampling_period``: Sampling period for frequency output (optional). This must be given to obtain the correct corresponding frequency array (see outputs).

Outputs:

* ``cwtmatr``: A tensor containing the transformation matrix.

* ``freqs``: An array with information about the frequencies corresponding to the scales in ``cwtmatr``.

Example Usage:

.. code-block::

    import torch
    import ptwt
    import numpy as np
    import scipy.signal as signal

    t = np.linspace(-2, 2, 800, endpoint=False)
    sig = signal.chirp(t, f0=1, f1=12, t1=2, method="linear")
    widths = np.arange(1, 31)
    cwtmatr, freqs = ptwt.cwt(
        torch.from_numpy(sig), widths, "mexh", sampling_period=(4 / 800) * np.pi
    )

This code snippet demonstrates how to compute the CWT of a chirp signal using the Mexican-Hat wavelet ("mexh"), specifying the scales and sampling period. The resulting ``cwtmatr`` contains the transformed coefficients, and ``freqs`` provides information about the frequencies corresponding to the scales in ``cwtmatr``.
