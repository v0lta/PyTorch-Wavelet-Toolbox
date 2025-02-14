.. _common-wavelets:

Common wavelets and their properties
------------------------------------

A key property of the wavelet transform is its invertibility. Additionally, we expect an alias-free representation.
Standard literature like :cite:`strang1996wavelets` formulates the perfect reconstruction
and alias cancellation conditions to satisfy both requirements.
For an analysis filter coefficient vector :math:`\mathbf{h}`
the equations below use the polynomial :math:`H(z) = \sum_n h(n)z^{-n}`.
We construct :math:`F(z)` the same way using the synthesis filter coefficients in :math:`\mathbf{f}`.
To guarantee perfect reconstruction the filters must respect

.. math::
    H_\mathcal{A}(z)F_\mathcal{A}(z) + H_\mathcal{D}(-z)F_\mathcal{D}(z) = 2z^{-l}.

Similarly

.. _eq-alias:

.. math::
  F_\mathcal{A}(z)H_\mathcal{A}(-z) + F_\mathcal{D}(z)H_\mathcal{D}(-z) = 0

guarantees alias cancellation.

Filters that satisfy both equations qualify as wavelets. Lets consider i.e. a Daubechies wavelet and a Symlet:

.. _fig-sym6:

.. figure:: figures/sym6.png
   :scale: 45 %
   :alt: sym6 filter values
   :align: center

   Visualization of the Symlet 6 filter coefficients.


.. _fig-db6:

.. figure:: figures/db6.png
   :scale: 45 %
   :alt: 2d wavelet packet transform computation diagram.
   :align: center

   Visualization of the Daubechies 6 filter coefficients.

:numref:`fig-sym6` and :numref:`fig-db6` visualize the Daubechies and Symlet filters of 6th degree.
Compared to the Daubechies Wavelet family, their Symlet cousins have more mass at the center.
:numref:`fig-sym6` illustrates this fact. Large deviations occur around the fifth filter in the center,
unlike the Daubechies' six filters in :numref:`fig-db6`.
Consider the sign patterns in :numref:`fig-db6`.
The decomposition highpass (orange) and the reconstruction lowpass (green) filters display an alternating sign pattern.
This behavior is a possible solution to the alias cancellation condition.
To understand why substitute :math:`F_\mathcal{A}(z) = H_\mathcal{D}(-z)` and :math:`F_\mathcal{D} = -H_\mathcal{A}(-z)`
into the perfect reconstruction condition :cite:`strang1996wavelets`.
:math:`F_\mathcal{A}(z) = H_\mathcal{D}(-z)` requires an opposing sign
at even and equal signs at odd powers of the polynomial.
