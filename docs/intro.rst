Understanding wavelets
======================

This supplementary section summarizes key wavelet facts as a convenience for the hasty reader.
See, for example, :cite:`strang1996wavelets,mallat1999wavelet` or :cite:`jensen2001ripples` for excellent detailed introductions to the topic.

The fwt relies on convolution operations with filter pairs.


.. _fig-fwt:

.. figure:: figures/fwt.png
   :scale: 50 %
   :alt: fast wavelet transform computation diagram.
   :align: center

   Overview of the fwt computation.
   

:numref:`fig-fwt` illustrates the process. :math:`\mathbf{h}_\mathcal{L}` denotes the analysis low-pass filter. :math:`\mathbf{h}_\mathcal{H}` the analysis high pass filter.  :math:`\mathbf{f}_\mathcal{L}` and :math:`\mathbf{f}_\mathcal{H}` the synthesis filer pair. :math:`\downarrow_2` denotes downsampling with a factor of two, :math:`\uparrow_2` means upsampling. In machine learning terms, the analysis transform relies on stride two convolutions. The synthesis or inverse transform on the right works with stride two transposed convolutions. :math:`\mathbf{H}_{k}` and :math:`\mathbf{F}_{k}` with :math:`k \in [\mathcal{L}, \mathcal{H}]` denote the corresponding convolution operators.


.. math::
  \mathbf{x}_s * \mathbf{h}_k = \mathbf{c}_{k, s+1}

with :math:`k \in [\mathcal{L}, \mathcal{H}]` and :math:`s \in \mathbb{N}_0` the set of natural numbers, where :math:`\mathbf{x}_0` is equal to
the original input signal :math:`\mathbf{x}`. At higher scales, the fwt uses the low-pass filtered result as input,
:math:`\mathbf{x}_s = \mathbf{c}_{\mathcal{L}, s}` if :math:`s > 0`. 
The dashed arrow indicates that we could continue to expand the fwt tree here.

The wpt additionally expands the high-frequency part of the tree.

.. _fig-wpt:

.. figure:: figures/packets_1d.png
   :scale: 50 %
   :alt: wavelet packet transform computation diagram.
   :align: center

   Scematic drawing of the full wpt in a single dimension. Compared to figure~\ref{fig:fwt}, the high-pass filtered side of the tree is expanded, too.

A comparison of :numref:`fig-fwt` and :numref:`fig-wpt` illustrates this difference.
Whole expansion is not the only possible way to construct a wavelet packet tree. See :cite:`jensen2001ripples` for a discussion of other options.
In both figures, capital letters denote convolution operators. These may be expressed as Toeplitz matrices :cite:`strang1996wavelets`.
The matrix nature of these operators explains the capital boldface notation.
Coefficient subscripts record the path that leads to a particular coefficient.

We construct filter quadruples from the original filter pairs to process two-dimensional inputs. The process uses outer products :cite:`vyas2018multiscale`:

.. math::
    \mathbf{h}_{a} = \mathbf{h}_\mathcal{L}\mathbf{h}_\mathcal{L}^T,
    \mathbf{h}_{h} = \mathbf{h}_\mathcal{L}\mathbf{h}_\mathcal{H}^T,
    \mathbf{h}_{v} = \mathbf{h}_\mathcal{H}\mathbf{h}_\mathcal{L}^T,
    \mathbf{h}_{d} = \mathbf{h}_\mathcal{H}\mathbf{h}_\mathcal{H}^T

With :math:`a` for approximation, :math:`h` for horizontal, :math:`v` for vertical, and :math:`d` for diagonal :cite:`lee2019pywavelets`.
We can construct a wpt-tree for images with these two-dimensional filters.


.. image:: figures/packets_2d.png
   :scale: 45 %
   :alt: 2d wavelet packet transform computation diagram.
   :align: center

Two dimensional \acf{wpt} computation overview. :math:`\mathbf{X}` and :math:`\hat{\mathbf{X}}` denote input image and
reconstruction respectively.
Figure~\ref{fig:wpt2d} illustrates the computation of a full two-dimensional wavelet packet tree.
More formally, the process initially evaluates

.. math::
    \mathbf{x}_0 *_2 \mathbf{h}_j = \mathbf{c}_{j, 1}

with :math:`\mathbf{x}_0` equal to an input image :math:`\mathbf{X}`, :math:`j \in [a,h,v,d]`, and :math:`*_2` for two-dimensional convolution. At higher scales, all resulting coefficients from previous scales serve as inputs. The four filters repeatedly convolved with all outputs to build the full tree. The inverse transforms work analogously. We refer to the standard literature :cite:`jensen2001ripples,strang1996wavelets` for an extended discussion.

Compared to the \ac{fwt}, the high-frequency half of the tree is subdivided into more bins, yielding a fine-grained view of the entire spectrum.
We always show analysis and synthesis transforms to stress that all wavelet transforms are lossless. Synthesis transforms reconstruct the original input based on the results from the analysis transform.

Common wavelets and their properties
------------------------------------

A key property of the wavelet transform is its invertibility. Additionally, we expect an alias-free representation.
Standard literature like :cite:`strang1996wavelets` formulates the perfect reconstruction
and alias cancellation conditions to satisfy both requirements. For an analysis filter coefficient vector :math:`\mathbf{h}` the equations below use the polynomial :math:`H(z) = \sum_n h(n)z^{-n}`. We construct :math:`F(z)` the same way using the synthesis filter coefficients in :math:`\mathbf{f}`. To guarantee perfect reconstruction the filters must respect 

.. math::
    H_\mathcal{L}(z)F_\mathcal{L}(z) + H_\mathcal{H}(-z)F_\mathcal{H}(z) = 2z^{-l}.

Similarly

.. math::
  F_\mathcal{L}(z)H_\mathcal{L}(-z) + F_\mathcal{H}(z)H_\mathcal{H}(-z) = 0 

guarantees alias cancellation.

Filters that satisfy both equations qualify as wavelets. Daubechies wavelets and Symlets appear in this paper.

.. figure:: figures/sym6.png
   :scale: 45 %
   :alt: sym6 filter values
   :align: center

Visualization of the Symlet 6 filter coefficients.


.. figure:: figures/db6.png
   :scale: 45 %
   :alt: 2d wavelet packet transform computation diagram.
   :align: center

Visualization of the Daubechies 6 filter coefficients.

Figures~\ref{fig:sym6_vis} and \ref{fig:db6_vis} visualize the Daubechies and Symlet filters of 6th degree.
Compared to the Daubechies Wavelet family, their Symlet cousins have more mass at the center. Figure~\ref{fig:sym6_vis} illustrates this fact. Large deviations occur around the fifth filter in the center, unlike the Daubechies' six filters in Figure~\ref{fig:db6_vis}.
Consider the sign patterns in Figure~\ref{fig:db6_vis}. The decomposition highpass (orange) and the reconstruction lowpass (green) filters display an alternating sign pattern. This behavior is a possible solution to the alias cancellation condition. To understand why substitute :math:`F_\mathcal{L}(z) = H_\mathcal{H}(-z)` and :math:`F_\mathcal{H} = -H_\mathcal{L}(-z)` into equation~\ref{eq:alias_cancellation} :cite:`strang1996wavelets`. :math:`F_\mathcal{L}(z) = H_\mathcal{H}(-z)` requires an opposing sign at even and equal signs at odd powers of the polynomial.



.. bibliography::