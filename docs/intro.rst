Understanding wavelets
======================

This supplementary section summarizes key wavelet facts as a convenience for the hasty reader.
See, for example, \cite{strang1996wavelets,mallat1999wavelet} or \cite{jensen2001ripples} for excellent detailed introductions to the topic.

The \acf{fwt} relies on convolution operations with filter pairs.
\begin{figure}
\centering
\includestandalone[scale=0.9]{./figures/supplementary/fwt}
\caption{Overview of the \acf{fwt} computation. $\mathbf{h}_\mathcal{L}$ denotes the analysis low-pass filter.
$\mathbf{h}_\mathcal{H}$ the analysis high pass filter.  $\mathbf{f}_\mathcal{L}$ and $\mathbf{f}_\mathcal{H}$
the synthesis filer pair. $\downarrow_2$ denotes downsampling with a factor of two, $\uparrow_2$
means upsampling. In machine learning terms, the analysis transform relies on stride two convolutions.
The synthesis or inverse transform on the right works with stride two transposed convolutions.
$\mathbf{H}_{k}$ and $\mathbf{F}_{k}$ with $k \in [\mathcal{L}, \mathcal{H}]$ denote the corresponding convolution
operators.}
\label{fig:fwt}
\end{figure}
Figure~\ref{fig:fwt} illustrates the process. The forward or analysis transform
works with a low-pass $\mathbf{h}_\mathcal{L}$ and a high-pass filter $\mathbf{h}_\mathcal{H}$.
The analysis transform repeatedly convolves with both filters,
\begin{align} \label{eq:fwt}
  \mathbf{x}_s * \mathbf{h}_k = \mathbf{c}_{k, s+1}
\end{align}
with $k \in [\mathcal{L}, \mathcal{H}]$ and $s \in \mathbb{N}_0$ the set of natural numbers, where $\mathbf{x}_0$ is equal to
the original input signal $\mathbf{x}$. At higher scales, the \ac{fwt} uses the low-pass filtered result as input,
$\mathbf{x}_s = \mathbf{c}_{\mathcal{L}, s}$ if $s > 0$. 
The dashed arrow indicates that we could continue to expand the \ac{fwt} tree here.

The \acf{wpt} additionally expands the high-frequency part of the tree.
\begin{figure}
\centering
\includestandalone[scale=0.9]{./figures/supplementary/packets_1d}  
\caption{Scematic drawing of the full \acf{wpt} in a single dimension.
Compared to figure~\ref{fig:fwt}, the high-pass filtered side of the tree is expanded, too.}
\label{fig:wpt}
\end{figure}
A comparison of figure~\ref{fig:fwt} and \ref{fig:wpt} illustrates this difference.
Whole expansion is not the only possible way to construct a wavelet packet tree. See \cite{jensen2001ripples} for a discussion of other options.
In both figures, capital letters denote convolution operators. These may be expressed as Toeplitz matrices \cite{strang1996wavelets}.
The matrix nature of these operators explains the capital boldface notation.
Coefficient subscripts record the path that leads to a particular coefficient.

We construct filter quadruples from the original filter pairs to process two-dimensional inputs. The process uses outer products \cite{vyas2018multiscale}:
\begin{align}
\mathbf{h}_{a} = \mathbf{h}_\mathcal{L}\mathbf{h}_\mathcal{L}^T,
\mathbf{h}_{h} = \mathbf{h}_\mathcal{L}\mathbf{h}_\mathcal{H}^T,
\mathbf{h}_{v} = \mathbf{h}_\mathcal{H}\mathbf{h}_\mathcal{L}^T,
\mathbf{h}_{d} = \mathbf{h}_\mathcal{H}\mathbf{h}_\mathcal{H}^T
\end{align}
With $a$ for approximation, $h$ for horizontal, $v$ for vertical, and $d$ for diagonal \cite{lee2019pywavelets}.
We can construct a \ac{wpt}-tree for images with these two-dimensional filters. 
\begin{figure}
\includestandalone[scale=0.9]{./figures/supplementary/packets_2d}  
\caption{Two dimensional \acf{wpt} computation overview. $\mathbf{X}$ and $\hat{\mathbf{X}}$ denote input image and
reconstruction respectively.}
\label{fig:wpt2d}
\end{figure}
Figure~\ref{fig:wpt2d} illustrates the computation of a full two-dimensional wavelet packet tree.
More formally, the process initially evaluates
\begin{align}
\mathbf{x}_0 *_2 \mathbf{h}_j = \mathbf{c}_{j, 1}
\end{align}
with $\mathbf{x}_0$ equal to an input image $\mathbf{X}$, $j \in [a,h,v,d]$, and $*_2$ for two-dimensional convolution. At higher scales, all resulting coefficients from previous scales serve as inputs. The four filters repeatedly convolved with all outputs to build the full tree. The inverse transforms work analogously. We refer to the standard literature \cite{jensen2001ripples,strang1996wavelets} for an extended discussion.

Compared to the \ac{fwt}, the high-frequency half of the tree is subdivided into more bins, yielding a fine-grained view of the entire spectrum.
We always show analysis and synthesis transforms to stress that all wavelet transforms are lossless. Synthesis transforms reconstruct the original input based on the results from the analysis transform.

\subsection{Common wavelets and their properties}\label{sec:db_and_sym}
A key property of the wavelet transform is its invertibility. Additionally, we expect an alias-free representation.
Standard literature like \cite{strang1996wavelets} formulates the perfect reconstruction
and alias cancellation conditions to satisfy both requirements. For an analysis filter coefficient vector $\mathbf{h}$ the equations below use the polynomial $H(z) = \sum_n h(n)z^{-n}$. We construct $F(z)$ the same way using the synthesis filter coefficients in $\mathbf{f}$. To guarantee perfect reconstruction the filters must respect 
\begin{align}
H_\mathcal{L}(z)F_\mathcal{L}(z) + H_\mathcal{H}(-z)F_\mathcal{H}(z) = 2z^{-l}.
\end{align}
Similarly
\begin{align} \label{eq:alias_cancellation}
  F_\mathcal{L}(z)H_\mathcal{L}(-z) + F_\mathcal{H}(z)H_\mathcal{H}(-z) = 0 
\end{align}
guarantees alias cancellation.

Filters that satisfy both equations qualify as wavelets. Daubechies wavelets and Symlets appear in this paper.
\begin{figure}
  \includestandalone{./figures/supplementary/sym6}
  \caption{Visualization of the Symlet 6 filter coefficients.}
  \label{fig:sym6_vis}
\end{figure} 
\begin{figure}
  \includestandalone{./figures/supplementary/db6}
  \caption{Visualization of the Daubechies 6 filter coefficients.}
\label{fig:db6_vis}
\end{figure}
Figures~\ref{fig:sym6_vis} and \ref{fig:db6_vis} visualize the Daubechies and Symlet filters of 6th degree.
Compared to the Daubechies Wavelet family, their Symlet cousins have more mass at the center. Figure~\ref{fig:sym6_vis} illustrates this fact. Large deviations occur around the fifth filter in the center, unlike the Daubechies' six filters in Figure~\ref{fig:db6_vis}.
Consider the sign patterns in Figure~\ref{fig:db6_vis}. The decomposition highpass (orange) and the reconstruction lowpass (green) filters display an alternating sign pattern. This behavior is a possible solution to the alias cancellation condition. To understand why substitute $F_\mathcal{L}(z) = H_\mathcal{H}(-z)$ and $F_\mathcal{H} = -H_\mathcal{L}(-z)$ into equation~\ref{eq:alias_cancellation}\cite{strang1996wavelets}. $F_\mathcal{L}(z) = H_\mathcal{H}(-z)$ requires an opposing sign at even and equal signs at odd powers of the polynomial.