********************************
Pytorch Wavelet Toolbox (`ptwt`)
********************************

.. image:: https://github.com/v0lta/PyTorch-Wavelet-Toolbox/actions/workflows/tests.yml/badge.svg 
    :target: https://github.com/v0lta/PyTorch-Wavelet-Toolbox/actions/workflows/tests.yml
    :alt: GitHub Actions

.. image:: https://readthedocs.org/projects/pytorch-wavelet-toolbox/badge/?version=latest
    :target: https://pytorch-wavelet-toolbox.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/pyversions/ptwt
    :target: https://pypi.org/project/ptwt/
    :alt: PyPI Versions

.. image:: https://img.shields.io/pypi/v/ptwt
    :target: https://pypi.org/project/ptwt/
    :alt: PyPI - Project

.. image:: https://img.shields.io/pypi/l/ptwt
    :target: https://github.com/v0lta/PyTorch-Wavelet-Toolbox/blob/main/LICENSE
    :alt: PyPI - License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black code style




Welcome to the PyTorch wavelet toolbox. This package implements:

- the fast wavelet transform (fwt) implemented in ``wavedec``.
- the inverse fwt can be used by calling ``waverec``.
- the 2d fwt is called ``wavedec2``
- and inverse 2d fwt ``waverec2``.
- 1d sparse-matrix fast wavelet transforms with boundary filters.
- 2d sparse-matrix transforms with boundary filters (experimental).
- single and two-dimensional wavelet packet forward transforms.
- adaptive wavelet support (experimental).

This toolbox supports pywt-wavelets. Complete documentation is available:
https://pytorch-wavelet-toolbox.readthedocs.io/en/latest/ptwt.html


**Installation**

Install the toolbox via pip or clone this repository. In order to use ``pip``, type:

.. code-block:: sh

    $ pip install ptwt
  

You can remove it later by typing ``pip uninstall ptwt``.

Example usage:
""""""""""""""
**Single dimensional transform**

One way to compute fast wavelet transforms is to rely on padding and
convolution. Consider the following example: 

.. code-block:: python

  import torch
  import numpy as np
  import pywt
  import ptwt  # use " from src import ptwt " if you cloned the repo instead of using pip.
  
  # generate an input of even length.
  data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0])
  data_torch = torch.from_numpy(data.astype(np.float32))
  wavelet = pywt.Wavelet('haar')
  
  # compare the forward fwt coefficients
  print(pywt.wavedec(data, wavelet, mode='zero', level=2))
  print(ptwt.wavedec(data_torch, wavelet, mode='zero', level=2))
  
  # invert the fwt.
  print(ptwt.waverec(ptwt.wavedec(data_torch, wavelet, mode='zero', level=2), wavelet))


The functions ``wavedec`` and ``waverec`` compute the 1d-fwt and its inverse.
Internally both rely on ``conv1d``, and its transposed counterpart ``conv_transpose1d``
from the ``torch.nn.functional`` module. This toolbox supports discrete wavelets
see also ``pywt.wavelist(kind='discrete')``. 
Daubechies-Wavelets ``db-x`` or symlets ``sym-x`` are usually a good starting point. 

**Two-dimensional transform**

Analog to the 1d-case ``wavedec2`` and ``waverec2`` rely on 
``conv2d``, and its transposed counterpart ``conv_transpose2d``.
To test an example run:


.. code-block:: python

  import ptwt, pywt, torch
  import numpy as np
  import scipy.misc

  face = np.transpose(scipy.misc.face(),
                          [2, 0, 1]).astype(np.float64)
  pytorch_face = torch.tensor(face).unsqueeze(1)
  coefficients = ptwt.wavedec2(pytorch_face, pywt.Wavelet("haar"),
                                  level=2, mode="constant")
  reconstruction = ptwt.waverec2(coefficients, pywt.Wavelet("haar"))
  np.max(np.abs(face - reconstruction.squeeze(1).numpy()))



**Boundary Wavelets with Sparse-Matrices**

In addition to convolution and padding approaches,
sparse-matrix-based code with boundary wavelet support is available.
In contrast to padding, boundary wavelets do not add extra pixels at 
the edges.
Internally, boundary wavelet support relies on ``torch.sparse.mm``.
Generate 1d sparse matrix forward and backward transforms with the
`MatrixWavedec` and `MatrixWaverec` classes.
Reconsidering the 1d case, try:

.. code-block:: python

  import torch
  import numpy as np
  import pywt
  import ptwt  # use " from src import ptwt " if you cloned the repo instead of using pip.
  
  # generate an input of even length.
  data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0])
  data_torch = torch.from_numpy(data.astype(np.float32))
  # forward
  matrix_wavedec = ptwt.MatrixWavedec(pywt.Wavelet("haar"), level=2)
  coeff = matrix_wavedec(data_torch)
  print(coeff)
  # backward 
  matrix_waverec = ptwt.MatrixWaverec(pywt.Wavelet("haar"), level=2)
  rec = matrix_waverec(coeff)
  print(rec)


The process for the 2d transforms `MatrixWavedec2d`, `MatrixWaverec2d`,
works similarly.


**Adaptive** **Wavelets**

Experimental code to train an adaptive wavelet layer in PyTorch is available in the `examples` folder. In addition to static wavelets
from pywt,

- Adaptive product-filters
- and optimizable orthogonal-wavelets are supported.

See https://github.com/v0lta/PyTorch-Wavelet-Toolbox/tree/main/examples for a complete implementation.


**Testing**

The `tests` folder contains multiple tests to allow independent verification of this toolbox. After cloning the
repository, and moving into the main directory, and installing `tox` with `pip install tox` run:

.. code-block:: sh

  $ tox -e py



📖 Citation
"""""""""""

If you find this work useful, please consider citing:

.. code-block::

  @phdthesis{handle:20.500.11811/9245,
    urn: https://nbn-resolving.org/urn:nbn:de:hbz:5-63361,
    author = {{Moritz Wolter}},
    title = {Frequency Domain Methods in Recurrent Neural Networks for Sequential Data Processing},
    school = {Rheinische Friedrich-Wilhelms-Universität Bonn},
    year = 2021,
    month = jul,
    url = {https://hdl.handle.net/20.500.11811/9245}
  }

