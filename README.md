## Pytorch Wavelet Toolbox (`ptwt`)

<p align="center">
  <a href="https://github.com/v0lta/PyTorch-Wavelet-Toolbox/actions/workflows/tests.yml">
    <img src="https://github.com/v0lta/PyTorch-Wavelet-Toolbox/actions/workflows/tests.yml/badge.svg"
         alt="GitHub Actions">

<a href="https://pypi.org/project/ptwt/">
    <img src="https://img.shields.io/pypi/v/ptwt"
         alt="PyPI - Project">
  </a>
  
</p>


Welcome to the PyTorch (adaptive) wavelet toolbox. This package implements:

- the fast wavelet transform (fwt) (wavedec)
- the inverse fwt (waverec)
- the 2d fwt wavedec2
- the inverse 2d fwt waverec2.
- single and two-dimensional wavelet packet forward transforms.
- adaptive wavelet support (experimental).
- sparse matrix fast wavelet transforms (experimental).

#### Installation

Install the toolbox via pip or clone this repository. In order to use `pip`, type:

``` shell
$ pip install ptwt
```

You can remove it later by typing ```pip uninstall ptwt```.

#### Example usage:

```python
import torch
import numpy as np
import pywt
import ptwt  # from src import ptwt instead if you cloned the repo instead of using pip.

# generate an input of even length.
data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
data_torch = torch.from_numpy(data.astype(np.float32))
wavelet = pywt.Wavelet('haar')

# compare the forward fwt coefficients
print(pywt.wavedec(data, wavelet, mode='zero', level=2))
print(ptwt.wavedec(data_torch, wavelet, mode='zero', level=2))

# invert the fwt.
print(ptwt.waverec(ptwt.wavedec(data_torch, wavelet, mode='zero', level=2), wavelet))
```

#### Unit Tests

The `tests` folder contains multiple tests to allow independent verification of this toolbox. After cloning the
repository, and moving into the main directory, and installing `tox` with `pip install tox` run:

```shell
$ tox -e py
```

#### Adaptive Wavelets (experimental)

Code to train an adaptive wavelet layer in PyTorch is available in the `examples` folder. In addition to static wavelets
from pywt,

- Adaptive product-filters
- and optimizable orthogonal-wavelets are supported.

#### Sparse-Matrix-multiplication Transform (experimental).

In addition to convolution-based fwt implementations matrix-based code is available. Continuing the example above try:

```python
# forward
coeff, fwt_matrix = ptwt.matrix_wavedec(data_torch, wavelet, level=2)
print(coeff)
# backward 
rec, ifwt_matrix = ptwt.matrix_waverec(coeff, wavelet, level=2)
print(rec)
```
