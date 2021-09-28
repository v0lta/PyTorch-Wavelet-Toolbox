## Pytorch Wavelet Toolbox (`ptwt`)

<p align="center">
  <a href="https://github.com/v0lta/PyTorch-Wavelet-Toolbox/actions/workflows/tests.yml">
    <img src="https://github.com/v0lta/PyTorch-Wavelet-Toolbox/actions/workflows/tests.yml/badge.svg"
         alt="GitHub Actions">
  </a>
  <a href="https://pypi.org/project/ptwt/">
    <img src="https://img.shields.io/pypi/pyversions/ptwt"
         alt="PyPI Versions">
  </a>

  <a href="https://pypi.org/project/ptwt/">
    <img src="https://img.shields.io/pypi/v/ptwt"
         alt="PyPI - Project">
  </a>
  
  <a href="https://github.com/v0lta/PyTorch-Wavelet-Toolbox/blob/main/LICENSE">
    <img alt="PyPI - License" src="https://img.shields.io/pypi/l/ptwt">
  </a>
</p>



Welcome to the PyTorch (adaptive) wavelet toolbox. This package implements:

- the fast wavelet transform (fwt) implemented in ```wavedec```.
- the inverse fwt can be used by calling ```waverec```.
- the 2d fwt is called ```wavedec2```
- and inverse 2d fwt ```waverec2```.
- single and two-dimensional wavelet packet forward transforms.
- 1d sparse-matrix fast wavelet transforms with boundary filters.
- adaptive wavelet support (experimental).
- 2d boundary filters (experimental).

This toolbox supports pywt-wavelets. 
  
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
import ptwt  # use " from src import ptwt " if you cloned the repo instead of using pip.

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

#### Transform by Sparse-Matrix-multiplication:

In additionally sparse-matrix-based code is available. Continuing the example above try:

```python
# forward
coeff, fwt_matrix = ptwt.matrix_wavedec(data_torch, wavelet, level=2)
print(coeff)
# backward 
rec, ifwt_matrix = ptwt.matrix_waverec(coeff, wavelet, level=2)
print(rec)
```

#### Adaptive Wavelets (experimental)

Code to train an adaptive wavelet layer in PyTorch is available in the `examples` folder. In addition to static wavelets
from pywt,

- Adaptive product-filters
- and optimizable orthogonal-wavelets are supported.


#### Unit Tests

The `tests` folder contains multiple tests to allow independent verification of this toolbox. After cloning the
repository, and moving into the main directory, and installing `tox` with `pip install tox` run:

```shell
$ tox -e py
```


#### 📖 Citation
If you find this work useful please consider citing:
```
@phdthesis{handle:20.500.11811/9245,
  urn: https://nbn-resolving.org/urn:nbn:de:hbz:5-63361,
  author = {{Moritz Wolter}},
  title = {Frequency Domain Methods in Recurrent Neural Networks for Sequential Data Processing},
  school = {Rheinische Friedrich-Wilhelms-Universität Bonn},
  year = 2021,
  month = jul,
  url = {https://hdl.handle.net/20.500.11811/9245}
}
```
