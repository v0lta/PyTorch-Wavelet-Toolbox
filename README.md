## Pytorch Wavelet Toolbox (ptwt)
Welcome to the PyTorch adaptive wavelet toolbox.
This package implements:

- the fast wavelet transform (fwt) (wavedec)
- the inverse fwt (waverec)
- the 2d fwt wavedec2
- the inverse 2d fwt waverec2.
- adaptive wavelet support (experimental).
- sparse matrix fast wavelet transforms (experimental).

#### Example usage:
``` python
    import torch
    import numpy as np
    import pywt
    import ptwt

    # generate an input of even length.
    data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
    data_torch = torch.from_numpy(data.astype(np.float32))
    wavelet = pywt.Wavelet('haar')

    # compare the forward fwt coefficients
    print(pywt.wavedec(data, wavelet, mode='zero', level=2))
    print(ptwt.wavedec(data_torch, wavelet, mode='zero', level=2))

    # invert the fwt.
    print(ptwt.waverec(ptwt.wavedec(data_torch, wavelet, mode='zero',
        level=2), wavelet))
```

#### Unit tests
The `tests` folder contains multiple tests, to allow independent
verification of this toolbox. Run 

``` python
    pytest
```
to evaluate all of them.


#### Adaptive wavelets (experimental)
Code to train an adaptive wavelet layer in PyTorch is available in
the `examples` folder. In addition to static wavelets from pywt,

- Adaptive product-filters

- and optimizable orthogonal-wavelets are supported.

#### Sparse-Matrix-multiplication transform (experimental).
In addition to convolution-based fwt implementations 
matrix-based code is available. Continuing the example above
try:
``` python
    # forward
    coeff, fwt_matrix = ptwt.matrix_wavedec(data_torch, wavelet, level=2)
    print(coeff)
    # backward 
    rec, ifwt_matrix = ptwt.matrix_waverec(coeff, wavelet, level=2)
    print(rec)
```

#### Known issues
PyTorch and pywt pad slightly differently. All is well with zero padding 
on even length sequences. Not all pywt padding modes exist in PyTorch.
For odd sequence lengths the edge coefficients may differ.
These differences *do not* affect invertibility!
