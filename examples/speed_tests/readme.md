### Ptwt - Speed - Tests

To run the speed tests install [pywt](https://pywavelets.readthedocs.io/en/latest/install.html)
and [pytorch-wavelets](https://github.com/fbcotter/pytorch_wavelets).

To execute the speed tests for the single-dimensional case run:
```bash
python timeitconv_1d.py
```
it produces the plot below:

![1d-speed](figs/dim1.png)

For the two-2d case run:
```bash
python timeitconv_2d.py
```