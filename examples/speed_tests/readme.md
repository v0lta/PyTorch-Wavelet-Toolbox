### Ptwt - Speed - Tests

To run the speed tests install [pywt](https://pywavelets.readthedocs.io/en/latest/install.html)
and [pytorch-wavelets](https://github.com/fbcotter/pytorch_wavelets).

To execute the speed tests for the single-dimensional case run:
```bash
python timeitconv_1d.py
```
it produces the plot below:

![1d-speed](figs/dim1.png)

For the two-2d fast wavelet decomposition case run:
```bash
python timeitconv_2d.py
```

![2d-speed](figs/dim2.png)

Finally use
```bash
python timeitconv_3d.py
```
for the three dimensional case. It should produce something like the output below:

![3d-speed](figs/dim3.png)