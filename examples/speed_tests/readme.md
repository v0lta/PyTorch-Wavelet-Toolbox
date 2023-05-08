### Ptwt - Speed - Tests

To run the speed tests install [pywt](https://pywavelets.readthedocs.io/en/latest/install.html)
and [pytorch-wavelets](https://github.com/fbcotter/pytorch_wavelets).
The numbers below were measured using an NVIDIA RTX A4000 graphics card and an Intel(R) Xeon(R) W-2235 CPU @ 3.80GHz.

To execute the speed tests for the single-dimensional case run:
```bash
python timeitconv_1d.py
```
it produces the output and plot below:

```bash
1d-pywt-cpu:0.25634 +- 0.00826
1d-ptwt-cpu:0.41659 +- 0.02594
1d-ptwt-cpu-jit:0.42266 +- 0.02404
1d-ptwt-gpu:0.01812 +- 0.17063
1d-ptwt-jit:0.00090 +- 0.00242
```

![1d-speed](figs/dim1.png)

For the two-2d fast wavelet decomposition case run:
```bash
python timeitconv_2d.py
```
Result:
```bash
2d-pywt-cpu:0.52735 +- 0.02435
2d-pytorch_wavelets-cpu:0.31621 +- 0.02328
2d-pytorch_wavelets-gpu:0.00558 +- 0.04472
2d-ptwt-cpu:0.21758 +- 0.01946
2d-ptwt-gpu:0.00130 +- 0.00409
2d-ptwt-jit:0.00870 +- 0.08016
```


![2d-speed](figs/dim2.png)

Finally use
```bash
python timeitconv_3d.py
```
for the three dimensional case. It should produce something like the output below:

```bash
3d-pywt-cpu:0.83785 +- 0.03117
3d-ptwt-cpu:0.35998 +- 0.03619
3d-ptwt-gpu:0.00499 +- 0.04062
3d-ptwt-jit:0.01852 +- 0.17757
```

![3d-speed](figs/dim3.png)