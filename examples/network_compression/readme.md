#### Adaptive Wavelets

```mnist_compression.py``` trains a CNN on MNIST with an adaptive-wavelet
compressed linear layer. The wavelets in the linear layer are learned using gradient descent.

See https://arxiv.org/pdf/2004.09569v3.pdf for a detailed description of the method.

Running this example requires the following steps, which takes care of installing everything

```console
$ git clone https://github.com/v0lta/PyTorch-Wavelet-Toolbox.git
$ cd PyTorch-Wavelet-Toolbox/examples/network_compression
$ uv run mnist_compression.py
```
