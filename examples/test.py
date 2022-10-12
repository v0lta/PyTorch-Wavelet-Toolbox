import torch
from torch import nn
import numpy as np
import src.ptwt as ptwt
from src.ptwt.continuous_transform import _ComplexMorletWavelet


class ScalogramLoss(nn.Module):
    """Complex Continuous Wavelet Transform Loss"""
    def __init__(self, wavelet, octaves=9, octave_divs=24, alpha=0.5, eps=1e-8):
        super().__init__()
        self.scales = 2**(torch.arange(octave_divs*octaves)/octave_divs + 1)
        self.wavelet = wavelet
        self.alpha = alpha
        self.eps = eps

    def forward(self, x, x_hat):
        S, _ = ptwt.continuous_transform.cwt(x, self.scales, self.wavelet)
        S_hat, _ = ptwt.continuous_transform.cwt(x_hat, self.scales, self.wavelet)
        S, S_hat = S.abs(), S_hat.abs() #take the magnitude of the complex wavelet transform

        linear_term = nn.functional.l1_loss(S, S_hat)
        log_term = nn.functional.l1_loss((S + self.eps).log2(), (S_hat + self.eps).log2())

        return self.alpha * linear_term + (1 - self.alpha) * log_term



def main():
    duration = 2 # seconds
    fs = 44100
    sig = np.sin(np.arange(int(fs*duration))*2*np.pi*440/fs)
    sig = torch.Tensor(sig).cuda()
    
    #reconstruct signal starting from random noise
    sig_hat = torch.randn_like(sig, requires_grad=True).cuda()

    wavelet = _ComplexMorletWavelet(name='cmor0.5-0.5').cuda()

    optim = torch.optim.Adam([sig_hat], lr=1e-3)
    loss_fn = ScalogramLoss(wavelet=wavelet).cuda()
    
    iterations = 500
    for it in range(iterations):
        loss = loss_fn(sig, sig_hat)
        loss.backward()
        optim.step()
        print(f'{it}: {loss.item()}')
       

if __name__ == "__main__":
    main()
