from typing import Tuple

import torch
import pywt


def cwt(
    data: torch.tensor, scales: torch.tensor, wavelet: pywt.ContinuousWavelet
) -> Tuple[torch.tensor, torch.tensor]:
    pass
