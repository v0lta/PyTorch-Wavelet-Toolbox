"""Check volumetric packets against PyWavelets, including edited leaves."""

import numpy as np
import pytest
import pywt
import torch

from ptwt import WaveletPacket3D


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_partial_tree(device: str) -> None:
    """A mixed-depth tree can be edited and reconstructed on non-trailing axes."""
    rng = np.random.default_rng(123)
    data = rng.standard_normal((9, 2, 12, 11))
    axes = (2, 0, 3)
    x = torch.tensor(data, device=device, requires_grad=True)
    wp = WaveletPacket3D(x, "db2", axes=axes, maxlevel=2)
    ref = pywt.WaveletPacketND(data, "db2", mode="reflect", axes=axes, maxlevel=2)
    key = "adaaad"
    np.testing.assert_allclose(wp[key].detach().cpu(), ref[key].data, atol=1e-12)
    assert len(wp) == 17
    torch.testing.assert_close(wp.reconstruct()[""], x)
    wp[key] = wp[key] * 0
    ref[key].data *= 0
    result = wp.reconstruct()[""]
    np.testing.assert_allclose(
        result.detach().cpu(), ref.reconstruct(False), atol=1e-12
    )
    (grad,) = torch.autograd.grad(result.square().sum(), x)
    assert grad.isfinite().all()


def test_transform_resets_tree() -> None:
    """Maximum depth and cached branches belong to the current input."""
    x = torch.ones(8, 8, 8)
    wp = WaveletPacket3D(None, "haar")
    with pytest.raises(ValueError):
        wp["aaa"]
    wp.transform(x)
    assert wp["aaaaaaaaa"].numel() == 1
    wp.transform(x * 2, maxlevel=1)
    assert list(wp) == [""]
    with pytest.raises(KeyError):
        wp["aaaaaa"]
    with pytest.raises(ValueError):
        wp["ab"]
