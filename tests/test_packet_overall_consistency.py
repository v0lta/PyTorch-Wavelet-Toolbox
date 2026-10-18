"""Test case based on NiclasPi's: https://github.com/v0lta/PyTorch-Wavelet-Toolbox/issues/137"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pywt
import torch

import ptwt
from ptwt.constants import ExtendedBoundaryMode

H, W = 8, 8
WAVELET = "haar"

PRECISION_ESP = np.finfo(np.float32).resolution

# natural order uses flat int positions.
# [(0, 0), (0, 1), (1, 0), (1, 1)]
# means [a, v, h, d]
# see also https://github.com/PyWavelets/pywt/blob/c9b542b2d2c37d28ed1d9bd49d178d2769608905/pywt/_multidim.py#L153
#
# {'aa': <coeffs>  # A(LL) - approx. on 1st dim, approx. on 2nd dim
#  'ad': <coeffs>  # V(LH) - approx. on 1st dim, det. on 2nd dim
#  'da': <coeffs>  # H(HL) - det. on 1st dim, approx. on 2nd dim
#  'dd': <coeffs>  # D(HH) - det. on 1st dim, det. on 2nd dim
# }


GridPos = int | tuple[int, int]
Grid = dict[GridPos, tuple[str, float]]


def make_varies_along_w() -> np.ndarray:
    """Nyquist-frequency pattern across columns, constant down each column."""
    row = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    return np.tile(row, (H, 1))


def make_varies_along_h() -> np.ndarray:
    """Nyquist-frequency pattern down rows, constant across each row."""
    col = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    return np.tile(col.reshape(-1, 1), (1, W))


def build_pywt_wp(signal_np: np.ndarray) -> Any:
    """Build a pywavelet tree."""
    return pywt.WaveletPacket2D(
        data=signal_np, wavelet=WAVELET, mode="reflect", maxlevel=1
    )


def build_ptwt_wp(
    signal_np: np.ndarray, mode: ExtendedBoundaryMode, separable: bool
) -> ptwt.WaveletPacket2D:
    """Build a pyTorch-Wavelet package tree."""
    signal = torch.from_numpy(signal_np).to(torch.float64)
    return ptwt.WaveletPacket2D(
        data=signal.unsqueeze(0),
        wavelet=WAVELET,
        mode=mode,
        maxlevel=1,
        orthogonalization="qr",
        separable=separable,
    )


def pywt_grid(wp: Any, order: str) -> Grid:
    """Official pywt.WaveletPacket2D, level 1, order='natural' (flat) or 'freq' (rows x cols)."""
    nodes = wp.get_level(1, order=order)
    if order == "freq":
        return {
            (r, c): (node.path, float(np.abs(node.data).sum()))
            for r, row in enumerate(nodes)
            for c, node in enumerate(row)
        }
    return {
        i: (node.path, float(np.abs(node.data).sum())) for i, node in enumerate(nodes)
    }


def ptwt_grid(wp: ptwt.WaveletPacket2D, order: str) -> Grid:
    """ptwt.WaveletPacket2D, level 1, order='natural' (flat) or 'freq' (rows x cols)."""
    if order == "freq":
        return {
            (r, c): (node, wp[node].abs().sum().item())
            for r, row in enumerate(wp.get_level(1, order="freq"))
            for c, node in enumerate(row)
        }
    return {
        i: (node, wp[node].abs().sum().item())
        for i, node in enumerate(wp.get_level(1, order="natural"))
    }


def _assert_grid_table(
    title: str, pywt_g: Grid, ptwt_g: Grid, positions: Sequence[GridPos]
) -> None:
    """Assert that the packages match."""
    print(f"  {title}")
    print(f"    {'pos':<8}{'pywt':<16}{'ptwt':<16}")
    for pos in positions:
        pywt_label, pywt_val = pywt_g[pos]
        ptwt_label, ptwt_val = ptwt_g[pos]
        print(
            f"    {pos!s:<8}"
            f"{pywt_label:<4}{pywt_val:>9.3f}   "
            f"{ptwt_label:<4}{ptwt_val:>9.3f}   "
        )
        assert np.allclose(pywt_val, ptwt_val, rtol=PRECISION_ESP)


def compare(
    signal_np: np.ndarray,
    label: str,
    mode: ExtendedBoundaryMode,
    separable: bool,
) -> None:
    pywt_wp = build_pywt_wp(signal_np)
    ptwt_wp = build_ptwt_wp(signal_np, mode=mode, separable=separable)

    print(f"\n{label}  [ptwt mode={mode!r}, separable={separable}]")
    print(
        "  pywt natural order:",
        [node.path for node in pywt_wp.get_level(1, order="natural")],
    )
    print("  ptwt natural order:", ptwt_wp.get_level(1, order="natural"))
    print(
        "  pywt freq order:",
        [[n.path for n in node] for node in pywt_wp.get_level(1, order="freq")],
    )
    print("  ptwt freq order:", ptwt_wp.get_level(1, order="freq"))

    natural_positions: list[GridPos] = [0, 1, 2, 3]
    _assert_grid_table(
        "natural order comparison:",
        pywt_grid(pywt_wp, order="natural"),
        ptwt_grid(ptwt_wp, order="natural"),
        natural_positions,
    )

    freq_positions: list[GridPos] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    _assert_grid_table(
        "freq order comparison:",
        pywt_grid(pywt_wp, order="freq"),
        ptwt_grid(ptwt_wp, order="freq"),
        freq_positions,
    )


def test_consistency() -> None:
    varies_along_w = make_varies_along_w()
    varies_along_h = make_varies_along_h()

    modes: list[ExtendedBoundaryMode] = ["reflect", "boundary"]
    for mode in modes:
        for separable in [False, True]:
            compare(
                varies_along_w,
                "Signal varies along W (columns), constant along H",
                mode,
                separable,
            )
            compare(
                varies_along_h,
                "Signal varies along H (rows), constant along W",
                mode,
                separable,
            )
