"""Test the wavelet packet code."""

# Created on Fri Apr 6 2021 by moritz (wolter@cs.uni-bonn.de)

from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import pytest
import pywt
import torch
from scipy import datasets

from ptwt.constants import ExtendedBoundaryMode
from ptwt.packets import WaveletPacket, WaveletPacket2D, get_freq_order


def _compare_trees1(
    wavelet_str: str,
    max_lev: Optional[int] = 3,
    pywt_boundary: str = "zero",
    ptwt_boundary: ExtendedBoundaryMode = "zero",
    length: int = 256,
    batch_size: int = 1,
    transform_mode: bool = False,
    multiple_transforms: bool = False,
) -> None:
    data = np.random.rand(batch_size, length)
    wavelet = pywt.Wavelet(wavelet_str)

    if transform_mode:
        twp = WaveletPacket(
            None, wavelet, mode=ptwt_boundary, maxlevel=max_lev
        ).transform(torch.from_numpy(data), maxlevel=max_lev)
    else:
        twp = WaveletPacket(
            torch.from_numpy(data), wavelet, mode=ptwt_boundary, maxlevel=max_lev
        )

    # if multiple_transform flag is set, recalculcate the packets
    if multiple_transforms:
        twp.transform(torch.from_numpy(data), maxlevel=max_lev)

    nodes = twp.get_level(twp.maxlevel)
    twp_lst = []
    for node in nodes:
        twp_lst.append(twp[node])
    torch_res = torch.cat(twp_lst, -1).numpy()

    np_batches = []
    for batch_index in range(batch_size):
        wp = pywt.WaveletPacket(
            data=data[batch_index],
            wavelet=wavelet,
            mode=pywt_boundary,
            maxlevel=max_lev,
        )
        nodes = [node.path for node in wp.get_level(wp.maxlevel, "freq")]
        np_lst = []
        for node in nodes:
            np_lst.append(wp[node].data)
        np_res = np.concatenate(np_lst, -1)
        np_batches.append(np_res)
    np_batches = np.stack(np_batches, 0)
    assert wp.maxlevel == twp.maxlevel
    assert np.allclose(torch_res, np_batches)


def _compare_trees2(
    wavelet_str: str,
    max_lev: Optional[int],
    pywt_boundary: str = "zero",
    ptwt_boundary: ExtendedBoundaryMode = "zero",
    height: int = 256,
    width: int = 256,
    batch_size: int = 1,
    transform_mode: bool = False,
    multiple_transforms: bool = False,
) -> None:
    face = datasets.face()[:height, :width]
    face = np.mean(face, axis=-1).astype(np.float64)
    wavelet = pywt.Wavelet(wavelet_str)
    batch_list = []
    for _ in range(batch_size):
        wp_tree = pywt.WaveletPacket2D(
            data=face,
            wavelet=wavelet,
            mode=pywt_boundary,
            maxlevel=max_lev,
        )
        # Get the full decomposition
        wp_keys = list(product(["a", "h", "v", "d"], repeat=wp_tree.maxlevel))
        np_packets = []
        for node in wp_keys:
            np_packet = wp_tree["".join(node)].data
            np_packets.append(np_packet)
        np_packets = np.stack(np_packets, 0)
        batch_list.append(np_packets)
    batch_np_packets = np.stack(batch_list, 0)

    # get the PyTorch decomposition
    pt_data = torch.stack([torch.from_numpy(face)] * batch_size, 0)

    if transform_mode:
        ptwt_wp_tree = WaveletPacket2D(
            None, wavelet=wavelet, mode=ptwt_boundary
        ).transform(pt_data, maxlevel=max_lev)
    else:
        ptwt_wp_tree = WaveletPacket2D(
            pt_data, wavelet=wavelet, mode=ptwt_boundary, maxlevel=max_lev
        )

    # if multiple_transform flag is set, recalculcate the packets
    if multiple_transforms:
        ptwt_wp_tree.transform(pt_data, maxlevel=max_lev)

    packets = []
    for node in wp_keys:
        packet = ptwt_wp_tree["".join(node)]
        packets.append(packet)
    packets_pt = torch.stack(packets, 1).numpy()
    assert wp_tree.maxlevel == ptwt_wp_tree.maxlevel
    assert np.allclose(packets_pt, batch_np_packets)


@pytest.mark.slow
@pytest.mark.parametrize("max_lev", [1, 2, 3, 4, None])
@pytest.mark.parametrize(
    "wavelet_str", ["haar", "db2", "db3", "db4", "db5", "db6", "db7", "db8"]
)
@pytest.mark.parametrize("boundary", ["zero", "reflect"])
@pytest.mark.parametrize("batch_size", [2, 1])
@pytest.mark.parametrize("transform_mode", [False, True])
@pytest.mark.parametrize("multiple_transforms", [False, True])
def test_2d_packets(
    max_lev: Optional[int],
    wavelet_str: str,
    boundary: ExtendedBoundaryMode,
    batch_size: int,
    transform_mode: bool,
    multiple_transforms: bool,
) -> None:
    """Ensure pywt and ptwt produce equivalent wavelet 2d packet trees."""
    _compare_trees2(
        wavelet_str,
        max_lev,
        pywt_boundary=boundary,
        ptwt_boundary=boundary,
        batch_size=batch_size,
        transform_mode=transform_mode,
        multiple_transforms=multiple_transforms,
    )


@pytest.mark.slow
@pytest.mark.parametrize("max_lev", [1, 2, 3, 4, None])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("transform_mode", [False, True])
@pytest.mark.parametrize("multiple_transforms", [False, True])
def test_boundary_matrix_packets2(
    max_lev: Optional[int],
    batch_size: int,
    transform_mode: bool,
    multiple_transforms: bool,
) -> None:
    """Ensure the 2d - sparse matrix haar tree and pywt-tree are the same."""
    _compare_trees2(
        "db1",
        max_lev,
        "zero",
        "boundary",
        batch_size=batch_size,
        transform_mode=transform_mode,
        multiple_transforms=multiple_transforms,
    )


@pytest.mark.slow
@pytest.mark.parametrize("max_lev", [1, 2, 3, 4, None])
@pytest.mark.parametrize(
    "wavelet_str", ["haar", "db2", "db3", "db4", "db5", "db6", "db7", "db8"]
)
@pytest.mark.parametrize("boundary", ["zero", "reflect", "constant"])
@pytest.mark.parametrize("batch_size", [2, 1])
@pytest.mark.parametrize("transform_mode", [False, True])
@pytest.mark.parametrize("multiple_transforms", [False, True])
def test_1d_packets(
    max_lev: int,
    wavelet_str: str,
    boundary: str,
    batch_size: int,
    transform_mode: bool,
    multiple_transforms: bool,
) -> None:
    """Ensure pywt and ptwt produce equivalent wavelet 1d packet trees."""
    _compare_trees1(
        wavelet_str,
        max_lev,
        pywt_boundary=boundary,
        ptwt_boundary=boundary,
        batch_size=batch_size,
        transform_mode=transform_mode,
        multiple_transforms=multiple_transforms,
    )


@pytest.mark.slow
@pytest.mark.parametrize("max_lev", [1, 2, 3, 4, None])
@pytest.mark.parametrize("transform_mode", [False, True])
@pytest.mark.parametrize("multiple_transforms", [False, True])
def test_boundary_matrix_packets1(
    max_lev: Optional[int], transform_mode: bool, multiple_transforms: bool
) -> None:
    """Ensure the 2d - sparse matrix haar tree and pywt-tree are the same."""
    _compare_trees1(
        "db1",
        max_lev,
        "zero",
        "boundary",
        transform_mode=transform_mode,
        multiple_transforms=multiple_transforms,
    )


@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize("wavelet_str", ["db2"])
@pytest.mark.parametrize("pywt_boundary", ["zero"])
def test_freq_order(level: int, wavelet_str: str, pywt_boundary: str) -> None:
    """Test the packets in frequency order."""
    face = datasets.face()
    wavelet = pywt.Wavelet(wavelet_str)
    wp_tree = pywt.WaveletPacket2D(
        data=np.mean(face, axis=-1).astype(np.float64),
        wavelet=wavelet,
        mode=pywt_boundary,
    )
    # Get the full decomposition
    freq_tree = wp_tree.get_level(level, "freq")
    freq_order = get_freq_order(level)

    for order_list, tree_list in zip(freq_tree, freq_order):
        for order_el, tree_el in zip(order_list, tree_list):
            print(
                level,
                order_el.path,
                "".join(tree_el),
                order_el.path == "".join(tree_el),
            )
            assert order_el.path == "".join(tree_el)


def test_packet_harbo_lvl3() -> None:
    """From Jensen, La Cour-Harbo, Rippels in Mathematics, Chapter 8 (page 89)."""
    data = np.array([56.0, 40.0, 8.0, 24.0, 48.0, 48.0, 40.0, 16.0])

    class _MyHaarFilterBank(object):
        @property
        def filter_bank(self) -> Tuple[List[float], ...]:
            """Unscaled Haar wavelet filters."""
            return (
                [1 / 2, 1 / 2.0],
                [-1 / 2.0, 1 / 2.0],
                [1 / 2.0, 1 / 2.0],
                [1 / 2.0, -1 / 2.0],
            )

    wavelet = pywt.Wavelet("unscaled Haar Wavelet", filter_bank=_MyHaarFilterBank())

    twp = WaveletPacket(torch.from_numpy(data), wavelet, mode="reflect")
    twp_nodes = twp.get_level(3)
    twp_lst = []
    for node in twp_nodes:
        twp_lst.append(torch.squeeze(twp[node]))
    torch_res = torch.stack(twp_lst).numpy()
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode="reflect")
    pywt_nodes = [node.path for node in wp.get_level(3, "freq")]
    np_lst = []
    for node in pywt_nodes:
        np_lst.append(wp[node].data)
    np_res = np.concatenate(np_lst)
    assert np.allclose(torch_res, np_res)


def test_access_errors_1d() -> None:
    """Test expected access errors for 1d packets."""
    twp = WaveletPacket(None, "haar")
    with pytest.raises(ValueError):
        twp["a"]

    twp.transform(torch.from_numpy(np.random.rand(1, 20)))

    with pytest.raises(KeyError):
        twp["a" * 100]


def test_access_errors_2d() -> None:
    """Test expected access errors for 2d packets."""
    face = datasets.face()
    face = np.mean(face, axis=-1).astype(np.float64)

    twp = WaveletPacket2D(None, "haar")
    with pytest.raises(ValueError):
        twp["a"]

    twp.transform(torch.from_numpy(face))

    with pytest.raises(KeyError):
        twp["a" * 100]


@pytest.mark.slow
@pytest.mark.parametrize("level", [1, 3])
@pytest.mark.parametrize("base_key", ["a", "d"])
@pytest.mark.parametrize("shape", [[1, 64, 63], [3, 64, 64], [1, 128]])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "sym4"])
@pytest.mark.parametrize("axis", (1, -1))
def test_inverse_packet_1d(
    level: int, base_key: str, shape: List[int], wavelet: str, axis: int
) -> None:
    """Test the 1d reconstruction code."""
    signal = np.random.randn(*shape)
    mode = "reflect"
    wp = pywt.WaveletPacket(signal, wavelet, mode=mode, maxlevel=level, axis=axis)
    ptwp = WaveletPacket(
        torch.from_numpy(signal), wavelet, mode=mode, maxlevel=level, axis=axis
    )
    wp[base_key * level].data *= 0
    ptwp[base_key * level].data *= 0
    wp.reconstruct(update=True)
    ptwp.reconstruct()
    assert np.allclose(wp[""].data, ptwp[""].numpy()[..., : shape[-2], : shape[-1]])


@pytest.mark.slow
@pytest.mark.parametrize("level", [1, 3])
@pytest.mark.parametrize("base_key", ["a", "h", "d"])
@pytest.mark.parametrize("size", [(32, 32, 32), (32, 32, 31, 64)])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "sym4"])
@pytest.mark.parametrize("axes", [(-2, -1), (-1, -2), (1, 2), (2, 0), (0, 2)])
def test_inverse_packet_2d(
    level: int,
    base_key: str,
    size: Tuple[int, ...],
    wavelet: str,
    axes: Tuple[int, int],
) -> None:
    """Test the 2d reconstruction code."""
    signal = np.random.randn(*size)
    mode = "reflect"
    wp = pywt.WaveletPacket2D(signal, wavelet, mode=mode, maxlevel=level, axes=axes)
    ptwp = WaveletPacket2D(
        torch.from_numpy(signal), wavelet, mode=mode, maxlevel=level, axes=axes
    )
    wp[base_key * level].data *= 0
    ptwp[base_key * level].data *= 0
    wp.reconstruct(update=True)
    ptwp.reconstruct()
    assert np.allclose(wp[""].data, ptwp[""].numpy()[: size[0], : size[1], : size[2]])


def test_inverse_boundary_packet_1d() -> None:
    """Test the 2d boundary reconstruction code."""
    signal = np.random.randn(1, 16)
    wp = pywt.WaveletPacket(signal, "haar", mode="zero", maxlevel=2)
    ptwp = WaveletPacket(torch.from_numpy(signal), "haar", mode="boundary", maxlevel=2)
    wp["aa"].data *= 0
    ptwp["aa"].data *= 0
    wp.reconstruct(update=True)
    ptwp.reconstruct()
    assert np.allclose(wp[""].data, ptwp[""].numpy()[:, :16])


def test_inverse_boundary_packet_2d() -> None:
    """Test the 2d boundary reconstruction code."""
    size = (16, 16)
    level = 2
    base_key = "h"
    wavelet = "haar"
    signal = np.random.randn(1, size[0], size[1])
    wp = pywt.WaveletPacket2D(signal, wavelet, mode="zero", maxlevel=level)
    ptwp = WaveletPacket2D(
        torch.from_numpy(signal), wavelet, mode="boundary", maxlevel=level
    )
    wp[base_key * level].data *= 0
    ptwp[base_key * level].data *= 0
    wp.reconstruct(update=True)
    ptwp.reconstruct()
    assert np.allclose(wp[""].data, ptwp[""].numpy()[:, : size[0], : size[1]])


@pytest.mark.slow
@pytest.mark.parametrize("axes", ((-2, -1), (1, 2), (2, 1)))
def test_separable_conv_packets_2d(axes: Tuple[int, int]) -> None:
    """Ensure the 2d separable conv code is ok."""
    wavelet = "db2"
    signal = np.random.randn(1, 32, 32, 32)
    ptwp = WaveletPacket2D(
        torch.from_numpy(signal),
        wavelet,
        mode="reflect",
        maxlevel=2,
        axes=axes,
        separable=True,
    )
    ptwp.reconstruct()
    assert np.allclose(signal, ptwp[""].data[:, :32, :32, :32])
