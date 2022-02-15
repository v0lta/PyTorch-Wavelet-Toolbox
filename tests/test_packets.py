"""Test the wavelet packet code."""
# Created on Fri Apr 6 2021 by moritz (wolter@cs.uni-bonn.de)
from itertools import product

import numpy as np
import pytest
import pywt
import torch
from scipy import misc
from src.ptwt.packets import WaveletPacket, WaveletPacket2D, get_freq_order


def _compare_trees1(
    wavelet_str: str,
    max_lev: int = 3,
    pywt_boundary: str = "zero",
    ptwt_boundary: str = "zero",
    length: int = 256,
    batch_size: int = 1,
):
    data = np.random.rand(batch_size, length)
    wavelet = pywt.Wavelet(wavelet_str)
    twp = WaveletPacket(torch.from_numpy(data), wavelet, mode=ptwt_boundary)
    nodes = twp.get_level(max_lev)
    twp_lst = []
    for node in nodes:
        twp_lst.append(twp[node])
    torch_res = torch.cat(twp_lst, -1).numpy()

    np_batches = []
    for batch_index in range(batch_size):
        wp = pywt.WaveletPacket(
            data=data[batch_index], wavelet=wavelet, mode=pywt_boundary
        )
        nodes = [node.path for node in wp.get_level(max_lev, "freq")]
        np_lst = []
        for node in nodes:
            np_lst.append(wp[node].data)
        np_res = np.concatenate(np_lst, -1)
        np_batches.append(np_res)
    np_batches = np.stack(np_batches, 0)
    assert np.allclose(torch_res, np_batches)


def _compare_trees2(
    wavelet_str: str,
    max_lev: int,
    pywt_boundary: str = "zero",
    ptwt_boundary: str = "zero",
    height: int = 256,
    width: int = 256,
    batch_size: int = 1,
):

    face = misc.face()[:height, :width]
    face = np.mean(face, axis=-1).astype(np.float64)
    wavelet = pywt.Wavelet(wavelet_str)
    batch_list = []
    for _ in range(batch_size):
        wp_tree = pywt.WaveletPacket2D(
            data=face,
            wavelet=wavelet,
            mode=pywt_boundary,
        )
        # Get the full decomposition
        wp_keys = list(product(["a", "h", "v", "d"], repeat=max_lev))
        np_packets = []
        for node in wp_keys:
            np_packet = wp_tree["".join(node)].data
            np_packets.append(np_packet)
        np_packets = np.stack(np_packets, 0)
        batch_list.append(np_packets)
    batch_np_packets = np.stack(batch_list, 0)

    # get the PyTorch decomposition
    pt_data = torch.stack([torch.from_numpy(face)] * batch_size, 0)
    ptwt_wp_tree = WaveletPacket2D(pt_data, wavelet=wavelet, mode=ptwt_boundary)
    packets = []
    for node in wp_keys:
        packet = ptwt_wp_tree["".join(node)]
        packets.append(packet)
    packets_pt = torch.stack(packets, 1).numpy()
    assert np.allclose(packets_pt, batch_np_packets)


@pytest.mark.slow
@pytest.mark.parametrize("max_lev", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "wavelet_str", ["haar", "db2", "db3", "db4", "db5", "db6", "db7", "db8"]
)
@pytest.mark.parametrize("boundary", ["zero", "reflect"])
@pytest.mark.parametrize("batch_size", [2, 1])
def test_2d_packets(max_lev, wavelet_str, boundary, batch_size):
    """Ensure pywt and ptwt produce equivalent wavelet 2d packet trees."""
    _compare_trees2(
        wavelet_str,
        max_lev,
        pywt_boundary=boundary,
        ptwt_boundary=boundary,
        batch_size=batch_size,
    )


@pytest.mark.slow
@pytest.mark.parametrize("max_lev", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_boundary_matrix_packets2(max_lev, batch_size):
    """Ensure the 2d - sparse matrix haar tree and pywt-tree are the same."""
    _compare_trees2("db1", max_lev, "zero", "boundary", batch_size=batch_size)


@pytest.mark.slow
@pytest.mark.parametrize("max_lev", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "wavelet_str", ["haar", "db2", "db3", "db4", "db5", "db6", "db7", "db8"]
)
@pytest.mark.parametrize("boundary", ["zero", "reflect"])
@pytest.mark.parametrize("batch_size", [2, 1])
def test_1d_packets(max_lev, wavelet_str, boundary, batch_size):
    """Ensure pywt and ptwt produce equivalent wavelet 1d packet trees."""
    _compare_trees1(
        wavelet_str,
        max_lev,
        pywt_boundary=boundary,
        ptwt_boundary=boundary,
        batch_size=batch_size,
    )


@pytest.mark.slow
@pytest.mark.parametrize("max_lev", [1, 2, 3, 4])
def test_boundary_matrix_packets1(max_lev):
    """Ensure the 2d - sparse matrix haar tree and pywt-tree are the same."""
    _compare_trees1("db1", max_lev, "zero", "boundary")


@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize("wavelet_str", ["db2"])
@pytest.mark.parametrize("pywt_boundary", ["zero"])
def test_freq_order(level, wavelet_str, pywt_boundary):
    """Test the packets in frequency order."""
    face = misc.face()
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


def test_packet_harbo_lvl3():
    """From Jensen, La Cour-Harbo, Rippels in Mathematics, Chapter 8 (page 89)."""
    data = np.array([56.0, 40.0, 8.0, 24.0, 48.0, 48.0, 40.0, 16.0])

    class _MyHaarFilterBank(object):
        @property
        def filter_bank(self):
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




