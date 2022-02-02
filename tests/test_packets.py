"""Test the wavelet packet code."""
# Created on Fri Apr 6 2021 by moritz (wolter@cs.uni-bonn.de)
from itertools import product

import numpy as np
import pytest
import pywt
import torch
from scipy import misc
from src.ptwt.packets import WaveletPacket, WaveletPacket2D, get_freq_order


def test_packet_harbo_lvl3():
    """From Jensen, La Cour-Harbo, Rippels in Mathematics, Chapter 8 (page 89)."""
    w = [56.0, 40.0, 8.0, 24.0, 48.0, 48.0, 40.0, 16.0]

    class MyHaarFilterBank(object):
        @property
        def filter_bank(self):
            return (
                [1 / 2, 1 / 2.0],
                [-1 / 2.0, 1 / 2.0],
                [1 / 2.0, 1 / 2.0],
                [1 / 2.0, -1 / 2.0],
            )

    wavelet = pywt.Wavelet("unscaled Haar Wavelet", filter_bank=MyHaarFilterBank())
    data = torch.tensor(w)
    twp = WaveletPacket(data, wavelet, mode="reflect")
    nodes = twp.get_level(3)
    twp_lst = []
    for node in nodes:
        twp_lst.append(torch.squeeze(twp[node]))
    res = torch.stack(twp_lst).numpy()
    wp = pywt.WaveletPacket(data=np.array(w), wavelet=wavelet, mode="reflect")
    nodes = [node.path for node in wp.get_level(3, "freq")]
    np_lst = []
    for node in nodes:
        np_lst.append(wp[node].data)
    viz = np.concatenate(np_lst)

    err = np.mean(np.abs(res - viz))
    assert err < 1e-8


def _compare_trees(
    wavelet_str: str,
    max_lev: int,
    pywt_boundary: str = "zero",
    ptwt_boundary: str = "zero",
):
    face = misc.face()[256:512, 256:512]
    wavelet = pywt.Wavelet(wavelet_str)
    wp_tree = pywt.WaveletPacket2D(
        data=np.mean(face, axis=-1).astype(np.float64),
        wavelet=wavelet,
        mode=pywt_boundary,
    )
    # Get the full decomposition
    wp_keys = list(product(["a", "v", "d", "h"], repeat=max_lev))
    count = 0
    img_rows = None
    img = []
    for node in wp_keys:
        packet = np.squeeze(wp_tree["".join(node)].data)
        if img_rows is not None:
            img_rows = np.concatenate([img_rows, packet], axis=1)
        else:
            img_rows = packet
        count += 1
        if count >= np.sqrt(len(wp_keys)):
            count = 0
            img.append(img_rows)
            img_rows = None
    img_pywt = np.concatenate(img, axis=0)
    pt_data = torch.unsqueeze(
        torch.from_numpy(np.mean(face, axis=-1).astype(np.float64)), 0
    )
    ptwt_wp_tree = WaveletPacket2D(pt_data, wavelet=wavelet, mode=ptwt_boundary)
    # get the PyTorch decomposition
    count = 0
    img_pt = []
    img_rows_pt = None
    for node in wp_keys:
        packet = torch.squeeze(ptwt_wp_tree["".join(node)])
        if img_rows_pt is not None:
            img_rows_pt = torch.cat([img_rows_pt, packet], axis=1)
        else:
            img_rows_pt = packet
        count += 1
        if count >= np.sqrt(len(wp_keys)):
            count = 0
            img_pt.append(img_rows_pt)
            img_rows_pt = None
    img_pt = torch.cat(img_pt, axis=0).numpy()
    abs_err_img = np.abs(img_pt - img_pywt)
    abs_err = np.mean(abs_err_img)
    print(
        wavelet_str,
        max_lev,
        "total error",
        abs_err,
        ["ok" if abs_err < 1e-4 else "failed!"],
    )
    assert np.allclose(img_pt, img_pywt)


@pytest.mark.slow
@pytest.mark.parametrize("max_lev", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "wavelet_str", ["haar", "db2", "db3", "db4", "db5", "db6", "db7", "db8"]
)
@pytest.mark.parametrize("boundary", ["zero", "reflect"])
def test_2d_packets(max_lev, wavelet_str, boundary):
    """Ensure pywt and ptwt produce equivalent wavelet packet trees."""
    _compare_trees(wavelet_str, max_lev, pywt_boundary=boundary, ptwt_boundary=boundary)


@pytest.mark.slow
@pytest.mark.parametrize("max_lev", [1, 2, 3, 4])
def test_boundary_matrix_packets(max_lev):
    """Ensure the sparse matrix haar tree and pywt-tree are the same."""
    _compare_trees("db1", max_lev, "zero", "boundary")


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
