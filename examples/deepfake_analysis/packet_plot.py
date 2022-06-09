import os
from itertools import product
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
import ptwt
import pywt

import matplotlib.pyplot as plt


def get_freq_order(level: int):
    """Get the frequency order for a given packet decomposition level.
    Adapted from:
    https://github.com/PyWavelets/pywt/blob/master/pywt/_wavelet_packets.py
    The code elements denote the filter application order. The filters
    are named following the pywt convention as:
    a - LL, low-low coefficients
    h - LH, low-high coefficients
    v - HL, high-low coefficients
    d - HH, high-high coefficients
    """
    wp_natural_path = list(product(["a", "h", "v", "d"], repeat=level))

    def _get_graycode_order(level, x="a", y="d"):
        graycode_order = [x, y]
        for _ in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [
                y + path for path in graycode_order[::-1]
            ]
        return graycode_order

    def _expand_2d_path(path):
        expanded_paths = {"d": "hh", "h": "hl", "v": "lh", "a": "ll"}
        return (
            "".join([expanded_paths[p][0] for p in path]),
            "".join([expanded_paths[p][1] for p in path]),
        )

    nodes: dict = {}
    for (row_path, col_path), node in [
        (_expand_2d_path(node), node) for node in wp_natural_path
    ]:
        nodes.setdefault(row_path, {})[col_path] = node
    graycode_order = _get_graycode_order(level, x="l", y="h")
    nodes_list: list = [nodes[path] for path in graycode_order if path in nodes]
    wp_frequency_path = []
    for row in nodes_list:
        wp_frequency_path.append([row[path] for path in graycode_order if path in row])
    return wp_frequency_path, wp_natural_path


def generate_frequency_packet_image(packet_array: np.ndarray, degree: int):
    """Create a ready-to-polt image with frequency-order packages.
       Given a packet array in natural order, creat an image which is
       ready to plot in frequency order.
    Args:
        packet_array (np.ndarray): [packet_no, packet_height, packet_width]
            in natural order.
        degree (int): The degree of the packet decomposition.
    Returns:
        [np.ndarray]: The image of shape [original_height, original_width]
    """
    wp_freq_path, wp_natural_path = get_freq_order(degree)

    image = []
    # go through the rows.
    for row_paths in wp_freq_path:
        row = []
        for row_path in row_paths:
            index = wp_natural_path.index(row_path)
            packet = packet_array[index]
            row.append(packet)
        image.append(np.concatenate(row, -1))
    return np.concatenate(image, 0)


def load_image(path_to_file: str) -> torch.Tensor:
    image = Image.open(path_to_file)
    tensor = torch.from_numpy(np.nan_to_num(np.array(image), posinf=255, neginf=0))
    return tensor

def process_images(tensor: torch.Tensor, paths: list) -> torch.Tensor:
    tensor = torch.mean(tensor/255., -1)
    packets = ptwt.WaveletPacket2D(tensor, pywt.Wavelet("Haar"))

    packet_list = []
    for node in paths:
        packet = torch.squeeze(packets["".join(node)], dim=1)
        packet_list.append(packet)
    wp_pt = torch.stack(packet_list, dim=1)
    # return wp_pt
    return torch.log(torch.abs(wp_pt) + 1e-12)


def load_images(path: str) -> list:
    image_list = []
    for root, _, files in os.walk(path, topdown=False):
        for name in tqdm(files):
            path = os.path.join(root, name)
            packets = load_image(path)
            image_list.append(packets)
    return image_list 


if __name__ == '__main__':
    frequency_path, natural_path = get_freq_order(level=3)
    print("Loading ffhq images:")
    ffhq_images = load_images("./ffhq_style_gan/source_data/A_ffhq")
    print("processing ffhq")
    ffhq_images = torch.stack(ffhq_images).split(2500)
    ffhq_packets = []
    for image_batch in tqdm(ffhq_images):
        ffhq_packets.append(process_images(image_batch, natural_path))

    mean_packets_ffhq = torch.mean(torch.cat(ffhq_packets), 0)
    del ffhq_images
    del ffhq_packets


    print("Loading style-gan images")
    gan_images = load_images("./ffhq_style_gan/source_data/B_stylegan")
    print("processing style-gan")
    gan_images = torch.stack(gan_images).split(2500)
    gan_packets = []
    for image_batch in tqdm(gan_images):
        gan_packets.append(process_images(image_batch, natural_path))

    mean_packets_gan = torch.mean(torch.cat(gan_packets), 0)
    del gan_images
    del gan_packets

    plot_ffhq = generate_frequency_packet_image(mean_packets_ffhq, 3)
    plot_gan = generate_frequency_packet_image(mean_packets_gan, 3)

    fig = plt.figure(figsize=(9,3))
    fig.add_subplot(1, 2, 1)
    plt.imshow(plot_ffhq, vmax=1.5, vmin=-7)
    plt.title("real")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    fig.add_subplot(1, 2, 2)
    plt.imshow(plot_gan, vmax=1.5, vmin=-7)
    plt.title("fake")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    plt.show()

    plt.plot(torch.mean(mean_packets_ffhq, (1, 2)).flatten().numpy(), label='real')
    plt.plot(torch.mean(mean_packets_gan, (1, 2)).flatten().numpy(), label='fake')
    plt.xlabel('mean packets')
    plt.ylabel('magnitude')
    plt.legend()
    plt.show()

    print("stop")