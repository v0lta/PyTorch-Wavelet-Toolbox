# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "ptwt",
#     "tensorboard",
#     "torch",
#     "torchvision",
#     "pystow",
# ]
#
# [tool.uv.sources]
# ptwt = { path = "../../" }
# ///

# Originally created by moritz (wolter@cs.uni-bonn.de) on 17/12/2019
# at https://github.com/v0lta/Wavelet-network-compression/blob/master/mnist_compression.py
# based on https://github.com/pytorch/examples/blob/master/mnist/main.py

import argparse
import collections
from typing import Literal

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pystow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms
from wavelet_linear import WaveletLayer

from ptwt.wavelets_learnable import ProductFilter, WaveletFilter

HERE = Path(__file__).parent.resolve()
MODULE = pystow.module("torchvision", "mnist")


class Net(nn.Module):
    def __init__(
        self,
        compression: Literal["None", "Wavelet"],
        *,
        wavelet: WaveletFilter | None = None,
        wave_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.wavelet = wavelet
        if compression == "None":
            self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
            self.fc2 = torch.nn.Linear(500, 10)
            self.do_dropout = True
        elif compression == "Wavelet":
            assert wavelet is not None, "initial wavelet must be set."
            self.fc1 = WaveletLayer(
                wavelet=wavelet, scales=6, depth=800, dropout=wave_dropout
            )
            self.fc2 = torch.nn.Linear(800, 10)
            self.do_dropout = False
        else:
            raise ValueError(f"invalid compression: {compression}")

        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.max_pool_2s_k2 = torch.nn.MaxPool2d(2)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool_2s_k2(x)
        x = self.conv2(x)
        x = self.max_pool_2s_k2(x)
        if self.do_dropout:
            x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        if self.do_dropout:
            x = self.dropout2(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

    def wavelet_loss(self):
        if self.wavelet is None:
            raise ValueError
        acl, _, _ = self.wavelet.alias_cancellation_loss()
        prl, _, _ = self.wavelet.perfect_reconstruction_loss()
        return acl + prl


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if args.compression == "Wavelet":
            wvl = model.wavelet_loss()
            loss = loss + wvl * args.wave_loss_weight
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            msg = "Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
            if args.compression == "Wavelet":
                msg += f", wvl-loss: {wvl.item():.6f}"
            print(msg)


def test(args, model, device, test_loader, test_writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    wvl_loss = model.wavelet_loss()

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    test_writer.add_scalar("test_correct", correct, epoch)
    test_writer.add_scalar("test_loss", test_loss, epoch)
    test_writer.add_scalar(
        "test_acc", 100.0 * correct / len(test_loader.dataset), epoch
    )
    test_writer.add_scalar("wvl_loss", wvl_loss, epoch)
    return wvl_loss, 100.0 * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=250,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="Wavelet",
        help="Choose the compression mode, None, Wavelet, Fastfood",
    )
    parser.add_argument(
        "--wave_loss_weight",
        type=float,
        default=1.0,
        help="Weight term of the wavelet loss",
    )
    parser.add_argument(
        "--wave_dropout",
        type=float,
        default=0.5,
        help="Wavelet layer dropout probability.",
    )

    args = parser.parse_args()
    print(args)

    if args.no_cuda:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    torch.manual_seed(args.seed)

    device = torch.device(device)

    kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            MODULE.base,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            MODULE.base,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    if args.compression == "Wavelet":
        CustomWavelet = collections.namedtuple(
            "Wavelet", ["dec_lo", "dec_hi", "rec_lo", "rec_hi", "name"]
        )
        # init_wavelet = ProductFilter(
        #     dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
        #     dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
        #     rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
        #     rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
        #     )

        # random init
        init_wavelet = ProductFilter(
            torch.rand(size=[6], requires_grad=True) / 2 - 0.25,
            torch.rand(size=[6], requires_grad=True) / 2 - 0.25,
            torch.rand(size=[6], requires_grad=True) / 2 - 0.25,
            torch.rand(size=[6], requires_grad=True) / 2 - 0.25,
        )

    else:
        init_wavelet = None

    model = Net(
        compression=args.compression,
        wavelet=init_wavelet,
        wave_dropout=args.wave_dropout,
    ).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    test_wvl_lst = []
    test_acc_lst = []
    test_wvl_loss, test_acc = test(args, model, device, test_loader, writer, 0)
    test_wvl_lst.append(test_wvl_loss.item())
    test_acc_lst.append(test_acc)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_wvl_loss, test_acc = test(args, model, device, test_loader, writer, epoch)
        test_wvl_lst.append(test_wvl_loss.item())
        test_acc_lst.append(test_acc)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    n_params = sum(np.prod(p.shape) for p in model.parameters() if p.requires_grad)
    print(f"the model has {n_params:,} parameters")

    # plt.semilogy(test_wvl_lst)
    # plt.semilogy(test_acc_lst)
    # plt.legend(['wavlet loss', 'accuracy'])
    # plt.show()

    plt.plot(model.wavelet.filter_bank.dec_lo.detach().cpu().numpy(), "-*")
    plt.plot(model.wavelet.filter_bank.dec_hi.detach().cpu().numpy(), "-*")
    plt.plot(model.wavelet.filter_bank.rec_lo.detach().cpu().numpy(), "-*")
    plt.plot(model.wavelet.filter_bank.rec_hi.detach().cpu().numpy(), "-*")
    plt.legend(["H_0", "H_1", "F_0", "F_1"])
    plt.savefig(HERE.joinpath("plot.svg"))


if __name__ == "__main__":
    main()
