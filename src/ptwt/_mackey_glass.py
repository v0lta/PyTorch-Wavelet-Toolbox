"""Generate artificial time-series data for debugging purposes."""

import matplotlib.pyplot as plt
import torch


def generate_mackey(
    batch_size: int = 100,
    tmax: int = 200,
    delta_t: float = 1.0,
    rnd: bool = True,
    device: torch.device = "cuda",
) -> torch.Tensor:
    """Generate synthetic training data using the Mackey system of equations.

    dx/dt = beta*(x'/(1+x'))
    See http://www.scholarpedia.org/article/Mackey-Glass_equation
    for reference.

    The system is simulated using a forward euler scheme
    (https://en.wikipedia.org/wiki/Euler_method).

    Args:
        batch_size (int): The number of simulated series to return.
            Defaults to 100.
        tmax (int): Total time to simulate. Defaults to 200.
        delta_t (float): Size of the time step. Defaults to 1.0.
        rnd (bool): If true use a random initial state.
            Defaults to True.
        device (torch.device): Choose cpu or cuda.
            Defaults to "cuda".

    Returns:
        torch.Tensor: A Tensor of shape [batch_size, time, 1],
    """
    steps = int(tmax / delta_t) + 200

    # multi-dimensional data.
    def mackey(x, tau, gamma=0.1, beta=0.2, n=10):
        return beta * x[:, -tau] / (1 + torch.pow(x[:, -tau], n)) - gamma * x[:, -1]

    tau = int(17 * (1 / delta_t))
    x0 = torch.ones([tau], device=device)
    x0 = torch.stack(batch_size * [x0], dim=0)
    if rnd:
        # print('Mackey initial state is random.')
        x0 += torch.empty(x0.shape, device=device).uniform_(-0.1, 0.1)
    else:
        x0 += [-0.01, 0.02]

    x = x0
    # forward_euler
    for _ in range(steps):
        res = torch.unsqueeze(x[:, -1] + delta_t * mackey(x, tau), -1)
        x = torch.cat([x, res], -1)
    discard = 200 + tau
    return x[:, discard:]


def _blockify(data: torch.Tensor, block_length: int) -> torch.Tensor:
    """Blockify the input data series.

       Blocks in the output are replaced with mean values its mean.

    Args:
        data (torch.Tensor): The block free input
        block_length (int): The desired length of the anomaly.

    Returns:
        torch.Tensor: Corrupted output.
    """
    batch_size = data.shape[0]
    steps = data.shape[-1] // block_length
    block_signal = []
    for block_no in range(steps):
        start = block_no * block_length
        stop = (block_no + 1) * block_length
        block_mean = torch.mean(data[:, start:stop], dim=-1)
        block = block_mean * torch.ones([batch_size, block_length], device=data.device)
        block_signal.append(block)
    return torch.cat(block_signal).transpose(0, 1)


class MackeyGenerator(object):
    """Generates lorenz attractor data in 1 or 3d on the GPU."""

    def __init__(
        self,
        batch_size,
        tmax,
        delta_t,
        block_size=None,
        restore_and_plot=False,
        device="cuda",
    ):
        """Create a Mackey-Glass simulator.

        Args:
            batch_size: Total number of samples to generate.
            tmax: Total simulation time.
            delta_t: Time step.
            block_size: Length of mean blocks. Defaults to None.
            restore_and_plot: Deactivate random init. Defaults to False.
            device: Choose cpu or cuda. Defaults to "cuda".
        """
        self.batch_size = batch_size
        self.tmax = tmax
        self.delta_t = delta_t
        self.block_size = block_size
        self.restore_and_plot = restore_and_plot
        self.device = device

    def __call__(self):
        """Simulate a batch and return the result."""
        data_nd = generate_mackey(
            tmax=self.tmax,
            delta_t=self.delta_t,
            batch_size=self.batch_size,
            rnd=not self.restore_and_plot,
            device=self.device,
        )
        data_nd = torch.unsqueeze(data_nd, -1)
        if self.block_size:
            data_nd = _blockify(data_nd, self.block_size)
        # print('data_nd_shape', data_nd.shape)
        return data_nd


def _main():
    mackey = generate_mackey(tmax=1200, delta_t=0.1, rnd=True, device="cuda")
    block_mackey = _blockify(mackey, 100)
    print(mackey.shape)
    plt.plot(mackey[0, :].cpu().numpy())
    plt.plot(block_mackey[0, :].cpu().numpy())
    # tikz.save('mackey.tex')
    plt.show()


if __name__ == "__main__":
    _main()
