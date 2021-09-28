# Created by moritz wolter@cs.uni-bonn.de, 14.05.20
# Inspired by Ripples in Mathematics, Jensen and La Cour-Harbo, Chapter 7.7
# import pywt
import torch
from abc import ABC, abstractmethod


class WaveletFilter(ABC):
    """Interface for learnable wavelets.
    Each wavelet has a filter bank a loss function
    and comes with functionality the test the perfect
    reconstruction and anti-aliasing conditions.
    """

    @property
    @abstractmethod
    def filter_bank(self):
        raise NotImplementedError

    @abstractmethod
    def wavelet_loss(self):
        return self.alias_cancellation_loss() \
               + self.perfect_reconstruction_loss()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    # @abstractmethod
    # def parameters(self):
    #     raise NotImplementedError

    def pf_alias_cancellation_loss(self) -> list:
        """Strang+Nguyen 105: F0(z) = H1(-z); F1(z) = -H0(-z)
        Alternating sign convention from 0 to N see Strang overview
        on the back of the cover.
        """
        dec_lo, dec_hi, rec_lo, rec_hi = self.filter_bank
        m1 = torch.tensor([-1], device=dec_lo.device, dtype=dec_lo.dtype)
        length = dec_lo.shape[0]
        mask = torch.tensor(
            [torch.pow(m1, n) for n in range(length)][::-1],
            device=dec_lo.device,
            dtype=dec_lo.dtype,
        )
        err1 = rec_lo - mask * dec_hi
        err1s = torch.sum(err1 * err1)

        length = dec_lo.shape[0]
        mask = torch.tensor(
            [torch.pow(m1, n) for n in range(length)][::-1],
            device=dec_lo.device,
            dtype=dec_lo.dtype,
        )
        err2 = rec_hi - m1 * mask * dec_lo
        err2s = torch.sum(err2 * err2)
        return err1s + err2s, err1, err2

    def alias_cancellation_loss(self):
        """Implementation of the ac-loss as described on page 104 of Strang+Nguyen.
        F0(z)H0(-z) + F1(z)H1(-z) = 0"""
        dec_lo, dec_hi, rec_lo, rec_hi = self.filter_bank
        m1 = torch.tensor([-1], device=dec_lo.device, dtype=dec_lo.dtype)
        length = dec_lo.shape[0]
        mask = torch.tensor(
            [torch.pow(m1, n) for n in range(length)][::-1],
            device=dec_lo.device,
            dtype=dec_lo.dtype,
        )
        # polynomial multiplication is convolution, compute p(z):
        pad = dec_lo.shape[0] - 1
        p_lo = torch.nn.functional.conv1d(
            dec_lo.unsqueeze(0).unsqueeze(0) * mask,
            torch.flip(rec_lo, [-1]).unsqueeze(0).unsqueeze(0),
            padding=pad,
        )

        pad = dec_hi.shape[0] - 1
        p_hi = torch.nn.functional.conv1d(
            dec_hi.unsqueeze(0).unsqueeze(0) * mask,
            torch.flip(rec_hi, [-1]).unsqueeze(0).unsqueeze(0),
            padding=pad,
        )

        p_test = p_lo + p_hi
        zeros = torch.zeros(
            p_test.shape, device=p_test.device, dtype=p_test.dtype)
        errs = (p_test - zeros) * (p_test - zeros)
        return torch.sum(errs), p_test, zeros

    def perfect_reconstruction_loss(self) -> list:
        """Strang 107: Assuming alias cancellation holds:
        P(z) = F(z)H(z)
        Product filter P(z) + P(-z) = 2.
        However since alias cancellation is implemented as soft constraint:
        P_0 + P_1 = 2
        Somehow numpy and torch implement convolution differently.
        For some reason the machine learning people call cross-correlation
        convolution.
        https://discuss.pytorch.org/t/numpy-convolve-and-conv1d-in-pytorch/12172
        Therefore for true convolution one element needs to be flipped.
        """
        dec_lo, dec_hi, rec_lo, rec_hi = self.filter_bank
        # polynomial multiplication is convolution, compute p(z):
        pad = dec_lo.shape[0] - 1
        p_lo = torch.nn.functional.conv1d(
            dec_lo.unsqueeze(0).unsqueeze(0),
            torch.flip(rec_lo, [-1]).unsqueeze(0).unsqueeze(0),
            padding=pad,
        )

        pad = dec_hi.shape[0] - 1
        p_hi = torch.nn.functional.conv1d(
            dec_hi.unsqueeze(0).unsqueeze(0),
            torch.flip(rec_hi, [-1]).unsqueeze(0).unsqueeze(0),
            padding=pad,
        )

        p_test = p_lo + p_hi
        two_at_power_zero = torch.zeros(
            p_test.shape, device=p_test.device, dtype=p_test.dtype
        )
        # numpy comparison for debugging.
        # np.convolve(self.init_wavelet.filter_bank[0],
        #             self.init_wavelet.filter_bank[2])
        # np.convolve(self.init_wavelet.filter_bank[1],
        #             self.init_wavelet.filter_bank[3])
        two_at_power_zero[..., p_test.shape[-1] // 2] = 2
        # square the error
        errs = (p_test - two_at_power_zero) * (p_test - two_at_power_zero)
        return torch.sum(errs), p_test, two_at_power_zero


class ProductFilter(WaveletFilter, torch.nn.Module):
    def __init__(
        self,
        dec_lo: torch.Tensor,
        dec_hi: torch.Tensor,
        rec_lo: torch.Tensor,
        rec_hi: torch.Tensor,
    ):
        super().__init__()
        self.dec_lo = torch.nn.Parameter(dec_lo)
        self.dec_hi = torch.nn.Parameter(dec_hi)
        self.rec_lo = torch.nn.Parameter(rec_lo)
        self.rec_hi = torch.nn.Parameter(rec_hi)

    @property
    def filter_bank(self):
        return self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi

    # def parameters(self):
    #     return [self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi]

    def __len__(self):
        return self.dec_lo.shape[-1]

    def product_filter_loss(self):
        return self.perfect_reconstruction_loss()[0] \
               + self.alias_cancellation_loss()[0]

    def wavelet_loss(self):
        return self.product_filter_loss()


class SoftOrthogonalWavelet(ProductFilter, torch.nn.Module):
    def __init__(
        self,
        dec_lo: torch.Tensor,
        dec_hi: torch.Tensor,
        rec_lo: torch.Tensor,
        rec_hi: torch.Tensor,
    ):
        super().__init__(dec_lo, dec_hi, rec_lo, rec_hi)

    def rec_lo_orthogonality_loss(self):
        """See Strang p. 148/149 or Harbo p. 80.
        Since L is a convolution matrix, LL^T can be evaluated
        trough convolution.
        :return: A tensor with the orthogonality constraint value."""
        filt_len = self.dec_lo.shape[-1]
        pad_dec_lo = torch.cat(
            [
                self.dec_lo,
                torch.zeros(
                    [
                        filt_len,
                    ],
                    device=self.dec_lo.device,
                ),
            ],
            -1,
        )
        res = torch.nn.functional.conv1d(
            pad_dec_lo.unsqueeze(0).unsqueeze(0),
            self.dec_lo.unsqueeze(0).unsqueeze(0),
            stride=2,
        )
        test = torch.zeros_like(res.squeeze(0).squeeze(0))
        test[0] = 1
        err = res - test
        return torch.sum(err * err)

    def filt_bank_orthogonality_loss(self):
        """On Page 79 of the Book Ripples in Mathematics
        by Jensen la Cour-Harbo the constraint
        g0[k] = h0[-k] and g1[k] = h1[-k] for orthogonal filters
        is presented. A measurement
        is implemented below."""
        eq0 = self.dec_lo - self.rec_lo.flip(-1)
        eq1 = self.dec_hi - self.rec_hi.flip(-1)
        seq0 = torch.sum(eq0 * eq0)
        seq1 = torch.sum(eq1 * eq1)
        # print(eq0, eq1)
        return seq0 + seq1

    def wavelet_loss(self):
        return self.product_filter_loss() + self.filt_bank_orthogonality_loss()


class HardOrthogonalWavelet(WaveletFilter, torch.nn.Module):
    def __init__(self, init_tensor: torch.Tensor):
        self.dec_lo = torch.nn.Parameter(init_tensor)
        m1 = torch.tensor(
            [-1], device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        length = self.dec_lo.shape[0]
        self.mask = torch.tensor(
            [torch.pow(m1, n) for n in range(length)][::-1],
            device=self.dec_lo.device,
            dtype=self.dec_lo.dtype,
        )

    def _construct_filter_bank(self):
        dec_hi = self.mask * self.dec_lo.flip(-1)
        rec_lo = self.dec_lo.flip(-1)
        rec_hi = dec_hi.flip(-1)
        return dec_hi, rec_lo, rec_hi

    def __len__(self):
        return self.dec_lo.shape[-1]

    def cuda(self):
        self.dec_lo.cuda()

    def cpu(self):
        self.dec_lo.cpu()

    # def parameters(self):
    #     return [self.dec_lo]

    @property
    def filter_bank(self):
        dec_hi, rec_lo, rec_hi = self._construct_filter_bank()
        return self.dec_lo, dec_hi, rec_lo, rec_hi

    def rec_lo_orthogonality_loss(self):
        """See Strang p. 148/149 or Harbo p. 80.
        Since L is a convolution matrix, LL^T can be evaluated
        trough convolution.
        :return: A tensor with the orthogonality constraint value."""
        filt_len = self.dec_lo.shape[-1]
        pad_dec_lo = torch.cat(
            [
                self.dec_lo,
                torch.zeros(
                    [
                        filt_len,
                    ],
                    device=self.dec_lo.device,
                ),
            ],
            -1,
        )
        res = torch.nn.functional.conv1d(
            pad_dec_lo.unsqueeze(0).unsqueeze(0),
            self.dec_lo.unsqueeze(0).unsqueeze(0),
            stride=2,
        )
        test = torch.zeros_like(res.squeeze(0).squeeze(0))
        test[0] = 1
        err = res - test
        return torch.sum(err * err)

    def filt_bank_orthogonality_loss(self):
        """On Page 79 of the Book Ripples in Mathematics
        by Jensen la Cour-Harbo the constraint
        g0[k] = h0[-k] and g1[k] = h1[-k] for orthogonal filters
        is presented. A measurement
        is implemented below."""
        dec_hi, rec_lo, rec_hi = self._construct_filter_bank()
        eq0 = self.dec_lo - rec_lo.flip(-1)
        eq1 = dec_hi - rec_hi.flip(-1)
        seq0 = torch.sum(torch.abs(eq0))
        seq1 = torch.sum(torch.abs(eq1))
        # print(eq0, eq1)
        return seq0 + seq1

    def wavelet_loss(self):
        return self.rec_lo_orthogonality_loss()
