import pywt
import torch
from src.learnable_wavelets import SoftOrthogonalWavelet

def run_list(lst, orth=True):
    for ws in lst:
        wavelet = pywt.Wavelet(ws)
        orthwave = SoftOrthogonalWavelet(torch.tensor(wavelet.dec_lo),
                                     torch.tensor(wavelet.dec_hi),
                                     torch.tensor(wavelet.rec_lo),
                                     torch.tensor(wavelet.rec_hi))
        prl = orthwave.perfect_reconstruction_loss()[0]
        acl = orthwave.alias_cancellation_loss()[0]
        assert prl < 1e-10
        assert acl < 1e-10
        pacl = orthwave.pf_alias_cancellation_loss()[0]

        orth = orthwave.filt_bank_orthogonality_loss()
        print(ws, 'prl, %.5f | acl, %.5f | pfacl, %.5f | orth, %.5f '
              % (prl.item(), acl.item(), pacl.item(), orth.item()))
        if orth is True:
            assert orth < 1e-10




def test_db_wavelet_loss():
    lst = pywt.wavelist(family='db')
    run_list(lst, orth=True)

def test_sym_wavelet_loss():
    lst = pywt.wavelist(family='sym')
    run_list(lst, orth=True)

def test_coif_wavelet_loss():
    lst = pywt.wavelist(family='coif')
    run_list(lst, orth=True)

def test_bior_wavelet_loss():
    lst = pywt.wavelist(family='bior')
    run_list(lst, orth=False)


def test_rbio_wavelet_loss():
    lst = pywt.wavelist(family='rbio')
    run_list(lst, orth=False)


# def test_dmey_wavelet_loss():
#     lst = pywt.wavelist(family='dmey')
#     run_list(lst, orth=True)




if __name__ == '__main__':
    test_db_wavelet_loss()
    test_sym_wavelet_loss()
    test_bior_wavelet_loss()
    test_rbio_wavelet_loss()
    test_coif_wavelet_loss()