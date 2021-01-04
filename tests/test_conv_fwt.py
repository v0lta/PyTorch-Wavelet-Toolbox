import torch
import numpy as np
import pywt
import scipy.misc
from tests.mackey_glass import MackeyGenerator
from examples.learnable_wavelets import SoftOrthogonalWavelet
from src.conv_transform import conv_fwt, conv_ifwt, conv_fwt_2d, conv_ifwt_2d
from src.conv_transform import flatten_2d_coeff_lst

# import matplotlib.pyplot as plt


def test_conv_fwt_haar_lvl2():
    data = [1., 2., 3., 4., 5., 6., 7., 8., 9.,
            10., 11., 12., 13., 14., 15., 16.]
    npdata = np.array(data)
    ptdata = torch.tensor(data).unsqueeze(0).unsqueeze(0)

    # -------------------------- Haar wavelet tests --------------------- #
    wavelet = pywt.Wavelet('haar')
    coeffs = pywt.wavedec(data, wavelet, level=2)
    coeffs2 = conv_fwt(ptdata, wavelet, scales=2)
    # print(coeffs)
    # print(coeffs2)
    assert len(coeffs) == len(coeffs2)
    err = np.mean(np.abs(np.concatenate(coeffs)
                  - torch.cat(coeffs2, -1).squeeze().numpy()))
    print('haar coefficient error scale 2', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4


def test_conv_fwt_haar_lvl4():
    generator = MackeyGenerator(batch_size=24, tmax=512,
                                delta_t=1, device='cpu')
    mackey_data_1 = torch.squeeze(generator())
    wavelet = pywt.Wavelet('haar')
    ptcoeff = conv_fwt(mackey_data_1.unsqueeze(1), wavelet, scales=4)
    pycoeff = pywt.wavedec(mackey_data_1[0, :].numpy(), wavelet, level=4)
    ptcoeff = torch.cat(ptcoeff, -1)[0, :].numpy()
    pycoeff = np.concatenate(pycoeff)
    err = np.mean(np.abs(pycoeff - ptcoeff))
    print('haar coefficient error scale 4:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

    res = conv_ifwt(conv_fwt(mackey_data_1.unsqueeze(1), wavelet), wavelet)
    err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
    print('haar reconstruction error scale 4:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

def test_conv_fwt_db2_lvl1():
    data = [1., 2., 3., 4., 5., 6., 7., 8., 9.,
            10., 11., 12., 13., 14., 15., 16.]
    npdata = np.array(data)
    ptdata = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    # ------------------------- db2 wavelet tests ----------------------------
    wavelet = pywt.Wavelet('db2')
    coeffs = pywt.wavedec(data, wavelet, level=1, mode='reflect')
    coeffs2 = conv_fwt(ptdata, wavelet, scales=1)
    ccoeffs = np.concatenate(coeffs, -1)
    ccoeffs2 = torch.cat(coeffs2, -1).numpy()
    err = np.mean(np.abs(ccoeffs - ccoeffs2))
    print('db2 coefficient error scale 1:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4
    rec = conv_ifwt(coeffs2, wavelet)
    err = np.mean(np.abs(npdata - rec.numpy()))
    print('db2 reconstruction error scale 1:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4


def test_conv_fwt_db5_lvl3():
    generator = MackeyGenerator(batch_size=24, tmax=512,
                                delta_t=1, device='cpu')

    mackey_data_1 = torch.squeeze(generator())
    wavelet = pywt.Wavelet('db5')
    ptcoeff = conv_fwt(mackey_data_1.unsqueeze(1), wavelet, scales=3)
    pycoeff = pywt.wavedec(mackey_data_1[0, :].numpy(), wavelet, level=3)
    cptcoeff = torch.cat(ptcoeff, -1)[0, :]
    cpycoeff = np.concatenate(pycoeff, -1)
    err = np.mean(np.abs(cpycoeff - cptcoeff.numpy()))
    print('db5 coefficient error scale 3:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    # assert err < 1e-4  # fixme!
    # print([c.shape for c in pycoeff])
    # print([cp.shape for cp in cptcoeff])
    # plt.semilogy(cpycoeff)
    # plt.semilogy(cptcoeff.numpy())
    # plt.show()

    res = conv_ifwt(conv_fwt(mackey_data_1.unsqueeze(1), wavelet, scales=3),
                    wavelet)
    err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
    print('db5 reconstruction error scale 3:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

    res = conv_ifwt(conv_fwt(mackey_data_1.unsqueeze(1), wavelet, scales=4),
                    wavelet)
    err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
    print('db5 reconstruction error scale 4:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4


def test_orth_wavelet():
    generator = MackeyGenerator(batch_size=24, tmax=512,
                                delta_t=1, device='cpu')

    mackey_data_1 = torch.squeeze(generator())
    # orthogonal wavelet object test
    wavelet = pywt.Wavelet('db5')
    orthwave = SoftOrthogonalWavelet(torch.tensor(wavelet.rec_lo),
                                     torch.tensor(wavelet.rec_hi),
                                     torch.tensor(wavelet.dec_lo),
                                     torch.tensor(wavelet.dec_hi))
    res = conv_ifwt(conv_fwt(mackey_data_1.unsqueeze(1), orthwave), orthwave)
    err = torch.mean(torch.abs(mackey_data_1 - res.detach())).numpy()
    print('orth reconstruction error scale 4:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

def test_2d_haar():
    # ------------------------- 2d haar wavelet tests -----------------------
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet('haar')

    # single level haar - 2d
    coeff2d_pywt = pywt.dwt2(face, wavelet)
    coeff2d = conv_fwt_2d(pt_face, wavelet, scales=1)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    err = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    print('haar 2d coeff err,', err, ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

    # single level 2d haar inverse
    rec = conv_ifwt_2d(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('haar 2d rec err', err, ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

def test_2d_db2():
    # single level db2 - 2d
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet('db2')
    coeff2d_pywt = pywt.dwt2(face, wavelet, mode='reflect')
    coeff2d = conv_fwt_2d(pt_face, wavelet, scales=1)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    err = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    print('db5 2d coeff err,', err, ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

    # single level db2 - 2d inverse.
    rec = conv_ifwt_2d(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('db5 2d rec err,', err, ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

def test_2d_haar_multi():
    # multi level haar - 2d
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet('haar')
    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode='reflect', level=5)
    coeff2d = conv_fwt_2d(pt_face, wavelet, scales=5)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    err = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    # plt.plot(flat_lst); plt.show()
    # plt.plot(flat_lst2); plt.show()
    print('haar 2d scale 5 coeff err,', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

    # inverse multi level Harr - 2d
    rec = conv_ifwt_2d(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('haar 2d scale 5 rec err,', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

def test_2d_db5():
    # max db5
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet('db5')
    coeff2d = conv_fwt_2d(pt_face, wavelet)
    rec = conv_ifwt_2d(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('db 5 scale max rec err,', err, ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4



if __name__ == '__main__':
    test_conv_fwt_haar_lvl2()
    test_conv_fwt_haar_lvl4()
    test_conv_fwt_db2_lvl1()
    test_conv_fwt_db5_lvl3()
    test_orth_wavelet()
    test_2d_haar()
    test_2d_db2()
    test_2d_haar_multi()
    test_2d_db5()