import torch
import numpy as np
import pywt
import scipy.misc
from tests.mackey_glass import MackeyGenerator
from src.ptwt.learnable_wavelets import SoftOrthogonalWavelet
from src.ptwt.conv_transform import wavedec, waverec, wavedec2, waverec2
from src.ptwt.conv_transform import flatten_2d_coeff_lst




def test_conv_fwt_haar_lvl2():
    data = [1., 2., 3., 4., 5., 6., 7., 8., 9.,
            10., 11., 12., 13., 14., 15., 16.]
    # npdata = np.array(data)
    ptdata = torch.tensor(data).unsqueeze(0).unsqueeze(0)

    wavelet = pywt.Wavelet('haar')
    coeffs = pywt.wavedec(data, wavelet, level=2)
    coeffs2 = wavedec(ptdata, wavelet, level=2)
    assert len(coeffs) == len(coeffs2)
    err = np.mean(np.abs(np.concatenate(coeffs)
                  - torch.cat(coeffs2, -1).squeeze().numpy()))
    print('haar coefficient error scale 2', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4


def test_conv_fwt_haar_lvl2_odd():
    data = [1., 2., 3., 4., 5., 6., 7., 8., 9.,
            10., 11., 12., 13., 14., 15., 16., 17.]
    ptdata = torch.tensor(data).unsqueeze(0).unsqueeze(0)

    wavelet = pywt.Wavelet('haar')
    coeffs = wavedec(ptdata, wavelet, level=2)
    rec = waverec(coeffs, wavelet)
    err = np.mean(np.abs((ptdata - rec[:, 1:]).numpy()))
    assert err < 1e-4


def test_conv_fwt_haar_lvl4():
    generator = MackeyGenerator(batch_size=24, tmax=512,
                                delta_t=1, device='cpu')
    mackey_data_1 = torch.squeeze(generator())
    wavelet = pywt.Wavelet('haar')
    ptcoeff = wavedec(mackey_data_1.unsqueeze(1), wavelet, level=4)
    pycoeff = pywt.wavedec(mackey_data_1[0, :].numpy(), wavelet, level=4)
    ptcoeff = torch.cat(ptcoeff, -1)[0, :].numpy()
    pycoeff = np.concatenate(pycoeff)
    err = np.mean(np.abs(pycoeff - ptcoeff))
    print('haar coefficient error scale 4:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

    res = waverec(wavedec(mackey_data_1.unsqueeze(1), wavelet), wavelet)
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
    coeffs2 = wavedec(ptdata, wavelet, level=1, mode='reflect')
    ccoeffs = np.concatenate(coeffs, -1)
    ccoeffs2 = torch.cat(coeffs2, -1).numpy()
    err = np.mean(np.abs(ccoeffs - ccoeffs2))
    print('db2 coefficient error scale 1:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4
    rec = waverec(coeffs2, wavelet)
    err = np.mean(np.abs(npdata - rec.numpy()))
    print('db2 reconstruction error scale 1:', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4


def test_conv_fwt_db5_lvl3():
    generator = MackeyGenerator(batch_size=24, tmax=512,
                                delta_t=1, device='cpu')

    mackey_data_1 = torch.squeeze(generator())
    wavelet = pywt.Wavelet('db5')
    for mode in ['reflect', 'zero']:
        ptcoeff = wavedec(mackey_data_1.unsqueeze(1), wavelet, level=3,
                          mode=mode)
        pycoeff = pywt.wavedec(mackey_data_1[0, :].numpy(), wavelet, level=3,
                               mode=mode)
        cptcoeff = torch.cat(ptcoeff, -1)[0, :]
        cpycoeff = np.concatenate(pycoeff, -1)
        err = np.mean(np.abs(cpycoeff - cptcoeff.numpy()))
        print('db5 coefficient error scale 3:', err,
              ['ok' if err < 1e-4 else 'failed!'], 'mode', mode)

        res = waverec(wavedec(mackey_data_1.unsqueeze(1), wavelet, level=3,
                              mode=mode),
                      wavelet)
        err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
        print('db5 reconstruction error scale 3:', err,
              ['ok' if err < 1e-4 else 'failed!'], 'mode', mode)
        assert err < 1e-4

        res = waverec(wavedec(mackey_data_1.unsqueeze(1), wavelet, level=4,
                              mode=mode),
                      wavelet)
        err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
        print('db5 reconstruction error scale 4:', err,
              ['ok' if err < 1e-4 else 'failed!'], 'mode', mode)
        assert err < 1e-4


def test_ripples_haar_lvl3():
    """ Compute example from page 7 of
        Ripples in Mathematics, Jensen, la Cour-Harbo
    """

    class MyHaarFilterBank(object):
        @property
        def filter_bank(self):
            return ([1/2, 1/2.], [-1/2., 1/2.],
                    [1/2., 1/2.], [1/2., -1/2.])

    data = [56., 40., 8., 24., 48., 48., 40., 16.]
    wavelet = pywt.Wavelet('unscaled Haar Wavelet',
                           filter_bank=MyHaarFilterBank())
    ptdata = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    coeffs = wavedec(ptdata, wavelet, level=3)
    # print(coeffs)
    assert torch.squeeze(coeffs[0]).numpy() == 35.
    assert torch.squeeze(coeffs[1]).numpy() == -3.
    assert (torch.squeeze(coeffs[2]).numpy() == [16., 10.]).all()
    assert (torch.squeeze(coeffs[3]).numpy() == [8., -8., 0., 12.]).all()


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
    res = waverec(wavedec(mackey_data_1.unsqueeze(1), orthwave), orthwave)
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
    coeff2d = wavedec2(pt_face, wavelet, level=1)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    err = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    print('haar 2d coeff err,', err, ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

    # single level 2d haar inverse
    rec = waverec2(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('haar 2d rec err', err, ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4


def test_2d_db2():
    # single level db2 - 2d
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet('db2')
    coeff2d_pywt = pywt.dwt2(face, wavelet, mode='reflect')
    coeff2d = wavedec2(pt_face, wavelet, level=1)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    err = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    print('db5 2d coeff err,', err, ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

    # single level db2 - 2d inverse.
    rec = waverec2(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('db5 2d rec err,', err, ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4


def test_2d_haar_multi():
    # multi level haar - 2d
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet('haar')
    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode='reflect', level=5)
    coeff2d = wavedec2(pt_face, wavelet, level=5)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    err = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    # plt.plot(flat_lst); plt.show()
    # plt.plot(flat_lst2); plt.show()
    print('haar 2d scale 5 coeff err,', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

    # inverse multi level Harr - 2d
    rec = waverec2(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('haar 2d scale 5 rec err,', err,
          ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4


def test_2d_db5():
    # max db5
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet('db5')
    coeff2d = wavedec2(pt_face, wavelet)
    rec = waverec2(coeff2d, wavelet)
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
