# Created by moritz wolter, 14.04.20
import torch
import numpy as np
import pywt
from util.mackey_glass import MackeyGenerator
from util.learnable_wavelets import OrthogonalWavelet
import matplotlib.pyplot as plt


def get_filter_tensors(wavelet, flip):
    def create_tensor(filter):
        if flip:
            if type(filter) is torch.Tensor:
                return filter.flip(-1).unsqueeze(0)
            else:
                return torch.tensor(filter[::-1]).unsqueeze(0)
        else:
            if type(filter) is torch.Tensor:
                return filter.unsqueeze(0)
            else:
                return torch.tensor(filter).unsqueeze(0)
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo = create_tensor(dec_lo)
    dec_hi = create_tensor(dec_hi)
    rec_lo = create_tensor(rec_lo)
    rec_hi = create_tensor(rec_hi)
    return dec_lo, dec_hi, rec_lo, rec_hi


def get_pad(data_len, wavelet):
    """ Compute the required padding.
    :param data: The input tensor.
    :param wavelet: The wavelet filters used.
    :return: The numbers to attach on the edges of the input.
    """
    filt_len = wavelet.__len__()
    pywt_coeff_no = (data_len + filt_len - 1) // 2
    pywt_len = pywt.dwt_coeff_len(data_len, filt_len, mode='reflect')
    assert pywt_coeff_no == pywt_len, 'padding error.'
    pad = (2*filt_len - 3)//2
    if data_len % 2 != 0:
        pad += 1
    pt_coeff_no = (data_len + 2*pad - (filt_len - 1) - 1) // 2 + 1
    assert pt_coeff_no == pywt_len, 'padding error, this is a bug please open an issue on github.'
    return pad


def fwt_pad(data, wavelet):
    """ Pad the input signal to make the fwt matrix work.
    :param data: Input data [batch_size, 1, time]
    :param wavelet: The input wavelet following the pywt wavelet format.
    :return: The padded input data
    """
    # following https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
    pad = get_pad(data.shape[-1], wavelet)

    # print('fwt pad', data.shape, pad)
    data_pad = torch.nn.functional.pad(data, [pad, pad],
                                       mode='reflect')
    return data_pad


def fwt_pad2d(data, wavelet):
    padx = get_pad(data.shape[-2], wavelet)
    pady = get_pad(data.shape[-1], wavelet)
    data_pad = torch.nn.functional.pad(data, [padx, padx, pady, pady],
                                       mode='reflect')
    return data_pad


def outer(a, b):
    """ Torch implementation of numpy's outer for vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul*b_mul


def flatten_2d_coeff_lst(coeff_lst_2d, flatten_tensors=True):
    flat_coeff_lst = []
    for coeff in coeff_lst_2d:
        if type(coeff) is tuple:
            for c in coeff:
                if flatten_tensors:
                    flat_coeff_lst.append(c.flatten())
                else:
                    flat_coeff_lst.append(c)
        else:
            if flatten_tensors:
                flat_coeff_lst.append(coeff.flatten())
            else:
                flat_coeff_lst.append(coeff)
    return flat_coeff_lst


def construct_2d_filt(lo, hi):
    ll = outer(lo, lo)
    lh = outer(hi, lo)
    hl = outer(lo, hi)
    hh = outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    filt = filt.unsqueeze(1)
    return filt


def conv_fwt_2d(data, wavelet, scales: int=None):
    """ 2d non-seperated fwt """
    # dec_lo, dec_hi, _, _ = wavelet.filter_bank
    # filt_len = len(dec_lo)
    # dec_lo = torch.tensor(dec_lo[::-1]).unsqueeze(0)
    # dec_hi = torch.tensor(dec_hi[::-1]).unsqueeze(0)
    dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True)
    filt_len = dec_lo.shape[-1]

    dec_filt = construct_2d_filt(lo=dec_lo, hi=dec_hi)

    if scales is None:
        scales = pywt.dwtn_max_level([data.shape[-1], data.shape[-2]], wavelet)

    result_lst = []
    res_ll = data
    for s in range(scales):
        if filt_len > 2:
            res_ll = fwt_pad2d(res_ll, wavelet)
        res = torch.nn.functional.conv2d(res_ll, dec_filt, stride=2)
        res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, 1)
        result_lst.append((res_lh, res_hl, res_hh))
    result_lst.append(res_ll)
    return result_lst[::-1]


def conv_ifwt_2d(coeffs, wavelet):
    """ 2d non separated ifwt"""
    # _, _, rec_lo, rec_hi = wavelet.filter_bank
    # filt_len = len(rec_lo)
    # rec_lo = torch.tensor(rec_lo).unsqueeze(0)
    # rec_hi = torch.tensor(rec_hi).unsqueeze(0)
    _, _, rec_lo, rec_hi = get_filter_tensors(wavelet, flip=False)
    filt_len = rec_lo.shape[-1]
    rec_filt = construct_2d_filt(rec_lo, rec_hi)

    res_ll = coeffs[0]
    for c_pos, res_lh_hl_hh in enumerate(coeffs[1:]):
        res_ll = torch.cat([res_ll, res_lh_hl_hh[0], res_lh_hl_hh[1], res_lh_hl_hh[2]], 1)
        res_ll = torch.nn.functional.conv_transpose2d(res_ll, rec_filt, stride=2)

        # remove the padding
        padl = (2*filt_len - 3)//2
        padr = (2*filt_len - 3)//2
        padt = (2*filt_len - 3)//2
        padb = (2*filt_len - 3)//2
        if c_pos < len(coeffs)-2:
            pred_len = res_ll.shape[-1] - (padl + padr)
            next_len = coeffs[c_pos+2][0].shape[-1]
            pred_len2 = res_ll.shape[-2] - (padt + padb)
            next_len2 = coeffs[c_pos+2][0].shape[-2]
            if next_len != pred_len:
                padl += 1
                pred_len = res_ll.shape[-1] - (padl + padr)
                assert next_len == pred_len, 'padding error, please open an issue on github '
            if next_len2 != pred_len2:
                padt += 1
                pred_len2 = res_ll.shape[-2] - (padt + padb)
                assert next_len2 == pred_len2, 'padding error, please open an issue on github '
        if padt > 0:
            res_ll = res_ll[..., padt:, :]
        if padb > 0:
            res_ll = res_ll[..., :-padb, :]
        if padl > 0:
            res_ll = res_ll[..., padl:]
        if padr > 0:
            res_ll = res_ll[..., :-padr]
    return res_ll


def conv_fwt(data, wavelet, scales: int = None):
    dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True)
    filt_len = dec_lo.shape[-1]
    # dec_lo = torch.tensor(dec_lo[::-1]).unsqueeze(0)
    # dec_hi = torch.tensor(dec_hi[::-1]).unsqueeze(0)
    filt = torch.stack([dec_lo, dec_hi], 0)

    if scales is None:
        scales = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_lst = []
    res_lo = data
    for s in range(scales):
        #if filt_len > 2:
        res_lo = fwt_pad(res_lo, wavelet)
        res = torch.nn.functional.conv1d(res_lo, filt, stride=2)
        res_lo, res_hi = torch.split(res, 1, 1)
        result_lst.append(res_hi.squeeze(1))
    result_lst.append(res_lo.squeeze(1))
    return result_lst[::-1]


def conv_ifwt(coeffs, wavelet):
    # _, _, rec_lo, rec_hi = wavelet.filter_bank
    # filt_len = len(rec_lo)
    # rec_lo = torch.tensor(rec_lo).unsqueeze(0)
    # rec_hi = torch.tensor(rec_hi).unsqueeze(0)
    _, _, rec_lo, rec_hi = get_filter_tensors(wavelet, flip=False)
    filt_len = rec_lo.shape[-1]

    filt = torch.stack([rec_lo, rec_hi], 0)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        # print('shapes', res_lo.shape, res_hi.shape)
        res_lo = torch.stack([res_lo, res_hi], 1)
        res_lo = torch.nn.functional.conv_transpose1d(res_lo, filt, stride=2).squeeze(1)

        # remove the padding
        padl = (2*filt_len - 3)//2
        padr = (2*filt_len - 3)//2
        if c_pos < len(coeffs)-2:
            pred_len = res_lo.shape[-1] - (padl + padr)
            nex_len = coeffs[c_pos+2].shape[-1]
            if nex_len != pred_len:
                padl += 1
                pred_len = res_lo.shape[-1] - (padl + padr)
                assert nex_len == pred_len, 'padding error, please open an issue on github '
            # ensure correct padding removal.
        # if padl > 0 and padr > 0:
        #     res_lo = res_lo[..., padl:-padr]
        if padl > 0:
            res_lo = res_lo[..., padl:]
        if padr > 0:
            res_lo = res_lo[..., :-padr]
    return res_lo


if __name__ == '__main__':
    data = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]
    npdata = np.array(data)
    ptdata = torch.tensor(data).unsqueeze(0).unsqueeze(0)

    generator = MackeyGenerator(batch_size=24, tmax=512, delta_t=1, device='cpu')

    # -------------------------- Haar wavelet tests ------------------------------ #
    wavelet = pywt.Wavelet('haar')
    coeffs = pywt.wavedec(data, wavelet, level=2)
    coeffs2 = conv_fwt(ptdata, wavelet, scales=2)
    # print(coeffs)
    # print(coeffs2)
    assert len(coeffs) == len(coeffs2)
    err = np.mean(np.abs(np.concatenate(coeffs) - torch.cat(coeffs2, -1).squeeze().numpy()))
    print('haar coefficient error scale 2', err, ['ok' if err < 1e-4 else 'failed!'])

    mackey_data_1 = torch.squeeze(generator())
    wavelet = pywt.Wavelet('haar')
    ptcoeff = conv_fwt(mackey_data_1.unsqueeze(1), wavelet, scales=4)
    pycoeff = pywt.wavedec(mackey_data_1[0, :].numpy(), wavelet, level=4)
    ptcoeff = torch.cat(ptcoeff, -1)[0, :].numpy()
    pycoeff = np.concatenate(pycoeff)
    err = np.mean(np.abs(pycoeff - ptcoeff))
    print('haar coefficient error scale 4:', err, ['ok' if err < 1e-4 else 'failed!'])
    # plt.semilogy(ptcoeff)
    # plt.semilogy(pycoeff)
    # plt.show()

    res = conv_ifwt(conv_fwt(mackey_data_1.unsqueeze(1), wavelet), wavelet)
    err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
    print('haar reconstruction error scale 4:', err, ['ok' if err < 1e-4 else 'failed!'])
    #plt.plot(res[0, :])
    # plt.show()

    # ------------------------- db2 wavelet tests --------------------------------
    wavelet = pywt.Wavelet('db2')
    coeffs = pywt.wavedec(data, wavelet, level=1, mode='reflect')
    coeffs2 = conv_fwt(ptdata, wavelet, scales=1)
    # pywt_len = 16 + wavelet.dec_len - 1
    # print(pywt_len, pywt_len//2)
    # print([c.shape for c in coeffs])
    # print([c.shape for c in coeffs2])
    ccoeffs = np.concatenate(coeffs, -1)
    ccoeffs2 = torch.cat(coeffs2, -1).numpy()
    err = np.mean(np.abs(ccoeffs - ccoeffs2))
    print('db2 coefficient error scale 1:', err, ['ok' if err < 1e-4 else 'failed!'])
    # plt.plot(coeffs)
    # plt.plot(coeffs2.numpy())
    # plt.show()
    # coeffs2_lst = [c.unsqueeze(0) for c in coeffs2]
    rec = conv_ifwt(coeffs2, wavelet)
    err = np.mean(np.abs(npdata - rec.numpy()))
    print('db2 reconstruction error scale 1:', err, ['ok' if err < 1e-4 else 'failed!'])

    mackey_data_1 = torch.squeeze(generator())
    wavelet = pywt.Wavelet('db5')
    ptcoeff = conv_fwt(mackey_data_1.unsqueeze(1), wavelet, scales=3)
    pycoeff = pywt.wavedec(mackey_data_1[0, :].numpy(), wavelet, level=3)
    cptcoeff = torch.cat(ptcoeff, -1)[0, :]
    cpycoeff = np.concatenate(pycoeff, -1)
    err = np.mean(np.abs(cpycoeff - cptcoeff.numpy()))
    print('db5 coefficient error scale 3:', err, ['ok' if err < 1e-4 else 'failed!'])  # fixme!
    # print([c.shape for c in pycoeff])
    # print([cp.shape for cp in cptcoeff])
    # plt.semilogy(cpycoeff)
    # plt.semilogy(cptcoeff.numpy())
    # plt.show()

    res = conv_ifwt(conv_fwt(mackey_data_1.unsqueeze(1), wavelet, scales=3), wavelet)
    err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
    print('db5 reconstruction error scale 3:', err, ['ok' if err < 1e-4 else 'failed!'])

    # print([coef.shape[-1] for coef in conv_fwt(mackey_data_1.unsqueeze(1), wavelet, scales=4)])
    # print([coef.shape[-1] for coef in pywt.wavedec(mackey_data_1[0, :].numpy(), wavelet, level=4, mode='reflect')])
    res = conv_ifwt(conv_fwt(mackey_data_1.unsqueeze(1), wavelet, scales=4), wavelet)
    err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
    # plt.plot(mackey_data_1[0, :])
    # plt.plot(res[0, :])
    # plt.show()
    print('db5 reconstruction error scale 4:', err, ['ok' if err < 1e-4 else 'failed!'])

    # orthogonal wavelet object test
    orthwave = OrthogonalWavelet(torch.tensor(wavelet.rec_lo))
    res = conv_ifwt(conv_fwt(mackey_data_1.unsqueeze(1), orthwave), orthwave)
    err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
    print('orth reconstruction error scale 4:', err, ['ok' if err < 1e-4 else 'failed!'])

    # ------------------------- 2d haar wavelet tests --------------------------------
    import scipy.misc
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

    # single level 2d haar inverse
    rec = conv_ifwt_2d(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('haar 2d rec err', err, ['ok' if err < 1e-4 else 'failed!'])

    # single level db2 - 2d
    wavelet = pywt.Wavelet('db2')
    coeff2d_pywt = pywt.dwt2(face, wavelet, mode='reflect')
    coeff2d = conv_fwt_2d(pt_face, wavelet, scales=1)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    # print([c.shape for c in flatten_2d_coeff_lst(coeff2d_pywt, flatten_tensors=False)])
    # print([c.shape for c in flatten_2d_coeff_lst(coeff2d, flatten_tensors=False)])
    err = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    print('db5 2d coeff err,', err, ['ok' if err < 1e-4 else 'failed!'])

    # single level db2 - 2d inverse.
    rec = conv_ifwt_2d(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('db5 2d rec err,', err, ['ok' if err < 1e-4 else 'failed!'])

    # multi level haar - 2d
    wavelet = pywt.Wavelet('haar')
    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode='reflect', level=5)
    coeff2d = conv_fwt_2d(pt_face, wavelet, scales=5)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    # print([c.shape for c in flatten_2d_coeff_lst(coeff2d_pywt, flatten_tensors=False)])
    # print([list(c.shape) for c in flatten_2d_coeff_lst(coeff2d, flatten_tensors=False)])
    err = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    # plt.plot(flat_lst); plt.show()
    # plt.plot(flat_lst2); plt.show()
    print('haar 2d scale 5 coeff err,', err, ['ok' if err < 1e-4 else 'failed!'])

    # inverse multi level Harr - 2d
    rec = conv_ifwt_2d(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('haar 2d scale 5 rec err,', err, ['ok' if err < 1e-4 else 'failed!'])

    # max db5
    wavelet = pywt.Wavelet('db5')
    coeff2d = conv_fwt_2d(pt_face, wavelet)
    rec = conv_ifwt_2d(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print('db 5 scale max rec err,', err, ['ok' if err < 1e-4 else 'failed!'])
