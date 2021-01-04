# Created by moritz at 14.04.20
"""
Implement an fwt using strang nguyen p. 32
"""

import pywt
import torch
import numpy as np
import time
import tikzplotlib
import matplotlib.pyplot as plt
from util.mackey_glass import MackeyGenerator


def cat_sparse_identity_matrix(sparse_matrix, new_length):
    """ Concatenate a sparse input matrix and a sparse identity matrix.
    :param sparse_matrix: The input matrix.
    :param new_length: The length up to which the diagonal should be elongated.
    :return: Square [input, eye] matrix of size [new_length, new_length]
    """
    # assert square matrix.
    assert sparse_matrix.shape[0] == sparse_matrix.shape[1], 'wavelet matrices are square'
    assert new_length > sparse_matrix.shape[0], 'cant add negatively many entries.'
    x = torch.arange(sparse_matrix.shape[0], new_length)
    y = torch.arange(sparse_matrix.shape[0], new_length)
    extra_indices = torch.stack([x, y])
    extra_values = torch.ones([new_length - sparse_matrix.shape[0]])
    new_indices = torch.cat([sparse_matrix.coalesce().indices(), extra_indices], -1)
    new_values = torch.cat([sparse_matrix.coalesce().values(), extra_values], -1)
    new_matrix = torch.sparse.FloatTensor(new_indices, new_values)
    return new_matrix


# construct the FWT analysis matrix.
def construct_a(wavelet, length, wrap=True):
    """ Construct a sparse matrix to compute a matrix based fwt.
    Following page 31 of the Strang Nguyen Wavelets and Filter Banks book.
    :param wavelet: The wavelet coefficients stored in a wavelet object.
    :param length: The number of entries in the input signal.
    :param wrap: Filter wrap around produces square matrices.
    :return: A the sparse fwt matrix
    """
    dec_lo, dec_hi, _, _ = wavelet.filter_bank
    filt_len = len(dec_lo)
    # right hand filtering and decimation matrix
    # set up the indices.

    h = length//2
    w = length

    # x = []; y = []
    # for i in range(0, h):
    #     for j in range(filt_len):
    #         x.append(i)
    #         y.append((j+2*i) % w)
    # for k in range(0, h):
    #     for j in range(filt_len):
    #         x.append(k + h)
    #         y.append((j+2*k) % w)

    xl = np.stack([np.arange(0, h)]*filt_len).T.flatten()
    yl = np.concatenate([np.arange(0, filt_len)]*h) + 2*xl
    xb = xl + h
    yb = yl
    x = np.concatenate([xl, xb])
    y = np.concatenate([yl, yb])
    if wrap:
        y = y % w
    a_indices = torch.from_numpy(np.stack([x, y]).astype(np.int))
    al_entries = torch.tensor([dec_lo[::-1]]*h).flatten()
    ab_entries = torch.tensor([dec_hi[::-1]]*h).flatten()
    a_entries = torch.cat([al_entries, ab_entries])
    a_ten = torch.sparse.FloatTensor(a_indices, a_entries)
    # left hand filtering and decimation matrix
    return a_ten


def matrix_fwt(data, wavelet, scales: int = None):
    """ Compute the sparse matrix fast wavlet transform.
    :param wavelet: A wavelet object in pywave
    :param data: Batched input data [batch_size, time]
    :param scales: The desired scales up to which to compute the fwt.
    :return: The wavelet coefficients in a single vector.
    """
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    assert len(dec_lo) == len(dec_hi), 'All filters hast have the same length.'
    assert len(dec_hi) == len(rec_lo), 'All filters hast have the same length.'
    assert len(rec_lo) == len(rec_hi), 'All filters hast have the same length.'
    filt_len = len(dec_lo)

    length = data.shape[1]

    if scales is None:
        scales = int(np.log2(length))
    else:
        assert scales > 0, 'scales must be a positive integer.'
    ar = construct_a(wavelet, length)
    if scales == 1:
        coefficients = torch.sparse.mm(ar, data.T)
        return torch.split(coefficients, coefficients.shape[0]//2), [ar]
    al2 = construct_a(wavelet, length//2)
    al2 = cat_sparse_identity_matrix(al2, length)
    if scales == 2:
        coefficients = torch.sparse.mm(al2, torch.sparse.mm(ar, pt_data.T))
        return torch.split(coefficients, [length//4, length//4, length//2]), [ar, al2]
    ar3 = construct_a(wavelet, length//4)
    ar3 = cat_sparse_identity_matrix(ar3, length)
    if scales == 3:
        coefficients = torch.sparse.mm(ar3, torch.sparse.mm(al2, torch.sparse.mm(ar, pt_data.T)))
        return torch.split(coefficients, [length//8, length//8, length//4, length//2]), [ar, al2, ar3]
    fwt_mat_lst = [ar, al2, ar3]
    split_lst = [length//2, length//4, length//8]
    for s in range(4, scales+1):
        if split_lst[-1] < filt_len:
            break
        an = construct_a(wavelet, split_lst[-1])
        an = cat_sparse_identity_matrix(an, length)
        fwt_mat_lst.append(an)
        new_split_size = length//np.power(2, s)
        split_lst.append(new_split_size)
    coefficients = data.T
    for fwt_mat in fwt_mat_lst:
        coefficients = torch.sparse.mm(fwt_mat, coefficients)
    split_lst.append(length//np.power(2, scales))
    return torch.split(coefficients, split_lst[::-1]), fwt_mat_lst


def construct_s(wavelet, length, wrap=True):
    # construct the FWT synthesis matrix.
    _, _, rec_lo, rec_hi = wavelet.filter_bank
    filt_len = len(rec_lo)
    # right hand filtering and decimation matrix
    # set up the indices.
    h = length//2
    w = length
    yl = np.stack([np.arange(0, h)]*filt_len).T.flatten()
    xl = np.concatenate([np.arange(0, filt_len)]*h) + 2*yl
    xb = xl
    yb = yl + h
    x = np.concatenate([xl, xb])
    y = np.concatenate([yl, yb])
    if wrap:
        x = x % w
    s_indices = torch.from_numpy(np.stack([x, y]).astype(np.int))
    sl_entries = torch.tensor([rec_lo]*h).flatten()
    sb_entries = torch.tensor([rec_hi]*h).flatten()
    s_entries = torch.cat([sl_entries, sb_entries])
    s_ten = torch.sparse.FloatTensor(s_indices, s_entries)
    # left hand filtering and decimation matrix
    return s_ten


def matrix_ifwt(coefficients, wavelet, scales: int = None):
    _, _, rec_lo, rec_hi = wavelet.filter_bank

    # if the coefficients come in a list concatenate!
    if type(coefficients) is tuple:
        coefficients = torch.cat(coefficients, 0)

    filt_len = len(rec_lo)
    #if filt_len > 2:
    #    data = fwt_pad(data, wavelet)
    length = coefficients.shape[0]

    if scales is None:
        scales = int(np.log2(length))
    else:
        assert scales > 0, 'scales must be a positive integer.'

    ifwt_mat_lst = []
    split_lst = [length]
    for s in range(1, scales+1):
        if split_lst[-1] < filt_len:
            break
        sn = construct_s(wavelet, split_lst[-1])
        if s > 1:
            sn = cat_sparse_identity_matrix(sn, length)
        ifwt_mat_lst.append(sn)
        new_split_size = length//np.power(2, s)
        split_lst.append(new_split_size)
    reconstruction = coefficients
    for ifwt_mat in ifwt_mat_lst[::-1]:
        reconstruction = torch.sparse.mm(ifwt_mat, reconstruction)
    return reconstruction.T, ifwt_mat_lst[::-1]


if __name__ == '__main__':
    # ---------------------------- matrix construction tests --------------------------#
    a_db1 = construct_a(pywt.Wavelet('db1'), 8)
    s_db1 = construct_s(pywt.Wavelet('db1'), 8)
    err = np.mean(np.abs(torch.sparse.mm(a_db1, s_db1.to_dense()).numpy() - np.eye(8)))
    print('db1 8 inverse error', err)

    a_db2 = construct_a(pywt.Wavelet('db2'), 64, wrap=True)
    s_db2 = construct_s(pywt.Wavelet('db2'), 64, wrap=True)
    test_eye = torch.sparse.mm(a_db2, s_db2.to_dense()).numpy()
    err = np.mean(np.abs(test_eye - np.eye(64)))
    print('db2 64 inverse error', err)
    # plt.imshow(a_db2.to_dense()); plt.show()
    # plt.imshow(s_db2.to_dense()); plt.show()
    # plt.imshow(test_eye); plt.show()

    a_db2_66 = construct_a(pywt.Wavelet('db2'), 66, wrap=True)
    s_db2_66 = construct_s(pywt.Wavelet('db2'), 66, wrap=True)
    test_eye = torch.sparse.mm(a_db2_66, s_db2_66.to_dense()).numpy()
    err = np.mean(np.abs(test_eye - np.eye(66)))
    print('db2 66 inverse error', err)

    a_db2 = construct_a(pywt.Wavelet('db8'), 128)
    s_db2 = construct_s(pywt.Wavelet('db8'), 128)
    test_eye = torch.sparse.mm(a_db2, s_db2.to_dense()).numpy()
    err = np.mean(np.abs(test_eye - np.eye(128)))
    print('db8 128 inverse error', err)

    # ----------------------------------- Haar fwt-ifwt tests --------------------------------------
    wavelet = pywt.Wavelet('haar')
    data2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    data = np.array([1, 2, 3, 4])

    # level 1
    coeffs = pywt.dwt(data2, wavelet)
    print(coeffs[0], coeffs[1])
    # AR = construct_ar(wavelet, data2.shape[0])
    pt_data = torch.from_numpy(data2.astype(np.float32)).unsqueeze(0)
    coeffsmat1, _ = matrix_fwt(pt_data, wavelet, 1)
    print(coeffsmat1[0].T.numpy(), coeffsmat1[1].T.numpy())
    print(np.sum(np.abs(coeffs[0] - coeffsmat1[0].squeeze().numpy())) < 0.00001,
          np.sum(np.abs(coeffs[1] - coeffsmat1[1].squeeze().numpy())) < 0.00001)

    coeffs2 = pywt.wavedec(data2, wavelet, level=2)
    coeffsmat2, _ = matrix_fwt(pt_data, wavelet, 2)
    print(coeffs2[0], coeffs2[1], coeffs2[2])
    print(coeffsmat2[0].T.numpy().astype(np.float32),
          coeffsmat2[1].T.numpy().astype(np.float32),
          coeffsmat2[2].T.numpy().astype(np.float32))
    print(np.sum(np.abs(coeffs2[0] - coeffsmat2[0].squeeze().numpy())) < 0.00001,
          np.sum(np.abs(coeffs2[1] - coeffsmat2[1].squeeze().numpy())) < 0.00001,
          np.sum(np.abs(coeffs2[2] - coeffsmat2[2].squeeze().numpy())) < 0.00001)

    coeffs3 = pywt.wavedec(data2, wavelet, level=3)
    coeffsmat3, mat_3_lst = matrix_fwt(pt_data, wavelet, 3)
    print(np.sum(np.abs(coeffs3[0] - coeffsmat3[0].squeeze().numpy())) < 0.00001,
          np.sum(np.abs(coeffs3[1] - coeffsmat3[1].squeeze().numpy())) < 0.00001,
          np.sum(np.abs(coeffs3[2] - coeffsmat3[2].squeeze().numpy())) < 0.00001,
          np.sum(np.abs(coeffs3[3] - coeffsmat3[3].squeeze().numpy())) < 0.00001)


    reconstructed_data, ifwt_mat_3_lst = matrix_ifwt(coeffsmat3, wavelet, 3)
    print('abs ifwt 3 reconstruction error', torch.mean(torch.abs(pt_data - reconstructed_data)))

    plot = False
    save = False
    if plot:
        operator_mat = mat_3_lst[0].to_dense()
        inv_operator_mat = ifwt_mat_3_lst[0].to_dense()
        for mat_no in range(1, len(mat_3_lst)): # mat_lst[1:]:
            operator_mat = torch.sparse.mm(mat_3_lst[mat_no], operator_mat)
            inv_operator_mat = torch.sparse.mm(ifwt_mat_3_lst[mat_no], inv_operator_mat)

        #  ---- fwt sparse matrix viz.
        import scipy.sparse as sparse
        for mat_no, wvl_mat in enumerate(mat_3_lst):
            plt.spy(sparse.csr_matrix(wvl_mat.to_dense().cpu().numpy()))
            if save:
                tikzplotlib.save('wvl_mat' + str(mat_no) + '.tex', standalone=True)
            plt.show()
        #
        # plt.imshow(operator_mat.cpu().numpy())
        # plt.show()
        plt.spy(sparse.csr_matrix(operator_mat.cpu().numpy()))
        if save:
            tikzplotlib.save('wvl_mat' + str(len(mat_3_lst)) + '.tex', standalone=True)
        plt.show()

        # ---- ifwt sparse matrix viz
        for imat_no, iwvl_mat in enumerate(ifwt_mat_3_lst):
            plt.spy(sparse.csr_matrix(iwvl_mat.to_dense().cpu().numpy()))
            if save:
                tikzplotlib.save('iwvl_mat' + str(imat_no) + '.tex', standalone=True)
            plt.show()
        plt.spy(sparse.csr_matrix(inv_operator_mat.cpu().numpy()))
        if save:
            tikzplotlib.save('iwvl_mat' + str(len(mat_3_lst)) + '.tex', standalone=True)
        plt.show()



    pd = {}
    pd['delta_t'] = 1
    pd['tmax'] = 1024
    pd['batch_size'] = 24

    generator = MackeyGenerator(batch_size=pd['batch_size'],
                                tmax=pd['tmax'],
                                delta_t=pd['delta_t'])
    wavelet = pywt.Wavelet('haar')
    pt_data = torch.squeeze(generator()).cpu()
    numpy_data = pt_data.cpu().numpy()
    pywt_start = time.time()
    coeffs_max = pywt.wavedec(numpy_data, wavelet, level=9)
    time_pywt = time.time() - pywt_start
    sparse_fwt_start = time.time()
    coeffs_mat_max, mat_lst = matrix_fwt(pt_data, wavelet, 9)
    time_sparse_fwt = time.time() - sparse_fwt_start

    test_lst = []
    for test_no in range(9):
        test_lst.append(np.sum(np.abs(coeffs_max[test_no]
                                      - coeffs_mat_max[test_no].T.numpy())) < 0.001)
    print(test_lst)
    print('time pywt', time_pywt)
    print('time_sparse_wt', time_sparse_fwt)

    # test the inverse fwt.
    reconstructed_data, ifwt_mat_lst = matrix_ifwt(coeffs_mat_max, wavelet, 9)
    print('abs ifwt reconstruction error', torch.mean(torch.abs(pt_data - reconstructed_data)))
    plt.plot(reconstructed_data[0, :].numpy())
    plt.show()

    operator_mat = mat_lst[0].to_dense()
    inv_operator_mat = ifwt_mat_lst[0].to_dense()
    for mat_no in range(1, len(mat_lst)): # mat_lst[1:]:
        operator_mat = torch.sparse.mm(mat_lst[mat_no], operator_mat)
        inv_operator_mat = torch.sparse.mm(ifwt_mat_lst[mat_no], inv_operator_mat)

    plt.imshow(torch.mm(inv_operator_mat, operator_mat).numpy()); plt.show()
    coeff = torch.sparse.mm(operator_mat, pt_data.T)
    data_rec = torch.sparse.mm(inv_operator_mat, coeff).T
    print('abs. reconstruction error:', np.mean(np.abs(data_rec.numpy() - numpy_data)))
    plt.plot(torch.sparse.mm(inv_operator_mat, torch.cat(coeffs_mat_max, 0)).T.numpy()[0, :])
    plt.show()

    import scipy.sparse as sparse
    plt.imshow(operator_mat.cpu().numpy())
    plt.show()
    plt.spy(sparse.csr_matrix(operator_mat.cpu().numpy()))
    plt.show()

    # ------------ db2 fwt-ifwt tests --------------------------------------
    pd = {}
    pd['delta_t'] = 1
    pd['tmax'] = 512
    pd['batch_size'] = 24
    wavelet = pywt.Wavelet('db2')
    generator = MackeyGenerator(batch_size=pd['batch_size'],
                                tmax=pd['tmax'],
                                delta_t=pd['delta_t'])
    pt_data = torch.squeeze(generator()).cpu()
    numpy_data = pt_data.cpu().numpy()
    coeffs_max = pywt.wavedec(numpy_data, wavelet, level=4)
    coeffs_mat_max, mat_lst = matrix_fwt(pt_data, wavelet, 4)

    plt.plot(coeffs_max[0][0, 1:])
    plt.plot(coeffs_mat_max[0].T.numpy()[0, :]); plt.show()
    reconstructed_data, ifwt_mat = matrix_ifwt(coeffs_mat_max, wavelet, 4)
    print('reconstruction error:', torch.mean(torch.abs(pt_data - reconstructed_data)))
