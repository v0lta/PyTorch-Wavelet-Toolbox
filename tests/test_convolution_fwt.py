import numpy as np
import pywt
import scipy.misc
import torch

from ptwt.conv_transform import (
    flatten_2d_coeff_lst,
    outer,
    wavedec,
    wavedec2,
    waverec,
    waverec2,
)
from ptwt.learnable_wavelets import SoftOrthogonalWavelet
from src.ptwt.mackey_glass import MackeyGenerator


def test_conv_fwt_haar_lvl2():
    data = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    # npdata = np.array(data)
    ptdata = torch.tensor(data).unsqueeze(0).unsqueeze(0)

    wavelet = pywt.Wavelet("haar")
    coeffs = pywt.wavedec(data, wavelet, level=2)
    coeffs2 = wavedec(ptdata, wavelet, level=2)
    assert len(coeffs) == len(coeffs2)
    err = np.mean(
        np.abs(np.concatenate(coeffs) - torch.cat(coeffs2, -1).squeeze().numpy())
    )
    print("haar coefficient error scale 2", err, ["ok" if err < 1e-4 else "failed!"])
    assert err < 1e-4


def test_conv_fwt_haar_lvl2_odd():
    data = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
    ]
    ptdata = torch.tensor(data).unsqueeze(0).unsqueeze(0)

    wavelet = pywt.Wavelet("haar")

    pycoeff = pywt.wavedec(data, wavelet, level=2, mode="reflect")
    cpycoeff = np.concatenate(pycoeff)
    ptcoeff = wavedec(ptdata, wavelet, level=2, mode="reflect")
    cptcoeff = torch.cat(ptcoeff, -1)[0, :].numpy()

    coeff_erorr = np.mean(np.abs(cptcoeff - cpycoeff))
    assert coeff_erorr < 1e-4
    rec = waverec(ptcoeff, wavelet)
    err = np.mean(np.abs((ptdata - rec[:, :-1]).numpy()))
    assert err < 1e-4


def test_conv_fwt_haar_lvl4():
    generator = MackeyGenerator(batch_size=8, tmax=128, delta_t=1, device="cpu")
    mackey_data_1 = torch.squeeze(generator())
    wavelet = pywt.Wavelet("haar")
    ptcoeff = wavedec(mackey_data_1.unsqueeze(1), wavelet, level=4)
    pycoeff = pywt.wavedec(mackey_data_1[0, :].numpy(), wavelet, level=4)
    ptcoeff = torch.cat(ptcoeff, -1)[0, :].numpy()
    pycoeff = np.concatenate(pycoeff)
    err = np.mean(np.abs(pycoeff - ptcoeff))
    print("haar coefficient error scale 4:", err, ["ok" if err < 1e-4 else "failed!"])
    assert err < 1e-4

    res = waverec(wavedec(mackey_data_1.unsqueeze(1), wavelet), wavelet)
    err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
    print(
        "haar reconstruction error scale 4:", err, ["ok" if err < 1e-4 else "failed!"]
    )
    assert err < 1e-4


def test_conv_fwt_db2_lvl1():
    data = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    npdata = np.array(data)
    ptdata = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    # ------------------------- db2 wavelet tests ----------------------------
    wavelet = pywt.Wavelet("db2")
    coeffs = pywt.wavedec(data, wavelet, level=1, mode="reflect")
    coeffs2 = wavedec(ptdata, wavelet, level=1, mode="reflect")
    ccoeffs = np.concatenate(coeffs, -1)
    ccoeffs2 = torch.cat(coeffs2, -1).numpy()
    err = np.mean(np.abs(ccoeffs - ccoeffs2))
    print("db2 coefficient error scale 1:", err, ["ok" if err < 1e-4 else "failed!"])
    assert err < 1e-4
    rec = waverec(coeffs2, wavelet)
    err = np.mean(np.abs(npdata - rec.numpy()))
    print("db2 reconstruction error scale 1:", err, ["ok" if err < 1e-4 else "failed!"])
    assert err < 1e-4


def test_conv_fwt_db5_lvl3():
    generator = MackeyGenerator(batch_size=8, tmax=128, delta_t=1, device="cpu")

    mackey_data_1 = torch.squeeze(generator())
    wavelet = pywt.Wavelet("db5")
    for mode in ["reflect", "zero"]:
        ptcoeff = wavedec(mackey_data_1.unsqueeze(1), wavelet, level=3, mode=mode)
        pycoeff = pywt.wavedec(mackey_data_1[0, :].numpy(), wavelet, level=3, mode=mode)
        cptcoeff = torch.cat(ptcoeff, -1)[0, :]
        cpycoeff = np.concatenate(pycoeff, -1)
        err = np.mean(np.abs(cpycoeff - cptcoeff.numpy()))
        print(
            "db5 coefficient error scale 3:",
            err,
            ["ok" if err < 1e-4 else "failed!"],
            "mode",
            mode,
        )

        res = waverec(
            wavedec(mackey_data_1.unsqueeze(1), wavelet, level=3, mode=mode), wavelet
        )
        err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
        print(
            "db5 reconstruction error scale 3:",
            err,
            ["ok" if err < 1e-4 else "failed!"],
            "mode",
            mode,
        )
        assert err < 1e-4

        res = waverec(
            wavedec(mackey_data_1.unsqueeze(1), wavelet, level=4, mode=mode), wavelet
        )
        err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
        print(
            "db5 reconstruction error scale 4:",
            err,
            ["ok" if err < 1e-4 else "failed!"],
            "mode",
            mode,
        )
        assert err < 1e-4


def test_conv_fwt():
    generator = MackeyGenerator(batch_size=8, tmax=128, delta_t=1, device="cpu")

    mackey_data_1 = torch.squeeze(generator())
    for level in [1, 2, 3, 4, 5, 6, None]:
        for wavelet_string in ["db1", "db2", "db3", "db4", "db5"]:
            for mode in ["reflect", "zero"]:
                wavelet = pywt.Wavelet("db5")
                ptcoeff = wavedec(
                    mackey_data_1.unsqueeze(1), wavelet, level=3, mode=mode
                )
                pycoeff = pywt.wavedec(
                    mackey_data_1[0, :].numpy(), wavelet, level=3, mode=mode
                )
                cptcoeff = torch.cat(ptcoeff, -1)[0, :]
                cpycoeff = np.concatenate(pycoeff, -1)
                err = np.mean(np.abs(cpycoeff - cptcoeff.numpy()))
                print(
                    "db5 coefficient error scale 3:",
                    err,
                    ["ok" if err < 1e-4 else "failed!"],
                    "mode",
                    mode,
                )

                res = waverec(
                    wavedec(mackey_data_1.unsqueeze(1), wavelet, level=3, mode=mode),
                    wavelet,
                )
                err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
                print(
                    "db5 reconstruction error scale 3:",
                    err,
                    ["ok" if err < 1e-4 else "failed!"],
                    "mode",
                    mode,
                )
                assert err < 1e-4

                res = waverec(
                    wavedec(mackey_data_1.unsqueeze(1), wavelet, level=4, mode=mode),
                    wavelet,
                )
                err = torch.mean(torch.abs(mackey_data_1 - res)).numpy()
                print(
                    "db5 reconstruction error scale 4:",
                    err,
                    ["ok" if err < 1e-4 else "failed!"],
                    "mode",
                    mode,
                )
                assert err < 1e-4


def test_ripples_haar_lvl3():
    """Compute example from page 7 of
    Ripples in Mathematics, Jensen, la Cour-Harbo
    """

    class MyHaarFilterBank(object):
        @property
        def filter_bank(self):
            return (
                [1 / 2, 1 / 2.0],
                [-1 / 2.0, 1 / 2.0],
                [1 / 2.0, 1 / 2.0],
                [1 / 2.0, -1 / 2.0],
            )

    data = [56.0, 40.0, 8.0, 24.0, 48.0, 48.0, 40.0, 16.0]
    wavelet = pywt.Wavelet("unscaled Haar Wavelet", filter_bank=MyHaarFilterBank())
    ptdata = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    coeffs = wavedec(ptdata, wavelet, level=3)
    # print(coeffs)
    assert torch.squeeze(coeffs[0]).numpy() == 35.0
    assert torch.squeeze(coeffs[1]).numpy() == -3.0
    assert (torch.squeeze(coeffs[2]).numpy() == [16.0, 10.0]).all()
    assert (torch.squeeze(coeffs[3]).numpy() == [8.0, -8.0, 0.0, 12.0]).all()


def test_orth_wavelet():
    generator = MackeyGenerator(batch_size=8, tmax=128, delta_t=1, device="cpu")

    mackey_data_1 = torch.squeeze(generator())
    # orthogonal wavelet object test
    wavelet = pywt.Wavelet("db5")
    orthwave = SoftOrthogonalWavelet(
        torch.tensor(wavelet.rec_lo),
        torch.tensor(wavelet.rec_hi),
        torch.tensor(wavelet.dec_lo),
        torch.tensor(wavelet.dec_hi),
    )
    res = waverec(wavedec(mackey_data_1.unsqueeze(1), orthwave), orthwave)
    err = torch.mean(torch.abs(mackey_data_1 - res.detach())).numpy()
    print(
        "orth reconstruction error scale 4:", err, ["ok" if err < 1e-4 else "failed!"]
    )
    assert err < 1e-4


def test_2d_haar_lvl1():
    # ------------------------- 2d haar wavelet tests -----------------------
    face = np.transpose(scipy.misc.face()[128:(512+128), 256:(512+256)],
                        [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet("haar")

    # single level haar - 2d
    coeff2d_pywt = pywt.dwt2(face, wavelet, mode="reflect")
    coeff2d = wavedec2(pt_face, wavelet, level=1, mode="reflect")
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    cerr = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    print("haar 2d coeff err,", cerr, ["ok" if cerr < 1e-4 else "failed!"])
    assert cerr < 1e-4

    # single level 2d haar inverse
    # ptwt_rec = pywt.waverec2(coeff2d_pywt, wavelet)
    rec = waverec2(coeff2d, wavelet).numpy().squeeze()
    err_img = np.abs(face - rec)
    err = np.mean(err_img)
    # err2 = np.mean(np.abs(face-ptwt_rec))
    print("haar 2d rec err", err, ["ok" if err < 1e-4 else "failed!"])
    assert err < 1e-4


def test_2d_db2_lvl1():
    # single level db2 - 2d
    face = np.transpose(scipy.misc.face()[128:(512+128), 256:(512+256)],
                        [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet("db2")
    coeff2d_pywt = pywt.dwt2(face, wavelet, mode="reflect")
    coeff2d = wavedec2(pt_face, wavelet, level=1)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    cerr = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    print("db5 2d coeff err,", cerr, ["ok" if cerr < 1e-4 else "failed!"])
    assert cerr < 1e-4

    # single level db2 - 2d inverse.
    rec = waverec2(coeff2d, wavelet)
    err = np.mean(np.abs(face - rec.numpy().squeeze()))
    print("db5 2d rec err,", err, ["ok" if err < 1e-4 else "failed!"])
    assert err < 1e-4


def test_2d_haar_multi():
    # multi level haar - 2d
    face = np.transpose(scipy.misc.face()[128:(512+128), 256:(512+256)],
                        [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet("haar")
    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode="reflect", level=5)
    coeff2d = wavedec2(pt_face, wavelet, level=5)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    cerr = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    # plt.plot(flat_lst); plt.show()
    # plt.plot(flat_lst2); plt.show()
    print("haar 2d scale 5 coeff err,", cerr, ["ok" if cerr < 1e-4 else "failed!"])
    assert cerr < 1e-4

    # inverse multi level Harr - 2d
    rec = waverec2(coeff2d, wavelet).numpy().squeeze()
    err = np.mean(np.abs(face - rec))
    print("haar 2d scale 5 rec err,", err, ["ok" if err < 1e-4 else "failed!"])
    assert err < 1e-4


def test_outer():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    b = torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0])
    res_t = outer(a, b)
    res_np = np.outer(a.numpy(), b.numpy())
    assert np.sum(np.abs(res_t.numpy() - res_np)) < 1e-7


def test_2d_wavedec_rec():
    # max db5
    for level in [1, 2, 3, 4, 5, None]:
        for wavelet_str in ["db2", "db3", "db4", "db5"]:

            face = np.transpose(scipy.misc.face()[128:(512+128), 256:(512+256)],
                                [2, 0, 1]).astype(np.float32)
            pt_face = torch.tensor(face).unsqueeze(1)
            wavelet = pywt.Wavelet(wavelet_str)
            coeff2d = wavedec2(pt_face, wavelet, mode="reflect", level=level)
            pywt_coeff2d = pywt.wavedec2(face, wavelet, mode="reflect", level=level)
            for pos, coeffs in enumerate(pywt_coeff2d):
                if type(coeffs) is tuple:
                    for tuple_pos, tuple_el in enumerate(coeffs):
                        assert (
                            tuple_el.shape
                            == torch.squeeze(coeff2d[pos][tuple_pos], 1).shape
                        ), "pywt and ptwt should produce the same shapes."
                else:
                    assert (
                        coeffs.shape == torch.squeeze(coeff2d[pos], 1).shape
                    ), "pywt and ptwt should produce the same shapes."
            flat_lst = np.concatenate(flatten_2d_coeff_lst(pywt_coeff2d), -1)
            flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
            cerr = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
            # plt.plot(flat_lst); plt.show()
            # plt.plot(flat_lst2); plt.show()
            print(
                "wavelet",
                wavelet_str,
                "level",
                str(level),
                "coeff err,",
                cerr,
                ["ok" if cerr < 1e-4 else "failed!"],
            )
            assert cerr < 1e-4

            rec = waverec2(coeff2d, wavelet)
            rec = rec.numpy().squeeze()
            err_img = np.abs(face - rec)
            err = np.mean(err_img)
            print(
                "wavelet",
                wavelet_str,
                "level",
                str(level),
                "rec   err,",
                err,
                ["ok" if err < 1e-4 else "failed!"],
            )
            assert err < 1e-4


if __name__ == "__main__":
    test_conv_fwt()
    test_2d_wavedec_rec()
