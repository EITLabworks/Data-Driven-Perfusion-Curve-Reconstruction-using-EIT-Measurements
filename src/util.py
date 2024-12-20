import os
from os.path import join
from typing import Union
from glob import glob

from scipy import signal
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import numpy as np


from scipy.signal import butter, filtfilt


def lowpass_filter(data, cutoff=10, fs=1000, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


# def normalize_aorta(y: np.ndarray, invert: bool = False):
#    y = np.array(y)
#    if invert:
#        y = y * 20.0 + 85.0
#    else:
#        y = (y - 85.0) / 20.0
#    return y


class AortaNormalizer:
    # TBD: min-max scaling, different scalings?
    def __init__(self, mode="old"):
        self.mode = mode
        # self.a_min = np.min(Y_true)
        # self.a_max = np.max(Y_true)
        # self.scale_factor = 1 / (self.a_max - self.a_min)

        # print(f"Init normalizer with\n min={self.a_min:.2f}, max={self.a_max:.2f}, scaling={self.scale_factor:.2f}.")

    def normalize_inverse(self, Y_norm):
        if self.mode == "old":
            Y_true = Y_norm * 20 + 85
        # else:
        #    Y_true = (Y_norm/self.scale_factor) + self.a_min
        return Y_true

    def update_scaling(self, aorta_min, aorta_max):
        self.a_min = aorta_min
        self.a_max = aorta_min
        self.scale_factor = 1 / (aorta_max - aorta_min)
        print(
            f"Updated normalizer to\n min={self.a_min:.2f}, max={self.a_max:.2f}, scaling={self.scale_factor:.2f}."
        )

    def normalize_forward(self, Y_true):
        if self.mode == "old":
            Y_norm = (Y_true - 85) / 20
        # else:
        #    Y_norm = (Y_true - self.a_min) * self.scale_factor
        return Y_norm


class DataLoader:
    def __init__(
        self,
        path: str,
        eit_length: int = 64,
        aorta_length: int = 1024,
        norm_eit: str = "block",
    ):
        self.path = path

        self.eit_length = eit_length
        self.aorta_length = aorta_length
        self.norm_eit = norm_eit
        self.init_path()

    def init_path(self):
        self.pigs_available = np.sort(os.listdir(self.path))
        self.pigs_nr_available = [int(pig.split("_")[1]) for pig in self.pigs_available]
        print(f"Data path: {self.path} contains {len(self.pigs_available)} pigs:")
        for pig in self.pigs_available:
            print(f"⋅ {pig}")

    def load_data(self, pig_nr: Union[int, list], shuffle: bool = True):
        if isinstance(pig_nr, list):
            n_pig_nr = len(pig_nr)
            print(f"Got a list of {n_pig_nr} pigs to load.")
        else:
            n_pig_nr = 1
            pig_nr = [pig_nr]
        pig_nr = np.array(pig_nr)
        for n in pig_nr:
            assert (
                n in self.pigs_nr_available
            ), f"Pig No. {n} is not available in {self.path}."

        self.pig_s_load_list = self.pigs_available[
            pig_nr - 1
        ]  # ["P_{0:02d}_PulHyp".format(i) for i in [pig_nr]]
        print("To load", self.pig_s_load_list)

        # initialize data lists
        X = list()
        Y = list()
        pigs = list()

        for pig in self.pig_s_load_list:
            X, Y, pigs = load_examples(X, Y, pigs, join(self.path, pig))

        # quality check
        rm_idx = quality_checks(X, Y, self.eit_length, self.aorta_length)
        for idx in sorted(rm_idx, reverse=True):
            del X[idx]
            del Y[idx]
            del pigs[idx]

        # shuffle
        N = len(Y)
        if shuffle:
            shuffle = np.arange(N)
            np.random.shuffle(shuffle)
        else:
            shuffle = np.arange(N)

        # resample eit signals to equal length
        X = resample_eit(X, self.eit_length)
        # normalize EIT signals
        X = normalize_eit(X, np.array(pigs), self.norm_eit)

        # resample/interpolate aorta signals to equal length aorta_length
        Y = [signal.resample(sample, self.aorta_length) for sample in Y]
        # aorta pressure curves without normalization
        Y = np.array(Y)

        # append empty axis for CNNs
        X = X[:, :, :, np.newaxis]
        Y = Y[..., np.newaxis]

        pigs = np.array(pigs)

        print(f"\nAorta curve min={np.min(Y):.2f}, max={np.max(Y):.2f}")
        print(f"Eit.shape={X.shape:}, Y.shape={Y.shape}")

        return X[shuffle, ...], Y[shuffle, ...], pigs[shuffle, ...]


def load_examples(X: list, y: list, pigs: list, path: str):
    print(f"Loading data from {path}")
    files = glob(join(path, "*.npz"), recursive=True)
    if len(files) == 0:
        raise Exception("No npz files found in directory")

    for filepath in files:
        tmp = np.load(filepath)
        X.append(tmp["eit_v"])
        y.append(tmp["aorta"])
        pigs.append(tmp["data_info"])

    return X, y, pigs


def quality_checks(X: list, y: list, eit_length=64, aorta_length=1024) -> list:
    idx = list()

    # quality checks for EIT data
    for n, eit in enumerate(X):
        if eit.shape[0] > eit_length or eit.shape[0] == 0:
            # print(f"dataset excluded: {eit.shape[0]=}>{eit_length}.")
            idx.append(n)

    # quality checks for aorta data
    for n, aorta in enumerate(y):
        if len(aorta) > aorta_length:
            # print(f"dataset excluded: {aorta.shape[0]=}>{aorta_length}.")
            idx.append(n)

    return idx


def resample_eit(X: list, eit_length: int = 64) -> np.ndarray:
    num_cores = 48
    eit_frame_length = X[0].shape[1]

    def worker(eit_arr, length, eit_frame_length):
        return np.array(
            [signal.resample(eit_arr[:, j], length) for j in range(eit_frame_length)]
        ).T

    X = Parallel(n_jobs=num_cores)(
        delayed(worker)(eit_block, eit_length, eit_frame_length) for eit_block in X
    )
    X = np.array(X)
    return X


def normalize_eit(X: np.ndarray, pigs: np.ndarray, norm_eit: str) -> np.ndarray:
    if norm_eit == "global":
        mx = np.mean(X, axis=(0, 1))
        sx = np.std(X, axis=(0, 1))
        X = (X - mx) / sx

    elif norm_eit == "block":
        le_p = LabelEncoder()
        le_b = LabelEncoder()
        le_p.fit(pigs[:, 0])

        for p in le_p.classes_:
            idx_p = np.where(pigs[:, 0] == p)
            le_b.fit(np.squeeze(pigs[idx_p, 1]))

            for b in le_b.classes_:
                idx = np.where((pigs[:, 0] == p) & (pigs[:, 1] == b))

                mx = np.mean(X[idx, :, :], axis=(0, 1))
                sx = np.std(X[idx, :, :], axis=(0, 1))
                X[idx, :, :] = (X[idx, :, :] - mx) / sx
    return X


def Pearson_correlation(Y_true, Y_pred) -> list:
    """
    The Pearson correlation coefficient [1]_ measures the linear relationship
    between two datasets. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear relationship.
    Positive correlations imply that as x increases, so does y. Negative
    correlations imply that as x increases, y decreases.

    Return:
    list of Pearson correlation coefficients
    """
    corr_number = list()
    for true, pred in zip(Y_true, Y_pred):
        p_nr, _ = pearsonr(true, pred)
        corr_number.append(p_nr)
    corr_number = np.array(corr_number)
    print("Pearson correlation coefficient mean", np.mean(corr_number))
    return corr_number
