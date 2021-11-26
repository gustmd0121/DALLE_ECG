import os
import numpy as np, sys, os
from scipy.io import loadmat
import ecg_plot

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    data_mean = np.mean(data, 1, keepdims=True)
    data_std = np.std(data, 1, keepdims=True)
    data -= data_mean
    data /= data_std
    return data

x = load_challenge_data("/home/data_storage/hschung/ECG_Training/ecg_dataset/WFDB_PTBXL/HR00940.mat")


ecg_plot.plot(x, sample_rate=500, title = "Original ECG")
ecg_plot.show()

