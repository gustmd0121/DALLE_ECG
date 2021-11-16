import torch
from utils.physionet_challenge_utility_script import *

x = load_challenge_data("/home/hschung/ecg/ECG_Training/ecg_dataset/WFDB_PTBXL/HR00001.mat")

print(x)