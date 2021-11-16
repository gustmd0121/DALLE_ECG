import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import pandas as pd
import numpy as np
import torch


# x = torch.load("/home/hschung/ecg/ECG_Training/ptbxl_normalized.pt") #LR 2e-4
#x = torch.load("results/vqvae_data_tue_sep_14_00_32_34_2021.pth")
#x = torch.load("results/vqvae_data_tue_sep_14_00_34_17_2021.pth")
# print(x)

x = pd.read_csv("/home/hschung/ecg/Captioning/RTLP/translations/en_df_round4.csv", index_col="report")

# print(x)

row_iterator = x.iterrows()
for value, row in row_iterator:
    # print(row.ecg_id)
    filename = "/home/hschung/ecg/ECG_Training/ecg_dataset/WFDB_PTBXL/HR" + str(row.ecg_id).zfill(5) + ".txt"
    with open(filename, 'w') as f:
        f.write(str(value))


