import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import torch


x = torch.load("/home/hschung/ecg/DALLE-pytorch/vae.pt") #LR 2e-4
#x = torch.load("results/vqvae_data_tue_sep_14_00_32_34_2021.pth")
#x = torch.load("results/vqvae_data_tue_sep_14_00_34_17_2021.pth")
print(x)