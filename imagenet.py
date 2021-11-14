import torch
import torchvision


trainset = torchvision.datasets.CIFAR10(root='/home/hschung/ecg/DALLE-pytorch/imagenet/', train=True, download=True)
