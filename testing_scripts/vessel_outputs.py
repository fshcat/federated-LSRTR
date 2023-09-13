import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

PROJECT_ROOT = "/common/users/rp1110/fishcat/Federated-LSRTR"
sys.path.append(PROJECT_ROOT)

from lsr_tensor import *
from lsr_bcd_regression import *
from federated_tests import *
from medmnist import VesselMNIST3D
import torch.nn.functional as f
from torchvision import transforms
from copy import copy
import numpy as np

def convert(x):
    return x.astype('float64')

if __name__ == "__main__":
    # Load data
    dataset = VesselMNIST3D(split="train", target_transform=convert, download=True)
    val_dataset = VesselMNIST3D(split="val", target_transform=convert, download=True)

    print("t fraction positive: ", sum(dataset[:, 0][1]) / len(dataset[:, 0][1]))
    print("t fraction negative: ", 1 - sum(dataset[:, 0][1]) / len(dataset[:, 0][1]))

    print("v fraction positive: ", sum(val_dataset[:, 0][1]) / len(val_dataset[:, 0][1]))
    print("v fraction negative: ", 1 - sum(val_dataset[:, 0][1]) / len(val_dataset[:, 0][1]))

    path_base = "../data/vessel_tuning_lin"

    name = 'lr10_mom0.9_steps20'
    lsr_ten_dot = LSR_tensor_dot.copy(torch.load(f"{path_base}/{name}/weights/lsr_dot_1.pt"), device=torch.device('cpu'))

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))

    for X, y in val_dataloader:
        y_pred = lsr_ten_dot.forward(X)

        for yp, yt in zip(y_pred, y):
            #yps = yp
            yps = torch.sigmoid(yp)
            #yps = torch.clamp(yp, 0.0, 1.0)
            print(f"{yps[0]} {yt[0]}")

