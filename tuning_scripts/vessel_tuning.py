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

def reverse(x):
    return 1 - x


def convert(x):
    return x
    #return x.astype(np.float64)

if __name__ == "__main__":
    # Load data
    vessel_dataset = VesselMNIST3D(split="train", target_transform=convert, download=True)
    vessel_val_dataset = VesselMNIST3D(split="val", target_transform=convert, download=True)

    print("t fraction positive: ", sum(vessel_dataset[:, 0][1]) / len(vessel_dataset[:, 0][1]))
    print("t fraction negative: ", 1 - sum(vessel_dataset[:, 0][1]) / len(vessel_dataset[:, 0][1]))

    print("v fraction positive: ", sum(vessel_val_dataset[:, 0][1]) / len(vessel_val_dataset[:, 0][1]))
    print("v fraction negative: ", 1 - sum(vessel_val_dataset[:, 0][1]) / len(vessel_val_dataset[:, 0][1]))


    # Tuning
    iters = 800
    n_workers = 16
    n_runs = 1

    path_base = "../data/vessel_tuning_log"
    loss_fn = logistic_loss
    #loss_fn = f.mse_loss
    aggregator_fn = avg_aggregation

    shape, ranks, separation_rank = (28, 28, 28), (4, 4, 4), 2
    base_steps = 5
    steps_multiplier = [1, 4, 16]
    lrs = [0.001]
    momentums = [0.0, 0.9, 0.99]

    failures = []

    for sm in steps_multiplier:
        for momentum in momentums:
            for lr in lrs:
                steps = base_steps * sm
                iterations = iters // sm

                hypers = {"max_rounds": 1, "max_iter": iterations, "batch_size": None, "lr": lr, "momentum": momentum, "steps": steps, "threshold": 0.0}
                lsr_dot_params = (shape, ranks, separation_rank, torch.float64, torch.device('cuda'))

                print(f"\nTraining lr {lr} steps {steps} iterations {iterations} momentum {momentum}")
                path = f"{path_base}/lr{int(lr*1000)}_mom{momentum}_steps{steps}"
                args = (lsr_bcd_regression, lsr_dot_params, (vessel_dataset, vessel_val_dataset, None), hypers, loss_fn, True)

                try:
                    run_test(path, n_runs, n_workers, *args, verbose=True)
                except:
                    print("FAILED")
                    failures.append((lr, momentum, steps, iterations))

    print("\nThe following parameters failed:")
    for l, m, s, i in failures:
        print(f"lr {l}, momentum {m}, steps {s}, iterations {i}")

