from datasets import *
from federated_algos import *
from lsr_tensor import *
from lsr_bcd_regression import *
import torch
import torch.nn.functional as f
import numpy as np
import torch.multiprocessing as mp

def run_trial(method, decomp_shape, *args):
    init_tensor_dot = LSR_tensor_dot(*decomp_shape)
    _, val_loss = method(init_tensor_dot, *args)
    return np.array(val_loss)

def run_test(n_runs, n_workers, method, decomp_shape, *args, verbose=True):
    results = []

    for r in range(n_runs):
        print(f"Run {r}")

        with mp.get_context('spawn').Pool(processes=n_workers) as pool:
            results.extend(pool.starmap(run_trial, [(method, decomp_shape, *args)] * n_workers))

    results = np.stack(results)
    return np.mean(results, axis=0), np.std(results, axis=0)

