from datasets import *
from federated_algos import *
from lsr_tensor import *
from lsr_bcd_regression import *
import torch
import torch.nn.functional as f
import numpy as np
import torch.multiprocessing as mp

def run_trial(method, tensor_params, *args):
    init_tensor_dot = LSR_tensor_dot(*tensor_params)
    _, val_loss = method(init_tensor_dot, *args)
    return np.array(val_loss)

def run_test(n_runs, n_workers, method, tensor_params, *args, verbose=True):
    results = []

    for r in range(n_runs):
        if verbose:
            print(f"Run {r}")

        arg_list = [(method, tensor_params, *args) for _ in range(n_workers)]
        with mp.get_context('spawn').Pool(processes=n_workers) as pool:
            results.extend(pool.starmap(run_trial, arg_list))

    results = np.stack(results)
    return np.mean(results, axis=0), np.std(results, axis=0)

def logistic_loss(y_pred, y):
    y_pred = torch.sigmoid(y_pred)
    return torch.mean(-1 * (y*torch.log(y_pred) + (1-y)*torch.log(1-y_pred)))

