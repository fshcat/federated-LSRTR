from datasets import *
from federated_algos import *
from lsr_tensor import *
from lsr_bcd_regression import *
import torch
import torch.nn.functional as f
import numpy as np
import torch.multiprocessing as mp
import os

def run_trial(method, tensor_params, *args):
    init_tensor_dot = LSR_tensor_dot(*tensor_params)
    final_lsr_dot, perf_info = method(init_tensor_dot, *args)
    return final_lsr_dot, perf_info

def run_test(path, n_runs, n_workers, method, tensor_params, *args, verbose=True):
    os.makedirs(f"{path}/weights", exist_ok=True)

    results = []

    for r in range(n_runs):
        if verbose:
            print(f"Run {r}")

        arg_list = [(method, tensor_params, *args) for _ in range(n_workers)]
        with mp.get_context('spawn').Pool(processes=n_workers) as pool:
            results.extend(pool.starmap(run_trial, arg_list))
    
    for i, (final_lsr_dot, _) in enumerate(results):
        torch.save(final_lsr_dot, f"{path}/weights/lsr_dot_{i}.pt")

    for key in results[0][1]:
        if len(results[0][1][key]) > 0:
            torch.save(torch.stack([pinfo[key] for _, pinfo in results]), f"{path}/{key}")

def logistic_loss(y_pred, y):
    y_pred = torch.sigmoid(y_pred)
    return torch.mean(-1 * (y*torch.log(y_pred) + (1-y)*torch.log(1-y_pred)))

