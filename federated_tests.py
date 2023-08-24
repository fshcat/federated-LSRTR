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
    lsr_dot_copy = LSR_tensor_dot.copy(final_lsr_dot, device=torch.device('cpu'))
    return lsr_dot_copy, perf_info

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

def run_method_trial(method, init_tensor_params, data, args, save_weights):
    init_tensor_dot = LSR_tensor_dot.copy(init_tensor_params)
    final_lsr_dot, perf_info = method(init_tensor_dot, data, *args)

    if save_weights:
        lsr_dot_copy = LSR_tensor_dot.copy(final_lsr_dot, device=torch.device('cpu'))
        return lsr_dot_copy, perf_info
    else:
        return perf_info

def run_combined_test(path, n_runs, n_trials, n_workers, data_fn, tensor_params, names, methods, arg_list,  verbose=True, save_weights=False):
    results = {}

    for r in range(n_runs):
        if verbose:
            print(f"Run {r}")

        data = data_fn()

        shape, ranks, separation_rank, dtype, device = tensor_params
        init_tensors = [LSR_tensor_dot(shape, ranks, separation_rank, dtype, device=torch.device('cpu')) for _ in range(n_trials)]
        for ten in init_tensors:
            ten.device = device

        for method, args, name in zip(methods, arg_list, names):
            method_results = []
            for i in range(n_trials):
                trial_args = [(method, init_tensors[i], data, args, save_weights) for _ in range(n_trials)]

            with mp.get_context('spawn').Pool(processes=n_workers) as pool:
                method_results.extend(pool.starmap(run_method_trial, trial_args))

            if name not in results:
                results[name] = method_results
            else:
                results[name].extend(method_results)
    
    for i, name in enumerate(names):
        os.makedirs(f"{path}/{name}/weights", exist_ok=True)
        method_results = results[name]

        if save_weights:
            for j, (final_lsr_dot, _) in enumerate(method_results):
                torch.save(final_lsr_dot, f"{path}/{name}/weights/lsr_dot_{j}.pt")
                
            for key in method_results[0][1]:
                if len(method_results[0][1][key]) > 0:
                    torch.save(torch.stack([pinfo[key] for _, pinfo in method_results]), f"{path}/{name}/{key}")
        else:
            for key in method_results[0]:
                if len(method_results[0][key]) > 0:
                    torch.save(torch.stack([pinfo[key] for pinfo in method_results]), f"{path}/{name}/{key}")

def logistic_loss(y_pred, y):
    y_pred = torch.sigmoid(y_pred)
    return torch.mean(-1 * (y*torch.log(y_pred) + (1-y)*torch.log(1-y_pred)))

