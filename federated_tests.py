import sys
from datasets import *
from federated_algos import *
from lsr_tensor import *
from lsr_bcd_regression import *
import torch
import torch.nn.functional as f
import numpy as np
import torch.multiprocessing as mp
import os
from torchvision import transforms
from medmnist import BreastMNIST, VesselMNIST3D

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
        print(f"final {key}: {torch.mean(torch.stack([pinfo[key] for _, pinfo in results]), axis=0)[-1]}")
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
            trial_args = [(method, init_tensors[i], data, args, save_weights) for i in range(n_trials)]

            torch.cuda.empty_cache()

            if n_workers == 1:
                for i, args in enumerate(trial_args):
                    method_results.append(run_method_trial(*args))
            else:
                with mp.get_context('spawn').Pool(processes=n_workers) as pool:
                    method_results.extend(pool.starmap(run_method_trial, trial_args))

            if name not in results:
                results[name] = method_results
            else:
                results[name].extend(method_results)
    
    for i, name in enumerate(names):
        os.makedirs(f"{path}/{name}/weights", exist_ok=True)
        method_results = results[name]
        print(f"Saving to {path}/{name}")

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

def run_synthetic_combined(path, n_runs, n_trials, n_workers, samples, clients, tensor_params, names, methods, arg_list, verbose=True, save_weights=False):
    shape, ranks, separation_rank, _, _ = tensor_params

    def data_fn(val_sample_size=500):
        synth_tensor = get_synth_tensor(shape, ranks, separation_rank)
        synth_dataset, synth_val_dataset = synthesize_data(synth_tensor, samples*clients, val_sample_size)
        synth_client_datasets = federate_dataset(synth_dataset, clients)
        return (synth_dataset, synth_val_dataset, synth_client_datasets)

    run_combined_test(path, n_runs, n_trials, n_workers, data_fn, tensor_params, names, methods, arg_list,  verbose=True, save_weights=False)
    print(f"FINISHED SIZE {samples} CLIENTS {clients}")

def run_vessel_combined(path, n_runs, n_trials, n_workers, fraction, clients, tensor_params, names, methods, arg_list, verbose=True, save_weights=False):
    shape, ranks, separation_rank, _, _ = tensor_params

    def data_fn():
        vessel_dataset = VesselMNIST3D(split="train", download=True)
        vessel_val_dataset = VesselMNIST3D(split="test", download=True)

        shuffle_inds = torch.randperm(len(vessel_dataset))
        frac_inds = shuffle_inds[0:int(len(vessel_dataset) * fraction)]
        frac_dataset = torch.utils.data.Subset(vessel_dataset, frac_inds)

        positive_inds = [idx for idx, target in enumerate(vessel_dataset.labels[frac_inds]) if target == 1]
        negative_inds = [idx for idx, target in enumerate(vessel_dataset.labels[frac_inds]) if target == 0]

        positive_dataset = torch.utils.data.Subset(frac_dataset, positive_inds)
        negative_dataset = torch.utils.data.Subset(frac_dataset, negative_inds)

        pos_fractions = [len(positive_dataset) // clients for i in range(clients)]
        pos_fractions[-1] += len(positive_dataset) - sum(pos_fractions)

        neg_fractions = [len(negative_dataset) // clients for i in range(clients)]
        neg_fractions[-1] += len(negative_dataset) - sum(neg_fractions)

        pos_clients = torch.utils.data.random_split(positive_dataset, pos_fractions)
        neg_clients = torch.utils.data.random_split(negative_dataset, neg_fractions)
        vessel_client_datasets = [torch.utils.data.ConcatDataset((pos, neg)) for pos, neg in zip(pos_clients, neg_clients)]

        return (frac_dataset, vessel_val_dataset, vessel_client_datasets)

    run_combined_test(path, n_runs, n_trials, n_workers, data_fn, tensor_params, names, methods, arg_list,  verbose=True, save_weights=False)
    print(f"FINISHED PERCENT {fraction * 100} CLIENTS {clients}")
    sys.stdout.flush()

def run_breast_combined(path, n_runs, n_trials, n_workers, fraction, clients, tensor_params, names, methods, arg_list, verbose=True, save_weights=False):
    shape, ranks, separation_rank, _, _ = tensor_params

    def data_fn():
        transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float64)])
        breast_dataset = BreastMNIST(split="train", download=True, transform=transform)
        breast_val_dataset = BreastMNIST(split="test", download=True, transform=transform)

        shuffle_inds = torch.randperm(len(breast_dataset))
        frac_inds = shuffle_inds[0:int(len(breast_dataset) * fraction)]
        frac_dataset = torch.utils.data.Subset(breast_dataset, frac_inds)

        positive_inds = [idx for idx, target in enumerate(breast_dataset.labels[frac_inds]) if target == 1]
        negative_inds = [idx for idx, target in enumerate(breast_dataset.labels[frac_inds]) if target == 0]

        positive_dataset = torch.utils.data.Subset(frac_dataset, positive_inds)
        negative_dataset = torch.utils.data.Subset(frac_dataset, negative_inds)

        pos_fractions = [len(positive_dataset) // clients for i in range(clients)]
        pos_fractions[-1] += len(positive_dataset) - sum(pos_fractions)

        neg_fractions = [len(negative_dataset) // clients for i in range(clients)]
        neg_fractions[-1] += len(negative_dataset) - sum(neg_fractions)

        pos_clients = torch.utils.data.random_split(positive_dataset, pos_fractions)
        neg_clients = torch.utils.data.random_split(negative_dataset, neg_fractions)
        breast_client_datasets = [torch.utils.data.ConcatDataset((pos, neg)) for pos, neg in zip(pos_clients, neg_clients)]

        return (frac_dataset, breast_val_dataset, breast_client_datasets)

    run_combined_test(path, n_runs, n_trials, n_workers, data_fn, tensor_params, names, methods, arg_list,  verbose=True, save_weights=False)
    print(f"FINISHED PERCENT {fraction * 100} CLIENTS {clients}")
    sys.stdout.flush()

def logistic_loss(y_pred, y):
    y_pred = torch.sigmoid(y_pred)
    pos_prop = torch.sum(y) / len(y)

    return torch.mean(-1 * ((1/pos_prop)*y*torch.log(y_pred) + (1/ (1 - pos_prop))*(1-y)*torch.log(1-y_pred)))
def logistic_loss(y_pred, y):
    y_pred = torch.sigmoid(y_pred)
    pos_prop = torch.sum(y) / len(y)

    return torch.mean(-1 * ((1/pos_prop)*y*torch.log(y_pred) + (1/ (1 - pos_prop))*(1-y)*torch.log(1-y_pred)))
