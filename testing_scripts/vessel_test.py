import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

PROJECT_ROOT = "/common/users/rp1110/fishcat/Federated-LSRTR"
sys.path.append(PROJECT_ROOT)

import torch
from federated_algos import *
from federated_tests import *

if __name__ == "__main__":
    # Comparison of federated algos
    shape, ranks, separation_rank = (28, 28, 28), (4, 4, 4), 2
    lsr_dot_params = (shape, ranks, separation_rank, torch.float64, torch.device('cuda'))

    loss_fn = logistic_loss
    aggregator_fn = avg_aggregation

    site_sizes = [1.0]
    client_nums = [6]

    iters = 400

    n_runs = 4
    n_trials = 8
    n_workers = 8

    path_base = "../data/vessel_final_2"
        
    methods = [BCD_avg_local, lsr_bcd_regression, BCD_federated_stepwise, BCD_federated_all_factors, BCD_federated_full_iteration, BCD_federated_full_iteration]
    names = ['local', 'centralized', 'step', 'factors_core', 'one_iter', 'five_iter']

    lr = 0.001
    steps = 20
    mom = 0.99

    gen_hypers = {"max_rounds": 1, "max_iter": iters, "batch_size": None, "lr": lr, "momentum": mom, "steps": steps, "threshold": 0.0}
    iter_hypers = {"max_rounds": iters, "max_iter": 1, "batch_size": None, "lr": lr, "momentum": mom, "steps": steps, "threshold": 0.0}
    iter_5_hypers = {"max_rounds": max(iters // 5, 1), "max_iter": 5, "batch_size": None, "lr": lr, "momentum": mom, "steps": steps, "threshold": 0.0}

    logistic = True
    arg_list = [(gen_hypers, loss_fn, logistic), # local
                (gen_hypers, loss_fn, logistic), # centralized
                (gen_hypers, loss_fn, aggregator_fn, logistic), # one step
                (gen_hypers, loss_fn, aggregator_fn, logistic), # factors core
                (iter_hypers, loss_fn, aggregator_fn, logistic), # one iter
                (iter_5_hypers, loss_fn, aggregator_fn, logistic)] # five iter

    trial_args = []

    print ("---STARTING---")
    for samples in site_sizes:
        for clients in client_nums:
            print(f"training | samples: {samples}, clients: {clients}")
            sized_names = [f"{name}_{int(samples*100)}_{clients}" for name in names]
            run_vessel_combined(path_base, n_runs, n_trials, n_workers, samples, clients, lsr_dot_params, sized_names, methods, arg_list)
            
