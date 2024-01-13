import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

PROJECT_ROOT = "/common/users/rp1110/fishcat/Federated-LSRTR"
sys.path.append(PROJECT_ROOT)

import torch
from federated_algos import *
from federated_tests import *

if __name__ == "__main__":
    # Comparison of federated algos
    shape, ranks, separation_rank = (64, 64), (8, 8), 2
    lsr_dot_params = (shape, ranks, separation_rank, torch.float32, torch.device('cuda'))

    loss_fn = f.mse_loss
    aggregator_fn = avg_aggregation

    site_sizes = [int(sys.argv[i]) for i in range(1,len(sys.argv))]
    client_nums = [8]

    iters = 200

    n_runs = 4
    n_trials = 16
    n_workers = 8

    path_base = "../data/synth_final"
        
    methods = [BCD_avg_local, lsr_bcd_regression, BCD_federated_stepwise, BCD_federated_all_factors, BCD_federated_full_iteration, BCD_federated_full_iteration]
    names = ['local', 'centralized', 'step', 'factors_core', 'one_iter', 'five_iter']

    gen_hypers = {"max_rounds": 1, "max_iter": iters, "batch_size": None, "lr": 0.001, "momentum": 0.9, "steps": 20, "threshold": 0.0}
    iter_hypers = {"max_rounds": iters, "max_iter": 1, "batch_size": None, "lr": 0.001, "momentum": 0.9, "steps": 20, "threshold": 0.0}
    iter_5_hypers = {"max_rounds": max(iters // 5, 1), "max_iter": 5, "batch_size": None, "lr": 0.001, "momentum": 0.9, "steps": 20, "threshold": 0.0}

    logistic = False
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
            sized_names = [f"{name}_{samples}_{clients}" for name in names]
            run_synthetic_combined(path_base, n_runs, n_trials, n_workers, samples, clients, lsr_dot_params, sized_names, methods, arg_list)
            
