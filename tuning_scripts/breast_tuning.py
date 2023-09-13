import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

PROJECT_ROOT = "/common/users/rp1110/fishcat/Federated-LSRTR"
sys.path.append(PROJECT_ROOT)

from lsr_tensor import *
from lsr_bcd_regression import *
from federated_tests import *
from medmnist import BreastMNIST
from torchvision import transforms
from copy import copy

if __name__ == "__main__":
    # Load data
    transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float64)])
    breast_dataset = BreastMNIST(split="train", download=True, transform=transform)
    breast_val_dataset = BreastMNIST(split="val", download=True, transform=transform)

    print("fraction positive: ", sum(breast_val_dataset[:, 0][1]) / len(breast_val_dataset[:, 0][1]))
    print("fraction negative: ", 1 - sum(breast_val_dataset[:, 0][1]) / len(breast_val_dataset[:, 0][1]))

    # Tuning
    iters = 200
    n_workers = 16
    n_runs = 1

    path_base = "../data/breast_tuning"
    loss_fn = logistic_loss
    aggregator_fn = avg_aggregation

    shape, ranks, separation_rank = (28, 28), (3, 3), 2
    base_steps = 5
    steps_multiplier = [1, 2, 4]
    lrs = [0.001, 0.01]
    momentums = [0.99, 0.9, 0.8, 0.0]

    failures = []

    for sm in steps_multiplier:
        for lr in lrs:
            for momentum in momentums:
                steps = base_steps * sm
                iterations = iters // sm

                hypers = {"max_rounds": 1, "max_iter": iterations, "batch_size": None, "lr": lr, "momentum": momentum, "steps": steps, "threshold": 0.0}
                lsr_dot_params = (shape, ranks, separation_rank, torch.float64, torch.device('cuda'))

                print(f"\nTraining lr {lr} steps {steps} iterations {iterations} momentum {momentum}")
                path = f"{path_base}/lr{int(lr*1000)}_mom{momentum}_steps{steps}"
                args = (lsr_bcd_regression, lsr_dot_params, (breast_dataset, breast_val_dataset, None), hypers, loss_fn, True)

                try:
                    run_test(path, n_runs, n_workers, *args, verbose=True)
                except:
                    failures.append((lr, momentum, steps, iterations))

    print("\nThe following parameters failed:")
    for l, m, s, i in failures:
        print(f"lr {l}, momentum {m}, steps {s}, iterations {i}")

