from lsr_tensor import *
from federated_algos import get_full_accuracy, get_full_loss
import torch

# Block coordinate descent optimization algorithm for LSR tensor regression
def lsr_bcd_regression(lsr_ten, loss_func, dataset, val_dataset, hypers, accuracy=False,\
                       verbose=False, optimize=True, true_param=None, adam=False):
    shape, ranks, sep_rank, order = lsr_ten.shape, lsr_ten.ranks, lsr_ten.separation_rank, lsr_ten.order
    batch_size = hypers["batch_size"]

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    perf_info = {}

    if batch_size is None:
        batch_size=len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        X, y = next(iter(dataloader))
        dataloader = [(X.to(device=lsr_ten.device), y.to(device=lsr_ten.device))]
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)


    params = lsr_ten.parameters()
    estim_error = []
    val_losses = []

    for iteration in range(hypers["max_iter"]):
        prev = lsr_ten.expand_to_tensor()

        # factor matrix updates
        for s in range(sep_rank):
            for k in range(len(ranks)): 
                if adam:
                    optimizer = torch.optim.Adam(lsr_ten.parameters(), lr=hypers["lr"], eps=1e-4)
                else:
                    optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=hypers["lr"], momentum=hypers["momentum"])

                x_combineds = []
                for X, y in dataloader:
                    X = X.to(device=lsr_ten.device)
                    y = y.to(device=lsr_ten.device)

                    X = torch.squeeze(X)
                    y = torch.squeeze(y)

                    x_combineds.append(lsr_ten.bcd_factor_update_x(s, k, X))

                for _ in range(hypers["steps"]):
                    for (X, y), x_combined in zip(dataloader, x_combineds):
                        X = X.to(device=lsr_ten.device)
                        y = y.to(device=lsr_ten.device)

                        X = torch.squeeze(X)
                        y = torch.squeeze(y)

                        optimizer.zero_grad()

                        if not optimize:
                            x_combined = lsr_ten.bcd_factor_update_x(s, k, X)

                        y_predicted = lsr_ten.bcd_factor_forward(s, k, x_combined, precombined=True)

                        loss = loss_func(y_predicted, y)
                        loss.backward()
                        optimizer.step()

                lsr_ten.orthonorm_factor(s, k)


        if adam:
            optimizer = torch.optim.Adam(lsr_ten.parameters(), lr=hypers["lr"], eps=1e-4)
        else:
            optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=hypers["lr"], momentum=hypers["momentum"])

        x_combineds = []
        for X, y in dataloader:
            X = X.to(device=lsr_ten.device)
            y = y.to(device=lsr_ten.device)

            X = torch.squeeze(X)
            y = torch.squeeze(y)

            x_combineds.append(lsr_ten.bcd_core_update_x(X))

        # core tensor update
        for _ in range(hypers["steps"]):
            for (X, y), x_combined in zip(dataloader, x_combineds):
                X = X.to(device=lsr_ten.device)
                y = y.to(device=lsr_ten.device)

                X = torch.squeeze(X)
                y = torch.squeeze(y)

                optimizer.zero_grad()

                if not optimize:
                    x_combined = lsr_ten.bcd_core_update_x(X)

                y_predicted = lsr_ten.bcd_core_forward(x_combined, precombined=True)
                loss = loss_func(y_predicted, y)
                loss.backward()
                optimizer.step()

        if true_param is not None:
            with torch.no_grad():
                error = torch.norm(lsr_ten.expand_to_tensor() - true_param) / torch.norm(true_param)

            estim_error.append(error.detach())

        val_losses.append(get_full_loss(lsr_ten, val_dataloader, loss_func))
        train_losses.append(get_full_loss(lsr_ten, dataloader, loss_func))


        if accuracy:
            val_accs.append(get_full_accuracy(lsr_ten, val_dataloader))
            train_accs.append(get_full_accuracy(lsr_ten, dataloader))

        if verbose:
            print(f"Iteration {iteration} | Validation Loss: {val_losses[-1]}")

        diff = torch.norm(lsr_ten.expand_to_tensor() - prev)
        if diff < hypers["threshold"]:
            break

    perf_info["val_loss"], perf_info["train_loss"] = torch.stack(val_losses), torch.stack(train_losses)
    if accuracy:
        perf_info["val_acc"], perf_info["train_acc"] = torch.stack(val_accs), torch.stack(train_accs)

    return lsr_ten, perf_info

def BCD_avg_local(lsr_ten, loss_func, client_datasets, val_dataset, hypers, accuracy=False,\
                       verbose=False, optimize=True, true_param=None, adam=False):

    perf_info_list = []
    avg_perf_info = {}

    best_val_loss = None
    return_tensor = None

    for client_dataset in client_datasets:
        final_tensor, perf_info = lsr_bcd_regression(lsr_ten, loss_func, client_dataset, val_dataset, hypers, accuracy, verbose, optimize, true_param, adam)

        if best_val_loss is None or perf_info["val_loss"][-1] < best_val_loss:
            best_val_loss = perf_info["val_loss"][-1]
            return_tensor = final_tensor

        perf_info_list.append(perf_info)

    avg_perf_info["val_loss"] = torch.mean(torch.stack([pinf["val_loss"] for pinf in perf_info_list], axis=0), axis=0)
    avg_perf_info["train_loss"] = torch.mean(torch.stack([pinf["train_loss"] for pinf in perf_info_list], axis=0), axis=0)

    if accuracy:
        avg_perf_info["val_acc"] = torch.mean(torch.stack([pinf["val_acc"] for pinf in perf_info_list], axis=0), axis=0)
        avg_perf_info["train_acc"] = torch.mean(torch.stack([pinf["train_acc"] for pinf in perf_info_list], axis=0), axis=0)

    return return_tensor, avg_perf_info



