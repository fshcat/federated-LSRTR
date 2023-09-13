import sklearn
import torch
import sys
from lsr_tensor import *
import torch.nn.functional as f

def logistic_loss(y_pred, y):
    y_pred = torch.sigmoid(y_pred)
    pos_prop = torch.sum(y) / len(y)

    return torch.mean(-1 * ((1/pos_prop)*y*torch.log(y_pred) + (1/ (1 - pos_prop))*(1-y)*torch.log(1-y_pred)))

def client_update_core(tensor, dataloader, optim_fn, loss_fn, steps):
    optimizer = optim_fn(tensor.parameters())

    x_combineds = []
    for X, y in dataloader:
        X = X.to(tensor.device)
        y = y.to(tensor.device)

        X = torch.squeeze(X)
        y = torch.squeeze(y)

        x_combineds.append(tensor.bcd_core_update_x(X))

    for _ in range(steps):
        for (X, y), x_combined in zip(dataloader, x_combineds):
            X = X.to(tensor.device)
            y = y.to(tensor.device)

            X = torch.squeeze(X)
            y = torch.squeeze(y)

            optimizer.zero_grad()
            y_predicted = tensor.bcd_core_forward(x_combined, precombined=True)
            loss = loss_fn(y_predicted, y)
            loss.backward()
            optimizer.step()

    return tensor.core_tensor

def client_update_factor(tensor, s, k, dataloader, optim_fn, loss_fn, steps):
    optimizer = optim_fn(tensor.parameters())

    x_combineds = []
    for X, y in dataloader:
        X = X.to(tensor.device)
        y = y.to(tensor.device)

        X = torch.squeeze(X)
        y = torch.squeeze(y)

        x_combineds.append(tensor.bcd_factor_update_x(s, k, X))

    for _ in range(steps):
        for (X, y), x_combined in zip(dataloader, x_combineds):
            X = X.to(tensor.device)
            y = y.to(tensor.device)

            X = torch.squeeze(X)
            y = torch.squeeze(y)

            optimizer.zero_grad()
            y_predicted = tensor.bcd_factor_forward(s, k, x_combined, precombined=True)
            loss = loss_fn(y_predicted, y)
            loss.backward()
            optimizer.step()

    return tensor.factor_matrices[s][k]

def client_update_factors(init_tensor, client_dataloader, optim_fn, loss_fn, steps, ortho=True):
    for s in range(len(init_tensor.factor_matrices)):
        for k in range(len(init_tensor.factor_matrices[s])):
            client_update_factor(init_tensor, s, k, client_dataloader, optim_fn, loss_fn, steps)
            if ortho:
                init_tensor.orthonorm_factor(s, k)

    return init_tensor.factor_matrices

def client_update_full(init_tensor, client_dataloader, optim_fn, loss_fn, steps, iterations):
    for i in range(iterations):
        init_tensor.factor_matrices = client_update_factors(init_tensor, client_dataloader, optim_fn, loss_fn, steps, ortho=True)
        init_tensor.core_tensor = client_update_core(init_tensor, client_dataloader, optim_fn, loss_fn, steps)
        
    return init_tensor

@torch.no_grad()
def avg_aggregation(tensor_list):
    return torch.nn.Parameter(torch.mean(torch.stack(tensor_list), dim=0))

@torch.no_grad()
def svd_aggregation(matrix_list):
    num_cols = matrix_list[0].shape[1]
    combined_matrix = torch.cat(matrix_list, dim=1)
    return torch.nn.Parameter(torch.linalg.svd(combined_matrix)[0][:, :num_cols])

def get_client_dataloaders(client_datasets, true_batch_size, device):
    client_dataloaders = []
    client_sizes = []
    for client_dataset in client_datasets:
        client_sizes.append(len(client_dataset))

        if true_batch_size is None:
            batch_size = len(client_dataset)
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
            X, y = next(iter(client_dataloader))
            client_dataloader = [(X.to(device=device), y.to(device=device))]
        else:
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

        client_dataloaders.append(client_dataloader)

    return client_dataloaders, client_sizes

@torch.no_grad()
def get_full_loss(lsr_tensor, dataloader, loss_fn):
    loss = 0 
    for X, y in dataloader:
        X = X.to(lsr_tensor.device)
        y = y.to(lsr_tensor.device)

        X = torch.squeeze(X)
        y = torch.squeeze(y)
        
        y_predicted = lsr_tensor.forward(X)
        loss += loss_fn(y_predicted, y) * len(X)

    if isinstance(dataloader, torch.utils.data.DataLoader):
        loss /= len(dataloader.dataset)
    else:
        loss /= len(dataloader[0][0])
    return loss.cpu()

@torch.no_grad()
def get_full_log_metrics(lsr_tensor, dataloader, sig=True):
    acc = 0 

    for X, y in dataloader:
        X = X.to(lsr_tensor.device)
        y = y.to(lsr_tensor.device)

        X = torch.squeeze(X)
        y = torch.squeeze(y)
        
        if sig:
            y_score = torch.sigmoid(lsr_tensor.forward(X))
        else:
            y_score = torch.clamp(y_score, 0.0, 1.0)

        y_predicted = y_score > 0.5

        # only makes sense for full GD, not batch. fix later
        precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(y.cpu(), y_predicted.cpu(), average='binary', zero_division=0.0)
        roc_auc = sklearn.metrics.roc_auc_score(y.cpu(), y_score.cpu())

        acc += torch.sum(y_predicted == y)

    if isinstance(dataloader, torch.utils.data.DataLoader):
        acc = acc / len(dataloader.dataset)
    else:
        acc = acc / len(dataloader[0][0])
    return acc.cpu(), torch.tensor(f1), torch.tensor(roc_auc)


def init_perf_info(logistic=False):
    perf_info = {"val_loss": [], "train_loss": []}

    if logistic:
        perf_info.update({"val_F1": [], "train_F1": [],\
                          "val_auc": [], "train_auc": [],\
                          "val_acc": [], "train_acc": []})

    return perf_info

@torch.no_grad()
def update_perf_info(perf_info, lsr_tensor, train_dataloader, val_dataloader, logistic=False):
    # bad, fix later
    if logistic:
        loss_fn = logistic_loss
    else:
        loss_fn = f.mse_loss

    perf_info["val_loss"].append(get_full_loss(lsr_tensor, val_dataloader, loss_fn))
    perf_info["train_loss"].append(get_full_loss(lsr_tensor, train_dataloader, loss_fn))

    if logistic:
        val_acc, val_F1, val_auc = get_full_log_metrics(lsr_tensor, val_dataloader)
        train_acc, train_F1, train_auc = get_full_log_metrics(lsr_tensor, train_dataloader)

        perf_info["val_acc"].append(val_acc)
        perf_info["train_acc"].append(train_acc)

        perf_info["val_F1"].append(val_F1)
        perf_info["train_F1"].append(train_F1)

        perf_info["val_auc"].append(val_auc)
        perf_info["train_auc"].append(train_auc)

    return perf_info

def stack_perf_info(perf_info):
    for key in perf_info:
        perf_info[key] = torch.stack(perf_info[key])

    return perf_info

def BCD_federated_stepwise(lsr_tensor, data, hypers, loss_fn, aggregator_fn, accuracy=False, verbose=False):
    train_dataset, val_dataset, client_datasets = data

    shape, ranks, separation_rank, order = lsr_tensor.shape, lsr_tensor.ranks, lsr_tensor.separation_rank, lsr_tensor.order
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    perf_info = init_perf_info(accuracy)

    batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(train_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=True, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    client_dataloaders, client_sizes = get_client_dataloaders(client_datasets, hypers["batch_size"], lsr_tensor.device)

    for iteration in range(hypers["max_iter"]):
        for s in range(separation_rank):
            for k in range(len(ranks)):
                client_outputs = []
                for client_dataloader in client_dataloaders:
                    init_tensor = LSR_tensor_dot.copy(lsr_tensor)
                    client_out = client_update_factor(init_tensor, s, k, client_dataloader, optim_fn, loss_fn, hypers["steps"])
                    client_outputs.append(client_out)

                with torch.no_grad():
                    lsr_tensor.factor_matrices[s][k][:, :] = aggregator_fn(client_outputs)[:, :]

                lsr_tensor.orthonorm_factor(s, k)

        client_outputs = []
        for client_dataloader in client_dataloaders:
            init_tensor = LSR_tensor_dot.copy(lsr_tensor)
            client_out = client_update_core(init_tensor, client_dataloader, optim_fn, loss_fn, hypers["steps"])
            client_outputs.append(client_out)

        lsr_tensor.core_tensor = avg_aggregation(client_outputs)
        perf_info = update_perf_info(perf_info, lsr_tensor, train_dataloader, val_dataloader, accuracy)

        if verbose:
            print(f"Iteration {iteration} | Validation Loss: {val_losses[-1]}")

    perf_info = stack_perf_info(perf_info)
    return lsr_tensor, perf_info

def BCD_federated_all_factors(lsr_tensor, data, hypers, loss_fn, aggregator_fn, accuracy=False, verbose=False, ortho_iteratively=True):
    train_dataset, val_dataset, client_datasets = data
    shape, ranks, separation_rank, order = lsr_tensor.shape, lsr_tensor.ranks, lsr_tensor.separation_rank, lsr_tensor.order
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    perf_info = init_perf_info(accuracy)

    batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(train_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=True, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    client_dataloaders, client_sizes = get_client_dataloaders(client_datasets, hypers["batch_size"], lsr_tensor.device)

    for iteration in range(hypers["max_iter"]):
        client_outputs = []
        for client_dataloader in client_dataloaders:
            init_tensor = LSR_tensor_dot.copy(lsr_tensor)
            client_out = client_update_factors(init_tensor, client_dataloader, optim_fn, loss_fn, hypers["steps"], ortho=ortho_iteratively)
            client_outputs.append(client_out)

        for s in range(separation_rank):
            for k in range(len(ranks)):
                client_factor = [client[s][k] for client in client_outputs]
                lsr_tensor.factor_matrices[s][k] = aggregator_fn(client_factor)
                lsr_tensor.orthonorm_factor(s, k)

        client_outputs = []
        for client_dataloader in client_dataloaders:
            init_tensor = LSR_tensor_dot.copy(lsr_tensor)
            client_out = client_update_core(init_tensor, client_dataloader, optim_fn, loss_fn, hypers["steps"])
            client_outputs.append(client_out)

        lsr_tensor.core_tensor = avg_aggregation(client_outputs)
        perf_info = update_perf_info(perf_info, lsr_tensor, train_dataloader, val_dataloader, accuracy)

        if verbose:
            print(f"Iteration {iteration} | Validation Loss: {val_losses[-1]}")

    perf_info = stack_perf_info(perf_info)
    return lsr_tensor, perf_info

def BCD_federated_full_iteration(lsr_tensor, data, hypers, loss_fn, aggregator_fn, accuracy=False, verbose=False):
    train_dataset, val_dataset, client_datasets = data

    shape, ranks, separation_rank, order = lsr_tensor.shape, lsr_tensor.ranks, lsr_tensor.separation_rank, lsr_tensor.order
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    perf_info = init_perf_info(accuracy)

    batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(train_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=True, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    client_dataloaders, client_sizes = get_client_dataloaders(client_datasets, hypers["batch_size"], lsr_tensor.device)
    train_data_size = sum(client_sizes)

    for comm_round in range(hypers["max_rounds"]):
        client_outputs = []
        for client_dataloader in client_dataloaders:
            init_tensor = LSR_tensor_dot.copy(lsr_tensor)
            client_out = client_update_full(init_tensor, client_dataloader, optim_fn, loss_fn, hypers["steps"], hypers["max_iter"])
            client_outputs.append(client_out)

        for s in range(separation_rank):
            for k in range(len(ranks)):
                client_factor = [client.factor_matrices[s][k] for client in client_outputs]
                lsr_tensor.factor_matrices[s][k] = aggregator_fn(client_factor)
                lsr_tensor.orthonorm_factor(s, k)

        client_cores = [client.core_tensor for client in client_outputs]
        lsr_tensor.core_tensor = avg_aggregation(client_cores)
        perf_info = update_perf_info(perf_info, lsr_tensor, train_dataloader, val_dataloader, accuracy)

        if verbose:
            print(f"Round {comm_round} | Validation Loss: {val_losses[-1]}")

    perf_info = stack_perf_info(perf_info)
    return lsr_tensor, perf_info

