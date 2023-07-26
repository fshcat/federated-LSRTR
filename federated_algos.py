import torch
import sys
from lsr_tensor import *

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
def get_full_accuracy(lsr_tensor, dataloader):
    acc = 0 
    for X, y in dataloader:
        X = X.to(lsr_tensor.device)
        y = y.to(lsr_tensor.device)

        X = torch.squeeze(X)
        y = torch.squeeze(y)
        
        y_predicted = lsr_tensor.forward(X) > 0.5
        acc += torch.sum(y_predicted == y)
    if isinstance(dataloader, torch.utils.data.DataLoader):
        acc = acc / len(dataloader.dataset)
    else:
        acc = acc / len(dataloader[0][0])
    return acc.cpu()

def BCD_federated_stepwise(lsr_tensor, client_datasets, val_dataset, hypers, loss_fn, aggregator_fn, accuracy=False, verbose=False):
    shape, ranks, separation_rank, order = lsr_tensor.shape, lsr_tensor.ranks, lsr_tensor.separation_rank, lsr_tensor.order
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    perf_info = {}

    val_batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, pin_memory=True, shuffle=False)

    client_dataloaders, client_sizes = get_client_dataloaders(client_datasets, hypers["batch_size"], lsr_tensor.device)
    train_data_size = sum(client_sizes)

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

        val_losses.append(get_full_loss(lsr_tensor, val_dataloader, loss_fn))
        train_losses.append(sum([get_full_loss(lsr_tensor, c_data, loss_fn) * c_size for c_data, c_size in zip(client_dataloaders, client_sizes)]) / train_data_size)

        if accuracy:
            val_accs.append(get_full_accuracy(lsr_tensor, val_dataloader))
            train_accs.append(sum([get_full_accuracy(lsr_tensor, c_data) * c_size for c_data, c_size in zip(client_dataloaders, client_sizes)]) / train_data_size)

        if verbose:
            print(f"Iteration {iteration} | Validation Loss: {val_losses[-1]}")

    perf_info["val_loss"], perf_info["train_loss"] = torch.stack(val_losses), torch.stack(train_losses)
    perf_info["val_acc"], perf_info["train_acc"] = torch.stack(val_accs), torch.stack(train_accs)

    return lsr_tensor, perf_info

def BCD_federated_all_factors(lsr_tensor, client_datasets, val_dataset, hypers, loss_fn, aggregator_fn, accuracy=False, verbose=False, ortho_iteratively=True):
    shape, ranks, separation_rank, order = lsr_tensor.shape, lsr_tensor.ranks, lsr_tensor.separation_rank, lsr_tensor.order
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    perf_info = {}

    val_batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, pin_memory=True, shuffle=False)

    client_dataloaders, client_sizes = get_client_dataloaders(client_datasets, hypers["batch_size"], lsr_tensor.device)
    train_data_size = sum(client_sizes)

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

        val_losses.append(get_full_loss(lsr_tensor, val_dataloader, loss_fn))
        train_losses.append(sum([get_full_loss(lsr_tensor, c_data, loss_fn) * c_size for c_data, c_size in zip(client_dataloaders, client_sizes)]) / train_data_size)

        if accuracy:
            val_accs.append(get_full_accuracy(lsr_tensor, val_dataloader))
            train_accs.append(sum([get_full_accuracy(lsr_tensor, c_data) * c_size for c_data, c_size in zip(client_dataloaders, client_sizes)]) / train_data_size)

        if verbose:
            print(f"Iteration {iteration} | Validation Loss: {val_losses[-1]}")

    perf_info["val_loss"], perf_info["train_loss"] = torch.stack(val_losses), torch.stack(train_losses)
    perf_info["val_acc"], perf_info["train_acc"] = torch.stack(val_accs), torch.stack(train_accs)

    return lsr_tensor, perf_info

def BCD_federated_full_iteration(lsr_tensor, client_datasets, val_dataset, hypers, loss_fn, aggregator_fn, accuracy=False, verbose=False):
    shape, ranks, separation_rank, order = lsr_tensor.shape, lsr_tensor.ranks, lsr_tensor.separation_rank, lsr_tensor.order
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    perf_info = {}

    val_batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, pin_memory=True, shuffle=False)

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

        val_losses.append(get_full_loss(lsr_tensor, val_dataloader, loss_fn))
        train_losses.append(sum([get_full_loss(lsr_tensor, c_data, loss_fn) * c_size for c_data, c_size in zip(client_dataloaders, client_sizes)]) / train_data_size)

        if accuracy:
            val_accs.append(get_full_accuracy(lsr_tensor, val_dataloader))
            train_accs.append(sum([get_full_accuracy(lsr_tensor, c_data) * c_size for c_data, c_size in zip(client_dataloaders, client_sizes)]) / train_data_size)

        if verbose:
            print(f"Round {comm_round} | Validation Loss: {val_losses[-1]}")

    perf_info["val_loss"], perf_info["train_loss"] = torch.stack(val_losses), torch.stack(train_losses)
    perf_info["val_acc"], perf_info["train_acc"] = torch.stack(val_accs), torch.stack(train_accs)

    return lsr_tensor, perf_info

