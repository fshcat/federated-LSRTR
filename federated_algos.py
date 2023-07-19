import torch
import sys
from lsr_tensor import *

def client_update_core(tensor, dataloader, optim_fn, loss_fn, steps):
    optimizer = optim_fn(tensor.parameters())

    for _ in range(steps):
        for X, y in dataloader:
            X = X.to(tensor.device)
            y = y.to(tensor.device)

            X = torch.squeeze(X)
            y = torch.squeeze(y)

            optimizer.zero_grad()
            y_predicted = tensor.bcd_core_forward(X)
            loss = loss_fn(y_predicted, y)
            loss.backward()
            optimizer.step()

    return tensor.core_tensor

def client_update_factor(tensor, s, k, dataloader, optim_fn, loss_fn, steps):
    optimizer = optim_fn(tensor.parameters())

    for _ in range(steps):
        for X, y in dataloader:
            X = X.to(tensor.device)
            y = y.to(tensor.device)

            X = torch.squeeze(X)
            y = torch.squeeze(y)

            optimizer.zero_grad()
            y_predicted = tensor.bcd_factor_forward(s, k, X)
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

def BCD_federated_stepwise(lsr_tensor, client_datasets, val_dataset, hypers, loss_fn, aggregator_fn, verbose=False):
    shape, ranks, separation_rank, order = lsr_tensor.shape, lsr_tensor.ranks, lsr_tensor.separation_rank, lsr_tensor.order
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    val_losses = []
    val_batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, pin_memory=True)

    client_dataloaders = []
    for client_dataset in client_datasets:
        batch_size = hypers["batch_size"]
        if batch_size is None:
            batch_size = len(client_dataset)
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, pin_memory=True)
            X, y = next(iter(client_dataloader))
            client_dataloader = [(X.to(device=lsr_tensor.device), y.to(device=lsr_tensor.device))]
        else:
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, pin_memory=True)

        client_dataloaders.append(client_dataloader)

    for iteration in range(hypers["max_iter"]):
        for s in range(separation_rank):
            for k in range(len(ranks)):
                client_outputs = []
                for client_dataloader in client_dataloaders:
                    init_tensor = LSR_tensor_dot.copy(lsr_tensor)
                    client_out = client_update_factor(init_tensor, s, k, client_dataloader, optim_fn, loss_fn, hypers["steps"])
                    client_outputs.append(client_out)

                lsr_tensor.factor_matrices[s][k] = aggregator_fn(client_outputs).contiguous()
                lsr_tensor.orthonorm_factor(s, k)

        client_outputs = []
        for client_dataloader in client_dataloaders:
            init_tensor = LSR_tensor_dot.copy(lsr_tensor)
            client_out = client_update_core(init_tensor, client_dataloader, optim_fn, loss_fn, hypers["steps"])
            client_outputs.append(client_out)

        lsr_tensor.core_tensor = avg_aggregation(client_outputs)

        val_loss = 0 
        with torch.no_grad():
            for X, y in val_dataloader:
                X = X.to(lsr_tensor.device)
                y = y.to(lsr_tensor.device)

                X = torch.squeeze(X)
                y = torch.squeeze(y)

                y_predicted = lsr_tensor.forward(X)
                val_loss += loss_fn(y_predicted, y) * len(X)
            val_loss /= len(val_dataset)

        if verbose:
            print(f"Iteration {iteration} | Validation Loss: {val_loss}")
        val_losses.append(val_loss.cpu()) 

    return lsr_tensor, val_losses

def BCD_federated_all_factors(lsr_tensor, client_datasets, val_dataset, hypers, loss_fn, aggregator_fn, verbose=False, ortho_iteratively=True):
    shape, ranks, separation_rank, order = lsr_tensor.shape, lsr_tensor.ranks, lsr_tensor.separation_rank, lsr_tensor.order
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    val_losses = []
    val_batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, pin_memory=True)

    client_dataloaders = []
    for client_dataset in client_datasets:
        batch_size = hypers["batch_size"]
        if batch_size is None:
            batch_size = len(client_dataset)
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, pin_memory=True)
            X, y = next(iter(client_dataloader))
            client_dataloader = [(X.to(device=lsr_tensor.device), y.to(device=lsr_tensor.device))]
        else:
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, pin_memory=True) 
        client_dataloaders.append(client_dataloader)

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

        val_loss = 0 
        with torch.no_grad():
            for X, y in val_dataloader:
                X = X.to(lsr_tensor.device)
                y = y.to(lsr_tensor.device)

                X = torch.squeeze(X)
                y = torch.squeeze(y)

                y_predicted = lsr_tensor.forward(X)
                val_loss += loss_fn(y_predicted, y) * len(X)
            val_loss /= len(val_dataset)

        if verbose:
            print(f"Iteration {iteration} | Validation Loss: {val_loss}")
        val_losses.append(val_loss.cpu()) 

    return lsr_tensor, val_losses

def BCD_federated_full_iteration(lsr_tensor, client_datasets, val_dataset, hypers, loss_fn, aggregator_fn, verbose=False):
    shape, ranks, separation_rank, order = lsr_tensor.shape, lsr_tensor.ranks, lsr_tensor.separation_rank, lsr_tensor.order
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    val_losses = []
    val_batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, pin_memory=True)

    client_dataloaders = []
    for client_dataset in client_datasets:
        batch_size = hypers["batch_size"]
        if batch_size is None:
            batch_size = len(client_dataset)
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, pin_memory=True)
            X, y = next(iter(client_dataloader))
            client_dataloader = [(X.to(device=lsr_tensor.device), y.to(device=lsr_tensor.device))]
        else:
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, pin_memory=True)

        client_dataloaders.append(client_dataloader)

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

        val_loss = 0 
        with torch.no_grad():
            for X, y in val_dataloader:
                X = X.to(lsr_tensor.device)
                y = y.to(lsr_tensor.device)

                X = torch.squeeze(X)
                y = torch.squeeze(y)
                
                y_predicted = lsr_tensor.forward(X)
                val_loss += loss_fn(y_predicted, y) * len(X)
            val_loss /= len(val_dataset)

        if verbose:
            print(f"Round {comm_round} | Validation Loss: {val_loss}")
        val_losses.append(val_loss.cpu()) 

    return lsr_tensor, val_losses
