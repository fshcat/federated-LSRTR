import torch
from lsr_tensor import *

def client_update_core(tensor, dataloader, optim_fn, loss_fn, steps):
    optimizer = optim_fn(tensor.parameters())

    for _ in range(steps):
        for X, y in dataloader:
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
            optimizer.zero_grad()
            y_predicted = tensor.bcd_factor_forward(s, k, X)
            loss = loss_fn(y_predicted, y)
            loss.backward()
            optimizer.step()

    return tensor.factor_matrices[s][k]

def client_update_factors(init_tensor, client_dataloader, optim_fn, loss_fn, steps, ortho=False):
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

def BCD_federated_stepwise(shape, ranks, separation_rank, client_datasets, val_dataset, hypers, loss_fn, aggregator_fn, verbose=False):
    lsr_tensor = LSR_tensor(shape, ranks, separation_rank)
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    val_losses = []
    val_batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size)

    for iteration in range(hypers["max_iter"]):
        for s in range(separation_rank):
            for k in range(len(ranks)):
                client_outputs = []
                for client_dataset in client_datasets:
                    init_tensor = LSR_tensor.copy(lsr_tensor)

                    batch_size = hypers["batch_size"]
                    if batch_size is None:
                        batch_size = len(client_dataset)

                    client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size)
                    client_out = client_update_factor(init_tensor, s, k, client_dataloader, optim_fn, loss_fn, hypers["steps"])
                    client_outputs.append(client_out)

                lsr_tensor.factor_matrices[s][k] = aggregator_fn(client_outputs)
                lsr_tensor.orthonorm_factor(s, k)

        client_outputs = []
        for client_dataset in client_datasets:
            init_tensor = LSR_tensor.copy(lsr_tensor)

            batch_size = hypers["batch_size"]
            if batch_size is None:
                batch_size = len(client_dataset)

            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size)
            client_out = client_update_core(init_tensor, client_dataloader, optim_fn, loss_fn, hypers["steps"])
            client_outputs.append(client_out)

        lsr_tensor.core_tensor = avg_aggregation(client_outputs)

        val_loss = 0 
        with torch.no_grad():
            for X, y in val_dataloader:
                y_predicted = lsr_tensor.forward(X)
                val_loss += loss_fn(y_predicted, y) * len(X)
            val_loss /= len(val_dataset)

        if verbose:
            print(f"Iteration {iteration} | Validation Loss: {val_loss}")
        val_losses.append(val_loss) 

    return lsr_tensor, val_losses

def BCD_federated_all_factors(shape, ranks, separation_rank, client_datasets, val_dataset, hypers, loss_fn, aggregator_fn, verbose=False):
    lsr_tensor = LSR_tensor(shape, ranks, separation_rank)
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    val_losses = []
    val_batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size)

    for iteration in range(hypers["max_iter"]):
        client_outputs = []
        for client_dataset in client_datasets:
            init_tensor = LSR_tensor.copy(lsr_tensor)

            batch_size = hypers["batch_size"]
            if batch_size is None:
                batch_size = len(client_dataset)

            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size)
            client_out = client_update_factors(init_tensor, client_dataloader, optim_fn, loss_fn, hypers["steps"])
            client_outputs.append(client_out)

        for s in range(separation_rank):
            for k in range(len(ranks)):
                client_factor = [client[s][k] for client in client_outputs]
                lsr_tensor.factor_matrices[s][k] = aggregator_fn(client_factor)
                lsr_tensor.orthonorm_factor(s, k)

        client_outputs = []
        for client_dataset in client_datasets:
            init_tensor = LSR_tensor.copy(lsr_tensor)

            batch_size = hypers["batch_size"]
            if batch_size is None:
                batch_size = len(client_dataset)

            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size)
            client_out = client_update_core(init_tensor, client_dataloader, optim_fn, loss_fn, hypers["steps"])
            client_outputs.append(client_out)

        lsr_tensor.core_tensor = avg_aggregation(client_outputs)

        val_loss = 0 
        with torch.no_grad():
            for X, y in val_dataloader:
                y_predicted = lsr_tensor.forward(X)
                val_loss += loss_fn(y_predicted, y) * len(X)
            val_loss /= len(val_dataset)

        if verbose:
            print(f"Iteration {iteration} | Validation Loss: {val_loss}")
        val_losses.append(val_loss) 

    return lsr_tensor, val_losses

def BCD_federated_full_iteration(shape, ranks, separation_rank, client_datasets, val_dataset, hypers, loss_fn, aggregator_fn, verbose=False):
    lsr_tensor = LSR_tensor(shape, ranks, separation_rank)
    optim_fn = lambda params: torch.optim.SGD(params, lr=hypers["lr"], momentum=hypers["momentum"])

    val_losses = []
    val_batch_size = hypers["batch_size"] if hypers["batch_size"] is not None else len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size)

    for comm_round in range(hypers["max_rounds"]):
        client_outputs = []
        for client_dataset in client_datasets:
            init_tensor = LSR_tensor.copy(lsr_tensor)

            batch_size = hypers["batch_size"]
            if batch_size is None:
                batch_size = len(client_dataset)

            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size)
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
                y_predicted = lsr_tensor.forward(X)
                val_loss += loss_fn(y_predicted, y) * len(X)
            val_loss /= len(val_dataset)

        if verbose:
            print(f"Round {comm_round} | Validation Loss: {val_loss}")
        val_losses.append(val_loss) 

    return lsr_tensor, val_losses
