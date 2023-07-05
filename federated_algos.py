import torch
from lsr_tensor import *

def client_update_core(tensor, dataloader, optimizer, loss_fn, steps):
    for _ in range(steps):
        for X, y in dataloader:
            optimizer.zero_grad()
            y_predicted = tensor.bcd_core_forward(X)
            loss = loss_fn(y_predicted, y)
            loss.backward()
            optimizer.step()

    return tensor.core_tensor

def client_update_factor(tensor, s, k, dataloader, optimizer, loss_fn, steps):
    for _ in range(steps):
        for X, y in dataloader:
            optimizer.zero_grad()
            y_predicted = tensor.bcd_factor_forward(s, k, X)
            loss = loss_fn(y_predicted, y)
            loss.backward()
            optimizer.step()

    return tensor.factor_matrices[s][k]

def client_update_factors(init_tensor, client_dataloader, optimizer, loss_fn, steps):
    for s in range(len(init_tensor.factor_matrices)):
        for k in range(len(init_tensor.factor_matrices[s])):
            client_update_factor(init_tensor, s, k, client_dataloader, optimizer, loss_fn, steps)
            init_tensor.orthonorm_factor(s, k)

    return init_tensor.factor_matrices

# TODO: Reset optimizer state between iterations
def client_update_full(init_tensor, client_dataloader, optimizer, loss_fn, steps, iterations):
    for i in range(iterations):
        client_update_factors(init_tensor, client_dataloader, optimizer, loss_fn, steps)
        client_update_core(init_tensor, client_dataloader, optimizer, loss_fn, steps)

    return init_tensor

@torch.no_grad()
def avg_aggregation(tensor_list):
    return torch.mean(torch.stack(tensor_list))

@torch.no_grad()
def svd_aggregation(matrix_list):
    num_cols = matrix_list[0].shape[1]
    combined_matrix = torhc.cat(matrix_list, dim=1)
    return torch.linalg.svd(combined_matrix)[0][:, :num_cols]

def BCD_federated_stepwise(shape, ranks, separation_rank, client_datasets, hypers, loss_fn, aggregator_fn):
    lsr_tensor = LSR_tensor(shape, ranks, separation_rank)

    for iteration in range(hypers["max_rounds"]):
        for s in range(separation_rank):
            for k in range(len(ranks)):
                client_outputs = []
                for client_dataset in client_datasets:
                    init_tensor = torch.clone(lsr_tensor)
                    client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=hypers["batch_size"])
                    optimizer = torch.optim.SGD(init_tensor.parameters(), lr=hypers["lr"], mometum=hypers["momentum"])
                    client_out = client_update_factor(init_tensor, s, k, client_dataloader, optimizer, loss_fn, hypers["steps"])
                    client_outputs.append(client_out)

                lsr_tensor.factor_matrices[s][k] = aggregator_fn(client_outputs)
                lsr_tensor.orthonorm_factor(s, k)

        client_outputs = []
        for client_dataset in client_datasets:
            init_tensor = torch.clone(lsr_tensor)
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=hypers["batch_size"])
            optimizer = torch.optim.SGD(init_tensor.parameters(), lr=hypers["lr"], mometum=hypers["momentum"])
            client_out = client_update_core(init_tensor, client_dataloader, optimizer, loss_fn, hypers["steps"])
            client_outputs.append(client_out)

        lsr_tensor.core_tensor = avg_aggregation(client_outputs)

    return lsr_tensor

def BCD_federated_all_factors(shape, ranks, separation_rank, client_datasets, hypers, loss_fn, aggregator_fn):
    lsr_tensor = LSR_tensor(shape, ranks, separation_rank)

    for iteration in range(hypers["max_rounds"]):
        client_outputs = []
        for client_dataset in client_datasets:
            init_tensor = torch.clone(lsr_tensor)
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=hypers["batch_size"])
            optimizer = torch.optim.SGD(init_tensor.parameters(), lr=hypers["lr"], mometum=hypers["momentum"])
            client_out = client_update_factors(init_tensor, client_dataloader, optimizer, loss_fn, hypers["steps"])
            client_outputs.append(client_out)

        for s in range(separation_rank):
            for k in range(len(ranks)):
                client_factor = [client[s][k] for client in client_outputs]
                lsr_tensor.factor_matrices[s][k] = aggregator_fn(client_factor)
                lsr_tensor.orthonorm_factor(s, k)

        client_outputs = []
        for client_dataset in client_datasets:
            init_tensor = torch.clone(lsr_tensor)
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=hypers["batch_size"])
            optimizer = torch.optim.SGD(init_tensor.parameters(), lr=hypers["lr"], mometum=hypers["momentum"])
            client_out = client_update_core(init_tensor, client_dataloader, optimizer, loss_fn, hypers["steps"])
            client_outputs.append(client_out)

        lsr_tensor.core_tensor = avg_aggregation(client_outputs)

    return lsr_tensor

def BCD_federated_full_iteration(shape, ranks, separation_rank, client_datasets, hypers, loss_fn, aggregator_fn):
    lsr_tensor = LSR_tensor(shape, ranks, separation_rank)

    for iteration in range(hypers["max_rounds"]):
        client_outputs = []
        for client_dataset in client_datasets:
            init_tensor = torch.clone(lsr_tensor)
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=hypers["batch_size"])
            optimizer = torch.optim.SGD(init_tensor.parameters(), lr=hypers["lr"], mometum=hypers["momentum"])
            client_out = client_update_full(init_tensor, client_dataloader, optimizer, loss_fn, hypers["steps"], hypers["max_iter"])
            client_outputs.append(client_out)

        for s in range(separation_rank):
            for k in range(len(ranks)):
                client_factor = [client.factor_matrices[s][k] for client in client_outputs]
                lsr_tensor.factor_matrices[s][k] = aggregator_fn(client_factor)
                lsr_tensor.orthonorm_factor(s, k)

        client_cores = [client.core_tensor for client in client_outputs]
        lsr_tensor.core_tensor = avg_aggregation(client_cores)

    return lsr_tensor
