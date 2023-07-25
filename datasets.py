from lsr_tensor import *
import torch.nn.functional as f
import torch

# Create synthetic data using the given tensor as the underlying parameters of the distribution
@torch.no_grad()
def synthesize_data(shape, ranks, separation_rank, train_num, val_num):
    x_stdev = 1
    y_stdev = 0.05

    with torch.no_grad():
        true_lsr = LSR_tensor_dot(shape, ranks, separation_rank)
        f.normalize(true_lsr.core_tensor, p=2, dim=0, out=true_lsr.core_tensor)
        true_lsr.core_tensor *= (5 / torch.sqrt(torch.sqrt(torch.prod(torch.tensor(ranks)))))

    x_train = torch.randn((train_num, *shape)) * x_stdev
    x_val = torch.randn((val_num, *shape)) * x_stdev

    y_train = true_lsr.forward(x_train) + torch.randn_like(true_lsr.forward(x_train)) * y_stdev
    y_val = true_lsr.forward(x_val) + torch.randn_like(true_lsr.forward(x_val)) * y_stdev

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

    return train_dataset, val_dataset

@torch.no_grad()
def federate_dataset(dataset, num_clients):
    indices = torch.randperm(len(dataset))
    client_size = len(dataset) / num_clients
    client_datasets = []

    for i in range(num_clients):
        start_ind = int(client_size * i)
        end_ind = int(client_size * (i + 1)) if i != num_clients - 1 else len(indices)
        client_datasets.append(torch.utils.data.Subset(dataset, indices[start_ind:end_ind]))

    return client_datasets

