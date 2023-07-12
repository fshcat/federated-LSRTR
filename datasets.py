import torch

# Create synthetic data using the given tensor as the underlying parameters of the distribution
@torch.no_grad()
def synthesize_data(true_tensor, sample_size, shape, x_stdev, y_stdev):
    x = torch.randn((sample_size, *shape)) * x_stdev
    y = true_tensor(x) + torch.randn_like(true_tensor(x)) * y_stdev
    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset

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

