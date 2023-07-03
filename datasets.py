import torch

# Create synthetic data using the given tensor as the underlying parameters of the distribution
def synthesize_data(true_tensor, sample_size, shape, x_stdev, y_stdev):
    x = torch.randn((sample_size, *shape)) * x_stdev
    y = true_tensor(x) + torch.randn_like(true_tensor(x)) * y_stdev
    dataset = torch.utils.data.TensorDataset(xs, ys)
    return dataset


