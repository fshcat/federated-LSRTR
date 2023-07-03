from lsr_tensor import *
import torch

# Block coordinate descent optimization algorithm for LSR tensor regression
def lsr_bcd_regression(loss_func, dataset, shape, ranks, sep_rank, lr=0.01, momentum=0.9, step_epochs=5, batch_size=None,\ 
                       threshold=0.01, max_iter=100, init_zero=True, ortho=True, true_param=None,\
                       verbose=False):
    order = len(shape)

    if batch_size is None:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        X, y = next(iter(dataloader))
        dataloader = [(X, y)]
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    lsr_ten = LSR_tensor(shape, ranks, sep_rank, init_zero=init_zero)
    params = lsr_ten.parameters()

    estim_error = []

    for iteration in range(max_iter):
        prev = lsr_ten.expand_to_tensor()

        # factor matrix updates
        for s in range(sep_rank):
            for k in range(len(ranks)): 
                optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=lr, momentum=momentum)

                for _ in range(step_epochs):
                    for X, y in dataloader:
                        optimizer.zero_grad()
                        y_predicted = lsr_ten.bcd_factor_forward(s, k, X)
                        loss = loss_func(y_predicted, y)
                        loss.backward()
                        optimizer.step()

                if ortho:
                    lsr_ten.orthonorm_factor(s, k)


        optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=lr, momentum=momentum)

        # core tensor update
        for _ in range(step_epochs):
            for X, y in dataloader:
                optimizer.zero_grad()
                y_predicted = lsr_ten.bcd_core_forward(X)
                loss = loss_func(y_predicted, y)
                loss.backward()
                optimizer.step()

        if true_param is not None:
            estim_error.append(torch.norm(lsr_ten.expand_to_tensor() - true_param) / torch.norm(true_param))

        # Stop if the change in the LSR tensor is under the convergence threshold
        diff = torch.norm(lsr_ten.expand_to_tensor() - prev)
        if diff < threshold:
            break

        if verbose and iteration % (max_iter // 50) == 0:
            print(f"Iteration {iteration} | Delta: {diff}, Training Loss: {loss}")

    return lsr_ten, estim_error

