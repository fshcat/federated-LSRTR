from lsr_tensor import *
import torch

# Block coordinate descent optimization algorithm for LSR tensor regression
def lsr_bcd_regression(loss_func, Xs, ys, shape, ranks, sep_rank, lr=0.01, momentum=0.9, step_epochs=5, max_iter=100, batch_size=None, init_zero=True, ortho=True, debug=False):
    order = len(shape)

    if batch_size is None:
        batch_size = len(Xs)

    dataset = torch.utils.data.TensorDataset(Xs, ys)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    X, y = Xs, ys

    lsr_ten = LSR_tensor(shape, ranks, sep_rank, init_zero=init_zero)
    params = lsr_ten.parameters()

    for iteration in range(max_iter):
        prev = lsr_ten.expand_to_tensor()

        if debug:
            test_core = lsr_ten.core_tensor.detach()

        # factor matrix updates
        for s in range(sep_rank):
            for k in range(len(ranks)): 
                if debug:
                    test_factor = lsr_ten.factor_matrices[s][k-1].detach()

                optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=lr, momentum=momentum)

                for _ in range(step_epochs):
                    optimizer.zero_grad()
                    y_predicted = lsr_ten.bcd_factor_forward(s, k, X)
                    loss = loss_func(y_predicted, y)
                    loss.backward()
                    optimizer.step()

                if ortho:
                    lsr_ten.orthonorm_factor(s, k)

                if debug:
                    assert torch.equal(test_factor, lsr_ten.factor_matrices[s][k-1]), "Factor matrix update step updated other matrices"

        if debug:
            assert torch.equal(test_core, lsr_ten.core_tensor), "Factor matrix update steps updated core tensor"

        optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=lr, momentum=momentum)

        if debug:
            test_factor = lsr_ten.factor_matrices[0][0].detach()

        # core tensor update
        for _ in range(step_epochs):
            optimizer.zero_grad()
            y_predicted = lsr_ten.bcd_core_forward(X)
            loss = loss_func(y_predicted, y)
            loss.backward()
            optimizer.step()

        if debug:
            assert torch.equal(test_factor, lsr_ten.factor_matrices[0][0]), "Core update step updated factor matrix"

        if iteration % (max_iter // 10) == 0:
            print(f"iteration {iteration} diff: ", torch.norm(lsr_ten.expand_to_tensor() - prev))

    return lsr_ten

