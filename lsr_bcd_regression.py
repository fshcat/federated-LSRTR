from lsr_tensor import *
import torch

# Block coordinate descent optimization algorithm for LSR tensor regression
def lsr_bcd_regression(loss_func, dataset, shape, ranks, sep_rank, lr=0.01, momentum=0.9, step_epochs=5, max_iter=100, threshold=0.01, batch_size=None, init_zero=True, ortho=True, verbose=False, debug=False):
    order = len(shape)

    if batch_size is None:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        X, y = next(iter(dataloader))
        dataloader = [(X, y)]
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    lsr_ten = LSR_tensor(shape, ranks, sep_rank, init_zero=init_zero)
    params = lsr_ten.parameters()

    for iteration in range(max_iter):
        prev = lsr_ten.expand_to_tensor()

        # factor matrix updates
        for s in range(sep_rank):
            for k in range(len(ranks)): 
                optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=lr, momentum=momentum)

                if debug:
                    p1 = torch.clone(lsr_ten.factor_matrices[s][k].data)

                for _ in range(step_epochs):
                    for X, y in dataloader:
                        optimizer.zero_grad()
                        y_predicted = lsr_ten.bcd_factor_forward(s, k, X)
                        loss = loss_func(y_predicted, y)
                        loss.backward()
                        optimizer.step()

                if debug:
                    p2 = torch.clone(lsr_ten.factor_matrices[s][k].data)

                if ortho:
                    lsr_ten.orthonorm_factor(s, k)

                if debug and torch.norm(p2 - lsr_ten.factor_matrices[s][k].detach()) > 1.0:
                    ind = torch.randint(shape[k], (1, ))
                    print()
                    print("before grad: ", p1[ind])
                    print("after grad: ", p2[ind])
                    print("after ortho: ", lsr_ten.factor_matrices[s][k][ind])

        optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=lr, momentum=momentum)

        # core tensor update
        for _ in range(step_epochs):
            for X, y in dataloader:
                optimizer.zero_grad()
                y_predicted = lsr_ten.bcd_core_forward(X)
                loss = loss_func(y_predicted, y)
                loss.backward()
                optimizer.step()

        diff = torch.norm(lsr_ten.expand_to_tensor() - prev)
        if diff < threshold:
            break

        if verbose and iteration % (max_iter // 50) == 0:
            print(f"Iteration {iteration} | Delta: {diff}, Training Loss: {loss}")

    return lsr_ten

