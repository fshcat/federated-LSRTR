from lsr_tensor import *
import torch

# Block coordinate descent optimization algorithm for LSR tensor regression
def lsr_bcd_regression(loss_func, dataset, lsr_ten, lr=0.01, momentum=0.9, step_epochs=5, batch_size=None,\
                       threshold=1e-6, max_iter=200, init_zero=False, ortho=True, true_param=None, val_dataset=None,\
                       verbose=False, adam=False):
    shape, ranks, sep_rank, order = lsr_ten.shape, lsr_ten.ranks, lsr_ten.separation_rank, lsr_ten.order

    if batch_size is None:
        batch_size=len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        X, y = next(iter(dataloader))
        dataloader = [(X, y)]
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)


    params = lsr_ten.parameters()
    estim_error = []
    val_losses = []

    for iteration in range(max_iter):
        prev = lsr_ten.expand_to_tensor()

        # factor matrix updates
        for s in range(sep_rank):
            for k in range(len(ranks)): 
                if adam:
                    optimizer = torch.optim.Adam(lsr_ten.parameters(), lr=lr, eps=1e-4)
                else:
                    optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=lr, momentum=momentum)

                for _ in range(step_epochs):
                    for X, y in dataloader:
                        X = torch.squeeze(X)
                        y = torch.squeeze(y)

                        optimizer.zero_grad()
                        y_predicted = lsr_ten.bcd_factor_forward(s, k, X)

                        loss = loss_func(y_predicted, y)
                        loss.backward()
                        optimizer.step()

                if ortho:
                    lsr_ten.orthonorm_factor(s, k)


        if adam:
            optimizer = torch.optim.Adam(lsr_ten.parameters(), lr=lr, eps=1e-4)
        else:
            optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=lr, momentum=momentum)

        # core tensor update
        for _ in range(step_epochs):
            for X, y in dataloader:
                X = torch.squeeze(X)
                y = torch.squeeze(y)

                optimizer.zero_grad()
                y_predicted = lsr_ten.bcd_core_forward(X)
                loss = loss_func(y_predicted, y)
                loss.backward()
                optimizer.step()

        if true_param is not None:
            with torch.no_grad():
                error = torch.norm(lsr_ten.expand_to_tensor() - true_param) / torch.norm(true_param)

            estim_error.append(error.detach())

        if val_dataset is not None:
            val_loss = 0 

            with torch.no_grad():
                for X, y in val_dataloader:
                    X = torch.squeeze(X)
                    y = torch.squeeze(y)

                    y_predicted = lsr_ten.forward(X)
                    val_loss += loss_func(y_predicted, y) * len(X)
                val_loss /= len(val_dataset)

            val_losses.append(val_loss) 


        # Stop if the change in the LSR tensor is under the convergence threshold
        diff = torch.norm(lsr_ten.expand_to_tensor() - prev)
        if diff < threshold:
            break

        if verbose and iteration % max(1, max_iter // 50) == 0:
            loss_type = "Training Batch"
            if val_dataset is not None:
                loss = val_loss
                loss_type = "Validation"

            print(f"Iteration {iteration} | Delta: {diff}, {loss_type} Loss: {loss}")

    diagnostics = {"estimation_error": estim_error, "val_loss": val_losses}
    return lsr_ten, diagnostics

