from lsr_tensor import *
import torch

# Block coordinate descent optimization algorithm for LSR tensor regression
def lsr_bcd_regression(lsr_ten, loss_func, dataset, val_dataset, hypers,\
                       verbose=False, true_param=None, adam=False):
    shape, ranks, sep_rank, order = lsr_ten.shape, lsr_ten.ranks, lsr_ten.separation_rank, lsr_ten.order
    batch_size = hypers["batch_size"]

    if batch_size is None:
        batch_size=len(dataset)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        X, y = next(iter(dataloader))
        dataloader = [(X.to(device=lsr_ten.device), y.to(device=lsr_ten.device))]
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)


    params = lsr_ten.parameters()
    estim_error = []
    val_losses = []

    for iteration in range(hypers["max_iter"]):
        prev = lsr_ten.expand_to_tensor()

        # factor matrix updates
        for s in range(sep_rank):
            for k in range(len(ranks)): 
                if adam:
                    optimizer = torch.optim.Adam(lsr_ten.parameters(), lr=hypers["lr"], eps=1e-4)
                else:
                    optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=hypers["lr"], momentum=hypers["momentum"])

                x_combineds = []
                for X, y in dataloader:
                    x_combineds.append(lsr_ten.bcd_factor_update_x(s, k, X))

                for _ in range(hypers["steps"]):
                    for (X, y), x_combined in zip(dataloader, x_combineds):
                        X = X.to(device=lsr_ten.device)
                        y = y.to(device=lsr_ten.device)

                        X = torch.squeeze(X)
                        y = torch.squeeze(y)

                        optimizer.zero_grad()
                        y_predicted = lsr_ten.bcd_factor_forward(s, k, x_combined, precombined=True)

                        loss = loss_func(y_predicted, y)
                        loss.backward()
                        optimizer.step()

                lsr_ten.orthonorm_factor(s, k)


        if adam:
            optimizer = torch.optim.Adam(lsr_ten.parameters(), lr=hypers["lr"], eps=1e-4)
        else:
            optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=hypers["lr"], momentum=hypers["momentum"])

        x_combineds = []
        for X, y in dataloader:
            x_combineds.append(lsr_ten.bcd_core_update_x(X))

        # core tensor update
        for _ in range(hypers["steps"]):
            for (X, y), x_combined in zip(dataloader, x_combineds):
                X = X.to(device=lsr_ten.device)
                y = y.to(device=lsr_ten.device)

                X = torch.squeeze(X)
                y = torch.squeeze(y)

                optimizer.zero_grad()
                y_predicted = lsr_ten.bcd_core_forward(x_combined, precombined=True)
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
                    X = X.to(device=lsr_ten.device)
                    y = y.to(device=lsr_ten.device)

                    X = torch.squeeze(X)
                    y = torch.squeeze(y)

                    y_predicted = lsr_ten.forward(X)
                    val_loss += loss_func(y_predicted, y) * len(X)
                val_loss /= len(val_dataset)

            val_losses.append(val_loss.cpu()) 


        # Stop if the change in the LSR tensor is under the convergence threshold
        diff = torch.norm(lsr_ten.expand_to_tensor() - prev)
        if diff < hypers["threshold"]:
            break

        if verbose and iteration % max(1, hypers["max_iter"] // 50) == 0:
            loss_type = "Training Batch"
            if val_dataset is not None:
                loss = val_loss
                loss_type = "Validation"

            print(f"Iteration {iteration} | Delta: {diff}, {loss_type} Loss: {loss}")

    diagnostics = {"estimation_error": estim_error, "val_loss": val_losses}
    return lsr_ten, diagnostics["val_loss"]

