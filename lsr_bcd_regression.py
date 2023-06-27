def nn_lsr_regression(loss_func, Xs, ys, shape, ranks, sep_rank, steps=5, epochs=100, batch_size=128):
    order = len(shape)

    dataset = torch.utils.data.TensorDataset(Xs, ys)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    lsr_ten = LSR_tensor(shape, ranks, sep_rank, init_zero=True)
    params = lsr_ten.parameters()

    for e in range(epochs):
        prev = lsr_ten.expand_to_tensor()

        for s in range(sep_rank):
            for k in range(len(ranks)): 
                optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=0.01, momentum=0.9)

                for _ in range(steps):
                    for X, y in dataloader:
                        optimizer.zero_grad()
                        y_predicted = lsr_ten.bcd_factor_forward(s, k, X)
                        loss = loss_func(y_predicted, y)
                        loss.backward()
                        optimizer.step()

                lsr_ten.orthonorm_factor(s, k)

        optimizer = torch.optim.SGD(lsr_ten.parameters(), lr=0.001, momentum=0.9)

        for _ in range(steps):
            for X, y in dataloader:
                optimizer.zero_grad()
                y_predicted = lsr_ten.bcd_core_forward(X)
                loss = loss_func(y_predicted, y)
                loss.backward()
                optimizer.step()

        print(f"epoch {e} diff: ", torch.norm(lsr_ten.expand_to_tensor() - prev))

    return lsr_ten

