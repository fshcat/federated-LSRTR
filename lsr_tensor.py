class LSR_tensor(torch.nn.Module):
    def __init__(self, shape, ranks, separation_rank, init_zero=False):
        super(LSR_tensor, self).__init__()
        self.shape = shape
        self.ranks = ranks
        self.separation_rank = separation_rank        
        self.order = len(shape)

        if init_zero:
            self.core_tensor = torch.nn.parameter.Parameter(torch.zeros(ranks))
        else:
            self.core_tensor = torch.nn.parameter.Parameter(torch.normal(0, 1, size=ranks))

        self.factor_matrices = torch.nn.ModuleList()

        for s in range(separation_rank):
            factors_s = torch.nn.ParameterList()

            for k in range(self.order):
                if init_zero:
                    factors_s.append(torch.zeros((shape[k], ranks[k])))
                else:
                    factor_matrix_A = torch.normal(torch.zeros((shape[k], ranks[k])), torch.ones((shape[k], ranks[k])))
                    factor_matrix_B = torch.linalg.qr(factor_matrix_A)[0]
                    factors_s.append(factor_matrix_B)

            self.factor_matrices.append(factors_s)

    def expand_to_tensor(self, skip_term=None):
        full_lsr_tensor = 0

        for s, term_s_factors in enumerate(self.factor_matrices):
            if s == skip_term:
                continue

            full_lsr_tensor += ten.tenalg.multi_mode_dot(self.core_tensor, term_s_factors)

        return full_lsr_tensor

    def forward(self, x):
        return ten.tenalg.inner(x, self.expand_to_tensor(), n_modes=self.order)

    @torch.no_grad()
    def bcd_factor_update_x(self, s, k, x):
        omega = ten.base.partial_tensor_to_vec(
                ten.base.partial_unfold(x, mode=k) @
                (ten.tenalg.kronecker(self.factor_matrices[s], skip_matrix=k, reverse=True) @
                    ten.base.unfold(self.core_tensor, k).T)
                )
        gamma = ten.tenalg.inner(x, expand_lsr(self.core_tensor, self.factor_matrices, skip_term=s), n_modes=self.order)
        x_combined = torch.cat((omega, torch.unsqueeze(gamma, axis=1)), axis=1)
        return x_combined

    @torch.no_grad()
    def bcd_core_update_x(self, x):
        x_vec = torch.unsqueeze(ten.base.partial_tensor_to_vec(x), axis=2)

        kron_factor_sum = 0
        for term_s_factors in self.factor_matrices:
            kron_factor_sum += ten.tenalg.kronecker(term_s_factors, reverse=False).T

        x_combined = kron_factor_sum @ x_vec
        x_combined = torch.squeeze(x_combined, axis=2)
        return x_combined

    def bcd_core_forward(self, x):
        x_combined = self.bcd_core_update_x(x)
        core_vec = ten.base.tensor_to_vec(self.core_tensor)
        return ten.tenalg.inner(x_combined, core_vec, n_modes=1)

    def bcd_factor_forward(self, s, k, x):
        x_combined = self.bcd_factor_update_x(s, k, x)
        factor_expanded = torch.cat((ten.base.tensor_to_vec(self.factor_matrices[s][k]), torch.ones((1))))
        return ten.tenalg.inner(x_combined, factor_expanded, n_modes=1)

    @torch.no_grad()
    def orthonorm_factor(self, s, k):
        trash = torch.empty((self.ranks[k], self.ranks[k]))
        torch.linalg.qr(self.factor_matrices[s][k], out=(self.factor_matrices[s][k], trash))

