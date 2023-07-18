import torch
import tensorly as ten
import tltorch as tlt

# Class for Low Separation Rank tensor decomposition
class LSR_tensor_dot(torch.nn.Module):
    def __init__(self, shape, ranks, separation_rank, init_zero=False, dtype=torch.float32):
        super(LSR_tensor_dot, self).__init__()
         
        self.shape = shape
        self.ranks = ranks
        self.separation_rank = separation_rank        
        self.order = len(shape)

        # Initialize core tensor as either zeros or independent standard gaussians
        if init_zero:
            self.core_tensor = torch.nn.parameter.Parameter(torch.zeros(ranks, dtype=dtype))
        else:
            self.core_tensor = torch.nn.parameter.Parameter(torch.normal(0, 1, size=ranks, dtype=dtype))

        self.factor_matrices = torch.nn.ModuleList()

        # Initialize all factor matrices
        for s in range(separation_rank):
            factors_s = torch.nn.ParameterList()

            for k in range(self.order):
                if init_zero:
                    factors_s.append(torch.zeros((shape[k], ranks[k]), dtype=dtype))
                else:
                    factor_matrix_A = torch.normal(torch.zeros((shape[k], ranks[k]), dtype=dtype), torch.ones((shape[k], ranks[k]), dtype=dtype))

                    # Orthonormalize matrix
                    factor_matrix_B = torch.linalg.qr(factor_matrix_A)
                    factors_s.append(factor_matrix_B[0])

            self.factor_matrices.append(factors_s)

    # Create a new LSR_tensor using the parameters from the LSR_tensor provided
    @classmethod
    @torch.no_grad()
    def copy(cls, old_tensor):
        new_tensor = cls(old_tensor.shape, old_tensor.ranks, old_tensor.separation_rank)
        for s in range(old_tensor.separation_rank):
            for k in range(len(old_tensor.ranks)):
                new_tensor.factor_matrices[s][k] = torch.clone(old_tensor.factor_matrices[s][k])

        new_tensor.core_tensor = torch.nn.Parameter(torch.clone(old_tensor.core_tensor))
        return new_tensor

    # Expand core tensor and factor matrices to full tensor, optionally excluding
    # a given term from the separation rank decomposition
    def expand_to_tensor(self, skip_term=None):
        full_lsr_tensor = 0

        for s, term_s_factors in enumerate(self.factor_matrices):
            if s == skip_term:
                continue

            full_lsr_tensor += ten.tenalg.multi_mode_dot(self.core_tensor, term_s_factors)

        return full_lsr_tensor

    # Regular forward pass
    def forward(self, x):
        return ten.tenalg.inner(x, self.expand_to_tensor(), n_modes=self.order)

    # Absorb all factor matrices and core tensor into the input tensor except for matrix s, k
    # Used during a factor matrix update step of block coordiante descent
    @torch.no_grad()
    def bcd_factor_update_x(self, s, k, x):
        omega = ten.base.partial_tensor_to_vec(
                ten.base.partial_unfold(x, mode=k) @
                (ten.tenalg.kronecker(self.factor_matrices[s], skip_matrix=k, reverse=True) @
                    ten.base.unfold(self.core_tensor, k).T)
                )
        gamma = ten.tenalg.inner(x, self.expand_to_tensor(skip_term=s), n_modes=self.order)
        x_combined = torch.cat((omega, torch.unsqueeze(gamma, axis=1)), axis=1)
        return x_combined

    # Absorb all factor matrices the input tensor (not the core tensor)
    # Used during a core tensor update step of block coordiante descent
    @torch.no_grad()
    def bcd_core_update_x(self, x):
        x_vec = torch.unsqueeze(ten.base.partial_tensor_to_vec(x), axis=2)

        kron_factor_sum = 0
        for term_s_factors in self.factor_matrices:
            kron_factor_sum += ten.tenalg.kronecker(term_s_factors, reverse=False).T

        x_combined = kron_factor_sum @ x_vec
        x_combined = torch.squeeze(x_combined, axis=2)
        return x_combined

    # Block coordinate descent core tensor update step 
    def bcd_core_forward(self, x):
        x_combined = self.bcd_core_update_x(x)
        core_vec = ten.base.tensor_to_vec(self.core_tensor)
        return ten.tenalg.inner(x_combined, core_vec, n_modes=1)

    # Block coordinate descent factor matrix update step 
    def bcd_factor_forward(self, s, k, x):
        x_combined = self.bcd_factor_update_x(s, k, x)
        factor_expanded = torch.cat((ten.base.tensor_to_vec(self.factor_matrices[s][k]), torch.ones((1))))
        return ten.tenalg.inner(x_combined, factor_expanded, n_modes=1)

    # Orthonormalize the columns of a factor matrix
    @torch.no_grad()
    def orthonorm_factor(self, s, k):
        q, r = torch.linalg.qr(self.factor_matrices[s][k], mode='reduced')
        r_signs = torch.sign(torch.eye(self.ranks[k]) * r)
        self.factor_matrices[s][k][:, :self.ranks[k]] = (q @ r_signs)[:, :self.ranks[k]]

