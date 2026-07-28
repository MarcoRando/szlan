import sys
from math import sqrt

import torch
sys.path.append("../../")

from szlan.optimizer.szlan import Optimizer


class CMAES(Optimizer):

    def __init__(self, population, sigma, seed, dtype=torch.float64, device='cpu', mu=None, weights=None):
        self.population = population
        self.n = self.population.shape[0]
        self.d = self.population.shape[1]

        self.generator = torch.Generator(device).manual_seed(seed)
        self.device = device
        self.dtype = dtype

        self.mean = self.population.mean(dim=0).clone()
        self.sigma = torch.as_tensor(sigma, dtype=dtype, device=device)

        d = self.d
        self.mu = mu if mu is not None else self.n // 2

        if weights is None:
            idx = torch.arange(1, self.mu + 1, dtype=dtype, device=device)
            raw_weights = torch.log(torch.tensor(self.mu + 0.5, dtype=dtype, device=device)) - torch.log(idx)
            self.weights = raw_weights / raw_weights.sum()
        else:
            self.weights = weights.to(dtype=dtype, device=device)

        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        self.c_sigma = (self.mu_eff + 2) / (d + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0.0, sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / d) / (d + 4 + 2 * self.mu_eff / d)
        self.c_1 = 2 / ((d + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((d + 2) ** 2 + self.mu_eff))
        self.chi_n = sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        self.p_sigma = torch.zeros(d, dtype=dtype, device=device)
        self.p_c = torch.zeros(d, dtype=dtype, device=device)
        self.C = torch.eye(d, dtype=dtype, device=device)

        self.counteval = 0

        self._B = None
        self._D = None
        self._z = None
        self._y = None

    def ask(self):
        C = 0.5 * (self.C + self.C.T)
        eigvals, eigvecs = torch.linalg.eigh(C)
        eigvals = eigvals.clamp_min(1e-20)

        self._B = eigvecs
        self._D = eigvals.sqrt()

        z = torch.randn((self.n, self.d), generator=self.generator, device=self.device, dtype=self.dtype)
        y = z @ (self._B * self._D).T

        self.population = self.mean[None, :] + self.sigma * y
        self._z = z
        self._y = y
        return self.population

    def tell(self, X, y):
        order = torch.argsort(y)
        X_sel = X[order][:self.mu]
        y_sel = self._y[order][:self.mu]

        old_mean = self.mean
        self.mean = (self.weights[:, None] * X_sel).sum(dim=0)

        y_mean = (self.weights[:, None] * y_sel).sum(dim=0)

        C_inv_sqrt = (self._B * (1.0 / self._D)) @ self._B.T

        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
            sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (C_inv_sqrt @ y_mean)

        p_sigma_norm = self.p_sigma.norm()
        self.counteval += 1
        hs = float(
            p_sigma_norm / sqrt(1 - (1 - self.c_sigma) ** (2 * self.counteval))
            < (1.4 + 2 / (self.d + 1)) * self.chi_n
        )

        self.p_c = (1 - self.c_c) * self.p_c + \
            hs * sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_mean

        artmp = (X_sel - old_mean[None, :]) / self.sigma
        rank_mu = torch.einsum('i,ij,ik->jk', self.weights, artmp, artmp)

        self.C = (1 - self.c_1 - self.c_mu) * self.C \
            + self.c_1 * (torch.outer(self.p_c, self.p_c) + (1 - hs) * self.c_c * (2 - self.c_c) * self.C) \
            + self.c_mu * rank_mu

        self.sigma = self.sigma * torch.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1))