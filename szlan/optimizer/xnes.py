from math import log, sqrt

import torch

from szlan.optimizer.opt import Optimizer


class XNES(Optimizer):

    def __init__(self, population, sigma, seed = 1231415, dtype=torch.float64, device='cpu', eta_mu=1.0, eta_sigma=None, eta_B=None):
        super().__init__(device=device, dtype=dtype, seed=seed)

        self.population = population
        self.n = self.population.shape[0]
        self.d = self.population.shape[1]

        self.mean = self.population.mean(dim=0).clone()
        self.sigma = torch.as_tensor(sigma, dtype=dtype, device=device)
        self.B = torch.eye(self.d, dtype=dtype, device=device)

        d = self.d
        self.eta_mu = eta_mu
        self.eta_sigma = eta_sigma if eta_sigma is not None else (3.0 / 5.0) * (3 + log(d)) / (d * sqrt(d))
        self.eta_B = eta_B if eta_B is not None else self.eta_sigma

        ranks = torch.arange(1, self.n + 1, dtype=dtype, device=device)
        u = torch.clamp(log(self.n / 2.0 + 1.0) - torch.log(ranks), min=0.0)
        u = u / u.sum()
        self.utilities = u - 1.0 / self.n

        self._z = None

    def ask(self):
        z = torch.randn((self.n, self.d), generator=self.generator, device=self.device, dtype=self.dtype)
        A = self.sigma * self.B
        self.population = self.mean[None, :] + z @ A.T
        self._z = z
        return self.population

    def tell(self, X, y):
        order = torch.argsort(y)
        z_sorted = self._z[order]
        u = self.utilities

        G_delta = (u[:, None] * z_sorted).sum(dim=0)

        I = torch.eye(self.d, dtype=self.dtype, device=self.device)
        outer = torch.einsum('ij,ik->ijk', z_sorted, z_sorted)
        G_M = (u[:, None, None] * (outer - I[None, :, :])).sum(dim=0)

        G_sigma = torch.trace(G_M) / self.d
        G_B = G_M - G_sigma * I

        A = self.sigma * self.B
        self.mean = self.mean + self.eta_mu * (A @ G_delta)

        self.sigma = self.sigma * torch.exp(0.5 * self.eta_sigma * G_sigma)
        self.B = self.B @ torch.matrix_exp(0.5 * self.eta_B * G_B)