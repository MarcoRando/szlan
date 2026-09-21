from math import sqrt

import torch

from szlan.optimizer.opt import Optimizer


class CBO(Optimizer):

    def __init__(self, population, dt, lam, alpha, sigma, seed = 1231415, dtype=torch.float64, device='cpu'):
        super().__init__(device=device, dtype=dtype, seed=seed)
        self.population = population
        self.dt = dt
        self.lam = lam
        self.alpha = alpha
        self.sigma = sigma

        self.n = self.population.shape[0]
        self.d = self.population.shape[1]

        self.current_cons_point = None
        
    def ask(self):
        if self.current_cons_point is None:
            return self.population
        return torch.vstack([self.population, self.current_cons_point])
    
    def tell(self, X, y):
    
        if X.shape[0] > self.n:
            y = y[:-1]
            X = X[:-1, :]
                
        log_w = - self.alpha * (y - y.min())

        weights = torch.softmax(log_w, dim=0)

        cons_point = (weights[:, None] * X).sum(dim=0).reshape(1, -1)
        
        self.current_cons_point = cons_point

        xi = torch.randn((self.n, self.d), generator=self.generator, device=self.population.device, dtype=self.population.dtype)
        drift = self.lam * self.dt * (cons_point - self.population)
        brw_motion = self.sigma * sqrt(self.dt) * (self.population - cons_point).abs() * xi

        self.population = self.population - drift + brw_motion

    