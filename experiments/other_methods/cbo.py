import sys 


from math import sqrt

import torch
sys.path.append("../../")

from szlan.optimizer.szlan import Optimizer


class CBO(Optimizer):

    def __init__(self, population, dt, lam, alpha, sigma, use_cons, seed, dtype=torch.float64, device='cpu'):
        self.population = population
        self.dt = dt
        self.lam = lam
        self.alpha = alpha
        self.sigma = sigma
        self.use_cons = use_cons
        self.n = self.population.shape[0]
        self.d = self.population.shape[1]

        self.generator = torch.Generator(device).manual_seed(seed)

        self.current_cons_point = None
        
    def ask(self):
        if self.current_cons_point is None or not self.use_cons:
            return self.population
        return torch.vstack([self.population, self.current_cons_point])

    def tell(self, X, y):
    
        if self.current_cons_point is not None and self.use_cons:
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

    