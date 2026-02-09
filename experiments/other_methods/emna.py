import sys 


from math import sqrt

import torch
sys.path.append("../../")

from szlan.optimizer.szlan import Optimizer


class EMNA(Optimizer):

    def __init__(self, population, mu = 0.5, isotropic = True, min_sigma=1e-5, seed=123123, dtype=torch.float64, device='cpu'):
        super().__init__(dtype=dtype, device=device)
        self.population = population

        self.isotropic = isotropic

        self.n = self.population.shape[0]
        self.d = self.population.shape[1]

        self.min_sigma = min_sigma
        self.mu = max(1, int(mu * self.n))

        self.generator = torch.Generator(device).manual_seed(seed)
        self.fX = torch.full((self.n,), float("inf"), device=self.device, dtype=self.dtype)

        
    def ask(self):
        return self.population
    
    def tell(self, X, y):
    
        _, idx = torch.topk(y, self.mu, largest=False)

        mean = torch.mean(self.population[idx, :], dim=0)
        diffs = self.population[idx, :] - mean
        sigma = torch.sqrt(diffs.square().mean(dim=0))
        if self.isotropic:
            sigma = torch.mean(sigma).unsqueeze(0)  
        sigma += self.min_sigma

        eps = torch.randn((self.n, self.d), generator=self.generator, device=self.device, dtype=self.dtype)
        self.population = mean + sigma * eps
            
        