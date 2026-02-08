import sys 


from math import sqrt

import torch
sys.path.append("../../")

from szlan.optimizer.szlan import Optimizer


class RS(Optimizer):

    def __init__(self, population, 
                 sigma=1.0,
                 seed = 131415, 
                 dtype=torch.float64, 
                 device='cpu'):
        super().__init__(device=device, dtype=dtype)
        self.population = population
        self.sigma = sigma

        self.n = self.population.shape[0]
        self.d = self.population.shape[1]
        self.generator = torch.Generator(device).manual_seed(seed)
                
    def ask(self):
        return self.population


    def tell(self, X, y):
        self.population = self.population[y.argmin(), :] + self.sigma * torch.randn(self.population.shape, generator=self.generator, device=self.device, dtype=self.dtype)
