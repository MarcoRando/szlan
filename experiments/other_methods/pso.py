import sys 


from math import sqrt

import torch
sys.path.append("../../")

from szlan.optimizer.szlan import Optimizer


class PSO(Optimizer):

    def __init__(self, population, 
                 inertia=0.729,
                 c1=1.49445,
                 c2=1.49445,
                 seed = 131415, 
                 dtype=torch.float64, 
                 device='cpu'):
        super().__init__(device=device, dtype=dtype)
        self.population = population
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

        self.n = self.population.shape[0]
        self.d = self.population.shape[1]

        self.generator = torch.Generator(device).manual_seed(seed)

        self.V = torch.zeros_like(self.population)

        self.pbest_X = self.population.clone()
        self.pbest_F = torch.full(
            (self.n,), float("inf"), device=self.population.device, dtype=self.population.dtype
        )


        self.gbest_X = self.population[0].clone()
        self.gbest_F = float("inf")

                
    def ask(self):
        return self.population


    def tell(self, X, y):
    
        improved = y < self.pbest_F
        self.pbest_F[improved] = y[improved]
        self.pbest_X[improved] = self.population[improved]


        best_idx = (self.pbest_F).argmin()
        best_val = self.pbest_F[best_idx]

        if best_val < self.gbest_F:
            self.gbest_F = best_val.item()
            self.gbest_X = self.pbest_X[best_idx].clone()

        r1 = torch.rand(self.n, self.d, device=self.device, dtype=self.dtype)
        r2 = torch.rand(self.n, self.d, device=self.device, dtype=self.dtype)

        self.V = (
            self.inertia * self.V
            + self.c1 * r1 * (self.pbest_X - self.population)
            + self.c2 * r2 * (self.gbest_X - self.population)
        )

        self.population = self.population + self.V