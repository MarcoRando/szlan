import sys 


from math import sqrt

import torch
sys.path.append("../../")

from szlan.optimizer.szlan import Optimizer
from enum import Enum


class DEPhase(Enum):
    INITIALIZATION = 0
    ITERATION = 1


class DifferentialEvolution(Optimizer):
    def __init__(self, population, F=0.8, CR=0.9, seed = 131415, dtype = torch.float64, device = 'cpu'):
        super().__init__(x0 = population[0], device=device, dtype=dtype, seed = seed)

        self.population = population.detach().clone()
        self.n, self.d = population.shape
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device).manual_seed(seed)
        self.F = F
        self.CR = CR
        self.phase = DEPhase.INITIALIZATION
        self.fX = torch.full(
            (self.n,), float("inf"), device=self.device, dtype=self.dtype
        )


        self._trial = None


    def ask(self):
        if self.phase == DEPhase.INITIALIZATION:
            return self.population
        
        # choose r1, r2, r3 distinct from i
        idx = torch.arange(self.n, device=self.device)

        r1 = torch.randint(0, self.n - 1, (self.n,), generator=self.generator, device=self.device) #sample_excluding()
        r2 = torch.randint(0, self.n - 1, (self.n,), generator=self.generator, device=self.device) #sample_excluding()
        r3 = torch.randint(0, self.n - 1, (self.n,), generator=self.generator, device=self.device) #sample_excluding()

        r1 += (r1 >= idx).long()
        r2 += (r2 >= idx).long()
        r3 += (r3 >= idx).long()
        
        mask = (r2 == r1)
        r2[mask] = (r2[mask] + 1) % self.n
        mask = (r3 == r1) | (r3 == r2)
        r3[mask] = (r3[mask] + 1) % self.n

        # mutation
        V = self.population[r1, :] + self.F * (self.population[r2, :] - self.population[r3, :])

        # crossover
        cross_mask = torch.rand(self.n, self.d, generator=self.generator, dtype=self.dtype, device=self.device) < self.CR
        j_rand = torch.randint(0, self.d, (self.n,), generator=self.generator, dtype=torch.long, device=self.device)
        cross_mask[torch.arange(self.n, dtype=torch.long), j_rand] = True

        U = torch.where(cross_mask, V, self.population)


        self._trial = U
        
        return U

    def tell(self, X, F_trial):
        if self.phase == DEPhase.ITERATION:

            F_trial = F_trial.detach()

            improved = F_trial <= self.fX
            self.population[improved] = self._trial[improved]
            self.fX[improved] = F_trial[improved]

        self.phase = DEPhase.ITERATION
        self._trial = None


