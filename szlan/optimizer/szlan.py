import numpy as np 

import torch

from typing import Callable
from enum import Enum
from szlan.optimizer.opt import Optimizer



class SZLanPhase(Enum):
    INITIALIZATION = 0
    ITERATE = 1
    GRADIENT_APPROX = 2


class SZLan(Optimizer):
    
    def __init__(self, 
                 population, # n x d
                 direction_generator,
                 h : float | Callable[[int], float]  = 1e-7,
                 gamma : float | Callable[[int], float] = 0.1,
                 beta : float | Callable[[int], float] = 1.0,
                 just_fd : bool = False,
                 seed : int = 121314,
                 device : str = "cpu",
                 dtype : torch.dtype = torch.float64
                ):
        super().__init__(device=device, dtype=dtype)
        self.direction_generator = direction_generator
        self.s, self.l, self.nrm_const, self.d = self.direction_generator.s, self.direction_generator.l, self.direction_generator.nrm_const, self.direction_generator.d
        self.P = self.direction_generator()
        self.num_particles =  population.shape[0]
        self.just_fd = just_fd
        self.h = h if isinstance(h, Callable) else lambda _: h
        self.gamma = gamma if isinstance(gamma, Callable) else lambda _: gamma
        self.beta = beta if isinstance(beta, Callable) else lambda _: beta
        
        self.population = population 
        self.phase = SZLanPhase.INITIALIZATION

        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.best = None
        self.k = 0

    def _approx_gradient(self, h):
        diff = (self.forward_values.reshape(self.num_particles, self.s, self.l) - self.current_values[:, None, None]) / h
        grad_contrib = (diff[...,None] * self.P).sum(dim=(1, 2)).div_(self.s * self.l).mul_(self.nrm_const)
        return grad_contrib

    def ask(self):
        h_k = self.h(self.k)
        if self.phase == SZLanPhase.INITIALIZATION:
            return self.population
        elif self.phase == SZLanPhase.GRADIENT_APPROX:
            return (self.population[:, None, None, :] + h_k * self.P[None, :, :, :]).reshape(-1, self.d)

        g_k = self._approx_gradient(h_k) 
        gamma_k = self.gamma(self.k)
        self.population.add_(gamma_k * g_k, alpha = -1)
        z_k = 0.0
        if not self.just_fd:
            z_k = torch.randn((self.num_particles, self.population.shape[1]), generator=self.generator, device=self.device, dtype=self.dtype)
            beta_k = self.beta(self.k)
            self.population.add_(np.sqrt(2 *  gamma_k / beta_k) * z_k)

        return self.population
    
    def tell(self, X, y):
        best_idx = y.argmin().item()#np.argmin(y)
        if self.best is None or y[best_idx] < self.best[1]:
            self.best = (X[best_idx, :], y[best_idx].item())
            
        if self.phase == SZLanPhase.INITIALIZATION or self.phase == SZLanPhase.ITERATE:
            self.current_values = y 
            self.P = self.direction_generator() 
            self.phase = SZLanPhase.GRADIENT_APPROX
            if self.phase == SZLanPhase.ITERATE:
                self.k += 1
        elif self.phase == SZLanPhase.GRADIENT_APPROX:
            self.forward_values = y
            self.phase = SZLanPhase.ITERATE
            

    def recommend(self):
        return self.best 


    