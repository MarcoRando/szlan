import numpy as np 

import torch

from math import sqrt

from typing import Callable
from enum import Enum
from szlan.optimizer.opt import Optimizer
from szlan.utils.utils import _ffd


class SZLanPhase(Enum):
    INITIALIZATION = 0
    ITERATE = 1
    GRADIENT_APPROX = 2


class SZLan(Optimizer):
    
    def __init__(self, 
                 x0 : torch.Tensor, 
                 direction_generator,
                 h : float | Callable[[int], float]  = 1e-7,
                 gamma : float | Callable[[int], float] = 0.1,
                 beta : float | Callable[[int], float] = 1.0,
                 seed : int = 121314,
                 device : str = "cpu",
                 dtype : torch.dtype = torch.float64
                ):
        super().__init__(x0 = x0, device=device, dtype=dtype, seed=seed)
        self.direction_generator = direction_generator
        self.P = self.direction_generator()

        self.h = h if isinstance(h, Callable) else lambda _: h
        self.gamma = gamma if isinstance(gamma, Callable) else lambda _: gamma
        self.beta = beta if isinstance(beta, Callable) else lambda _: beta
        
#        self.phase = SZLanPhase.INITIALIZATION

        self.k = 0

    # def _approx_gradient(self, h):
    #     diff = (self.forward_values.reshape(self.num_particles, self.s, self.l) - self.current_values[:, None, None]) / h
    #     grad_contrib = (diff[...,None] * self.P).sum(dim=(1, 2)).div_(self.s * self.l).mul_(self.nrm_const)
    #     return grad_contrib

    def ask(self):
        h_k = self.h(self.k)

        x_next = self.x0 + h_k * self.P
        
#        return x_next, self.x0 
        return torch.cat((x_next.view(-1, self.P.shape[-1]), self.x0.view(1, -1)), dim=0)
        
        # if self.phase == SZLanPhase.INITIALIZATION:
        #     return self.x0
        # elif self.phase == SZLanPhase.GRADIENT_APPROX:
        #     return (self.population[:, None, None, :] + h_k * self.P[None, :, :, :]).reshape(-1, self.d)

        # g_k = self._approx_gradient(h_k) 
        # gamma_k = self.gamma(self.k)
        # self.population.add_(gamma_k * g_k, alpha = -1)
        # z_k = 0.0
        # if not self.just_fd:
        #     z_k = torch.randn((self.num_particles, self.population.shape[1]), generator=self.generator, device=self.device, dtype=self.dtype)
        #     beta_k = self.beta(self.k)
        #     self.population.add_(np.sqrt(2 *  gamma_k / beta_k) * z_k)

        # return self.population
    
    def tell(self, y):
        # y -> [y]
        h_k = self.h(self.k)
        gamma_k = self.gamma(self.k)
        beta_k = self.beta(self.k)
        y_next, y_curr = y[:-1], y[-1]
#def _ffd(forward_values, current_values, directions, h, nrm_const = 1.0):
        
        g_k = _ffd(y_next.view(-1, 1), y_curr, self.P, h_k, nrm_const=self.direction_generator.nrm_const)
        z_k = torch.randn(self.x0.shape, dtype=self.x0.dtype, device=self.x0.device, generator=self.generator)
        self.x0 = self.x0 - gamma_k * g_k + sqrt(2 * gamma_k / beta_k) * z_k
        self.P = self.direction_generator()
        self.k += 1
        # if self.phase == SZLanPhase.INITIALIZATION or self.phase == SZLanPhase.ITERATE:
        #     self.current_values = y 
        #     self.P = self.direction_generator() 
        #     self.phase = SZLanPhase.GRADIENT_APPROX
        #     if self.phase == SZLanPhase.ITERATE:
        #         self.k += 1
        # elif self.phase == SZLanPhase.GRADIENT_APPROX:
        #     self.forward_values = y
        #     self.phase = SZLanPhase.ITERATE
            


    