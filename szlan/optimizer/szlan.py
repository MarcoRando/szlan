
from collections.abc import Callable
from enum import Enum
from math import sqrt

import torch

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
                 only_fd = False,
                 dtype : torch.dtype = torch.float64
                ):
        super().__init__(x0 = x0, device=device, dtype=dtype, seed=seed)
        self.direction_generator = direction_generator
        self.P = self.direction_generator()

        self.h = h if isinstance(h, Callable) else lambda _: h
        self.gamma = gamma if isinstance(gamma, Callable) else lambda _: gamma
        self.beta = beta if isinstance(beta, Callable) else lambda _: beta
        self.only_fd = only_fd
        
        self.k = 0

    def ask(self):
        h_k = self.h(self.k)
        x_next = self.x0 + h_k * self.P
        
        return torch.cat((x_next.view(-1, self.P.shape[-1]), self.x0.view(1, -1)), dim=0)
        
    
    def tell(self, X, y):

        h_k = self.h(self.k)
        gamma_k = self.gamma(self.k)
        beta_k = self.beta(self.k)
        y_next, y_curr = y[:-1], y[-1]
        
        g_k = _ffd(y_next.view(-1, 1), y_curr, self.P, h_k, nrm_const=self.direction_generator.nrm_const)
        z_k = torch.randn(self.x0.shape, dtype=self.x0.dtype, device=self.x0.device, generator=self.generator)
        self.x0 = self.x0 - gamma_k * g_k 
        
        if not self.only_fd:
            self.x0 += sqrt(2 * gamma_k / beta_k) * z_k
        self.P = self.direction_generator()
        self.k += 1
            


    