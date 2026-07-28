import sys
from collections.abc import Callable
from math import sqrt

import torch

sys.path.append("../../")

from szlan.optimizer.szlan import Optimizer


class Langevin(Optimizer):
    
    
    def __init__(self, 
                 x0 : torch.Tensor, 
                 grad,  
                 gamma,
                 beta,
                 device : str = "cpu", 
                 dtype : torch.dtype = torch.float64, 
                 seed : int = 123144):
        
        super().__init__(x0=x0, device=device, dtype=dtype, seed=seed)
        
        self.grad = grad
        self.gamma = gamma if isinstance(gamma, Callable) else lambda _ : gamma
        self.beta = beta if isinstance(beta, Callable) else lambda _ : beta
        self.k = 0
        
    def ask(self):
        return self.x
    
    def tell(self, X, y):
        z_k = torch.randn(self.x.shape, dtype=self.x.dtype, device=self.x.device, generator = self.generator)        
        gamma_k, beta_k = self.gamma(self.k), self.beta(self.k)
        grad_k = self.grad(self.x)

        self.x = self.x - gamma_k * grad_k + sqrt( 2 * gamma_k / beta_k) * z_k
        self.k += 1
        