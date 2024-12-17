import torch

from szlan.optimizer.opt import Optimizer


class SZLan(Optimizer):
    
    def __init__(self):
        super().__init__()
    
    def _approx_grad(self, f, x, h):
        
        pass
    
    
    def optimize(self, f, x0 : torch.Tensor, T : int, gamma, h):
        
        x_k = x0.clone()
        for k in range(T):
            gamma_k, h_k = gamma(k), h(k)
            g = self._approx_grad(f, x_k, h_k)
            x_k = x_k - gamma_k * g
        
        return x_k