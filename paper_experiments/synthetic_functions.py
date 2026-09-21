from math import pi, sqrt

import torch


class TargetFunction:
    
    def __init__(self, name, d, lam =1e-5,  dtype = torch.float64, device = 'cpu') -> None:
        self.name = name
        self.d = d
        self.lam = lam
        self.dtype = dtype
        self.device = device
        
    def __call__(self, x):
        raise NotImplementedError("This method is implemented in subclasses")
        
    def grad(self, x):
        raise NotImplementedError("This method is implemented in subclasses")
        
    def __str__(self):
        return f"{self.name}_{self.d}"
         
class Ackley(TargetFunction):
 
    def __init__(self, d, a: float = 20.0, b: float = 0.2, c: float = 2 * 3.141592653589793, lam : float=1e-5,
                 dtype=torch.float64, device='cpu'):
        super().__init__("Ackley", d, lam=lam, dtype=dtype, device=device)
        self.a = a
        self.b_coef = b
        self.c = c
 
    @property
    def bounds(self) -> tuple[float, float]:
        return (-32.768, 32.768)
 
    @property
    def x_star(self) -> torch.Tensor:
        return torch.zeros(self.d, dtype=self.dtype, device=self.device)
 
    @property
    def f_star(self) -> float:
        return 0.0
 
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        sum_sq   = (x ** 2).mean(-1)
        sum_cos  = torch.cos(self.c * x).mean(-1)
        term1 = -self.a * torch.exp(-self.b_coef * torch.sqrt(sum_sq))
        term2 = -torch.exp(sum_cos)
        return term1 + term2 + self.a + torch.tensor(1.0, dtype=self.dtype, device=self.device).exp() + self.lam * x.norm(p=2).square()
 
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        xv = x.detach().requires_grad_(True)
        f  = self(xv)
        f.sum().backward()
        return xv.grad.detach()
    
class Rastrigin(TargetFunction):
    
    def __init__(self, d, lam : float=1e-5, dtype=torch.float64, device='cpu'):
        super().__init__("Rastrigin", d, lam=lam, dtype=dtype, device=device)

    @property
    def bounds(self):
        return (-5.12, 5.12)

    @property
    def x_star(self):
        return torch.zeros(self.d, dtype=self.dtype, device=self.device)

    @property
    def f_star(self):
        return 0.0
        
    def __call__(self, x):
        return 10 * self.d + torch.sum(x**2 - 10 * torch.cos(2 * pi * x)) + self.lam * x.norm(p=2).square()

    def grad(self, x):
        xv = x.detach().requires_grad_(True)
        f = self(xv)
        f.sum().backward()
        return xv.grad.detach()


class Levy(TargetFunction):
    def __init__(self, d, lam = 1e-5, dtype=torch.float64, device='cpu'):
        super().__init__("Levy", d, lam=lam, dtype= dtype, device=device)

    @property
    def bounds(self):
        return (-10.0, 10.0)

    @property
    def x_star(self):
        return torch.ones(self.d, dtype=self.dtype, device=self.device)

    @property
    def f_star(self):
        return 0.0

    def __call__(self, x):

        w = 1.0 + (x - 1.0) / 4.0

        term1 = torch.sin(pi * w[0]) ** 2

        if w.shape[0] > 1:
            wi = w[:-1]
            mid = (wi - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(pi * wi + 1.0) ** 2)
            term_mid = torch.sum(mid, axis=-1)
        else:
            term_mid = torch.zeros((w.shape[0],), dtype=x.dtype, device=x.device )

        wd = w[-1]
        term_last = (wd - 1.0) ** 2 * (1.0 + torch.sin(2.0 * pi * wd) ** 2)

        return term1 + term_mid + term_last + self.lam * x.norm(p=2).square()

    def grad(self, x):
        xv = x.detach().requires_grad_(True)
        f = self(xv)
        f.backward()
        return xv.grad.detach()    

