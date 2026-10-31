import math

import torch


class SyntheticTarget:

    def __init__(self, name, x_star, bounds, lam=1e-3, seed = 1234):
        self.name = name
        self.x_star = x_star
        self.d = x_star.shape[-1]
        self.lam = lam

        self.bounds = bounds
        
        self.generator = torch.Generator(self.x_star.device).manual_seed(seed)

        # low, high = bounds
        # self.lower = torch.full_like(x_star, low)
        # self.upper = torch.full_like(x_star, high)

    def _add_regularization(self, x, fx):
        return fx + self.lam * ((x - self.x_star) ** 2).sum(1)

    def sample(self, n):
        u = torch.rand((n, self.d), dtype=self.x_star.dtype,device=self.x_star.device, generator=self.generator)
        return self.bounds[:, 0] + u * (self.bounds[:, 1] - self.bounds[:, 0])

    def __call__(self, x):
        raise NotImplementedError("Method implemented in subclasses!")


class Salomon(SyntheticTarget):
#    BOUNDS = (-100.0, 100.0)

    def __init__(self, d, lam=0.1, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-100.0, 100.0] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("Salomon", x_star, bounds=bounds, lam=lam, seed=seed)

    def __call__(self, x):
        r = torch.sqrt((x ** 2).sum(1))
        fx = 1.0 - torch.cos(2 * math.pi * r) + 0.1 * r
        return self._add_regularization(x, fx)


class Alpine1(SyntheticTarget):
#    BOUNDS = (-10.0, 10.0)

    def __init__(self, d, lam=1.0, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-10.0, 10.0] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("Alpine1", x_star, bounds=bounds, lam=lam, seed=seed)

    def __call__(self, x):
        fx = (x * torch.sin(x) + 0.1 * x).abs().sum(1)
        return self._add_regularization(x, fx)


class Ackley(SyntheticTarget):
#    BOUNDS = (-32.768, 32.768)

    def __init__(self, d, lam=0.1, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-32.768, 32.768] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("Ackley", x_star, bounds=bounds, lam=lam, seed=seed)
        self.a = 20.0
        self.b = 0.2
        self.c = 2 * math.pi

    def __call__(self, x):
        fx = (-self.a * torch.exp(-self.b * torch.sqrt((x ** 2).sum(1) / self.d))
              - torch.exp(torch.cos(self.c * x).sum(1) / self.d)
              + self.a + math.e)
        return self._add_regularization(x, fx)


class Rastrigin(SyntheticTarget):
#    BOUNDS = (-5.12, 5.12)

    def __init__(self, d, lam=0.1, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-5.12, 5.12] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("Rastrigin", x_star, bounds=bounds, lam=lam, seed=seed)

    def __call__(self, x):
        fx = 10.0 * x.shape[1] + (x ** 2 - 10.0 * torch.cos(2 * math.pi * x)).sum(1)
        return self._add_regularization(x, fx)


    
TARGETS = {
    'ackley' : Ackley,
    'rastrigin' : Rastrigin,
    'alpine1' : Alpine1
}