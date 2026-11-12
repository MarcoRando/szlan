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

    def __init__(self, d, lam=1e-3, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-100.0, 100.0] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("Salomon", x_star, bounds=bounds, lam=lam, seed=seed)

    def __call__(self, x):
        r = torch.sqrt((x ** 2).sum(1))
        fx = 1.0 - torch.cos(2 * math.pi * r) + 0.1 * r
        return self._add_regularization(x, fx)


class Alpine1(SyntheticTarget):
#    BOUNDS = (-10.0, 10.0)

    def __init__(self, d, lam=1e-3, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-10.0, 10.0] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("Alpine1", x_star, bounds=bounds, lam=lam, seed=seed)

    def __call__(self, x):
        fx = (x * torch.sin(x) + 0.1 * x).abs().sum(1)
        return self._add_regularization(x, fx)


class Ackley(SyntheticTarget):
#    BOUNDS = (-32.768, 32.768)

    def __init__(self, d, lam=1e-3, dtype=torch.float64, device='cuda', seed=1234):
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

    def __init__(self, d, lam=1e-3, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-5.12, 5.12] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("Rastrigin", x_star, bounds=bounds, lam=lam, seed=seed)

    def __call__(self, x):
        fx = 10.0 * x.shape[1] + (x ** 2 - 10.0 * torch.cos(2 * math.pi * x)).sum(1)
        return self._add_regularization(x, fx)

class ZigzagL1(SyntheticTarget):
    # f(x) = sum_i ( |x_i| + A * |x_i - round(x_i)| ), A > 1
    # Triangle-wave "Rastrigin": kinks at every integer and half-integer, local minima at the
    # integers (value |k|), global minimum at 0 with f = 0.

    def __init__(self, d, lam=1e-3, A=4.0, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-10.0, 10.0] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("ZigzagL1", x_star, bounds=bounds, lam=lam, seed=seed)
        self.A = A

    def __call__(self, x):
        fx = (x.abs() + self.A * (x - torch.round(x)).abs()).sum(1)
        return self._add_regularization(x, fx)


class AckleyL1(SyntheticTarget):
    # Ackley with the Euclidean norm replaced by the mean absolute value:
    # f(x) = -a exp(-b * mean|x_i|) - exp(mean cos(c x_i)) + a + e
    # Kinks on every hyperplane x_i = 0, many local minima, global minimum at 0 with f = 0.

    def __init__(self, d, lam=1e-3, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-32.768, 32.768] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("AckleyL1", x_star, bounds=bounds, lam=lam, seed=seed)
        self.a = 20.0
        self.b = 0.2
        self.c = 2 * math.pi

    def __call__(self, x):
        fx = (-self.a * torch.exp(-self.b * x.abs().mean(1))
              - torch.exp(torch.cos(self.c * x).mean(1))
              + self.a + math.e)
        return self._add_regularization(x, fx)

class TakagiL1(SyntheticTarget):
    # Truncated Takagi (blancmange) curve plus an L1 tilt, summed over coordinates:
    # f(x) = sum_i [ sum_{k=0}^{K} 2^{-k} * dist(2^k x_i, Z) + 0.1*|x_i| ]
    # Self-similar sawtooth: kinks at every dyadic rational j / 2^k (k <= K), local minima at the
    # integers (value 0.1*|n|), global minimum f = 0 at x = 0. Valid for any d >= 1.

    def __init__(self, d, lam=1e-3, K=4, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-5.0, 5.0] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("TakagiL1", x_star, bounds=bounds, lam=lam, seed=seed)
        k = torch.arange(K + 1, dtype=dtype, device=device)
        self.freq = 2.0 ** k          # 2^k
        self.weight = 2.0 ** (-k)     # 2^{-k}

    def __call__(self, x):
        y = x.unsqueeze(-1) * self.freq                      # (n, d, K+1)
        takagi = ((y - torch.round(y)).abs() * self.weight).sum(-1)   # (n, d)
        fx = (takagi + 0.1 * x.abs()).sum(1)
        return self._add_regularization(x, fx)
class NesterovChebyshevRosenbrock(SyntheticTarget):
    # f(x) = 1/4 |x_1 - 1| + sum_{i=1}^{d-1} | x_{i+1} - 2|x_i| + 1 |
    # Nesterov's non-smooth Rosenbrock: min f = 0 at x = (1, ..., 1); requires d >= 2.

    def __init__(self, d, lam=1e-3, dtype=torch.float64, device='cuda', seed=1234):
        assert d >= 2, "NesterovChebyshevRosenbrock requires d >= 2"
        x_star = torch.ones((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-5.0, 5.0] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("NesterovChebyshevRosenbrock", x_star, bounds=bounds, lam=lam, seed=seed)

    def __call__(self, x):
        fx = 0.25 * (x[:, 0] - 1.0).abs() \
             + (x[:, 1:] - 2.0 * x[:, :-1].abs() + 1.0).abs().sum(1)
        return self._add_regularization(x, fx)

class YangN2(SyntheticTarget):
    # f(x) = (sum_i |x_i|) * exp(-sum_i sin(x_i^2))
    # Standard Xin-She Yang N.2 benchmark. Non-smooth at x_i = 0 (kink from |x_i|),
    # highly multimodal from the sin(x^2) term, global minimum f = 0 at x = 0. Valid for any d >= 1.

    def __init__(self, d, lam=1e-3, dtype=torch.float64, device='cuda', seed=1234):
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        bounds = torch.tensor([[-10, 10] for _ in range(d)], dtype=dtype, device=device)
        super().__init__("YangN2", x_star, bounds=bounds, lam=lam, seed=seed)

    def __call__(self, x):
        fx = x.abs().sum(1) * torch.exp(-(torch.sin(x ** 2)).sum(1))
        return self._add_regularization(x, fx)

TARGETS = {
    'ackley': Ackley,
    'rastrigin': Rastrigin,
    'alpine1': Alpine1,
    'zigzag_l1': ZigzagL1,
    'ackley_l1': AckleyL1,
    'nesterov_rosenbrock': NesterovChebyshevRosenbrock,
    'yang_n2': YangN2,
    'takagi_l1': TakagiL1
}
 