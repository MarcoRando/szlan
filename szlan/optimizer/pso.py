import torch

from szlan.optimizer.opt import Optimizer


class PSO(Optimizer):

    def __init__(self,
                 x0: torch.Tensor,
                 n_particles: int = 50,
                 init_std: float = 1.0,
                 v_init_std: float = 0.0,
                 w: float  = 0.7298,
                 c1: float  = 1.49618,
                 c2: float  = 1.49618,
                 v_max: float | None = None,
                 batch_size: int | None = None,
                 seed: int = 1231415,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float64):
        super().__init__(device=device, dtype=dtype, seed=seed)
        assert n_particles > 0, "n_particles must be positive"

        x0 = x0.to(device=device, dtype=dtype).reshape(-1)
        d = x0.numel()
        P = torch.randn((n_particles, d), dtype=dtype, device=device, generator=self.generator)
        self.X = x0.view(1, -1) + init_std * P
        self.X[0] = x0                                   # keep the starting point in the swarm
        self.V = v_init_std * torch.randn((n_particles, d), dtype=dtype, device=device,
                                          generator=self.generator)

        self.n_particles = n_particles
        self.batch_size = n_particles if batch_size is None else min(batch_size, n_particles)
        assert self.batch_size > 0, "batch_size must be positive"

        self.w = w 
        self.c1 = c1 
        self.c2 = c2 
        self.v_max = v_max

        self.k = 0
        self.P_best = self.X.clone()                     # personal bests
        self.f_pbest = torch.full((n_particles,), float("inf"), dtype=dtype, device=device)
        self.x_best = x0.clone()                         # global best
        self.f_best = None
        self.x = self.x_best

        self._pos = 0
        self._stop = 0
        self._y = torch.empty(n_particles, dtype=dtype, device=device)
        self._pending = False

    def ask(self) -> torch.Tensor:
        if self._pending:
            raise RuntimeError("tell() must be called before the next ask()")
        self._stop = min(self._pos + self.batch_size, self.n_particles)
        self._pending = True
        return self.X[self._pos:self._stop].clone()

    def tell(self, X, y) -> bool:
        if not self._pending:
            raise RuntimeError("ask() must be called before tell()")
        y = y.reshape(-1).to(device=self.X.device, dtype=self.X.dtype)
        assert y.numel() == self._stop - self._pos, "number of values does not match the last ask()"

        self._y[self._pos:self._stop] = y
        self._pending = False
        self._pos = self._stop
        if self._pos < self.n_particles:
            return False

        improved = self._y < self.f_pbest
        self.f_pbest = torch.where(improved, self._y, self.f_pbest)
        self.P_best = torch.where(improved.view(-1, 1), self.X, self.P_best)

        g = self.f_pbest.argmin()
        if self.f_best is None or self.f_pbest[g] < self.f_best:
            self.f_best = self.f_pbest[g].clone()
            self.x_best = self.P_best[g].clone()
        self.x = self.x_best

        shape = self.X.shape
        r1 = torch.rand(shape, dtype=self.X.dtype, device=self.X.device, generator=self.generator)
        r2 = torch.rand(shape, dtype=self.X.dtype, device=self.X.device, generator=self.generator)
        self.V = (self.w * self.V
                  + self.c1 * r1 * (self.P_best - self.X)
                  + self.c2 * r2 * (self.x_best.view(1, -1) - self.X))
        if self.v_max is not None:
            self.V = self.V.clamp(-self.v_max, self.v_max)
        self.X = self.X + self.V

        self._pos = 0
        self.k += 1
        return True