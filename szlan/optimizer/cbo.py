from collections.abc import Callable

import torch

from szlan.optimizer.opt import Optimizer


class CBO(Optimizer):


    def __init__(self,
                 x0: torch.Tensor,
                 n_particles: int = 50,
                 init_std: float = 1.0,
                 lam: float  = 1.0,
                 sigma: float  = 1.0,
                 alpha: float = 10.0,
                 dt: float = 0.1,
                 anisotropic: bool = True,
                 batch_size: int | None = None,
                 seed: int = 1231415,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float64):
        super().__init__(device=device, dtype=dtype, seed=seed)
        assert n_particles > 0, "n_particles must be positive"
        assert dt > 0, "dt must be positive"

        x0 = x0.to(device=device, dtype=dtype).reshape(-1)
        P = torch.randn((n_particles, x0.numel()), dtype=dtype,
                        device=device, generator=self.generator)
        self.X = x0.view(1, -1) + init_std * P

        self.n_particles = n_particles
        self.batch_size = n_particles if batch_size is None else min(batch_size, n_particles)
        assert self.batch_size > 0, "batch_size must be positive"

        self.lam = lam 
        self.sigma = sigma 
        self.alpha = alpha 
        self.dt = dt
        self.anisotropic = anisotropic

        self.k = 0
        self.x = self.X.mean(dim=0)      # current consensus point
        self.x_best = None               # best particle seen so far
        self.f_best = None

        self._pos = 0                    # next particle index within the round
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
        X = X.reshape(y.numel(), -1).to(device=self.X.device, dtype=self.X.dtype)

        self._y[self._pos:self._stop] = y
        best = y.argmin()
        if self.f_best is None or y[best] < self.f_best:
            self.f_best = y[best].clone()
            self.x_best = X[best].clone()

        self._pending = False
        self._pos = self._stop
        if self._pos < self.n_particles:
            return False

        y_all = self._y

        w = torch.softmax(-self.alpha * (y_all - y_all.min()), dim=0)
        self.x = (w.view(-1, 1) * self.X).sum(dim=0)

        diff = self.X - self.x.view(1, -1)
        Z = torch.randn(self.X.shape, dtype=self.X.dtype, device=self.X.device, generator=self.generator)
        if self.anisotropic:
            noise = diff * Z
        else:
            noise = diff.norm(dim=1, keepdim=True) * Z
        self.X = self.X - self.lam * self.dt * diff + self.sigma *  noise

        self._pos = 0
        self.k += 1
        return True