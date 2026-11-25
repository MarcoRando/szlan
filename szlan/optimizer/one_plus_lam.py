from math import exp

import torch

from szlan.optimizer.opt import Optimizer


class OnePlusLamES(Optimizer):

    def __init__(self,
                 x0: torch.Tensor,
                 sigma: float = 0.1,
                 inc: float = 1.5,
                 batch_size: int = 1,
                 sigma_min: float = 0.0,
                 sigma_max: float = float("inf"),
                 seed: int = 1231415,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float64):
        super().__init__(device=device, dtype=dtype, seed=seed)
        assert batch_size > 0, "batch_size must be positive"
        assert sigma > 0, "sigma must be positive"
        assert inc > 1.0, "inc must be > 1"

        self.x = x0.to(device=device, dtype=dtype).reshape(-1).clone()
        self.sigma = sigma
        self.inc = inc
        self.dec = inc ** (-0.25)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.batch_size = batch_size

        self.k = 0
        self.f_x = None          # f(x), known after the first tell
        self._pending = False
        self._with_base = False

    def ask(self) -> torch.Tensor:
        if self._pending:
            raise RuntimeError("tell() must be called before the next ask()")
        Z = torch.randn((self.batch_size, self.x.numel()), dtype=self.x.dtype,
                        device=self.x.device, generator=self.generator)
        X = self.x.view(1, -1) + self.sigma * Z

        self._with_base = self.f_x is None
        if self._with_base:
            X = torch.cat((X, self.x.view(1, -1)), dim=0)
        self._pending = True
        return X

    def tell(self, X, y) -> bool:
        if not self._pending:
            raise RuntimeError("ask() must be called before tell()")
        y = y.reshape(-1).to(self.x.dtype)
        if self._with_base:
            y_cand, self.f_x = y[:-1], y[-1]
            X_cand = X[:-1]
        else:
            y_cand, X_cand = y, X
        assert y_cand.numel() == self.batch_size, "number of values does not match the last ask()"

        best = y_cand.argmin()
        if y_cand[best] <= self.f_x:
            success = bool(y_cand[best] < self.f_x)
            self.x = X_cand[best].reshape(-1).clone()
            self.f_x = y_cand[best]
        else:
            success = False

        self.sigma = min(max(self.sigma * (self.inc if success else self.dec), self.sigma_min), self.sigma_max)

        self._pending = False
        self.k += 1
        return True