from collections.abc import Callable
from enum import Enum
from math import sqrt

import torch

from szlan.optimizer.opt import Optimizer
from szlan.utils.utils import _ffd_accumulate


class SZLanPhase(Enum):
    INITIALIZATION = 0
    ITERATE = 1
    GRADIENT_APPROX = 2


class SZLan(Optimizer):

    def __init__(self,
                 x0: torch.Tensor,
                 direction_generator,
                 h: float | Callable[[int], float] = 1e-7,
                 gamma: float | Callable[[int], float] = 0.1,
                 beta: float | Callable[[int], float] = 1.0,
                 batch_size: int | None = None,
                 seed: int = 1231415,
                 device: str = "cpu",
                 only_fd=False,
                 dtype: torch.dtype = torch.float64):
        super().__init__(device=device, dtype=dtype, seed=seed)
        self.x0 = x0
        self.direction_generator = direction_generator

        self.h = h if isinstance(h, Callable) else lambda _: h
        self.gamma = gamma if isinstance(gamma, Callable) else lambda _: gamma
        self.beta = beta if isinstance(beta, Callable) else lambda _: beta
        self.only_fd = only_fd

        n = self.direction_generator.n_directions
        self.batch_size = n if batch_size is None else min(batch_size, n)
        assert self.batch_size > 0, "batch_size must be positive"

        self.k = 0
        self._pos = 0                       # next direction index within the round
        self._acc = torch.zeros_like(x0)    # running sum of (f(x+hp)-f(x))/h * p
        self._y_curr = None                 # f(x0), evaluated on the first ask of a round
        self._P = None                      # cached chunk between ask and tell
        self._stop = 0
        self._with_base = False


    @property
    def round_start(self) -> bool:
        return self._with_base

    def ask(self) -> torch.Tensor:
        if self._P is not None:
            raise RuntimeError("tell() must be called before the next ask()")
        stop = min(self._pos + self.batch_size, self.direction_generator.n_directions)
        P = self.direction_generator.directions(self._pos, stop).to(device=self.x0.device, dtype=self.x0.dtype)
        self._P, self._stop = P, stop
        self._with_base = self._y_curr is None

        X = self.x0.view(1, -1) + self.h(self.k) * P
        if self._with_base:
            X = torch.cat((X, self.x0.view(1, -1)), dim=0)
        return X

    def tell(self, X, y) -> bool:
        if self._P is None:
            raise RuntimeError("ask() must be called before tell()")
        h_k = self.h(self.k)
        y = y.reshape(-1).to(self.x0.dtype)
        if self._with_base:
            y_fwd, self._y_curr = y[:-1], y[-1]
        else:
            y_fwd = y
        assert y_fwd.numel() == self._P.shape[0], "number of values does not match the last ask()"

        self._acc += _ffd_accumulate(y_fwd, self._y_curr, self._P, h_k)
        self._P = None
        self._pos = self._stop
        n = self.direction_generator.n_directions
        if self._pos < n:
            return False

        gamma_k = self.gamma(self.k)
        beta_k = self.beta(self.k)
        g_k = self.direction_generator.nrm_const * self._acc / n
        self.x0 = self.x0 - gamma_k * g_k
        if not self.only_fd:
            z_k = torch.randn(self.x0.shape, dtype=self.x0.dtype, device=self.x0.device, generator=self.generator)
            self.x0 = self.x0 + sqrt(2 * gamma_k / beta_k) * z_k

        self._acc.zero_()
        self._y_curr = None
        self._pos = 0
        self.direction_generator.reset()
        self.k += 1
        return True