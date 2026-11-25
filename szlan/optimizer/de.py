import torch

from szlan.optimizer.opt import Optimizer


class DifferentialEvolution(Optimizer):
    

    def __init__(self,
                 x0: torch.Tensor,
                 n_particles: int = 50,
                 init_std: float = 1.0,
                 F: float = 0.8,
                 CR: float = 0.9,
                 strategy: str = "rand1",
                 batch_size: int | None = None,
                 seed: int = 1231415,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float64):
        super().__init__(device=device, dtype=dtype, seed=seed)
        assert strategy in ("rand1", "best1"), "strategy must be 'rand1' or 'best1'"
        assert n_particles >= (4 if strategy == "rand1" else 3), "n_particles too small for the strategy"
        assert 0.0 <= CR <= 1.0, "CR must be in [0, 1]"

        x0 = x0.to(device=device, dtype=dtype).reshape(-1)
        d = x0.numel()
        P = torch.randn((n_particles, d), dtype=dtype, device=device, generator=self.generator)
        self.X = x0.view(1, -1) + init_std * P
        self.X[0] = x0                                   # keep the starting point in the population

        self.n_particles = n_particles
        self.batch_size = n_particles if batch_size is None else min(batch_size, n_particles)
        assert self.batch_size > 0, "batch_size must be positive"

        self.F = F
        self.CR = CR
        self.strategy = strategy

        self.k = 0
        self.f_pop = torch.full((n_particles,), float("inf"), dtype=dtype, device=device)
        self.x_best = x0.clone()
        self.f_best = None
        self.x = self.x_best

        self._init_phase = True          # first pass evaluates the initial population
        self._U = None                   # trial vectors of the current generation
        self._pos = 0
        self._stop = 0
        self._y = torch.empty(n_particles, dtype=dtype, device=device)
        self._pending = False

    def _make_trials(self) -> torch.Tensor:
        NP, d = self.X.shape
        kw = dict(dtype=self.X.dtype, device=self.X.device, generator=self.generator)

        # distinct random indices, all different from i
        R = torch.rand((NP, NP), **kw)
        R.fill_diagonal_(2.0)                            # push i to the end of the ordering
        idx = R.argsort(dim=1)[:, :3]

        if self.strategy == "rand1":
            V = self.X[idx[:, 0]] + self.F * (self.X[idx[:, 1]] - self.X[idx[:, 2]])
        else:
            V = self.x_best.view(1, -1) + self.F * (self.X[idx[:, 0]] - self.X[idx[:, 1]])

        mask = torch.rand((NP, d), **kw) < self.CR
        j_rand = torch.randint(d, (NP,), device=self.X.device, generator=self.generator)
        mask[torch.arange(NP, device=self.X.device), j_rand] = True   # at least one gene from V
        return torch.where(mask, V, self.X)

    def ask(self) -> torch.Tensor:
        if self._pending:
            raise RuntimeError("tell() must be called before the next ask()")
        if not self._init_phase and self._U is None:
            self._U = self._make_trials()
        src = self.X if self._init_phase else self._U
        self._stop = min(self._pos + self.batch_size, self.n_particles)
        self._pending = True
        return src[self._pos:self._stop].clone()

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

        if self._init_phase:
            self.f_pop = self._y.clone()
            self._init_phase = False
        else:
            improved = self._y <= self.f_pop             # greedy selection
            self.f_pop = torch.where(improved, self._y, self.f_pop)
            self.X = torch.where(improved.view(-1, 1), self._U, self.X)
            self._U = None

        g = self.f_pop.argmin()
        if self.f_best is None or self.f_pop[g] < self.f_best:
            self.f_best = self.f_pop[g].clone()
            self.x_best = self.X[g].clone()
        self.x = self.x_best

        self._pos = 0
        self.k += 1
        return True