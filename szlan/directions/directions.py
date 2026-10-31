import torch


def _mix(*ints: int) -> int:
    mask = 0xFFFFFFFFFFFFFFFF
    h = 0x9E3779B97F4A7C15
    for i in ints:
        h = ((h ^ (int(i) & mask)) * 0xBF58476D1CE4E5B9) & mask
        h ^= h >> 31
    return h & 0x7FFFFFFFFFFFFFFF


class DirectionGenerator:
    def __init__(self,
                 d: int,  # dimension
                 l: int,  # number of directions per matrix
                 s: int,  # number of matrices
                 nrm_const: float = 1.0,
                 seed: int = 12131415,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float64):
        assert s > 0, "Number of matrices s must be greater than 0"
        assert d >= l > 0, "Dimension d must be >= number of directions l, which must be > 0"

        self.d = d
        self.l = l
        self.s = s
        self.nrm_const = nrm_const
        self.device = device
        self.dtype = dtype

        self._seed_gen = torch.Generator().manual_seed(seed)  # CPU, draws round seeds
        self._scratch = torch.Generator(device=device)         # reseeded on demand
        self.round_seed = 0
        self.reset()

    @property
    def n_directions(self) -> int:
        return self.s * self.l

    def reset(self):
        self.round_seed = int(torch.randint(0, 2**62, (1,), generator=self._seed_gen))

    def directions(self, start: int = 0, stop: int | None = None) -> torch.Tensor:
        stop = self.n_directions if stop is None else stop
        assert 0 <= start < stop <= self.n_directions, "invalid direction range"
        return self._directions(start, stop)

    def __call__(self) -> torch.Tensor:
        self.reset()
        return self.directions(0, self.n_directions).view(self.s, self.l, self.d)

    def _gen(self, *tags: int) -> torch.Generator:
        return self._scratch.manual_seed(_mix(self.round_seed, *tags))

    def _directions(self, start: int, stop: int) -> torch.Tensor:
        raise NotImplementedError("This method is implemented in subclasses")


class GaussianDirectionGenerator(DirectionGenerator):
    def __init__(self, d, l, s, seed=12131415, device="cpu", dtype=torch.float64):
        super().__init__(d=d, l=l, s=s, nrm_const=1.0, seed=seed, device=device, dtype=dtype)

    def _directions(self, start, stop):
        return torch.randn((stop - start, self.d), generator=self._gen(start), device=self.device, dtype=self.dtype)


class SphericalDirectionGenerator(DirectionGenerator):
    def __init__(self, d, l, s, seed=12131415, device="cpu", dtype=torch.float64):
        super().__init__(d=d, l=l, s=s, nrm_const=d, seed=seed, device=device, dtype=dtype)

    def _directions(self, start, stop):
        P = torch.randn((stop - start, self.d), generator=self._gen(start),device=self.device, dtype=self.dtype)
        return P / P.norm(dim=1, keepdim=True)


class HaarDirectionGenerator(DirectionGenerator):

    def __init__(self, d, l, s, seed=12131415, device="cpu", dtype=torch.float64):
        super().__init__(d=d, l=l, s=s, nrm_const=d, seed=seed, device=device, dtype=dtype)

    def _reflector(self, block: int, k: int) -> torch.Tensor:
        x = torch.randn(self.d - k, generator=self._gen(block, k), device=self.device, dtype=self.dtype)
        x = x / x.norm()
        v = x.clone()
        v[0] = x[0] + torch.where(x[0] >= 0, 1.0, -1.0)   # no host sync (stays on device)
        return v / v.norm()

    def _signs(self, block: int) -> torch.Tensor:
        r = torch.rand(self.l, generator=self._gen(block, self.d), device=self.device, dtype=self.dtype)
        return 1.0 - 2.0 * (r < 0.5).to(self.dtype)

    def _columns(self, block: int, cols: torch.Tensor) -> torch.Tensor:
        n = cols.numel()
        cols = cols.to(self.device)
        X = torch.zeros((self.d, n), device=self.device, dtype=self.dtype)
        X[cols, torch.arange(n, device=self.device)] = self._signs(block)[cols]

        for k in reversed(range(min(self.l, self.d - 1))):
            v = self._reflector(block, k)
            sub = X[k:]                                   # view: reflector only touches rows k:
            sub -= 2.0 * torch.outer(v, v @ sub)          # (I - 2 v v^T) sub
        return X.T

    def _directions(self, start, stop):
        out = torch.empty((stop - start, self.d), device=self.device, dtype=self.dtype)
        for b in range(start // self.l, (stop - 1) // self.l + 1):
            lo, hi = max(start, b * self.l), min(stop, (b + 1) * self.l)
            cols = torch.arange(lo - b * self.l, hi - b * self.l)
            out[lo - start:hi - start] = self._columns(b, cols)
        return out


GENERATORS = {
    "gaussian": GaussianDirectionGenerator,
    "spherical": SphericalDirectionGenerator,
    "structured": HaarDirectionGenerator,
}