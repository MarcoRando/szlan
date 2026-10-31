import torch


class Optimizer:

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float64, seed: int = 0):
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device).manual_seed(seed)