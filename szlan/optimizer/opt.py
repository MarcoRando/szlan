import torch


class Optimizer:
    
    def __init__(self, x0 : torch.Tensor, device : str = "cpu", dtype : torch.dtype = torch.float64, seed : int = 121341):
        self.x0 = x0
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device).manual_seed(seed)
    
    def ask(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def tell(self, X, y):
        raise NotImplementedError("Subclasses should implement this method.")

