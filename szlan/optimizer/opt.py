import torch

class Optimizer:
    
    def __init__(self, device : str = "cpu", dtype : torch.dtype = torch.float64):
        self.device = device
        self.dtype = dtype
    
    def ask(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def tell(self, x, y):
        raise NotImplementedError("Subclasses should implement this method.")

    # def recommend(self):
    #     raise NotImplementedError("Subclasses should implement this method.")