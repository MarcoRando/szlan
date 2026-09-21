import torch




class DirectionGenerator:
    def __init__(self, 
                 d : int, # dimension 
                 l : int, # number of directions
                 s : int, # number of matrices
                 nrm_const : float = 1.0,
                 seed : int = 12131415,
                 device : str = "cpu",
                 dtype : torch.dtype = torch.float64
                 ):
        assert s > 0, "Number of matrices s must be greater than 0"
        assert d >= l >0, "Dimension d must be greater or equals to number of directions l which must be greater than 0"

        self.d = d 
        self.l = l
        self.s = s
        self.nrm_const = nrm_const
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.device = device
        self.dtype = dtype

    def __call__(self):
        raise NotImplementedError("This method is implemented in subclasses")
    
    
class GaussianDirectionGenerator(DirectionGenerator):
    def __init__(self, 
                 d : int, 
                 l : int, 
                 s : int,
                 seed : int = 12131415,
                 device : str = "cpu",
                 dtype : torch.dtype = torch.float64                 
                 ):
        super().__init__(d=d, l=l, s=s, nrm_const=1.0, seed=seed, device=device, dtype=dtype)

    def __call__(self):
        return torch.randn((self.s, self.l, self.d), generator=self.generator, device=self.device, dtype=self.dtype)

class SphericalDirectionGenerator(DirectionGenerator):
    def __init__(self, 
                 d : int, 
                 l : int, 
                 s : int,
                 seed : int = 12131415,
                 device : str = "cpu",
                 dtype : torch.dtype = torch.float64                 
                 ):
        super().__init__(d=d, l=l, s=s, nrm_const=d, seed=seed, device=device, dtype=dtype)

    def __call__(self):
        P = torch.randn(size=(self.s,self.d, self.l), dtype=self.dtype, device=self.device, generator=self.generator)
        return (P / P.norm(p=2, dim=1, keepdim=True)).transpose(1,2)
        

