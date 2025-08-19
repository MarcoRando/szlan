import numpy as np 


class DirectionGenerator:
    def __init__(self, 
                 d : int, 
                 l : int, 
                 nrm_const : float = 1.0, 
                 seed : int = 12131415):
        self.d = d 
        self.l = l
        self.nrm_const = nrm_const
        self.rnd_state = np.random.RandomState(seed)

    def __call__(self):
        raise NotImplementedError("This method is implemented in subclasses")
    
class GaussianDirectionGenerator(DirectionGenerator):
    def __init__(self, 
                 d : int, 
                 l : int, 
                 seed : int = 12131415):
        super().__init__(d, l, 1/l, seed)

    def __call__(self):
        return self.rnd_state.randn(self.l, self.d)

class SphericalDirectionGenerator(DirectionGenerator):
    def __init__(self, 
                 d : int, 
                 l : int, 
                 seed : int = 12131415):
        super().__init__(d, l, d/l, seed)

    def __call__(self):
        P = self.rnd_state.randn(self.l, self.d)
        return  P /  np.linalg.norm(P, axis=1, ord=2, keepdims=True)

class QRDirectionGenerator(DirectionGenerator):
    def __init__(self, 
                 d : int, 
                 l : int, 
                 seed : int = 12131415):
        super().__init__(d, l, d/l, seed)

    def __call__(self):
        return np.linalg.qr(self.rnd_state.randn(self.d, self.l))[0].T


class CoordinateDirectionGenerator(DirectionGenerator):
    def __init__(self, 
                 d : int, 
                 l : int, 
                 seed : int = 12131415):
        super().__init__(d, l, d/l, seed)

    def __call__(self):
        indices = self.rnd_state.choice(self.d, self.l, replace=False)
        I = np.zeros((self.l, self.d))
        I[np.arange(self.l), indices] = 1.0
        return I
