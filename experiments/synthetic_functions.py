import torch
from math import sqrt, pi, e

from typing import Tuple

class TargetFunction:
    
    def __init__(self, name, d,  dtype = torch.float64, device = 'cpu') -> None:
        self.name = name
        self.d = d
        self.dtype = dtype
        self.device = device
        
    def __call__(self, x):
        raise NotImplementedError("This method is implemented in subclasses")
        
    def grad(self, x):
        raise NotImplementedError("This method is implemented in subclasses")
        
    def __str__(self):
        return f"{self.name}_{self.d}"

 
 
class Quadratic(TargetFunction):
 
    def __init__(self, d, seed=0, dtype=torch.float64, device='cpu'):
        super().__init__("Quadratic", d, dtype=dtype, device=device)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
 
        Q, _ = torch.linalg.qr(
            torch.randn(d, d, generator=rng, dtype=dtype, device=device)
        )
        lam = 1.0 + 9.0 * torch.rand(d, generator=rng, dtype=dtype, device=device)
        self.A = (Q * lam) @ Q.T                        # (d, d)
        self.b = torch.randn(d, generator=rng, dtype=dtype, device=device)  # (d,)


        self._x_star = -torch.linalg.solve(self.A, self.b)   # (d,)
        self._f_star = 0.5 * (self.b @ self._x_star)          # scalar
 
    @property
    def bounds(self) -> Tuple[float, float]:
        return (-10.0, 10.0)
 
    @property
    def x_star(self) -> torch.Tensor:
        return self._x_star
 
    @property
    def f_star(self) -> float:
        return self._f_star.item()
 
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        Ax = x @ self.A.T           # (..., d)
        return 0.5 * (Ax * x).sum(-1) + (x * self.b).sum(-1)
 
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T + self.b
 
 
class LeastSquares(TargetFunction):
     
    def __init__(
        self,
        d: int,
        L:   float = 10.0,   # largest  eigenvalue of A^T A  (smoothness)
        mu:  float =  1.0,   # smallest eigenvalue of A^T A  (strong convexity)
        seed: int  =  5551,
        dtype=torch.float64,
        device='cpu',
    ):
        assert mu > 0,  "mu must be strictly positive"
        assert L >= mu, "L must be >= mu" 
        super().__init__("LeastSquares", d, dtype=dtype, device=device)
        self.L     = L
        self.mu    = mu
        self.generator = torch.Generator(device=device).manual_seed(seed)
        

        self.A = torch.randn((self.d, self.d), dtype = self.dtype, device = self.device, generator=self.generator)
        self._x_star = torch.ones((self.d, ), dtype=self.dtype, device=self.device, requires_grad=False)

        U, S, V = self.A.svd()
        S = torch.linspace(sqrt(L), sqrt(mu), steps = self.A.shape[0], dtype = self.dtype, device = self.device, requires_grad=False)
        self.A = U @ S.diag() @ V
        self.y = self.A @ self.x_star 

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-10.0, 10.0)
 
    @property
    def x_star(self) -> torch.Tensor:
        return self._x_star
 
    @property
    def f_star(self) -> float:
        return 0.0

    def __call__(self, x):
        return 0.5 * (self.A @ x - self.y).norm(p=2)
    
    def grad(self, x):
        return self.A.T @ (self.A @ x - self.y)
        
 
class Rosenbrock(TargetFunction):
 
    def __init__(self, d, a: float = 1.0, b: float = 100.0,
                 dtype=torch.float64, device='cpu'):
        super().__init__("Rosenbrock", d, dtype=dtype, device=device)
        self.a = a
        self.b_coef = b
 
    @property
    def bounds(self) -> Tuple[float, float]:
        return (-5.0, 10.0)
 
    @property
    def x_star(self) -> torch.Tensor:
        return torch.ones(self.d, dtype=self.dtype, device=self.device)
 
    @property
    def f_star(self) -> float:
        return 0.0
 
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xi   = x[..., :-1]   # (..., d-1)
        xi1  = x[..., 1:]    # (..., d-1)
        term1 = self.b_coef * (xi1 - xi ** 2) ** 2
        term2 = (self.a - xi) ** 2
        return (term1 + term2).sum(-1)
 
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(x)
        xi  = x[..., :-1]
        xi1 = x[..., 1:]
        diff = xi1 - xi ** 2                               # (..., d-1)
 
        g[..., :-1] += -4 * self.b_coef * xi * diff - 2 * (self.a - xi)
        g[..., 1:]  +=  2 * self.b_coef * diff
        return g
 
 
 
class Ackley(TargetFunction):
 
    def __init__(self, d, a: float = 20.0, b: float = 0.2,
                 c: float = 2 * 3.141592653589793, lam : float=1e-5,
                 dtype=torch.float64, device='cpu'):
        super().__init__("Ackley", d, dtype=dtype, device=device)
        self.a = a
        self.b_coef = b
        self.c = c
        self.lam = lam
 
    @property
    def bounds(self) -> Tuple[float, float]:
        return (-32.768, 32.768)
 
    @property
    def x_star(self) -> torch.Tensor:
        return torch.zeros(self.d, dtype=self.dtype, device=self.device)
 
    @property
    def f_star(self) -> float:
        return 0.0
 
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        sum_sq   = (x ** 2).mean(-1)
        sum_cos  = torch.cos(self.c * x).mean(-1)
        term1 = -self.a * torch.exp(-self.b_coef * torch.sqrt(sum_sq))
        term2 = -torch.exp(sum_cos)
        return term1 + term2 + self.a + torch.tensor(1.0, dtype=self.dtype, device=self.device).exp() + self.lam * x.norm(p=2).square()
 
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        xv = x.detach().requires_grad_(True)
        f  = self(xv)
        f.sum().backward()
        return xv.grad.detach()
    
class Rastrigin(TargetFunction):
    
    def __init__(self, d, lam : float=1e-5, dtype=torch.float64, device='cpu'):
        super().__init__("Rastrigin", d, dtype=dtype, device=device)
        self.lam = lam 

    @property
    def bounds(self):
        return (-5.12, 5.12)

    @property
    def x_star(self):
        return torch.zeros(self.d, dtype=self.dtype, device=self.device)

    @property
    def f_star(self):
        return 0.0
        
    def __call__(self, x):
        return 10 * self.d + torch.sum(x**2 - 10 * torch.cos(2 * pi * x))

    def grad(self, x):
        xv = x.detach().requires_grad_(True)
        f = self(xv)
        f.sum().backward()
        return xv.grad.detach()


# # class RastriginFunction(TargetFunction):

# #     def __init__(self, d, regularization = 0.0, seed=121314):
# #         super().__init__("Rastrigin", d, regularization, seed)
# #         self.bounds = np.array([[-5.12, 5.12] for _ in range(d)])
# #         self.min_f = 0.0
# #         self.x0 = np.ones((1, self.d))
# #         self.x_star = np.zeros((1, self.d))


# #     def __call__(self, x):

# #         return self._add_regularization(10 * self.d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1), x)
 



# class AckleyFunction(TargetFunction):
#     def __init__(self, d, dtype = torch.float64, device = 'cpu'):
#         super().__init__("Ackley", d, dtype=dtype, device=device)

#         self.bounds = torch.tensor([[-2.768, 2.768] for _ in range(d)], dtype=dtype, device=device)
#         self.min_f = 0.0
#         self.x0 = torch.ones((1, self.d), dtype=dtype, device=device)
#         self.x_star = torch.zeros((1, self.d), dtype=dtype, device=device)
            
#     def __call__(self, x):
#         sum_sq = torch.sum(x ** 2, dim=1)
#         sum_cos = torch.sum(torch.cos(2 * pi * x), dim=1)
#         term1 = -20 * torch.exp(-0.2 * torch.sqrt(sum_sq / self.d))
#         term2 = -torch.exp(sum_cos / self.d)
#         return term1 + term2 + 20 + e 



# class ZigZag(TargetFunction):
#     def __init__(self, d, dtype = torch.float64, device = 'cpu'):
#         super().__init__("ZigZag", d, dtype=dtype, device=device)
#         self.bounds = torch.tensor([[-10.0, 10.0] for _ in range(d)], dtype=dtype, device=device)
#         self.x0 = torch.ones((1, self.d), dtype=dtype, device=device)
#         self.x_star = torch.zeros((1, self.d), dtype=dtype, device=device)
#         self.min_f = -(d - 1)

#     def __call__(self, x):
#         t = torch.remainder(x, 2.0)
#         return 1.0 - torch.sum(torch.abs(t - 1.0), dim=1) + (1/5) * torch.sum(x**2, dim=1)


# class ZigZagSmooth(TargetFunction):

#     def __init__(self, d, dtype=torch.float64, device='cpu'):
#         super().__init__("ZigZagSmooth", d, dtype=dtype, device=device)
# #        self.bounds = np.array([[-1.0, 1.0] for _ in range(d)])
#         self.bounds = torch.tensor([[-10.0, 10.0] for _ in range(d)], dtype=dtype, device=device)
#         self.min_f = -d
#         self.x0 = torch.ones((1, self.d), dtype=dtype, device=device)
#         self.x_star = torch.zeros((1, self.d), dtype=dtype, device=device)

#     def __call__(self, x):
#         t1 = -torch.sum(torch.cos(3*x), dim=1)
#         t2 = (1/5) * torch.sum(x**2, dim=1)
#         return t1 + t2


# class StyblinksiTangFunction(TargetFunction):
#     def __init__(self, d, dtype = torch.float64, device = 'cpu'):
#         super().__init__("StyblinksiTang", d, dtype=dtype, device=device)

#         self.bounds = torch.tensor([[-5.0, 5.0] for _ in range(d)], dtype=dtype, device=device)
#         self.min_f = -39.16599 * d 

#         self.x0 = torch.ones((1, self.d), dtype=dtype, device=device)
#         self.x_star = torch.full((1, self.d), -2.903534, dtype=dtype, device=device)
            
#     def __call__(self, x):
#         return 0.5 * torch.sum(x ** 4 - 16 * x ** 2 + 5 * x, dim=1)

# class SchwefelFunction(TargetFunction):
#     def __init__(self, d, dtype = torch.float64, device='cpu'):
#         super().__init__("Schwefel", d, dtype = dtype, device= device)
#         self.bounds = torch.tensor([[-500.0, 500.0] for _ in range(d)], dtype=dtype, device=device)
#         self.min_f = 0.0 
#         self.x0 = torch.ones((1, self.d), dtype=dtype, device=device)
#         self.x_star = torch.full((1, self.d), 420.9687, dtype=dtype, device=device)
            
#     def __call__(self, x):
#         return 418.9829 * self.d - torch.sum(x * x.abs().sqrt().sin(), dim=1) + 1e-3 * torch.linalg.norm(x - self.x_star, ord=2, dim=1).square()
# #        return self._add_regularization(418.9829 * self.d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1), x)


# class TridFunction(TargetFunction):

#     def __init__(self, d, dtype=torch.float64, device='cpu'):
#         super().__init__("Trid", d, dtype=dtype, device=device)
#         self.bounds = torch.tensor([[-d**2, d**2] for _ in range(d)], dtype=dtype, device=device)
#         self.min_f = - (d * (d + 4) * (d - 1)) / 6
#         self. x0 = torch.zeros((1, d), dtype=dtype, device=device)
    
#     def __call__(self, x):
#         term1 = torch.sum((x - 1)**2, dim=1)
#         term2 = torch.sum(x[:, 1:] * x[:, :-1], dim=1)
#         return term1 - term2 





# class RosenbrockFunction(TargetFunction):
#     def __init__(self, d, dtype = torch.float64, device='cpu'):
#         super().__init__("Rosenbrock", d, dtype=dtype, device=device)
#         self.bounds = torch.tensor([[-2.048, 2.048] for _ in range(d)], dtype=dtype, device=device)
#         self.min_f = 0.0
#         self.x0 = torch.full((1, self.d), 0.5, dtype=dtype, device=device)
#         self.x_star = torch.ones((1, self.d), dtype=dtype, device=device)
            

#     def __call__(self, x):
#         #x = x if x.shape[0] > 1 else x.unsqueeze(0) 
        
#         xi = x[:, :-1]
#         xi1 = x[:, 1:]
#         return torch.sum(100 * (xi1 - xi**2)**2 + (xi - 1)**2, dim=1)



# class GriewankFunction(TargetFunction):
#     def __init__(self, d, dtype=torch.float64, device='cpu'):
#         super().__init__("Griewank", d, dtype=dtype, device=device)
#         self.bounds = torch.tensor([[-600.0, 600.0] for _ in range(d)], dtype=dtype, device=device)
#         self.min_f = 0.0
#         self.x0 = torch.full((1, self.d), 1.0, dtype=dtype, device=device)
#         self.x_star = torch.zeros((1, self.d), dtype=dtype, device=device)

#     def __call__(self, x):
#         d = x.shape[-1]
#         indices = torch.arange(1, d + 1)
#         sum_sq = torch.sum(x ** 2, dim=-1) / 4000
#         prod_cos = torch.prod(torch.cos(x / torch.sqrt(indices)), dim=-1)
#         return 1 + sum_sq - prod_cos




# # class LeastSquares(TargetFunction):
    
# #     def __init__(self, d, mu, L, regularization = 0.0, seed = 1231415) -> None:
# #         assert mu < L
# #         super().__init__(name="LeastSquares", d = d, regularization=regularization, seed=seed)
# #         self.mu = mu
# #         self.L = L
# #         self.bounds = np.array([[-1, 1] for _ in range(d)])
# #         A = self.rnd_state.randn(self.d, self.d)
# #         Q, _ = np.linalg.qr(A)
# #         S = np.diag(np.linspace(sqrt(L), sqrt(mu), self.d))
# #         self.A = Q @ S @ Q.T

# #         self.x_star = self.rnd_state.randn(1, self.d) 
# #         self.x0 = np.ones((1, self.d))
# #         self.y = self.A @ self.x_star.T
# #         self.min_f = 0.0
        
# #     def __call__(self, x):
# #         fx = 0.5 * np.square(np.linalg.norm(self.A @ x.T - self.y, axis=0))
# #         return  fx

# #     def __str__(self):
# #         return f"{self.name}_{self.L}_{self.mu}_{self.d}"
    


# # class QuingFunction(TargetFunction):
# #     def __init__(self, d, regularization = 0.0, seed=121314):
# #         super().__init__("Quing", d, regularization, seed)
# #         self.bounds = np.array([[-500.0, 500.0] for _ in range(d)])
# #         self.min_f = 0.0
# #         self.x0 = np.ones((1, self.d))
# #         self.x_star = np.array([np.sqrt(i) for i in range(1, d + 1)]).reshape(1, -1)
            

# #     def __call__(self, x):
# #         d = x.shape[1]
# #         x = x if x.shape[0] > 1 else x.reshape((1, -1)) #unsqueeze(0)
# #         i = np.arange(1, d + 1).reshape((1, -1))#unsqueeze(0)  # shape (1, d)
# #         return self._add_regularization(np.sum((x ** 2 - i) ** 2, axis=1).squeeze(), x)




# # class TridFunction(TargetFunction):

# #     def __init__(self, d, regularization=0.0, seed=121314):
# #         super().__init__("Trid", d, regularization, seed)
# #         self.bounds = np.array([[-d**2, d**2] for _ in range(d)])
# #         self.min_f = - (d * (d + 4) * (d - 1)) / 6
# #         x0 = np.zeros((1, d))


# #         self.x0 = x0 #torch.ones((1, self.d))
    
# #     def __call__(self, x):
# #         term1 = np.sum(np.square(x - 1), axis=1)#, keepdims=True) #.square().sum(dim=1, keepdim=True)  
# #         term2 = np.sum(x[:, 1:] * x[:, :-1], axis=1)#, keepdims=True)  
# #         return self._add_regularization(term1 - term2, x)  # Compute function


# # class AckleyFunction(TargetFunction):
# #     def __init__(self, d, regularization = 0.0, seed=121314):
# #         super().__init__("Ackley", d, regularization, seed)
# #         self.bounds = np.array([[-32.768, 32.768] for _ in range(d)])
# #         self.min_f = 0.0
# #         self.x0 = np.ones((1, self.d))
# #         self.x_star = np.zeros((1, self.d))
            
# #     def __call__(self, x):
# #         d = x.shape[1]
# #         sum_sq = np.sum(x ** 2, axis=1)
# #         sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
# #         term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / d))
# #         term2 = -np.exp(sum_cos / d)
# #         return self._add_regularization(term1 + term2 + 20 + np.e + np.linalg.norm(x, axis=1)**2, x)

# #     def grad(self, x):
# # #        a=20, b=0.2, c=2*np.pi
# #         x = np.atleast_2d(x)  # Ensure shape (m, n)
# #         n = x.shape[1]


# #         s1 = np.mean(x**2, axis=1, keepdims=True)
# #         s2 = np.mean(np.cos(2 * np.pi * x), axis=1, keepdims=True)
        
# #         sqrt_s1 = np.sqrt(s1)
        
# #         # Avoid division by zero (at x=0)
# #         denom = np.where(sqrt_s1 == 0, 1, sqrt_s1)
        
# #         term1 = 20 * 0.2 * np.exp(-0.2 * sqrt_s1) * (x / (n * denom))
# #         term2 = (2*np.pi / n) * np.sin(2*np.pi * x) * np.exp(s2)
        
# #         grad = term1 + term2
        
# #         # If input was 1D, return 1D
# #         return grad 
        

# # class StyblinksiTangFunction(TargetFunction):
# #     def __init__(self, d,regularization = 0.0, seed=121314):
# #         super().__init__("StyblinksiTang", d,regularization, seed)
# #         self.bounds = np.array([[-5.0, 5.0] for _ in range(d)])
# #         self.min_f = -39.16599 * d 
# #         self.x0 = np.ones((1, self.d))
# #         self.x_star = np.full((1, self.d), -2.903534)
            
# #     def __call__(self, x):
# #         return self._add_regularization(np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1) / 2, x)


# # class RastriginFunction(TargetFunction):

# #     def __init__(self, d, regularization = 0.0, seed=121314):
# #         super().__init__("Rastrigin", d, regularization, seed)
# #         self.bounds = np.array([[-5.12, 5.12] for _ in range(d)])
# #         self.min_f = 0.0
# #         self.x0 = np.ones((1, self.d))
# #         self.x_star = np.zeros((1, self.d))


# #     def __call__(self, x):

# #         return self._add_regularization(10 * self.d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1), x)

# # class SchwefelFunction(TargetFunction):
# #     def __init__(self, d, regularization = 0.0, seed=121314):
# #         super().__init__("Schwefel", d, regularization,seed)
# #         self.bounds = np.array([[-500.0, 500.0] for _ in range(d)])
# #         self.min_f = 0.0 
# #         self.x0 = np.ones((1, self.d))
            
# #     def __call__(self, x):
# #         return self._add_regularization(418.9829 * self.d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1), x)

# # class LevyFunction(TargetFunction):
# #     def __init__(self, d, regularization = 0.0,seed=121314):
# #         super().__init__("Levy", d, regularization,seed)
# #         self.bounds = np.array([[-10.0, 10.0] for _ in range(d)])
# #         self.min_f = 0.0 
# #         self.x0 = np.zeros((1, self.d))
# #         self.x_star = np.ones((1, self.d))

# #     def __call__(self, x):

# #         w = 1.0 + (x - 1.0) / 4.0

# #         # Term 1: sin^2(pi * w1)
# #         term1 = np.sin(np.pi * w[:, 0]) ** 2

# #         # Middle sum over i = 1..d-1 (handle d=1 gracefully)
# #         if w.shape[1] > 1:
# #             wi = w[:, :-1]
# #             mid = (wi - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * wi + 1.0) ** 2)
# #             term_mid = np.sum(mid, axis=1)
# #         else:
# #             term_mid = np.zeros(w.shape[0])

# #         # Last term uses w_d
# #         wd = w[:, -1]
# #         term_last = (wd - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * wd) ** 2)

# #         return self._add_regularization(term1 + term_mid + term_last, x)  #+ 10 * np.square(np.linalg.norm(x, axis=1))

# #     def grad(self, x):

# #         x = np.atleast_2d(x)
# #         m, n = x.shape

# #         w = 1 + (x - 1) / 4.0

# #         grad = np.zeros_like(x)

# #         sin_pi_w = np.sin(np.pi * w)
# #         cos_pi_w = np.cos(np.pi * w)
# #         sin_pi_w1 = np.sin(np.pi * w + 1)
# #         cos_pi_w1 = np.cos(np.pi * w + 1)
# #         sin_2pi_wn = np.sin(2 * np.pi * w[:, -1:])
# #         cos_2pi_wn = np.cos(2 * np.pi * w[:, -1:])

# #         term1 = 2 * np.pi * sin_pi_w[:, 0:1] * cos_pi_w[:, 0:1]
# #         term2 = 2 * (w[:, 0:1] - 1) * (1 + 10 * sin_pi_w1[:, 0:1] ** 2)
# #         term3 = 20 * np.pi * (w[:, 0:1] - 1) ** 2 * sin_pi_w1[:, 0:1] * cos_pi_w1[:, 0:1]
# #         grad[:, 0:1] = 0.25 * (term1 + term2 + term3)

# #         if n > 2:
# #             wi = w[:, 1:-1]
# #             wi_prev = w[:, :-2]
# #             grad[:, 1:-1] = 0.25 * (
# #                 2 * (wi - 1) * (1 + 10 * np.sin(np.pi * wi + 1) ** 2)
# #                 + 20 * np.pi * (wi - 1) ** 2 * np.sin(np.pi * wi + 1) * np.cos(np.pi * wi + 1)
# #                 + 2 * np.pi * np.sin(np.pi * wi) * np.cos(np.pi * wi)
# #             )


# #         wn = w[:, -1:]
# #         grad[:, -1:] = 0.25 * (
# #             2 * (wn - 1) * (1 + np.sin(2 * np.pi * wn) ** 2)
# #             + 4 * np.pi * (wn - 1) ** 2 * np.sin(2 * np.pi * wn) * np.cos(2 * np.pi * wn)
# #             + 2 * np.pi * np.sin(np.pi * wn) * np.cos(np.pi * wn)
# #         )

# #         return grad




# # class ZigZagSmooth(TargetFunction):

# #     def __init__(self, d, regularization = 0.0, seed=121314):
# #         super().__init__("ZigZagSmooth", d, regularization, seed)
# #         self.bounds = np.array([[-1.0, 1.0] for _ in range(d)])
# #         self.min_f = -d
# #         self.x0 = np.ones((1, self.d))
# #         self.x_star = np.zeros((1, self.d))

# #     def __call__(self, x):
# #         t1 = -np.sum(np.cos(3*x), axis=1)
# #         t2 = (1/5) * np.sum(x**2, axis=1)

# #         return self._add_regularization(t1 + t2, x)

# # class ZigZag(TargetFunction):
# #     def __init__(self, d, regularization = 0.0, seed=121314):
# #         super().__init__("ZigZag", d, regularization, seed)
# #         self.bounds = np.array([[-2.0, 2.0] for _ in range(d)])
# #         self.x0 = np.ones((1, self.d))
# #         self.x_star = np.zeros((1, self.d))
# #         self.min_f = -(d - 1)

# #     def __call__(self, x):
# #         t = np.mod(x, 2.0)
# #         return self._add_regularization(1.0 - np.sum(np.abs(t - 1.0), axis=1) + (1/5) * np.sum(x**2, axis=1), x)

# # class SumExpTarget(TargetFunction):
# #     def __init__(self, d, regularization = 0.0, seed=121314):
# #         super().__init__("SumExp", d, regularization, seed)
# #         self.bounds = np.array([[-1.0, 1.0] for _ in range(d)])
# #         self.min_f = 0.0
# #         self.x0 = np.ones((1, self.d))
# #         self.x_star = np.zeros((1, self.d))
# #         self.v1 = 0.1
# #         self.v2 = 0.5

# #     def __call__(self, x):
# #         t1 = - np.sum(np.exp(-(x+2)**2/self.v1), axis=1)/np.sqrt(self.v1)
# #         t2 = - np.sum(1.5 * np.exp(-(x-2)**2/self.v2), axis=1)/np.sqrt(self.v2)
# #         t3 = 0.1 * np.sum(x**2, axis=1) + 1
# #         return self._add_regularization(t1 + t2 + t3, x)




# # class BukinFunction(TargetFunction):
# #     def __init__(self, regularization = 0.0, seed=121314):
# #         super().__init__("Bukin", 2, regularization=regularization, seed=seed)
# #         self.bounds = np.array([[-15.0, -5.0], [-3.0, 3.0]])
# #         self.min_f = 0.0 
# #         self.x0 = np.array([[-10.0, 1.0]])
            
# #     def __call__(self, x):
# #         x1 = x[:, 0]
# #         x2 = x[:, 1]
# #         term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2))
# #         term2 = 0.01 * np.abs(x1 + 10)
# #         return self._add_regularization(term1 + term2, x)

# # # class PycutestTarget(TargetFunction):
    
# # #     def __init__(self, pycutest_problem, dtype = torch.float64, device = 'cpu', seed = 121314, x0 = None) -> None:
# # #         self.fun = pycutest_problem# pycutest.import_problem(problem_name)        
# # #         super().__init__("CUTest", d = self.fun.n, dtype = dtype, device= device, seed=seed)
# # #         if x0 is None:
# # #             self.x0 = torch.from_numpy(self.fun.x0).to(device=device, dtype=dtype).reshape(1, -1)
# # #         else:
# # #             self.x0 = torch.from_numpy(x0).to(device=device, dtype=dtype).reshape(1, -1)
            
            
# # #     def __call__(self, x):
# # #         res = np.apply_along_axis(self.fun.obj, axis=1, arr=x.cpu().numpy())

# # #         fx = torch.from_numpy(res).to(dtype=self.dtype, device=self.device)
# # #         return  fx

# # #     def grad(self, x):
# # #         return torch.from_numpy(self.fun.grad(x.cpu().numpy().flatten())).to(device=self.device, dtype=self.dtype)

# # #     def close(self):
# # #         if hasattr(self.fun, 'close'):
# # #             self.fun.close()












