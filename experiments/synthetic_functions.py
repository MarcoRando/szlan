import numpy as np 


from math import sqrt

class TargetFunction:
    
    def __init__(self, name, d, seed=131415) -> None:
        self.name = name
        self.d = d
        self.rnd_state = np.random.RandomState(seed)
        
    def __call__(self, x):
        pass
        
    def close(self):
        pass
        
    def __str__(self):
        return f"{self.name}_{self.d}"

class LeastSquares(TargetFunction):
    
    def __init__(self, d, mu, L, seed = 1231415) -> None:
        assert mu < L
        super().__init__(name="LeastSquares", d = d, seed=seed)
        self.mu = mu
        self.L = L
        self.bounds = np.array([[-1, 1] for _ in range(d)])
        A = self.rnd_state.randn(self.d, self.d)
        Q, _ = np.linalg.qr(A)
        S = np.diag(np.linspace(sqrt(L), sqrt(mu), self.d))
        self.A = Q @ S @ Q.T

        self.x_star = self.rnd_state.randn(1, self.d) 
        self.x0 = np.ones((1, self.d))
        self.y = self.A @ self.x_star.T
        self.min_f = 0.0
        
    def __call__(self, x):
        fx = 0.5 * np.square(np.linalg.norm(self.A @ x.T - self.y, axis=0))
        return  fx

    def __str__(self):
        return f"{self.name}_{self.L}_{self.mu}_{self.d}"
    
class RosenbrockFunction(TargetFunction):
    def __init__(self, d, seed=121314):
        super().__init__("Rosenbrock", d,  seed)
        self.bounds = np.array([[-2.048, 2.048] for _ in range(d)]).reshape(1, -1)
        self.min_f = 0.0
        self.x0 = np.full((1, self.d), 0.5)
        self.x_star = np.ones((1, self.d))
            

    def __call__(self, x):
        #x = x if x.shape[0] > 1 else x.unsqueeze(0) 
        
        xi = x[:, :-1]
        xi1 = x[:, 1:]
        return np.sum(100 * (xi1 - xi**2)**2 + (xi - 1)**2, axis=1).reshape(-1)


class QuingFunction(TargetFunction):
    def __init__(self, d,  seed=121314):
        super().__init__("Quing", d, seed)
        self.bounds = np.array([[-500.0, 500.0] for _ in range(d)])
        self.min_f = 0.0
        self.x0 = np.ones((1, self.d))
            

    def __call__(self, x):
        d = x.shape[1]
        x = x if x.shape[0] > 1 else x.reshape((1, -1)) #unsqueeze(0)
        i = np.arange(1, d + 1).reshape((1, -1))#unsqueeze(0)  # shape (1, d)
        return np.sum((x ** 2 - i) ** 2, axis=1).squeeze()



class GriewankFunction(TargetFunction):
    def __init__(self, d, seed=121314):
        super().__init__("Griewank", d, seed)
        self.bounds = np.array([[-600.0, 600.0] for _ in range(d)])
        self.min_f = 0.0
        self.x0 = np.full((1, self.d), 1.0)

    def __call__(self, x):
        d = x.shape[-1]
        indices = np.arange(1, d + 1)
        sum_sq = np.sum(x ** 2, axis=-1) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(indices)), axis=-1)
        return 1 + sum_sq - prod_cos



class TridFunction(TargetFunction):

    def __init__(self, d, seed=121314):
        super().__init__("Trid", d, seed)
        self.bounds = np.array([[-d**2, d**2] for _ in range(d)])
        self.min_f = - (d * (d + 4) * (d - 1)) / 6
        x0 = np.zeros((1, d))


        self.x0 = x0 #torch.ones((1, self.d))
    
    def __call__(self, x):
        term1 = np.sum(np.square(x - 1), axis=1)#, keepdims=True) #.square().sum(dim=1, keepdim=True)  
        term2 = np.sum(x[:, 1:] * x[:, :-1], axis=1)#, keepdims=True)  
        return term1 - term2  # Compute function


class AckleyFunction(TargetFunction):
    def __init__(self, d, seed=121314):
        super().__init__("Ackley", d, seed)
        self.bounds = np.array([[-32.0, 32.0] for _ in range(d)])
        self.min_f = 0.0
        self.x0 = np.ones((1, self.d))
            
    def __call__(self, x):
        d = x.shape[1]
        sum_sq = np.sum(x ** 2, axis=1)
        sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return term1 + term2 + 20 + np.e 



class StyblinksiTangFunction(TargetFunction):
    def __init__(self, d, seed=121314):
        super().__init__("StyblinksiTang", d, seed)
        self.bounds = np.array([[-5.0, 5.0] for _ in range(d)])
        self.min_f = -39.16599 * d 
        self.x0 = np.ones((1, self.d))
            
    def __call__(self, x):
        return np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1) / 2


class SchwefelFunction(TargetFunction):
    def __init__(self, d, seed=121314):
        super().__init__("Schwefel", d, seed)
        self.bounds = np.array([[-500.0, 500.0] for _ in range(d)])
        self.min_f = 0.0 
        self.x0 = np.ones((1, self.d))
            
    def __call__(self, x):
        return 418.9829 * self.d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


class BukinFunction(TargetFunction):
    def __init__(self, seed=121314):
        super().__init__("Bukin", 2, seed)
        self.bounds = np.array([[-15.0, -5.0], [-3.0, 3.0]])
        self.min_f = 0.0 
        self.x0 = np.array([[-10.0, 1.0]])
            
    def __call__(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2))
        term2 = 0.01 * np.abs(x1 + 10)
        return term1 + term2

# class PycutestTarget(TargetFunction):
    
#     def __init__(self, pycutest_problem, dtype = torch.float64, device = 'cpu', seed = 121314, x0 = None) -> None:
#         self.fun = pycutest_problem# pycutest.import_problem(problem_name)        
#         super().__init__("CUTest", d = self.fun.n, dtype = dtype, device= device, seed=seed)
#         if x0 is None:
#             self.x0 = torch.from_numpy(self.fun.x0).to(device=device, dtype=dtype).reshape(1, -1)
#         else:
#             self.x0 = torch.from_numpy(x0).to(device=device, dtype=dtype).reshape(1, -1)
            
            
#     def __call__(self, x):
#         res = np.apply_along_axis(self.fun.obj, axis=1, arr=x.cpu().numpy())

#         fx = torch.from_numpy(res).to(dtype=self.dtype, device=self.device)
#         return  fx

#     def grad(self, x):
#         return torch.from_numpy(self.fun.grad(x.cpu().numpy().flatten())).to(device=self.device, dtype=self.dtype)

#     def close(self):
#         if hasattr(self.fun, 'close'):
#             self.fun.close()












