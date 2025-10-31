import numpy as np 


from math import sqrt

class TargetFunction:
    
    def __init__(self, name, d,  regularization = 0.0, seed=131415) -> None:
        self.name = name
        self.d = d
        self.rnd_state = np.random.RandomState(seed)
        self.regularization = regularization

    def _add_regularization(self, fx, x):
        return fx + self.regularization * np.linalg.norm(x, axis=1)**2
        
    def __call__(self, x):
        pass
        
    def close(self):
        pass
        
    def __str__(self):
        return f"{self.name}_{self.d}"

class LeastSquares(TargetFunction):
    
    def __init__(self, d, mu, L, regularization = 0.0, seed = 1231415) -> None:
        assert mu < L
        super().__init__(name="LeastSquares", d = d, regularization=regularization, seed=seed)
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
    def __init__(self, d, regularization = 0.0, seed=121314):
        super().__init__("Rosenbrock", d, regularization, seed)
        self.bounds = np.array([[-2.048, 2.048] for _ in range(d)]).reshape(1, -1)
        self.min_f = 0.0
        self.x0 = np.full((1, self.d), 0.5)
        self.x_star = np.ones((1, self.d))
            

    def __call__(self, x):
        #x = x if x.shape[0] > 1 else x.unsqueeze(0) 
        
        xi = x[:, :-1]
        xi1 = x[:, 1:]
        return self._add_regularization(np.sum(100 * (xi1 - xi**2)**2 + (xi - 1)**2, axis=1).reshape(-1), x)

    def grad(self, x):
        x = x if x.ndim > 1 else x.unsqueeze(0) 
        batch_size, d = x.shape
        grad = np.zeros_like(x)

        xi = x[:, :-1]
        xi1 = x[:, 1:]
        grad[:, :-1] += -400 * xi * (xi1 - xi**2) - 2 * (1 - xi)
        grad[:, 1:] += 200 * (xi1 - xi**2)

        return grad#.squeeze()


class QuingFunction(TargetFunction):
    def __init__(self, d, regularization = 0.0, seed=121314):
        super().__init__("Quing", d, regularization, seed)
        self.bounds = np.array([[-500.0, 500.0] for _ in range(d)])
        self.min_f = 0.0
        self.x0 = np.ones((1, self.d))
        self.x_star = np.array([np.sqrt(i) for i in range(1, d + 1)]).reshape(1, -1)
            

    def __call__(self, x):
        d = x.shape[1]
        x = x if x.shape[0] > 1 else x.reshape((1, -1)) #unsqueeze(0)
        i = np.arange(1, d + 1).reshape((1, -1))#unsqueeze(0)  # shape (1, d)
        return self._add_regularization(np.sum((x ** 2 - i) ** 2, axis=1).squeeze(), x)



class GriewankFunction(TargetFunction):
    def __init__(self, d, regularization = 0.0, seed=121314):
        super().__init__("Griewank", d, regularization, seed)
        self.bounds = np.array([[-600.0, 600.0] for _ in range(d)])
        self.min_f = 0.0
        self.x0 = np.full((1, self.d), 1.0)
        self.x_star = np.zeros((1, self.d))

    def __call__(self, x):
        d = x.shape[-1]
        indices = np.arange(1, d + 1)
        sum_sq = np.sum(x ** 2, axis=-1) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(indices)), axis=-1)
        return self._add_regularization(1 + sum_sq - prod_cos, x)



class TridFunction(TargetFunction):

    def __init__(self, d, regularization=0.0, seed=121314):
        super().__init__("Trid", d, regularization, seed)
        self.bounds = np.array([[-d**2, d**2] for _ in range(d)])
        self.min_f = - (d * (d + 4) * (d - 1)) / 6
        x0 = np.zeros((1, d))


        self.x0 = x0 #torch.ones((1, self.d))
    
    def __call__(self, x):
        term1 = np.sum(np.square(x - 1), axis=1)#, keepdims=True) #.square().sum(dim=1, keepdim=True)  
        term2 = np.sum(x[:, 1:] * x[:, :-1], axis=1)#, keepdims=True)  
        return self._add_regularization(term1 - term2, x)  # Compute function


class AckleyFunction(TargetFunction):
    def __init__(self, d, regularization = 0.0, seed=121314):
        super().__init__("Ackley", d, regularization, seed)
        self.bounds = np.array([[-32.768, 32.768] for _ in range(d)])
        self.min_f = 0.0
        self.x0 = np.ones((1, self.d))
        self.x_star = np.zeros((1, self.d))
            
    def __call__(self, x):
        d = x.shape[1]
        sum_sq = np.sum(x ** 2, axis=1)
        sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return self._add_regularization(term1 + term2 + 20 + np.e + np.linalg.norm(x, axis=1)**2, x)

    def grad(self, x):
#        a=20, b=0.2, c=2*np.pi
        x = np.atleast_2d(x)  # Ensure shape (m, n)
        n = x.shape[1]


        s1 = np.mean(x**2, axis=1, keepdims=True)
        s2 = np.mean(np.cos(2 * np.pi * x), axis=1, keepdims=True)
        
        sqrt_s1 = np.sqrt(s1)
        
        # Avoid division by zero (at x=0)
        denom = np.where(sqrt_s1 == 0, 1, sqrt_s1)
        
        term1 = 20 * 0.2 * np.exp(-0.2 * sqrt_s1) * (x / (n * denom))
        term2 = (2*np.pi / n) * np.sin(2*np.pi * x) * np.exp(s2)
        
        grad = term1 + term2
        
        # If input was 1D, return 1D
        return grad 
        

class StyblinksiTangFunction(TargetFunction):
    def __init__(self, d,regularization = 0.0, seed=121314):
        super().__init__("StyblinksiTang", d,regularization, seed)
        self.bounds = np.array([[-5.0, 5.0] for _ in range(d)])
        self.min_f = -39.16599 * d 
        self.x0 = np.ones((1, self.d))
        self.x_star = np.full((1, self.d), -2.903534)
            
    def __call__(self, x):
        return self._add_regularization(np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1) / 2, x)


class RastriginFunction(TargetFunction):

    def __init__(self, d, regularization = 0.0, seed=121314):
        super().__init__("Rastrigin", d, regularization, seed)
        self.bounds = np.array([[-5.12, 5.12] for _ in range(d)])
        self.min_f = 0.0
        self.x0 = np.ones((1, self.d))
        self.x_star = np.zeros((1, self.d))


    def __call__(self, x):

        return self._add_regularization(10 * self.d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1), x)

class SchwefelFunction(TargetFunction):
    def __init__(self, d, regularization = 0.0, seed=121314):
        super().__init__("Schwefel", d, regularization,seed)
        self.bounds = np.array([[-500.0, 500.0] for _ in range(d)])
        self.min_f = 0.0 
        self.x0 = np.ones((1, self.d))
            
    def __call__(self, x):
        return self._add_regularization(418.9829 * self.d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1), x)

class LevyFunction(TargetFunction):
    def __init__(self, d, regularization = 0.0,seed=121314):
        super().__init__("Levy", d, regularization,seed)
        self.bounds = np.array([[-10.0, 10.0] for _ in range(d)])
        self.min_f = 0.0 
        self.x0 = np.zeros((1, self.d))
        self.x_star = np.ones((1, self.d))

    def __call__(self, x):

        w = 1.0 + (x - 1.0) / 4.0

        # Term 1: sin^2(pi * w1)
        term1 = np.sin(np.pi * w[:, 0]) ** 2

        # Middle sum over i = 1..d-1 (handle d=1 gracefully)
        if w.shape[1] > 1:
            wi = w[:, :-1]
            mid = (wi - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * wi + 1.0) ** 2)
            term_mid = np.sum(mid, axis=1)
        else:
            term_mid = np.zeros(w.shape[0])

        # Last term uses w_d
        wd = w[:, -1]
        term_last = (wd - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * wd) ** 2)

        return self._add_regularization(term1 + term_mid + term_last, x)  #+ 10 * np.square(np.linalg.norm(x, axis=1))

    def grad(self, x):

        x = np.atleast_2d(x)
        m, n = x.shape

        w = 1 + (x - 1) / 4.0

        grad = np.zeros_like(x)

        sin_pi_w = np.sin(np.pi * w)
        cos_pi_w = np.cos(np.pi * w)
        sin_pi_w1 = np.sin(np.pi * w + 1)
        cos_pi_w1 = np.cos(np.pi * w + 1)
        sin_2pi_wn = np.sin(2 * np.pi * w[:, -1:])
        cos_2pi_wn = np.cos(2 * np.pi * w[:, -1:])

        term1 = 2 * np.pi * sin_pi_w[:, 0:1] * cos_pi_w[:, 0:1]
        term2 = 2 * (w[:, 0:1] - 1) * (1 + 10 * sin_pi_w1[:, 0:1] ** 2)
        term3 = 20 * np.pi * (w[:, 0:1] - 1) ** 2 * sin_pi_w1[:, 0:1] * cos_pi_w1[:, 0:1]
        grad[:, 0:1] = 0.25 * (term1 + term2 + term3)

        if n > 2:
            wi = w[:, 1:-1]
            wi_prev = w[:, :-2]
            grad[:, 1:-1] = 0.25 * (
                2 * (wi - 1) * (1 + 10 * np.sin(np.pi * wi + 1) ** 2)
                + 20 * np.pi * (wi - 1) ** 2 * np.sin(np.pi * wi + 1) * np.cos(np.pi * wi + 1)
                + 2 * np.pi * np.sin(np.pi * wi) * np.cos(np.pi * wi)
            )


        wn = w[:, -1:]
        grad[:, -1:] = 0.25 * (
            2 * (wn - 1) * (1 + np.sin(2 * np.pi * wn) ** 2)
            + 4 * np.pi * (wn - 1) ** 2 * np.sin(2 * np.pi * wn) * np.cos(2 * np.pi * wn)
            + 2 * np.pi * np.sin(np.pi * wn) * np.cos(np.pi * wn)
        )

        return grad




class ZigZagSmooth(TargetFunction):

    def __init__(self, d, regularization = 0.0, seed=121314):
        super().__init__("ZigZagSmooth", d, regularization, seed)
        self.bounds = np.array([[-1.0, 1.0] for _ in range(d)])
        self.min_f = -d
        self.x0 = np.ones((1, self.d))
        self.x_star = np.zeros((1, self.d))

    def __call__(self, x):
        t1 = -np.sum(np.cos(3*x), axis=1)
        t2 = (1/5) * np.sum(x**2, axis=1)

        return self._add_regularization(t1 + t2, x)

class ZigZag(TargetFunction):
    def __init__(self, d, regularization = 0.0, seed=121314):
        super().__init__("ZigZag", d, regularization, seed)
        self.bounds = np.array([[-2.0, 2.0] for _ in range(d)])
        self.x0 = np.ones((1, self.d))
        self.x_star = np.zeros((1, self.d))
        self.min_f = -(d - 1)

    def __call__(self, x):
        t = np.mod(x, 2.0)
        return self._add_regularization(1.0 - np.sum(np.abs(t - 1.0), axis=1) + (1/5) * np.sum(x**2, axis=1), x)

class SumExpTarget(TargetFunction):
    def __init__(self, d, regularization = 0.0, seed=121314):
        super().__init__("SumExp", d, regularization, seed)
        self.bounds = np.array([[-1.0, 1.0] for _ in range(d)])
        self.min_f = 0.0
        self.x0 = np.ones((1, self.d))
        self.x_star = np.zeros((1, self.d))
        self.v1 = 0.1
        self.v2 = 0.5

    def __call__(self, x):
        t1 = - np.sum(np.exp(-(x+2)**2/self.v1), axis=1)/np.sqrt(self.v1)
        t2 = - np.sum(1.5 * np.exp(-(x-2)**2/self.v2), axis=1)/np.sqrt(self.v2)
        t3 = 0.1 * np.sum(x**2, axis=1) + 1
        return self._add_regularization(t1 + t2 + t3, x)




class BukinFunction(TargetFunction):
    def __init__(self, regularization = 0.0, seed=121314):
        super().__init__("Bukin", 2, regularization=regularization, seed=seed)
        self.bounds = np.array([[-15.0, -5.0], [-3.0, 3.0]])
        self.min_f = 0.0 
        self.x0 = np.array([[-10.0, 1.0]])
            
    def __call__(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2))
        term2 = 0.01 * np.abs(x1 + 10)
        return self._add_regularization(term1 + term2, x)

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












