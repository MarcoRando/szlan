import sys 
import numpy as np
sys.path.append("../../")

from szlan.optimizer.szlan import Optimizer
from scipy.special import logsumexp as logsumexp_scp

class CBO(Optimizer):

#    def __init__(self, population, gamma, beta, direction_generator, h, seed):
    def __init__(self, population, dt, lam, alpha, sigma, use_cons, seed):
        self.population = population
        self.dt = dt
        self.lam = lam
        self.alpha = alpha
        self.sigma = sigma
        self.use_cons = use_cons
        self.n = self.population.shape[0]
        self.d = self.population.shape[1]
        self.rnd_state = np.random.RandomState(seed)
        self.k = 0
        self.best = None
        self.current_cons_point = None
        
    def ask(self):
        if self.current_cons_point is None or not self.use_cons:
            return self.population
        return np.vstack([self.population, self.current_cons_point])

    def tell(self, X, y):
        best_idx = np.argmin(y)
        if self.best is None or y[best_idx] < self.best[1]:
            self.best = (X[best_idx, :], y[best_idx])

        if self.current_cons_point is not None and self.use_cons:
            y = y[:-1]
            X = X[:-1, :]

        weights = (- self.alpha * y).reshape(-1, 1)

        coeffs = np.exp(weights - logsumexp_scp(weights, axis=0, keepdims=True)).reshape(-1, 1)
        cons_point = (X * coeffs).sum(axis=0, keepdims=True)
        self.current_cons_point = cons_point


#        cons_point = np.sum(X * weights, axis=0) / np.sum(weights)

        xi = self.rnd_state.randn(self.n, self.d)
        drift = self.lam * self.dt * (cons_point - self.population)
        brw_motion = self.sigma * np.sqrt(self.dt) * np.abs(cons_point - self.population) * xi

        self.population = self.population - drift + brw_motion
        

        # coeff_expan = tuple([Ellipsis] + [None for i in range(x.ndim-2)])
        # coeffs = np.exp(weights - logsumexp_scp(weights, axis=-1, keepdims=True))[coeff_expan]
        # self.check_coeffs(coeffs)
        # return (x * coeffs).sum(axis=1, keepdims=True), energy

        # if self.phase == CBOPhase.INITIALIZATION or self.phase == CBOPhase.ITERATE:
        #     self.current_values = y
        #     self.P = self.direction_generator() # l x d matrix of directions
        #     self.phase = CBOPhase.GRADIENT_APPROX
        # elif self.phase == CBOPhase.GRADIENT_APPROX:
        #     self.forward_values = y
        #     self.phase = CBOPhase.ITERATE
        self.k += 1

    def recommend(self):
        return self.best