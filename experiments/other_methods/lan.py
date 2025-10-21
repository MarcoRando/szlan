import sys 
import numpy as np
sys.path.append("../../")

from szlan.optimizer.szlan import Optimizer
from szlan.optimizer.szlan import SZLanPhase

class Langevin(Optimizer):

#    def __init__(self, population, gamma, beta, direction_generator, h, seed):
    def __init__(self, population, gamma, beta, grad_fun, seed):
        self.population = population
        self.gamma = gamma if isinstance(gamma, Callable) else lambda _: gamma
        self.beta = beta if isinstance(beta, Callable) else lambda _: beta
        self.grad_fun = grad_fun

        self.n = self.population.shape[0]
        self.d = self.population.shape[1]
        self.rnd_state = np.random.RandomState(seed)
        self.k = 0
        self.phase = SZLanPhase.INITIALIZATION
        self.best = None

        
    def ask(self):
        if self.phase == SZLanPhase.INITIALIZATION:
            return self.population

        z_k = self.rnd_state.randn(self.n, self.d)
        g_k = self.grad_fun(self.population) # n x d where every ros is a gradient approximation
        gamma_k = self.gamma(self.k)
        beta_k = self.beta(self.k)
        
        self.population = self.population - gamma_k * g_k + np.sqrt(2 *  gamma_k / beta_k) * z_k


        return self.population
    
    def tell(self, X, y):
        best_idx = np.argmin(y)
        if self.best is None or y[best_idx] < self.best[1]:
            self.best = (X[best_idx, :], y[best_idx])
            
        self.phase = SZLanPhase.ITERATE
        self.k += 1
            

    def recommend(self):
        return self.best 
