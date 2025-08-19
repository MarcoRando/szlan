import numpy as np 

from typing import Callable
from enum import Enum
from szlan.optimizer.opt import Optimizer


class SZLanPhase(Enum):
    INITIALIZATION = 0
    ITERATE = 1
    GRADIENT_APPROX = 2


class SZLan(Optimizer):
    
    def __init__(self, 
                 population, # n x d
                 direction_generator,
                 h : float | Callable[[int], float]  = 1e-7,
                 gamma : float | Callable[[int], float] = 0.1,
                 beta : float | Callable[[int], float] = 1.0,
                 seed : int = 121314
                ):
        self.direction_generator = direction_generator
        self.P = self.direction_generator()
        self.num_particles =  population.shape[0]
        
        self.h = h if isinstance(h, Callable) else lambda _: h
        self.gamma = gamma if isinstance(gamma, Callable) else lambda _: gamma
        self.beta = beta if isinstance(beta, Callable) else lambda _: beta
        
        self.population = population if population is not None else self._build_population()
        self.phase = SZLanPhase.INITIALIZATION
        self.rnd_state = np.random.RandomState(seed)
        self.best = None
        self.k = 0
        
    
    def _build_population(self):
        pass

    def _approx_gradient(self):
        h = self.h(self.k)
        forward_values = self.forward_values.reshape(self.population.shape[0], self.direction_generator.l)
        print(f"Forward values shape: {forward_values.shape}, Current values shape: {self.current_values.shape}")
        diff = (forward_values - self.current_values[:, None]) / h
        grads = self.direction_generator.nrm_const * (diff @ self.P)
        test_grads = np.zeros((self.population.shape[0], self.P.shape[1]))
        for i in range(self.population.shape[0]):
            for j in range(self.P.shape[0]):
                test_grads[i] += ((forward_values[i,j] - self.current_values[i])/h) * self.P[j] 
        test_grads = self.direction_generator.nrm_const * test_grads
        
        # print("GRADS - TEST GRADS: ", grads.flatten() - test_grads.flatten())

        return test_grads

    def ask(self):
        h_k = self.h(self.k)
        if self.phase == SZLanPhase.INITIALIZATION:
            return self.population
        elif self.phase == SZLanPhase.GRADIENT_APPROX:
            COMP_VALS=(self.population[:, None, :] + h_k * self.P[None, :, :]).reshape(-1, self.population.shape[1])
            SEQ_VALS = []
            for i in range(self.population.shape[0]):
                for j in range(self.P.shape[0]):
                    SEQ_VALS.append(self.population[i, :] + h_k * self.P[j, :])
            SEQ_VALS = np.array(SEQ_VALS).reshape(-1, self.population.shape[1])
            print(f"COMP_VALS shape: {COMP_VALS}, SEQ_VALS shape: {SEQ_VALS}")
            return COMP_VALS
        z_k = self.rnd_state.randn(self.num_particles, self.population.shape[1])
        g_k = self._approx_gradient() # n x d where every ros is a gradient approximation
        gamma_k = self.gamma(self.k)
        beta_k = self.beta(self.k)
        
        self.population = self.population - gamma_k * g_k + np.sqrt(2 * gamma_k / beta_k) * z_k
        return self.population
    
    def tell(self, X, y):
        best_idx = np.argmin(y)
        if self.best is None or y[best_idx] < self.best[1]:
            self.best = (X[best_idx, :], y[best_idx])
            
        if self.phase == SZLanPhase.INITIALIZATION or self.phase == SZLanPhase.ITERATE:
            self.current_values = y
            self.P = self.direction_generator() # l x d matrix of directions
            self.phase = SZLanPhase.GRADIENT_APPROX
        elif self.phase == SZLanPhase.GRADIENT_APPROX:
            self.forward_values = y
            self.phase = SZLanPhase.ITERATE
            self.k += 1
            

    def reccommend(self):
        return self.best


    