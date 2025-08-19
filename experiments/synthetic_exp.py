import numpy as np 

import sys 

sys.path.append("..")

from szlan.optimizer.szlan import SZLan, SZLanPhase
from szlan.direction_generators.direction_generators import CoordinateDirectionGenerator, GaussianDirectionGenerator, QRDirectionGenerator

def target(x):
    x = x.reshape(-1, 2)
    return np.sum(x**2, axis=1) #+ np.sum(np.sin(24*x), axis=1) + 43

def real_gradient(x):
    return 2 * x #+ 24 * np.cos(24*x)


population = np.array([
    [-5.0, -5.0],
    [5.0, 5.0]
])


h = 1e-5
d,l =2, 2
direction_generator = CoordinateDirectionGenerator(d=d, l=l)
for _ in range(10):
    P = direction_generator()
    print(P.T)
#P = direction_generator()
#print(P.T)
GRAD_T0 = np.zeros((population.shape[0], population.shape[1]))
for i in range(population.shape[0]):
    for j in range(P.shape[0]):
        GRAD_T0[i] += ((target(population[i, :] + h * P[j, :]) - target(population[i, :])) / h) * P[j, :]
print("Initial Gradient: ", GRAD_T0)

opt = SZLan(population=population,
            direction_generator=direction_generator,
            h=1e-10,
            gamma=0.1,
            beta=1.0,
            seed=121314)


for i in range(3):
    old_pop = opt.population.copy()
    X = opt.ask()
    y = target(X)
    if opt.phase == SZLanPhase.INITIALIZATION or opt.phase == SZLanPhase.ITERATE:
        print(f"Iteration {i+1}:  Values: {y} Best: {opt.best}")

    if opt.phase == SZLanPhase.ITERATE:
        grad = opt._approx_gradient()
        print(f"Approximate Gradient: {grad.flatten()}")
        print(f"Real Gradient: {real_gradient(old_pop.flatten())}")
        rgrad = real_gradient(old_pop.flatten())[0]
        print(np.linalg.norm(grad[0] - rgrad) / np.linalg.norm(rgrad) )
    opt.tell(X, y)


