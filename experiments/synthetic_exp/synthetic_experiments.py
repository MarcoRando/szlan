import numpy as np 

import sys 

sys.path.append("..")
sys.path.append("../../")
from szlan.optimizer.szlan import SZLan, SZLanPhase
from szlan.direction_generators.direction_generators import CoordinateDirectionGenerator, GaussianDirectionGenerator, QRDirectionGenerator

from synthetic_functions import *

import nevergrad as ng


rnd_state = np.random.RandomState(121314)



d = 10
l = d
population =  np.array([[5.0 for _ in range(d)],
                        [1.0 for _ in range(d)],
                        [3.0 for _ in range(d)],
                        [8.0 for _ in range(d)]])


h = 1e-5
direction_generator = QRDirectionGenerator(d=d, l=l)

target = GriewankFunction(d)

param = ng.p.Array(shape=(d,))


np.random.seed(12314)

optim = ng.optimizers.DifferentialEvolution(popsize=4, crossover='random')(parametrization=d, budget=1000)

for vec in population:
    cand = optim.parametrization.spawn_child(new_value=vec)
    optim.tell(cand, target(vec.reshape(1, -1)))

#cand = optim.parametrization.spawn_child(new_value=population[-1])
#optim.tell(cand, target(population[-1].reshape(1, -1)))


for i in range(996):
    candidate = optim.ask()
    value = target(candidate.value.reshape(1, -1))
    optim.tell(candidate, value)

    print(f"NG Best: {target(optim.recommend().value.reshape(1, -1))} Opt: {target.min_f}")

#exit()
#print(target( np.array([718.24131949, 713.0810815]).reshape(1,-1) ) )
#exit()



opt = SZLan(population=population,
            direction_generator=direction_generator,
            h=h,
            gamma=lambda k : 0.1,
            beta= lambda k : np.array([0.001, 0.01, 0.1, 10.0]).reshape(-1, 1) * np.log(k+2) / (k + 2),
            seed=121314)

print("Starting SZLan")
for i in range(1000):
    old_pop = opt.population.copy()
    X = opt.ask()
    y = target(X)
    if (opt.phase == SZLanPhase.INITIALIZATION or opt.phase == SZLanPhase.ITERATE) and opt.best is not None:
#        print(X)
        print(f"Iteration {i+1}:  Best: {opt.best[1]} Opt: {target.min_f}")

    opt.tell(X, y)


