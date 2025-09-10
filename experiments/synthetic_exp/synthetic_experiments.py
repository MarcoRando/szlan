import numpy as np 

import sys 

sys.path.append("..")
sys.path.append("../../")
from szlan.optimizer.szlan import SZLan, SZLanPhase
from szlan.direction_generators.direction_generators import CoordinateDirectionGenerator, GaussianDirectionGenerator, QRDirectionGenerator
import matplotlib.pyplot as plt
from synthetic_functions import *

import nevergrad as ng


rnd_state = np.random.RandomState(121314)



def run_experiment(target, opt, population, T, best_init, alg_cost, verbose=False):
    opt_values = [1.0]
    i = 0
    # if not isinstance(opt, SZLan):
    #     init_val = target(opt.recommend().value.reshape(1, -1))[0]
    #     print(init_val)

    while i < T:
        candidate = opt.ask()
        value = target(candidate.value.reshape(-1, d)) if not isinstance(candidate, np.ndarray) else target(candidate.reshape(-1, d))
        # if verbose:
        #     if opt.phase == SZLanPhase.ITERATE:
        #         print(f"[--] Current value: {value} [{value.shape}]")

        opt.tell(candidate, value)

        f_best = target(opt.recommend().value.reshape(1, -1)) if not isinstance(opt, SZLan) else target(opt.recommend()[0].reshape(1, -1))

        regret = (f_best[0] - target.min_f) / (best_init - target.min_f)
        cost = 1 if not isinstance(opt, SZLan) else  candidate.shape[0]
        opt_values += [regret for _ in range(cost)]
        print(opt_values)
#        exit()
        i= i + 1 if not isinstance(opt, SZLan) else i + candidate.shape[0]
        if verbose:
            print(f"[--] Best value observed: {f_best} [{target.min_f}]")
#        print(f"NG Best: {target(optim.recommend().value.reshape(1, -1))} Opt: {target.min_f}")
    return opt_values


d = 5 #25
l = d
n = 100
#np.array([
#     [2.0 for _ in range(d)],
#     [3.0 for _ in range(d)],
# ]) #
target = AckleyFunction(d, seed=121314)
population = (target.bounds[:, 1] - target.bounds[:, 0]) * rnd_state.rand(n, d) + target.bounds[:, 0]

values = target(population)
best_init = np.min(values)
best_init_idx = np.argsort(values)
population = population[best_init_idx, :].reshape(-1, d)


h = 1e-5
direction_generator = QRDirectionGenerator(d=d, l=l)




np.random.seed(12314)
T = 1000 #20000


optim_twopoint = ng.optimizers.DifferentialEvolution(popsize=n, F1=0.5, F2=0.5, crossover='twopoints')(parametrization=d, budget=T)
optim_cmaes = ng.optimizers.ParametrizedCMA(popsize=n, scale=0.9).set_name("CMA-ES", register=True)(parametrization=d, budget=T)
optim_pso = ng.optimizers.ConfPSO(popsize=n, omega=0.1, phip=0.1, phig=0.1)(parametrization=d, budget=T)
#optim_gpucb = ng.optimizers.ParametrizedBO()(parametrization=d, budget=T)


#optim_ngopt = ng.optimizers.NGOpt(parametrization=d, budget=T)


for vec in population:
    cand_twopoint = optim_twopoint.parametrization.spawn_child(new_value=vec)
    cand_cmaes = optim_cmaes.parametrization.spawn_child(new_value=vec)
    cand_pso = optim_pso.parametrization.spawn_child(new_value=vec)
    optim_twopoint.tell(cand_twopoint, target(vec.reshape(1, -1)))
    optim_cmaes.tell(cand_cmaes, target(vec.reshape(1, -1)))
    optim_pso.tell(cand_pso, target(vec.reshape(1, -1)))


beta = 1.0 #7.0 
gamma = 0.1

sz_lan = SZLan(population=population, #[:1, :].reshape(-1, d),
            direction_generator=direction_generator,
            h=h,
            gamma=lambda k : gamma,
            beta= lambda k : 0.1,
            seed=121314)



opt_values_cmaes = run_experiment(target, optim_cmaes, population, T - n, best_init, alg_cost=1, verbose=True)
# opt_values_szlan = run_experiment(target, sz_lan, population, T, best_init, alg_cost=l, verbose=True)

# opt_values_de_twopoint = run_experiment(target, optim_twopoint, population, T - n, best_init, alg_cost=1, verbose=False)
# opt_values_pso = run_experiment(target, optim_pso, population, T - n, best_init, alg_cost=1, verbose=False)



fig, ax = plt.subplots(1, 1)
ax.set_title(f"{target.name} [$d$ = {d}]")



#ax.plot(range(len(opt_values_de_rand)), opt_values_de_rand, label="DE [Rand]")
#ax.plot(range(len(opt_values_de_twopoint)), opt_values_de_twopoint, label="DE [2Points]")
ax.plot(range(len(opt_values_cmaes)), opt_values_cmaes, label="CMA-ES")
#ax.plot(range(len(opt_values_pso)), opt_values_pso, label="PSO")

#ax.plot(range(len(opt_values_ngopt)), opt_values_ngopt, label=f"NGOPT")
#ax.plot(range(len(opt_values_ozd)), opt_values_ozd, label=f"OZD")
#ax.plot(range(len(opt_values_szlan)), opt_values_szlan, label="SZLAN [$\\beta_k = 10.0$]")

ax.set_ylabel("normalized simple regret")
ax.set_xlabel("function evaluations")

ax.set_yscale('log')
ax.legend()
fig.savefig(f"./{target.name}_{d}_{l}_{n}_{T}.png".format(target_name=target.name, d=d, l=l, T=T), bbox_inches='tight')
plt.close(fig)

