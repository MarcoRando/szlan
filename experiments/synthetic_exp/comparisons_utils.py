import numpy as np 
import nevergrad as ng

from itertools import product



def get_optimizer(optimizer_name, params, population, d, h, seed):
    if optimizer_name == "szlan":
        l, gamma, beta, _, rep = params
        direction_seed = seed + 134 * rep
        opt_seed = seed + 473 * rep

        direction_generator = QRDirectionGenerator(d=d, l=l, seed=direction_seed)
        return SZLan(population=population, gamma=gamma, beta=beta , h=h, direction_generator=direction_generator, seed=opt_seed)
    elif optimizer_name == "cmaes":
        return ng.optimizers.ParametrizedCMA(popsize=params[1], scale=params[0])
    elif optimizer_name == "de_2p":
        return ng.optimizers.DifferentialEvolution(popsize=params[-2], scale=params[0], F1=params[1], F2=params[2], crossover='twopoints') #(parametrization=d, budget=budget)
    elif optimizer_name == "pso":
        return ng.optimizers.ConfPSO(popsize=params[-2], omega=params[0], phip=params[1], phig=params[2])
    raise Exception(f"Unknown optimizer {optimizer_name}")



def get_params_grid(optimizer_name, d, reps = 10):
    num_particles = [10, 100, 1000]
    if optimizer_name == "szlan":
        if d >= 10:
            num_directions = [2, d//3, d//2, int(4/5 * d), d]
        elif d > 5:
            num_directions = [2, d//2, d]
        else:
            num_directions = [d//2, d]
        gammas =  [1e-7, 1e-5,  1e-3, 1e-1, 1.0, 10.0] 
        betas =[1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3] 
        params = [num_directions, gammas, betas]
    elif optimizer_name == "cmaes":
        scales = np.linspace(0.1, 1.0, 6)
        params = [scales]
    elif optimizer_name == "de_2p":
        scales = np.linspace(0.1, 1.0, 6)
        diff_weights = np.linspace(0.1, 1.0, 6)
        params = [scales, diff_weights, diff_weights]
    elif optimizer_name == "pso":
        omegas = np.linspace(0.1, 1.0, 6)
        phis = np.linspace(0.1, 2.0, 6)
        params = [omegas, phis, phis]

    params += [num_particles, range(reps)]

    return list(product(*params))
