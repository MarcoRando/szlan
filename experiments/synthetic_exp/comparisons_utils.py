import numpy as np 

from itertools import product


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
