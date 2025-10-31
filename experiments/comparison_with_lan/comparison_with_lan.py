import os
import numpy as np 
import argparse as ap 

import sys 

sys.path.append("..")
sys.path.append("../../")
from szlan.optimizer.szlan import Optimizer, SZLan, SZLanPhase
from szlan.direction_generators.direction_generators import GaussianDirectionGenerator, SphericalDirectionGenerator, QRDirectionGenerator

from synthetic_functions import *
from utils import get_objective_function

from comparisons_utils import get_optimizer, get_params_grid

import nevergrad as ng

from itertools import product
from functools import partial
import multiprocessing as mp 


def run_lan(params, target, budget, d, n, seed, reps = 5):
    gamma, beta = params
    fvalues = [[] for _ in range(reps)]
    opt_gaps = [[] for _ in range(reps)]
    costs = [[] for _ in range(reps)]
    print("[LAN] Params: {}".format(params))

    for r in range(reps):
        rnd_state = np.random.RandomState(seed + 423 * r)
        population = (target.bounds[:, 1] - target.bounds[:, 0]) * rnd_state.rand(n, d) + target.bounds[:, 0]

        pop_x = population.copy()
        Y_0 = np.min(target(pop_x))
        init_optgap = Y_0 - target.min_f

        fvalues[r].append(np.min(target(pop_x)))
        opt_gaps[r].append((np.min(target(pop_x)) - target.min_f)/init_optgap )
        costs[r].append(pop_x.shape[0])
        num_evals = pop_x.shape[0]
        rnd_state = np.random.RandomState(seed + r * 987)
        while num_evals < budget:
            z_k = np.sqrt(2 * gamma / beta) * rnd_state.randn(d)
            pop_x = pop_x - gamma * target.grad(pop_x) + z_k
            pop_y = target(pop_x)
            fvalues[r].append(np.min(pop_y))
            opt_gaps[r].append((np.min(pop_y) - target.min_f)/init_optgap )
            num_evals += pop_x.shape[0]
            is_nan = np.isnan(fvalues[r][-1])
            if is_nan:
                return params, fvalues, opt_gaps, costs, is_nan

    return params, np.array(fvalues).reshape(reps, - 1), np.array(opt_gaps).reshape(reps, - 1), np.array(costs).reshape(reps, - 1), is_nan

def get_hseq(h, h_schedule):
    if h_schedule == 'sqrt':
        return lambda k : h * (1/sqrt(k + 1))
    elif h_schedule == 'linear':
        return lambda k : h * (1/(k + 1))
    return lambda k : h



def run_zlan(params, target, budget, d, l, n, seed, h=1e-5, h_schedule='constant', reps = 5):

    print("[SZ-LAN] Params: {}".format(params))

    fvalues = [[] for _ in range(reps)]
    opt_gaps = [[] for _ in range(reps)]
    costs = [[] for _ in range(reps)]
    for r in range(reps):
        szlan_params = (l, params[0], params[1], 'qr', r)
        rnd_state = np.random.RandomState(seed + 423 * r)
        population = (target.bounds[:, 1] - target.bounds[:, 0]) * rnd_state.rand(n, d) + target.bounds[:, 0]

        opt = get_optimizer('szlan', szlan_params, d=d, h=get_hseq(h, h_schedule), population=population, seed=seed + 123 * r)
        
        Y_0 = np.min(target(population))
        init_optgap = Y_0 - target.min_f
        X = population.copy()
        num_evals = 0
        while num_evals < budget:

            X = opt.ask()
            Y = target(X)
            opt.tell(X, Y)
            
            num_evals += X.shape[0]
            costs[r].append(X.shape[0])
            fvalues[r].append(np.min(Y))
            opt_gaps[r].append((np.min(Y) - target.min_f)/init_optgap )
            is_nan = np.isnan(fvalues[r][-1])
            if is_nan:
                return params, fvalues, opt_gaps, costs, is_nan

    return params, np.array(fvalues).reshape(reps, - 1), np.array(opt_gaps).reshape(reps, - 1), np.array(costs).reshape(reps, - 1), is_nan


if __name__ == "__main__":
    mp.set_start_method('spawn')
    seed = 123141


    fun_name = sys.argv[1]
    opt_name = sys.argv[2]
    assert opt_name in ["szlan", "lan"]

    d = int(sys.argv[3])
    n = int(sys.argv[4])

    l_codes = [2, d//3, d//2, int(2/3 * d), d]


    out_dir = sys.argv[5] + f"/szlan_results/comp_langevin/{fun_name}"
    os.makedirs(out_dir, exist_ok=True)

    num_workers = int(sys.argv[6])
    reps = int(sys.argv[7])
    budget = int(sys.argv[8]) #1000

    l = l_codes[int(sys.argv[9])] if opt_name == "szlan" else None
    h_schedule = None
    h = None
    if opt_name == "szlan":
        h = float(sys.argv[10])
        h_schedule = sys.argv[11]
        assert h_schedule in ["constant", "sqrt", "linear"]


    fun = get_objective_function(fun_name, d, 0.0, seed) #RosenbrockFunction(d=d, regularization=0.0, seed=seed)





    dir_type = 'qr'



    gammas = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    betas = [0.001, 0.01, 0.1, 1.0, 2.5, 5.0, 7.5, 10.0, 25.0, 50.0, 100.0, 1000.0]

    print("[--] Optimizer name {}\tl: {}\th: {}".format(opt_name, l, h))
    param_grid = product(gammas, betas)


    if opt_name == "szlan":
        fname = f"szlan_{fun_name}_{d}_{l}_{n}_{h}_{h_schedule}"
        exp_fun = partial(run_zlan, target=fun, budget=budget, d=d, l=l, n=n, seed=seed, h=h, h_schedule=h_schedule, reps = reps)
    else:
        fname = f"lan_{fun_name}_{d}_{n}"
        exp_fun = partial(run_lan, target=fun, budget=budget, d=d, n=n, seed=seed, reps = reps)

    best_result = (None, None, None, None, None)

    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered( exp_fun, param_grid):
            #params, fvalues, opt_gaps, is_nan
            params, fvalues, opt_gaps, costs, is_nan = result
            if not is_nan:
                mu_gap, std_gap = np.mean(opt_gaps, axis=0), np.std(opt_gaps, axis=0)
                upp_bound = mu_gap + std_gap
                best_idx = np.argmin(upp_bound)
                if best_result[0] is None or upp_bound[best_idx] < best_result[1]:
                    best_result = (params, upp_bound[best_idx], fvalues, opt_gaps, costs, is_nan)
                    print("[{}] new best: {} with {}".format(opt_name, upp_bound[best_idx], params))

    print("[--] Best params: {}".format(best_result[0]))

    values = best_result[2]
    opt_gaps = best_result[3]
    costs = best_result[4]
    mu_costs, std_costs = np.mean(costs, axis=0), np.std(costs, axis=0)

    with open(f"{out_dir}/{fname}_values.log", 'w') as f:
        for j in range(values.shape[1]):
            out_str = ",".join([str(x) for x in values[:, j]]) + f",{mu_costs[j]},{std_costs[j]}\n"
            f.write(out_str)

    with open(f"{out_dir}/{fname}_optgaps.log", 'w') as f:
        for j in range(opt_gaps.shape[1]):
            out_str = ",".join([str(x) for x in opt_gaps[:, j]]) + f",{mu_costs[j]},{std_costs[j]}\n"
            f.write(out_str)


