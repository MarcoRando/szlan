import os
import numpy as np 
import argparse as ap 

import sys 

sys.path.append("..")
sys.path.append("../../")
from szlan.optimizer.szlan import SZLan, SZLanPhase
from szlan.direction_generators.direction_generators import GaussianDirectionGenerator, SphericalDirectionGenerator, QRDirectionGenerator

from synthetic_functions import *
from utils import get_objective_function

from itertools import product
from functools import partial
import multiprocessing as mp 





def run_optimizer(params, target, budget, d, h, seed):
    l, gamma, beta, rep = params

    direction_seed = seed + 134 * rep
    opt_seed = seed + 473 * rep

    rnd_state = np.random.RandomState(seed + 961 * rep)
    population = (target.bounds[:, 1] - target.bounds[:, 0]) * rnd_state.rand(1, d) + target.bounds[:, 0] #target.x0 #np.array([ np.full((d,)) for _ in range(1)]).reshape(-1, d)
    direction_generator = QRDirectionGenerator(d=d, l=l, seed=direction_seed)
    opt = SZLan(population=population, gamma=gamma, beta=beta , h=h, direction_generator=direction_generator, seed=opt_seed)

    print("------------")
    print(f"Running SZLan with parameters: l={l}, gamma={gamma}, beta={beta}, h={h}")
    print(population)
    print("------------")

    num_evals = 0

    best_on_iterate = None
    fvalues = []

    while num_evals < budget:
        X = opt.ask()
        Y = target(X)
#        if np.any(np.isnan(Y)):
#            break
        if opt.phase == SZLanPhase.INITIALIZATION or opt.phase == SZLanPhase.ITERATE:
            fvalues.append(Y[0])
        if opt.phase == SZLanPhase.ITERATE and (best_on_iterate is None or Y < best_on_iterate):
            best_on_iterate = Y[0]       
        opt.tell(X, Y)


        num_evals += X.shape[0]

    return opt.best[1], best_on_iterate, fvalues, params




def run_experiment(args):
    target_name = args.fun_name
    d = args.d
    h = args.h
    budget = args.budget
    seed = args.seed
    num_workers = args.num_workers
    reps = args.reps
    output_directory = f"{args.out_dir}/szlan_results/changing_parameters/{target_name}"
    os.makedirs(output_directory, exist_ok=True)

    num_directions = [2, d//3, d//2 ]#, d//2, int(4/5 * d), d]
    gammas = np.logspace(-5, 0, 1)
    betas = np.logspace(-5, 0, 1)
    param_grid = list(product(num_directions, gammas, betas, range(reps)))

#    print(param_grid)
#    exit()
    target = get_objective_function(target_name, d=d, seed=seed)
    min_f = target(target.x_star)[0]
    print(f"Function minimum: {min_f}")

    run_opt = partial(run_optimizer, target=target, budget=budget, d=d, h=h, seed=seed)
    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered( run_opt, param_grid):
            f_found, f_found_on_iterate, fvalues, params = result
            l, gamma, beta, rep = params

            with open(f"{output_directory}/{target_name}_{d}.txt", "a") as f:
                f.write(f"{l},{rep},{gamma},{beta},{f_found},{f_found_on_iterate},{fvalues[0]},{min_f}\n")
                f.flush()
#            print("RESULT: ",result)

    #     for (i, resul) in pool.

#    print(f"Running experiment on {target_name} function in dimension {d} with budget {budget} and seed {seed}")
#    run_optimizer(target, budget, d, 0.01 * 1/d, 0.0001, 1e-7, seed)
    




if __name__ == "__main__":
    parser = ap.ArgumentParser()
    
    # Experiment Parameters
    parser.add_argument("fun_name", type=str, default="Rosenbrock", help="Objective function to optimize")
    parser.add_argument("--d", type=int, default=10, help="Dimension of the problem")
    parser.add_argument('--budget', type=int, default=1000, help='Function evaluation budget')
    parser.add_argument('--reps', type=int, default=10, help='Number of repetitions for the experiment')

    # Optimizer Parameters
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma parameter')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter')
    parser.add_argument('--h', type=float, default=1e-7, help='Finite difference step size')


    # General Parameters
    parser.add_argument('--num-workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--out-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    
    args = parser.parse_args()
    run_experiment(args)
    
    