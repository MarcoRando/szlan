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

import nevergrad as ng

from itertools import product
from functools import partial
import multiprocessing as mp 




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


def run_optimizer(params, optimizer_name, target, budget, d, h, seed):
    l, gamma, beta, n, rep = params

    direction_seed = seed + 134 * rep
    opt_seed = seed + 473 * rep

    rnd_state = np.random.RandomState(seed + 961 * rep)
    population = (target.bounds[:, 1] - target.bounds[:, 0]) * rnd_state.rand(n, d) + target.bounds[:, 0] #target.x0 #np.array([ np.full((d,)) for _ in range(1)]).reshape(-1, d)


    opt = get_optimizer(optimizer_name, params, population, d, h, seed)

    if not isinstance(opt, Optimizer):
        opt = opt(parametrization=d, budget=budget)
        for vec in population:
            cand = optim.parametrization.spawn_child(new_value=vec)
            opt.tell(cand, target(vec.reshape(1, -1)))
            budget-=1

    # direction_generator = QRDirectionGenerator(d=d, l=l, seed=direction_seed)
    # opt = SZLan(population=population, gamma=gamma, beta=beta , h=h, direction_generator=direction_generator, seed=opt_seed)

    print("------------")
    print(f"Running {optimizer_name} with parameters: {params}....")
    print("------------")

    num_evals = 0

    best_on_iterate = None
    fvalues = []

    while num_evals < budget:
        X = opt.ask()
#        Y = target(X)
        Y = target(X.value.reshape(-1, d)) if not isinstance(X, np.ndarray) else target(X.reshape(-1, d))

        if np.any(np.isnan(Y)):
            return np.nan, np.nan, fvalues + [np.nan], params

        if opt.phase == SZLanPhase.INITIALIZATION or opt.phase == SZLanPhase.ITERATE:

            fvalues.append(np.min(Y))
        if opt.phase == SZLanPhase.ITERATE and (best_on_iterate is None or np.min(Y) < best_on_iterate):
            best_on_iterate = np.min(Y)      
        opt.tell(X, Y)

        cost = 1 if not isinstance(opt, Optimizer) else  X.shape[0]
        num_evals += cost

    f_best = target(opt.recommend().value.reshape(1, -1)) if not isinstance(opt, Optimizer) else target(opt.recommend()[0].reshape(1, -1))


    return f_best, best_on_iterate, fvalues, params




def run_experiment(args):
    optimizer_name = args.optimizer
    target_name = args.fun_name
    d = args.d
    h = args.h
    budget = args.budget
    seed = args.seed
    num_workers = args.num_workers
    reps = args.reps
    output_directory = f"{args.out_dir}/szlan_results/comparison/{target_name}/{optimizer_name}"
    os.makedirs(output_directory + "/traces", exist_ok=True)

    param_grid = get_params_grid(optimizer_name, d, reps)

    filtered_param_grid = []
    for param in param_grid:
        if not os.path.exists(f"{output_directory}/traces/{optimizer_name}_{target_name}_{d}_{param[0]}_{param[1]}_{param[2]}_{param[3]}_{param[4]}_trace.txt"):
            filtered_param_grid.append(param)
        else:
            print(f"Skipping configuration {param}")
    param_grid = filtered_param_grid

    target = get_objective_function(target_name, d=d, seed=seed)
    min_f = target(target.x_star)[0]
    print(f"Function minimum: {min_f}")

    run_opt = partial(run_optimizer, optimizer_name=optimizer_name, target=target, budget=budget, d=d, h=h, seed=seed)
    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered( run_opt, param_grid):
            f_found, f_found_on_iterate, fvalues, params = result

            out_file_sumup = f"{output_directory}/{optimizer_name}_{target_name}_{d}_" + "_".join([str(x) for x in params[:-1]]) + ".txt"
            out_file_trace = f"{output_directory}/traces/{optimizer_name}_{target_name}_{d}_" + "_".join([str(x) for x in params]) + "_trace.txt"

            with open(f"{output_directory}/{out_file_sumup}", "a") as f:
                f.write(f"{rep},{f_found},{f_found_on_iterate},{fvalues[0]},{min_f}\n")
                f.flush()
            with open(f"{output_directory}/traces/{out_file_trace}", "a") as f:
                for i in range(len(fvalues)):
                    f.write(f"{fvalues[i]}\n")
                    f.flush()

    




if __name__ == "__main__":
    parser = ap.ArgumentParser()
    
    # Experiment Parameters
    parser.add_argument("optimizer", type=str, default="szlan", choices=["szlan", "cmaes", "de_2p", "pso"], help="Optimizer to use")
    parser.add_argument("fun_name", type=str, default="Rosenbrock", choices=["Rosenbrock", "Griewank", "Ackley", "Levy", "StyblinkskiTang"], help="Objective function to optimize")
    parser.add_argument("--d", type=int, default=10, help="Dimension of the problem")
    parser.add_argument('--budget', type=int, default=1000, help='Function evaluation budget')
    parser.add_argument('--reps', type=int, default=10, help='Number of repetitions for the experiment')

    # Optimizer Parameters
    parser.add_argument('--h', type=float, default=1e-7, help='Finite difference step size')

    # General Parameters
    parser.add_argument('--num-workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--out-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    
    args = parser.parse_args()
    run_experiment(args)
    
    