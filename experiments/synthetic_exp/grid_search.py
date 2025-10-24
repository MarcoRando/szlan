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



def run_optimizer(params, optimizer_name, target, budget, d, h, n, seed):
#    l, gamma, beta, n, rep = params

    rep = params[-1]

    direction_seed = seed + 134 * rep
    opt_seed = seed + 473 * rep
    min_f = target(target.x_star)[0]

    rnd_state = np.random.RandomState(seed+ 961 * rep)
    population = (target.bounds[:, 1] - target.bounds[:, 0]) * rnd_state.rand(n, d) + target.bounds[:, 0] #target.x0 #np.array([ np.full((d,)) for _ in range(1)]).reshape(-1, d)


    np.random.seed(opt_seed)
    opt = get_optimizer(optimizer_name, params, population, d, h, seed)


    fvalues = []
    normalized_fvalues = []
    costs =[]
    num_evals = 0

    if not isinstance(opt, Optimizer):
        opt = opt(parametrization=d, budget=budget )
        for vec in population:
            cand = opt.parametrization.spawn_child(new_value=vec)
            fx = target(vec.reshape(1, -1))
            fvalues.append(fx[0])
            opt.tell(cand, fx)
            num_evals += 1

        fvalues = [np.min(fvalues)]
        costs.append(num_evals)
    if len(fvalues) > 0:
        f_0 = fvalues[0]
        normalized_fvalues.append(1.0)

    while num_evals < budget:
        X = opt.ask()

        Y = target(X.value.reshape(-1, d)) if not isinstance(X, np.ndarray) else target(X.reshape(-1, d))

        if np.any(np.isnan(Y)):
            print("NAN")
            return np.nan, fvalues + [np.nan], normalized_fvalues + [np.nan], costs + [np.nan], params

        fvalues.append(np.min(Y))
        if len(fvalues) ==1:
            f_0 = fvalues[0]
            normalized_fvalues.append(1.0)
        else:
            opt_gap = (fvalues[-1] - min_f) / (f_0 - min_f)
            if normalized_fvalues[-1] > opt_gap:
                normalized_fvalues.append(opt_gap)
            else:
                normalized_fvalues.append(normalized_fvalues[-1])

        opt.tell(X, Y)

        cost = 1 if not isinstance(opt, Optimizer) else  X.shape[0]
        num_evals += cost
        costs.append(cost)

    f_best = target(opt.recommend().value.reshape(1, -1))[0] if not isinstance(opt, Optimizer) else target(opt.recommend()[0].reshape(1, -1))[0]

    return f_best, fvalues, normalized_fvalues, costs, params




def run_experiment(args):
    optimizer_name = args.optimizer
    target_name = args.fun_name
    d = args.d
    h = args.h
    reg = args.reg
    budget = args.budget #* (d + 1)
    seed = args.seed
    num_workers = args.num_workers
    num_particles = args.num_particles
    reps = args.reps
    output_directory = f"{args.out_dir}/szlan_results/comparison/{target_name}_{reg}/{optimizer_name}"
    os.makedirs(output_directory + "/traces", exist_ok=True)
    np.random.seed(seed)

    param_grid = get_params_grid(optimizer_name, d, reps)

    filtered_param_grid = []

    for param in param_grid:
        param_string = "_".join([str(x) for x in param])
        if not os.path.exists(f"{output_directory}/traces/{optimizer_name}_{target_name}_{d}_{param_string}_trace.txt"):
            filtered_param_grid.append(param)
        else:
            print(f"Skipping configuration {param}")
    param_grid = filtered_param_grid

    target = get_objective_function(target_name, d=d, reg=reg, seed=seed)
    target.regularization = 0.0
    min_f = target(target.x_star)[0]
    target.regularization = reg
    min_f_reg = target(target.x_star)[0]
    print(f"Function minimum: {min_f}")

    
    pool_results = {}
    pool_best_val = (None, None)

    run_opt = partial(run_optimizer, optimizer_name=optimizer_name, target=target, budget=budget, n=num_particles, d=d, h=h, seed=seed)
    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered( run_opt, param_grid):


            f_found, fvalues, normalized_best_fvalues, costs, params = result

            rep = params[-1]
            param_str = "_".join([str(x) for x in params[:-1]])
            if param_str not in pool_results:
                pool_results[param_str] = [(f_found,  fvalues, normalized_best_fvalues, costs, params)]
            elif len(pool_results[param_str]) < reps:
                pool_results[param_str].append((f_found,  fvalues, normalized_best_fvalues, costs, params))
                if len(pool_results[param_str]) == reps:



                    out_file_trace = f"{optimizer_name}_{target_name}_{d}_" + param_str + f"_{num_particles}_trace.txt"



                    mu_ffound = np.mean([x[0] for x in pool_results[param_str]])
                    std_ffound = np.std([x[0] for x in pool_results[param_str]])



                    if pool_best_val[0] is None or mu_ffound < pool_best_val[1]:
                        old_best_str = pool_best_val[0]
                        values = []
                        costs = []
                        norm_opt_gaps = []
                        for ris in pool_results[param_str]:
                            values.append(ris[1])
                            norm_opt_gaps.append(ris[2])
                            costs.append(ris[3])

                        min_len = min([len(val) for val in values])
                        if np.any([len(val) != min_len for val in values]):
                            continue

                        mu_values = np.mean(values, axis=0)
                        std_values = np.std(values, axis=0)

                        mu_normalized_best_fvalues = np.mean(norm_opt_gaps, axis=0)
                        std_normalized_best_fvalues = np.std(norm_opt_gaps, axis=0)

                        mu_costs = np.mean(costs, axis=0)
                        std_costs = np.std(costs, axis=0)
                        print(f"[{optimizer_name}] {params} f found: {f_found}\trep: {rep}\tmin f: {min_f}")

                        if old_best_str is not None:
                            print(old_best_str)
                            os.remove(f"{output_directory}/traces/{optimizer_name}_{target_name}_{d}_" + old_best_str + f"_{num_particles}_trace.txt")
                        with open(f"{output_directory}/traces/{out_file_trace}", "a") as f:
                            for i in range(len(mu_values)):
                                f.write(f"{mu_values[i]},{std_values[i]},{mu_values[0]},{std_values[0]},{mu_normalized_best_fvalues[i]},{std_normalized_best_fvalues[i]},{mu_costs[i]},{std_costs[i]},{min_f_reg},{min_f}\n")
                                f.flush()
                        pool_best_val = (param_str, mu_ffound)

    




if __name__ == "__main__":
    parser = ap.ArgumentParser()
    
    # Experiment Parameters
    parser.add_argument("optimizer", type=str, default="szlan", choices=["szlan", 'fd_gaus', 'fd_sph', 'fd_orth', "cmaes", "de_2p", "pso", 'cbo'], help="Optimizer to use")
    parser.add_argument("fun_name", type=str, default="Rosenbrock", choices=["Rosenbrock", "Griewank", "Ackley", "Levy", "StyblinkskiTang", "Rastrigin", "Quing"], help="Objective function to optimize")
    parser.add_argument("--d", type=int, default=10, help="Dimension of the problem")
    parser.add_argument('--budget', type=int, default=1000, help='Function evaluation budget')
    parser.add_argument('--reps', type=int, default=10, help='Number of repetitions for the experiment')
    parser.add_argument('--reg', type=float, default=0.0, help='Regularization parameter')

    # Optimizer Parameters
    parser.add_argument('--num_particles', type=int, default=10, help='Number of particles')
    parser.add_argument('--h', type=float, default=1e-7, help='Finite difference step size')

    # General Parameters
    parser.add_argument('--num-workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--out-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    
    args = parser.parse_args()
    run_experiment(args)
    
    